#!/usr/bin/env python3
"""
E1: Analysis module — parse nsys/ncu output, classify kernels, compute metrics.

All performance data comes from nsys/ncu CSV exports. No wall-time, no CUPTI.
"""

import csv
import io
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Kernel category classification ──────────────────────────────────────

# Ordered by priority: first-match wins
KERNEL_CATEGORY_RULES = [
    ("attention_fa2",       re.compile(r"flash_attn|flash_fwd|flash_bwd", re.I)),
    ("attention_sdpa",      re.compile(r"scaled_dot_product|sdpa|attention_forward", re.I)),
    ("attention_eager",     re.compile(r"attention.*eager|_attention", re.I)),
    ("gemm",                re.compile(r"gemm|gemv|matmul|linear|wgrad|dgrad|convolution|volta_|ampere_|turing_",
                                       re.I)),
    ("norm",                re.compile(r"rms_norm|layer_norm|group_norm|batch_norm|normalization", re.I)),
    ("softmax",             re.compile(r"softmax", re.I)),
    ("activation",          re.compile(r"silu|gelu|relu|swish|activation", re.I)),
    ("reduction",           re.compile(r"reduce|sum_kernel|mean_kernel|argmax|argmin", re.I)),
    ("elementwise",         re.compile(r"mul_|add_|sub_|div_|copy_|fill_|zero_|set_|abs_|neg_|clip",
                                       re.I)),
    ("pruning_cosine",      re.compile(r"cosine_similarity|cosine", re.I)),
    ("pruning_index",       re.compile(r"index_select|index_put|index_add|gather|scatter|take|where",
                                       re.I)),
    ("pruning_permute",     re.compile(r"permute|transpose|reshape|view|unsqueeze|squeeze|flatten",
                                       re.I)),
    ("pruning_topk",        re.compile(r"topk|sort|kthvalue|argsort", re.I)),
    ("pruning_mask",        re.compile(r"masked|mask_fill|nonzero", re.I)),
    ("pruning_other",       re.compile(r"pruning|prune|visi", re.I)),
    ("other",               re.compile(r".")),  # catch-all
]

# Category metadata for charting
CATEGORY_META = {
    "attention_fa2":       {"display": "Attention (FA2)",       "color": "#2ecc71"},
    "attention_sdpa":      {"display": "Attention (SDPA)",      "color": "#27ae60"},
    "attention_eager":     {"display": "Attention (Eager)",     "color": "#1abc9c"},
    "gemm":                {"display": "GEMM / MatMul",         "color": "#e74c3c"},
    "norm":                {"display": "Normalization",         "color": "#f39c12"},
    "softmax":             {"display": "Softmax",               "color": "#e67e22"},
    "activation":          {"display": "Activation",            "color": "#d35400"},
    "reduction":           {"display": "Reduction",             "color": "#9b59b6"},
    "elementwise":         {"display": "Elementwise",           "color": "#8e44ad"},
    "pruning_cosine":      {"display": "Pruning: Cosine Sim",   "color": "#3498db"},
    "pruning_index":       {"display": "Pruning: Index/Gather", "color": "#2980b9"},
    "pruning_permute":     {"display": "Pruning: Permute/View", "color": "#16a085"},
    "pruning_topk":        {"display": "Pruning: TopK/Sort",    "color": "#f1c40f"},
    "pruning_mask":        {"display": "Pruning: Mask/Fill",    "color": "#7f8c8d"},
    "pruning_other":       {"display": "Pruning: Other",        "color": "#95a5a6"},
    "other":               {"display": "Other",                 "color": "#bdc3c7"},
}


def categorize_kernel(name: str) -> str:
    """Classify a kernel name into a category. First-match wins."""
    for cat, pat in KERNEL_CATEGORY_RULES:
        if pat.search(name):
            return cat
    return "other"


# ── nsys CSV parsing ────────────────────────────────────────────────────

def parse_nsys_kernel_csv(csv_path: str) -> list[dict]:
    """Parse nsys `cuda_gpu_kern_sum` CSV export.

    Expected columns: Time(%), Total Time(ns), Instances, Avg(ns), Med(ns),
                      Min(ns), Max(ns), StdDev(ns), Name

    Returns list of dicts with keys:
        name, time_pct, total_time_ns, instances, avg_ns, med_ns, min_ns,
        max_ns, stddev_ns, category
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        # Detect if there's a leading metadata block (nsys prepends sometimes)
        # Read first line to check
        first = f.readline()
        f.seek(0)

        if first.lstrip().startswith('"') or first.lstrip().startswith('Time'):
            # CSV starts immediately
            reader = csv.DictReader(f)
        else:
            # Skip metadata lines until we find the header
            for line in f:
                if line.startswith('"Time') or line.startswith('Time'):
                    # We found the header; need to parse from here
                    # Use the remaining content as CSV
                    remaining = line + f.read()
                    reader = csv.DictReader(io.StringIO(remaining))
                    break
            else:
                # No header found
                print(f"[e1_analysis] WARNING: No CSV header found in {csv_path}",
                      flush=True)
                return []

        for row in reader:
            # Normalize column names (nsys may use different quoting).
            # "Total Time (ns)" → collapse to "total_time_ns"
            clean = {}
            for k, v in row.items():
                raw = k.strip().strip('"').lower()
                raw = raw.replace(" ", "_").replace("%", "pct")
                raw = raw.replace("(", "").replace(")", "")
                # Collapse multiple underscores to one, strip leading/trailing
                import re as _re
                raw = _re.sub(r'_+', '_', raw).strip('_')
                clean[raw] = v

            try:
                total_time_ns = float(clean.get("total_time_ns", clean.get("total_time", 0)))
                time_pct = float(clean.get("time_pct", clean.get("time", 0)))
                instances = int(float(clean.get("instances", 0)))
                kernel_name = clean.get("name", row.get("Name", "")).strip('"')
            except (ValueError, KeyError):
                continue

            rows.append({
                "name": kernel_name,
                "time_pct": time_pct,
                "total_time_ns": total_time_ns,
                "instances": instances,
                "avg_ns": _safe_float(clean.get("avg_ns", clean.get("avg", 0))),
                "med_ns": _safe_float(clean.get("med_ns", clean.get("med", 0))),
                "min_ns": _safe_float(clean.get("min_ns", clean.get("min", 0))),
                "max_ns": _safe_float(clean.get("max_ns", clean.get("max", 0))),
                "stddev_ns": _safe_float(clean.get("stddev_ns", clean.get("stddev", 0))),
                "category": categorize_kernel(kernel_name),
            })

    return rows


def _safe_float(val):
    """Parse float, returning 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ── nsys analysis ───────────────────────────────────────────────────────

def extract_top_kernels(kernels: list[dict], n: int = 10) -> list[dict]:
    """Return top-N kernels sorted by total_time_ns descending."""
    sorted_kernels = sorted(kernels, key=lambda k: k["total_time_ns"], reverse=True)
    return sorted_kernels[:n]


def compute_total_gpu_time(kernels: list[dict]) -> float:
    """Sum of all kernel total GPU times (ns)."""
    return sum(k["total_time_ns"] for k in kernels)


def aggregate_by_category(kernels: list[dict]) -> list[dict]:
    """Aggregate GPU time by kernel category.

    Returns list of dicts: category, total_time_ns, time_pct, display, color
    """
    agg = defaultdict(lambda: {"total_time_ns": 0.0, "instances": 0})
    total = compute_total_gpu_time(kernels)

    for k in kernels:
        cat = k["category"]
        agg[cat]["total_time_ns"] += k["total_time_ns"]
        agg[cat]["instances"] += k["instances"]

    result = []
    for cat, data in sorted(agg.items(), key=lambda x: x[1]["total_time_ns"], reverse=True):
        meta = CATEGORY_META.get(cat, CATEGORY_META["other"])
        result.append({
            "category": cat,
            "display": meta["display"],
            "color": meta["color"],
            "total_time_ns": data["total_time_ns"],
            "time_pct": (data["total_time_ns"] / total * 100) if total > 0 else 0.0,
            "instances": data["instances"],
        })

    return result


# ── ncu CSV parsing ─────────────────────────────────────────────────────

def parse_ncu_csv(csv_path: str) -> list[dict]:
    """Parse ncu `--csv` output.

    ncu --csv output has a multi-line format:
      - Header lines starting with '"ID","Kernel Name",...'
      - One row per kernel launch
      - Multiple sections possible

    Returns list of dicts with raw ncu metrics.
    """
    rows = []
    try:
        with open(csv_path, "r", newline="", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[e1_analysis] ERROR: ncu CSV not found: {csv_path}", flush=True)
        return []

    # ncu CSV has header lines interspersed with data
    # We find the first header line and parse from there
    lines = content.split("\n")

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') and '"Kernel Name"' in line:
            header_idx = i
            break

    if header_idx is None:
        print(f"[e1_analysis] WARNING: No ncu CSV header found in {csv_path}",
              flush=True)
        return []

    # Parse from header
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    for row in reader:
        # Keep raw row; bottleneck classification happens later
        if not row.get("Kernel Name", "").strip():
            continue
        rows.append(row)

    return rows


# ── ncu bottleneck classification ───────────────────────────────────────

# Key ncu metrics we care about (vary by GPU architecture)
# Common metric names for Ada Lovelace (RTX 4090):
NCU_COMPUTE_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_elapsed",
]
NCU_MEMORY_METRICS = [
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]
NCU_CACHE_METRICS = [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_read.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
]


def classify_bottleneck(metrics: dict) -> dict:
    """Classify a kernel's bottleneck type from ncu metrics.

    Args:
        metrics: dict of metric_name → float value

    Returns:
        dict with keys: bottleneck_class, u_compute, u_bw, o_sm, confidence
    """
    u_compute = _get_metric(metrics, NCU_COMPUTE_METRICS)
    u_bw = _get_metric(metrics, NCU_MEMORY_METRICS[:1])  # dram__throughput
    o_sm = _get_metric(metrics, ["smsp__warps_active.avg.pct_of_peak_sustained_elapsed"])

    if u_compute > 60 and u_bw < 60:
        bottleneck = "compute-bound"
        confidence = "high" if u_compute > 70 else "medium"
    elif u_bw > 60 and u_compute < 60:
        bottleneck = "memory-bound"
        confidence = "high" if u_bw > 70 else "medium"
    elif o_sm < 60 and u_bw < 30 and u_compute < 30:
        bottleneck = "latency-bound"
        confidence = "medium"
    else:
        # Mixed or undetermined
        if u_compute >= u_bw:
            bottleneck = "compute-bound"
        else:
            bottleneck = "memory-bound"
        confidence = "low"

    return {
        "bottleneck_class": bottleneck,
        "u_compute_pct": round(u_compute, 2),
        "u_bw_pct": round(u_bw, 2),
        "o_sm_pct": round(o_sm, 2),
        "confidence": confidence,
    }


def _get_metric(metrics: dict, candidates: list[str]) -> float:
    """Get first matching metric from ncu row. Returns 0.0 if none found."""
    # Try exact match first
    for c in candidates:
        if c in metrics:
            return _safe_float(metrics[c])
    # Try partial match (ncu CSV column names may be abbreviated)
    for c in candidates:
        for k, v in metrics.items():
            if c in k or k in c:
                return _safe_float(v)
    return 0.0


def compute_cache_metrics(ncu_rows: list[dict]) -> dict:
    """Compute aggregate cache-level metrics from ncu data.

    Returns:
        dict with L1, L2, DRAM byte totals and estimated time percentages
    """
    l1_read = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                  for k in r if "l1tex" in k.lower() and "read" in k.lower())
    l1_write = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                   for k in r if "l1tex" in k.lower() and "write" in k.lower())
    l2_read = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                  for k in r if "lts" in k.lower() and "read" in k.lower())
    l2_write = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                   for k in r if "lts" in k.lower() and "write" in k.lower())
    dram_read = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                    for k in r if "dram__bytes_read" in k.lower())
    dram_write = sum(_safe_float(r.get(k, 0)) for r in ncu_rows
                     for k in r if "dram__bytes_write" in k.lower())

    total_bytes = l1_read + l1_write + l2_read + l2_write + dram_read + dram_write

    return {
        "l1_bytes_read": l1_read,
        "l1_bytes_write": l1_write,
        "l2_bytes_read": l2_read,
        "l2_bytes_write": l2_write,
        "dram_bytes_read": dram_read,
        "dram_bytes_write": dram_write,
        "l1_pct": ((l1_read + l1_write) / total_bytes * 100) if total_bytes > 0 else 0,
        "l2_pct": ((l2_read + l2_write) / total_bytes * 100) if total_bytes > 0 else 0,
        "dram_pct": ((dram_read + dram_write) / total_bytes * 100) if total_bytes > 0 else 0,
    }


# ── Compute actual speedup ───────────────────────────────────────────────

def compute_speedup(nsys_data: dict[str, list[dict]],
                    baseline_config: str = "dense-fa2",
                    target_config: str = "visipruner-full") -> dict:
    """Compute actual GPU speedup from nsys data.

    actual_speedup = T_baseline / T_target

    Args:
        nsys_data: dict mapping config_name → list of kernel dicts
        baseline_config: config to use as baseline (denominator)
        target_config: config to compare (numerator denominator)

    Returns:
        dict with speedup, t_baseline_ms, t_target_ms, gap
    """
    t_baseline = compute_total_gpu_time(nsys_data.get(baseline_config, []))
    t_target = compute_total_gpu_time(nsys_data.get(target_config, []))

    if t_target <= 0:
        return {
            "baseline_config": baseline_config,
            "target_config": target_config,
            "t_baseline_ns": t_baseline,
            "t_target_ns": t_target,
            "t_baseline_ms": round(t_baseline / 1e6, 3),
            "t_target_ms": round(t_target / 1e6, 3),
            "actual_speedup": 1.0,
            "theoretical_speedup": 2.17,
            "gap": 2.17 - 1.0,
            "error": "t_target is zero or negative",
        }

    actual = t_baseline / t_target

    return {
        "baseline_config": baseline_config,
        "target_config": target_config,
        "t_baseline_ns": t_baseline,
        "t_target_ns": t_target,
        "t_baseline_ms": round(t_baseline / 1e6, 3),
        "t_target_ms": round(t_target / 1e6, 3),
        "actual_speedup": round(actual, 3),
        "theoretical_speedup": 2.17,
        "gap": round(2.17 - actual, 3),
    }


def _sqlite_table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _sqlite_duration_stats(con: sqlite3.Connection, table_name: str) -> dict:
    """Return count/sum/span duration stats for an nsys SQLite activity table."""
    if not _sqlite_table_exists(con, table_name):
        return {"count": 0, "sum_ms": 0.0, "span_ms": 0.0}

    count, total, span = con.execute(
        f"select count(*), coalesce(sum(end-start), 0), "
        f"coalesce(max(end)-min(start), 0) from {table_name}"
    ).fetchone()
    return {
        "count": int(count or 0),
        "sum_ms": round(float(total or 0) / 1e6, 3),
        "span_ms": round(float(span or 0) / 1e6, 3),
    }


def extract_nsys_timing_summary(sqlite_path: str,
                                kernels: Optional[list[dict]] = None,
                                nvtx_name: str = "e1_inference") -> dict:
    """Extract comparable timing metrics from an nsys SQLite export.

    cuda_gpu_kern_sum is useful for kernel attribution, but it is not
    end-to-end latency. This function also extracts the NVTX inference range
    duration and CUDA runtime spans so summaries can report the metric source
    explicitly.
    """
    summary = {
        "sqlite_path": sqlite_path,
        "nvtx_name": nvtx_name,
        "gpu_kernel_sum_ms": round(
            compute_total_gpu_time(kernels or []) / 1e6, 3
        ),
        "gpu_kernel_unique_count": len(kernels or []),
        "gpu_kernel_launch_count": 0,
        "gpu_kernel_span_ms": 0.0,
        "nvtx_inference_ms": 0.0,
        "nvtx_range_count": 0,
        "cuda_runtime_sum_ms": 0.0,
        "cuda_runtime_span_ms": 0.0,
        "memcpy_sum_ms": 0.0,
        "memset_sum_ms": 0.0,
        "sync_sum_ms": 0.0,
        "profiler_overhead_sum_ms": 0.0,
    }

    if not os.path.exists(sqlite_path):
        summary["error"] = f"sqlite not found: {sqlite_path}"
        return summary

    try:
        con = sqlite3.connect(sqlite_path)
    except sqlite3.Error as e:
        summary["error"] = f"cannot open sqlite: {e}"
        return summary

    try:
        kstats = _sqlite_duration_stats(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        summary["gpu_kernel_launch_count"] = kstats["count"]
        if summary["gpu_kernel_sum_ms"] <= 0:
            summary["gpu_kernel_sum_ms"] = kstats["sum_ms"]
        summary["gpu_kernel_span_ms"] = kstats["span_ms"]

        rstats = _sqlite_duration_stats(con, "CUPTI_ACTIVITY_KIND_RUNTIME")
        summary["cuda_runtime_sum_ms"] = rstats["sum_ms"]
        summary["cuda_runtime_span_ms"] = rstats["span_ms"]

        for table, key in [
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy_sum_ms"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset_sum_ms"),
            ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "sync_sum_ms"),
            ("CUPTI_ACTIVITY_KIND_OVERHEAD", "profiler_overhead_sum_ms"),
        ]:
            summary[key] = _sqlite_duration_stats(con, table)["sum_ms"]

        if _sqlite_table_exists(con, "NVTX_EVENTS"):
            # Most nsys exports store NVTX text through StringIds, but keep
            # fallback columns for compatibility with older exports.
            query = """
                select coalesce(max(n.end - n.start), 0), count(*)
                from NVTX_EVENTS n
                left join StringIds s on n.textId = s.id
                where coalesce(s.value, n.text, n.jsonText) = ?
                  and n.end > n.start
            """
            nvtx_ms, nvtx_count = con.execute(query, (nvtx_name,)).fetchone()
            summary["nvtx_inference_ms"] = round(float(nvtx_ms or 0) / 1e6, 3)
            summary["nvtx_range_count"] = int(nvtx_count or 0)
    except sqlite3.Error as e:
        summary["error"] = f"sqlite query failed: {e}"
    finally:
        con.close()

    return summary


def compute_timing_speedup(timing_data: dict[str, dict],
                           metric: str,
                           baseline_config: str = "dense-fa2",
                           target_config: str = "visipruner-full") -> dict:
    """Compute speedup from a named metric in nsys timing summaries."""
    baseline = timing_data.get(baseline_config, {})
    target = timing_data.get(target_config, {})
    t_baseline = _safe_float(baseline.get(metric, 0))
    t_target = _safe_float(target.get(metric, 0))

    result = {
        "metric": metric,
        "baseline_config": baseline_config,
        "target_config": target_config,
        "t_baseline_ms": round(t_baseline, 3),
        "t_target_ms": round(t_target, 3),
        "theoretical_speedup": 2.17,
    }

    if t_baseline <= 0 or t_target <= 0:
        return {
            **result,
            "actual_speedup": 0.0,
            "gap": 2.17,
            "error": f"non-positive timing: baseline={t_baseline}, target={t_target}",
        }

    actual = t_baseline / t_target
    return {
        **result,
        "actual_speedup": round(actual, 3),
        "gap": round(2.17 - actual, 3),
    }


# ── Report writing ──────────────────────────────────────────────────────

def write_top_kernels_csv(kernels: list[dict], output_path: str, n: int = 10):
    """Write top-N kernels to CSV."""
    top = extract_top_kernels(kernels, n)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "name", "category", "total_time_ns", "time_pct",
            "instances", "avg_ns", "med_ns", "min_ns", "max_ns", "stddev_ns",
        ])
        writer.writeheader()
        for i, k in enumerate(top, 1):
            writer.writerow({**k, "rank": i, "total_time_ns": int(k["total_time_ns"])})


def write_kernel_categories_csv(categories: list[dict], output_path: str):
    """Write kernel category aggregation to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "display", "total_time_ns", "time_pct", "instances", "color",
        ])
        writer.writeheader()
        for c in categories:
            writer.writerow({**c, "total_time_ns": int(c["total_time_ns"])})


def write_ncu_bottleneck_csv(results: list[dict], output_path: str):
    """Write ncu per-kernel bottleneck analysis to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "kernel_name", "bottleneck_class", "u_compute_pct",
            "u_bw_pct", "o_sm_pct", "confidence",
        ])
        for r in results:
            writer.writerow([
                r.get("kernel_name", ""),
                r.get("bottleneck_class", ""),
                r.get("u_compute_pct", 0),
                r.get("u_bw_pct", 0),
                r.get("o_sm_pct", 0),
                r.get("confidence", ""),
            ])


def write_gap_summary_json(summary: dict, output_path: str):
    """Write final gap summary to JSON."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


# ── Hotspot kernel selection ─────────────────────────────────────────────

def select_hotspot_kernels(kernels: list[dict],
                           min_time_pct: float = 3.0,
                           max_kernels: int = 5) -> list[str]:
    """Select hotspot kernel patterns for ncu analysis.

    Criteria:
      1. Single kernel GPU time percentage > min_time_pct
      2. Take up to max_kernels kernels

    Returns list of kernel name regex patterns for ncu --kernel-name.
    """
    sorted_kernels = sorted(kernels, key=lambda k: k["total_time_ns"], reverse=True)

    selected = []
    for k in sorted_kernels:
        if k["time_pct"] < min_time_pct:
            break
        if len(selected) >= max_kernels:
            break
        # Escape regex-special chars and use as literal pattern
        name = k["name"]
        # Take the base kernel name (before template params)
        base_name = name.split("<")[0].split("(")[0].strip()
        if base_name and base_name not in selected:
            selected.append(base_name)

    return selected


# ── Config-specific kernel patterns for ncu ─────────────────────────────

# Fallback patterns when nsys data isn't available yet
FALLBACK_NCU_PATTERNS = {
    "dense-fa2": [
        "flash_attn",
        "matmul",
        "gemm",
        "silu",
        "rms_norm",
    ],
    "dense-eager": [
        "scaled_dot_product",
        "matmul",
        "gemm",
        "silu",
        "rms_norm",
    ],
    "visipruner-full": [
        "matmul",
        "cosine_similarity",
        "index",
        "permute",
        "topk",
        "scaled_dot_product",
    ],
    "visipruner-shallow-only": [
        "matmul",
        "cosine_similarity",
        "index",
        "scaled_dot_product",
    ],
}
