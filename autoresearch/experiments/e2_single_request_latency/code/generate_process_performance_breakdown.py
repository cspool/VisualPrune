#!/usr/bin/env python3
"""Generate strict FX-process performance breakdowns from process NVTX traces.

The previous layer-total-to-FX-process projection workflow is 被抛弃或不需要使用.
This script attributes GPU kernels only by launch ownership:

process NVTX CPU range -> CUDA runtime call started inside the range
-> CUPTI kernel with matching correlationId.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path("autoresearch/experiments/e2_single_request_latency")
OUTPUT_DIR = EXPERIMENT_DIR / "output"
DEFAULT_EAGER_PACKAGE_DIR = OUTPUT_DIR / "visipruner_full_eager_layer_wise"
DEFAULT_PROCESS_PACKAGE_DIR = OUTPUT_DIR / "visipruner_full_eager_process_wise"
DEFAULT_SQLITE = DEFAULT_EAGER_PACKAGE_DIR / "nsys_sameinput_visipruner_full_eager_32tok.sqlite"
DEFAULT_LAYER_EVENTS = DEFAULT_EAGER_PACKAGE_DIR / "nsys_sameinput_visipruner_full_eager_32tok_layer_events.csv"
DEFAULT_HANDOFF = DEFAULT_PROCESS_PACKAGE_DIR / "FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md"
DEFAULT_DETAIL_CSV = (
    DEFAULT_PROCESS_PACKAGE_DIR / "nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv"
)
DEFAULT_DETAIL_JSON = (
    DEFAULT_PROCESS_PACKAGE_DIR / "nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json"
)
DEFAULT_PROCESS_CSV = DEFAULT_PROCESS_PACKAGE_DIR / "same_input_visipruner_full_eager_process_attribution.csv"
DEFAULT_REPORT = DEFAULT_PROCESS_PACKAGE_DIR / "SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md"
DEFAULT_AGGREGATE_REPORT = DEFAULT_PROCESS_PACKAGE_DIR / "SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md"

PROCESS_RE = re.compile(
    r"^visprune\.fx_process\.layer(?P<layer>\d+)\."
    r"(?P<phase>prefill|decode)\.fwd(?P<forward_id>\d+)\."
    r"(?P<event>event(?:\d+|_unknown))\."
    r"(?P<process_id>[^.]+)\.(?P<slug>[^.]+)"
    r"(?:\.part(?P<part>\d+))?$"
)
LAYER_RE = re.compile(r"^visprune\.layer(?P<layer>\d+)\.(?P<phase>prefill|decode)$")


@dataclass(frozen=True)
class NvtxRange:
    text: str
    start: int
    end: int
    global_tid: int | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) / 1e6


@dataclass(frozen=True)
class RuntimeCall:
    start: int
    end: int
    correlation_id: int | None
    name: str


@dataclass(frozen=True)
class Kernel:
    start: int
    end: int
    correlation_id: int | None
    name: str

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) / 1e6


@dataclass
class LaunchOwnedMetrics:
    cupti_ms: float = 0.0
    runtime_api_calls: int = 0
    kernel_instances: int = 0
    first_runtime_start_ns: int | None = None
    first_kernel_start_ns: int | None = None
    last_kernel_end_ns: int | None = None
    families: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    family_instances: Counter[str] = field(default_factory=Counter)
    kernel_names: Counter[str] = field(default_factory=Counter)

    @property
    def matched_kernel_families(self) -> list[str]:
        return sorted(family for family, value in self.families.items() if value > 0.0)

    @property
    def dominant_kernel_family(self) -> str:
        if not self.families:
            return "none"
        return max(self.families, key=lambda name: self.families[name])


def clean_cell(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        value = value[1:-1]
    return value.replace("<br>", "; ").strip()


def split_list(value: str) -> list[str]:
    cleaned = clean_cell(value)
    if not cleaned or cleaned in {"same", "-"}:
        return []
    parts = re.split(r",|;", cleaned)
    return [part.strip(" `") for part in parts if part.strip(" `")]


def kernel_family(name: str) -> str:
    lowered = (name or "").lower()
    if "_vp_fa_prefill_kernel" in lowered or "_vp_fa_last_query_weights_kernel" in lowered:
        return "triton_vp_fa"
    if "triton" in lowered:
        return "triton_other"
    if "gemvx::kernel" in lowered or "gemvnsp_kernel" in lowered or "gemv2t_kernel" in lowered:
        return "gemv_decode_cublas"
    if "flash::flash" in lowered:
        return "flash_attention"
    if "gemm" in lowered or "cutlass::kernel" in lowered or "cudnn" in lowered:
        return "gemm_tensorcore"
    if (
        "catarraybatchedcopy" in lowered
        or "copy_kernel" in lowered
        or "indexselect" in lowered
        or "index_select" in lowered
        or "gather" in lowered
    ):
        return "copy_gather_cat"
    if "cub::" in lowered or "reduce" in lowered or "scan" in lowered or "select" in lowered:
        return "selection_reduce_scan"
    if "softmax" in lowered:
        return "softmax"
    if (
        "elementwise" in lowered
        or "silu" in lowered
        or "sigmoid" in lowered
        or "rsqrt" in lowered
        or "pow" in lowered
        or "layer_norm" in lowered
        or "rms" in lowered
    ):
        return "elementwise_norm_activation"
    return "other"


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def format_pct(value: float, denom: float) -> str:
    if denom <= 0.0:
        return "n/a"
    return f"{(value / denom) * 100.0:.2f}%"


def parse_handoff_inventory(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    in_inventory = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "## Process Range Inventory":
            in_inventory = True
            continue
        if in_inventory and line.startswith("## ") and line != "## Process Range Inventory":
            break
        if not in_inventory or not line.startswith("|"):
            continue
        cells = [clean_cell(cell) for cell in line.strip("|").split("|")]
        if cells and cells[0] == "variant_scope":
            headers = cells
            continue
        if not headers or not cells or set(cells[0]) <= {"-"}:
            continue
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        if row.get("process_id"):
            rows.append(row)

    by_process_fragment: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        fragment_id = row.get("fragment_id", "whole")
        by_process_fragment[(row["process_id"], fragment_id)] = row
    return rows, by_process_fragment


def require_tables(conn: sqlite3.Connection, names: list[str]) -> None:
    available = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    missing = [name for name in names if name not in available]
    if missing:
        raise SystemExit(f"Missing Nsight sqlite tables: {', '.join(missing)}")


def read_nvtx_ranges(conn: sqlite3.Connection, like_pattern: str) -> list[NvtxRange]:
    rows = conn.execute(
        """
        SELECT n.start, n.end, n.globalTid,
               COALESCE(s.value, n.text, js.value, n.jsonText) AS text
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        LEFT JOIN StringIds AS js ON n.jsonTextId = js.id
        WHERE COALESCE(s.value, n.text, js.value, n.jsonText) LIKE ?
          AND n.end IS NOT NULL
        ORDER BY n.start
        """,
        (like_pattern,),
    ).fetchall()
    return [
        NvtxRange(
            text=row["text"],
            start=int(row["start"]),
            end=int(row["end"]),
            global_tid=int(row["globalTid"]) if row["globalTid"] is not None else None,
        )
        for row in rows
        if row["text"]
    ]


def read_runtime_calls(conn: sqlite3.Connection) -> list[RuntimeCall]:
    rows = conn.execute(
        """
        SELECT r.start, r.end, r.correlationId, COALESCE(s.value, '') AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS s ON r.nameId = s.id
        WHERE r.correlationId IS NOT NULL
        ORDER BY r.start
        """
    ).fetchall()
    return [
        RuntimeCall(
            start=int(row["start"]),
            end=int(row["end"]),
            correlation_id=int(row["correlationId"]) if row["correlationId"] is not None else None,
            name=row["name"] or "",
        )
        for row in rows
    ]


def read_kernels(conn: sqlite3.Connection) -> list[Kernel]:
    rows = conn.execute(
        """
        SELECT k.start, k.end, k.correlationId,
               COALESCE(d.value, s.value, m.value, '') AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS d ON k.demangledName = d.id
        LEFT JOIN StringIds AS s ON k.shortName = s.id
        LEFT JOIN StringIds AS m ON k.mangledName = m.id
        ORDER BY k.start
        """
    ).fetchall()
    return [
        Kernel(
            start=int(row["start"]),
            end=int(row["end"]),
            correlation_id=int(row["correlationId"]) if row["correlationId"] is not None else None,
            name=row["name"] or "",
        )
        for row in rows
    ]


def launch_owned_metrics(
    nvtx_range: NvtxRange,
    runtime_calls: list[RuntimeCall],
    runtime_starts: list[int],
    kernels_by_correlation: dict[int, list[Kernel]],
) -> LaunchOwnedMetrics:
    metrics = LaunchOwnedMetrics()
    seen_correlations: set[int] = set()
    start_idx = bisect_left(runtime_starts, nvtx_range.start)
    for runtime_call in runtime_calls[start_idx:]:
        if runtime_call.start >= nvtx_range.end:
            break
        metrics.runtime_api_calls += 1
        if metrics.first_runtime_start_ns is None:
            metrics.first_runtime_start_ns = runtime_call.start
        if runtime_call.correlation_id is None:
            continue
        correlation_id = runtime_call.correlation_id
        if correlation_id in seen_correlations:
            continue
        seen_correlations.add(correlation_id)
        for kernel in kernels_by_correlation.get(correlation_id, []):
            family = kernel_family(kernel.name)
            if metrics.first_kernel_start_ns is None or kernel.start < metrics.first_kernel_start_ns:
                metrics.first_kernel_start_ns = kernel.start
            if metrics.last_kernel_end_ns is None or kernel.end > metrics.last_kernel_end_ns:
                metrics.last_kernel_end_ns = kernel.end
            metrics.cupti_ms += kernel.duration_ms
            metrics.kernel_instances += 1
            metrics.families[family] += kernel.duration_ms
            metrics.family_instances[family] += 1
            metrics.kernel_names[kernel.name] += 1
    return metrics


def order_key_for_range(nvtx_range: NvtxRange, metrics: LaunchOwnedMetrics) -> tuple[int, str]:
    if metrics.first_kernel_start_ns is not None:
        return metrics.first_kernel_start_ns, "first_launch_owned_gpu_kernel_start"
    if metrics.first_runtime_start_ns is not None:
        return metrics.first_runtime_start_ns, "first_runtime_call_start"
    return nvtx_range.start, "process_nvtx_start"


def parse_process_range(range_text: str) -> dict[str, Any] | None:
    match = PROCESS_RE.match(range_text)
    if not match:
        return None
    part = match.group("part")
    fragment_id = f"part{int(part):02d}" if part is not None else "whole"
    return {
        "layer": int(match.group("layer")),
        "phase": match.group("phase"),
        "forward_id": int(match.group("forward_id")),
        "event": match.group("event"),
        "process_id": match.group("process_id"),
        "slug": match.group("slug"),
        "fragment_id": fragment_id,
    }


def find_parent_range(process_range: NvtxRange, layer_ranges: list[NvtxRange], layer: int, phase: str) -> NvtxRange | None:
    parent_text = f"visprune.layer{layer:02d}.{phase}"
    candidates = [
        rng
        for rng in layer_ranges
        if rng.text == parent_text and rng.start <= process_range.start and rng.end >= process_range.end
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda rng: rng.end - rng.start)


def load_layer_event_metadata(path: Path | None) -> dict[tuple[int, int], dict[str, str]]:
    if not path or not path.exists():
        return {}
    by_forward_layer: dict[tuple[int, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                forward_id = int(float(row.get("forward_id", "")))
                layer_idx = int(float(row.get("layer_idx", "")))
            except Exception:
                continue
            by_forward_layer[(forward_id, layer_idx)] = row
    return by_forward_layer


def validation_status(expected: list[str], matched: list[str], kernel_instances: int) -> str:
    if kernel_instances == 0:
        return "no_kernel"
    if not expected:
        return "partially_validated"
    expected_set = set(expected)
    matched_set = set(matched)
    if matched_set <= expected_set and matched_set:
        return "validated"
    if expected_set & matched_set:
        return "partially_validated"
    return "unexpected_kernel_family"


def combine_status(statuses: list[str]) -> str:
    if not statuses:
        return "unresolved"
    if all(status == "validated" for status in statuses):
        return "validated"
    if all(status == "no_kernel" for status in statuses):
        return "no_kernel"
    if any(status == "unexpected_kernel_family" for status in statuses):
        return "unexpected_kernel_family"
    if any(status == "unresolved" for status in statuses):
        return "unresolved"
    return "partially_validated"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def row_for_range(
    *,
    variant: str,
    parsed: dict[str, Any],
    nvtx_range: NvtxRange,
    metrics: LaunchOwnedMetrics,
    inventory: dict[str, str] | None,
    parent_range: NvtxRange | None,
    parent_metrics: LaunchOwnedMetrics | None,
    layer_event: dict[str, str] | None,
) -> dict[str, Any]:
    process_id = parsed["process_id"]
    fragment_id = parsed["fragment_id"]
    expected = split_list(inventory.get("expected_kernel_families", "") if inventory else "")
    matched = metrics.matched_kernel_families
    status = validation_status(expected, matched, metrics.kernel_instances)
    parent_cupti = parent_metrics.cupti_ms if parent_metrics else 0.0
    parent_nvtx = parent_range.duration_ms if parent_range else 0.0
    family_ms = "; ".join(
        f"{family}={value:.3f}" for family, value in sorted(metrics.families.items())
    )
    top_kernels = "; ".join(name for name, _ in metrics.kernel_names.most_common(5))
    notes = inventory.get("notes", "") if inventory else "No matching handoff inventory row."
    if status == "unexpected_kernel_family":
        notes = f"{notes} Matched families did not intersect expected families."
    if status == "no_kernel":
        notes = f"{notes} No launch-owned CUPTI kernel matched this NVTX range."
    gpu_order_key_ns, gpu_order_basis = order_key_for_range(nvtx_range, metrics)
    return {
        "variant": variant,
        "phase": parsed["phase"],
        "forward_id": parsed["forward_id"],
        "layer": parsed["layer"],
        "q_len": (layer_event or {}).get("q_len", ""),
        "kv_len": (layer_event or {}).get("kv_len", ""),
        "process_id": process_id,
        "process_title": inventory.get("process_title", process_id) if inventory else process_id,
        "fragment_id": fragment_id,
        "aggregation_key": inventory.get("aggregation_key", process_id) if inventory else process_id,
        "nvtx_range_name": nvtx_range.text,
        "fx_nodes": inventory.get("fx_nodes", "") if inventory else "",
        "fx_op_families": inventory.get("fx_op_families", "") if inventory else "",
        "expected_kernel_families": ", ".join(expected),
        "matched_kernel_families": ", ".join(matched),
        "CUPTI kernel ms": metrics.cupti_ms,
        "NVTX CPU ms": nvtx_range.duration_ms,
        "process_cupti_pct_in_parent": format_pct(metrics.cupti_ms, parent_cupti),
        "process_nvtx_pct_in_parent": format_pct(nvtx_range.duration_ms, parent_nvtx),
        "parent_layer_range": parent_range.text if parent_range else "missing",
        "parent_layer_CUPTI kernel ms": parent_cupti,
        "parent_layer_NVTX CPU ms": parent_nvtx,
        "runtime API calls": metrics.runtime_api_calls,
        "kernel instances": metrics.kernel_instances,
        "dominant kernel family": metrics.dominant_kernel_family,
        "kernel_family_ms": family_ms,
        "top_kernel_names": top_kernels,
        "process_nvtx_start_ns": nvtx_range.start,
        "process_nvtx_end_ns": nvtx_range.end,
        "first_runtime_start_ns": metrics.first_runtime_start_ns or "",
        "first_kernel_start_ns": metrics.first_kernel_start_ns or "",
        "last_kernel_end_ns": metrics.last_kernel_end_ns or "",
        "gpu_order_key_ns": gpu_order_key_ns,
        "gpu_order_basis": gpu_order_basis,
        "attribution method": "launch-owned correlationId",
        "validation status": status,
        "notes": notes,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["variant"],
            row["phase"],
            row["forward_id"],
            row["layer"],
            row["process_id"],
            row["aggregation_key"],
        )
        grouped[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, group in grouped.items():
        base = group[0]
        cupti = sum(float(item["CUPTI kernel ms"]) for item in group)
        nvtx = sum(float(item["NVTX CPU ms"]) for item in group)
        parent_cupti = float(base["parent_layer_CUPTI kernel ms"])
        parent_nvtx = float(base["parent_layer_NVTX CPU ms"])
        matched = sorted(
            {
                family.strip()
                for item in group
                for family in str(item["matched_kernel_families"]).split(",")
                if family.strip()
            }
        )
        expected = sorted(
            {
                family.strip()
                for item in group
                for family in str(item["expected_kernel_families"]).split(",")
                if family.strip()
            }
        )
        family_totals: dict[str, float] = defaultdict(float)
        for item in group:
            for entry in str(item["kernel_family_ms"]).split(";"):
                if "=" not in entry:
                    continue
                family, value = entry.split("=", 1)
                try:
                    family_totals[family.strip()] += float(value)
                except ValueError:
                    pass
        dominant = max(family_totals, key=family_totals.get) if family_totals else "none"
        fragment_ids = ", ".join(sorted(str(item["fragment_id"]) for item in group))
        order_values = [int(item["gpu_order_key_ns"]) for item in group if str(item.get("gpu_order_key_ns", "")).strip()]
        first_runtime_values = [
            int(item["first_runtime_start_ns"]) for item in group if str(item.get("first_runtime_start_ns", "")).strip()
        ]
        first_kernel_values = [
            int(item["first_kernel_start_ns"]) for item in group if str(item.get("first_kernel_start_ns", "")).strip()
        ]
        last_kernel_values = [
            int(item["last_kernel_end_ns"]) for item in group if str(item.get("last_kernel_end_ns", "")).strip()
        ]
        nvtx_start_values = [int(item["process_nvtx_start_ns"]) for item in group]
        nvtx_end_values = [int(item["process_nvtx_end_ns"]) for item in group]
        if first_kernel_values:
            gpu_order_key_ns = min(first_kernel_values)
            gpu_order_basis = "first_launch_owned_gpu_kernel_start"
        elif first_runtime_values:
            gpu_order_key_ns = min(first_runtime_values)
            gpu_order_basis = "first_runtime_call_start"
        else:
            gpu_order_key_ns = min(order_values) if order_values else min(nvtx_start_values)
            gpu_order_basis = "process_nvtx_start"
        aggregated.append(
            {
                "variant": base["variant"],
                "phase": base["phase"],
                "forward_id": base["forward_id"],
                "layer": base["layer"],
                "q_len": base["q_len"],
                "kv_len": base["kv_len"],
                "process_id": base["process_id"],
                "process_title": base["process_title"],
                "fragment_id": f"aggregated({fragment_ids})",
                "aggregation_key": base["aggregation_key"],
                "nvtx_range_name": "aggregated_fragments",
                "fx_nodes": base["fx_nodes"],
                "fx_op_families": base["fx_op_families"],
                "expected_kernel_families": ", ".join(expected),
                "matched_kernel_families": ", ".join(matched),
                "CUPTI kernel ms": cupti,
                "NVTX CPU ms": nvtx,
                "process_cupti_pct_in_parent": format_pct(cupti, parent_cupti),
                "process_nvtx_pct_in_parent": format_pct(nvtx, parent_nvtx),
                "parent_layer_range": base["parent_layer_range"],
                "parent_layer_CUPTI kernel ms": parent_cupti,
                "parent_layer_NVTX CPU ms": parent_nvtx,
                "runtime API calls": sum(int(item["runtime API calls"]) for item in group),
                "kernel instances": sum(int(item["kernel instances"]) for item in group),
                "dominant kernel family": dominant,
                "kernel_family_ms": "; ".join(
                    f"{family}={value:.3f}" for family, value in sorted(family_totals.items())
                ),
                "top_kernel_names": "",
                "process_nvtx_start_ns": min(nvtx_start_values),
                "process_nvtx_end_ns": max(nvtx_end_values),
                "first_runtime_start_ns": min(first_runtime_values) if first_runtime_values else "",
                "first_kernel_start_ns": min(first_kernel_values) if first_kernel_values else "",
                "last_kernel_end_ns": max(last_kernel_values) if last_kernel_values else "",
                "gpu_order_key_ns": gpu_order_key_ns,
                "gpu_order_basis": gpu_order_basis,
                "attribution method": "launch-owned correlationId",
                "validation status": combine_status([str(item["validation status"]) for item in group]),
                "notes": "Aggregated from fragment rows by aggregation_key.",
            }
        )
    return sorted(
        aggregated,
        key=lambda row: (
            str(row["phase"]),
            int(row["forward_id"]),
            int(row["layer"]),
            str(row["aggregation_key"]),
        ),
    )


def residual_rows(
    rows: list[dict[str, Any]],
    parent_ranges: dict[tuple[str, int, int], tuple[NvtxRange, LaunchOwnedMetrics]],
) -> list[dict[str, Any]]:
    process_sum: dict[tuple[str, int, int], dict[str, float]] = defaultdict(lambda: {"cupti": 0.0, "nvtx": 0.0})
    for row in rows:
        key = (str(row["parent_layer_range"]), int(row["forward_id"]), int(row["layer"]))
        process_sum[key]["cupti"] += float(row["CUPTI kernel ms"])
        process_sum[key]["nvtx"] += float(row["NVTX CPU ms"])

    residuals = []
    for key, (parent_range, parent_metrics) in parent_ranges.items():
        totals = process_sum.get(key, {"cupti": 0.0, "nvtx": 0.0})
        residuals.append(
            {
                "parent_layer_range": key[0],
                "forward_id": key[1],
                "layer": key[2],
                "parent_CUPTI kernel ms": parent_metrics.cupti_ms,
                "process_CUPTI kernel ms": totals["cupti"],
                "residual_CUPTI kernel ms": parent_metrics.cupti_ms - totals["cupti"],
                "parent_NVTX CPU ms": parent_range.duration_ms,
                "process_NVTX CPU ms": totals["nvtx"],
                "residual_NVTX CPU ms": parent_range.duration_ms - totals["nvtx"],
            }
        )
    return sorted(residuals, key=lambda row: abs(float(row["residual_CUPTI kernel ms"])), reverse=True)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    if not selected:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = format_ms(value)
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def gpu_execution_order_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        grouped[
            (
                str(row["parent_layer_range"]),
                int(row["forward_id"]),
                int(row["layer"]),
            )
        ].append(row)

    ordered: list[dict[str, Any]] = []
    phase_rank = {"prefill": 0, "decode": 1}
    for parent_key, rows in sorted(
        grouped.items(),
        key=lambda item: (
            phase_rank.get(str(item[1][0].get("phase", "")), 99),
            item[0][1],
            item[0][2],
            item[0][0],
        ),
    ):
        rows = sorted(
            rows,
            key=lambda row: (
                int(row.get("gpu_order_key_ns") or 0),
                int(row.get("process_nvtx_start_ns") or 0),
                str(row.get("process_id", "")),
            ),
        )
        base = min((int(row.get("gpu_order_key_ns") or 0) for row in rows), default=0)
        for idx, row in enumerate(rows, start=1):
            order_key = int(row.get("gpu_order_key_ns") or 0)
            out = dict(row)
            out["gpu_order"] = idx
            out["gpu_start_offset_us"] = f"{(order_key - base) / 1000.0:.3f}"
            out["parent_layer_range"] = parent_key[0]
            ordered.append(out)
    return ordered


def write_report(
    *,
    path: Path,
    variant: str,
    display_name: str,
    sqlite_path: Path,
    handoff_path: Path,
    detail_csv: Path,
    process_csv: Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
    handoff_rows: list[dict[str, str]],
) -> None:
    status_counts = Counter(str(row["validation status"]) for row in rows)
    matched_parents = {
        (row["parent_layer_range"], row["forward_id"], row["layer"])
        for row in rows
        if row["parent_layer_range"] != "missing"
    }
    top = sorted(aggregate, key=lambda row: float(row["CUPTI kernel ms"]), reverse=True)
    gpu_order_rows = gpu_execution_order_rows(aggregate)
    unexpected = [row for row in rows if row["validation status"] == "unexpected_kernel_family"]
    no_kernel = [row for row in rows if row["validation status"] == "no_kernel"]
    partially_validated = [row for row in rows if row["validation status"] == "partially_validated"]
    unresolved = [row for row in rows if row["validation status"] == "unresolved"]

    content = [
        f"# {display_name} SAME_INPUT Process-wise Performance Report",
        "",
        "## Sources",
        "",
        f"- Nsight sqlite: `{sqlite_path}`",
        f"- Handoff: `{handoff_path}`",
        f"- Fragment CSV: `{detail_csv}`",
        f"- Aggregated process CSV: `{process_csv}`",
        "",
        "## Handoff Summary",
        "",
        f"- Variant scope: `{variant}`",
        f"- Handoff inventory rows: {len(handoff_rows)}",
        "- Attribution method: launch-owned CUDA runtime `correlationId` to CUPTI kernels.",
        "- FX process data is used for semantic validation and process naming, not as a timing source.",
        "",
        "## Process-level NVTX Coverage",
        "",
        f"- Process NVTX fragment ranges: {len(rows)}",
        f"- Aggregated process rows: {len(aggregate)}",
        f"- Parent layer ranges with process coverage: {len(matched_parents)}",
        f"- Validation status counts: {dict(status_counts)}",
        "",
        "## Process Launch-owned Kernel Breakdown",
        "",
        markdown_table(
            top,
            [
                "CUPTI kernel ms",
                "NVTX CPU ms",
                "process_cupti_pct_in_parent",
                "process_nvtx_pct_in_parent",
                "phase",
                "forward_id",
                "layer",
                "process_id",
                "process_title",
                "fragment_id",
                "matched_kernel_families",
                "runtime API calls",
                "kernel instances",
                "validation status",
            ],
            limit=80,
        ),
        "",
        "## Representative Layer Process GPU Execution Order",
        "",
        "Rows are grouped by parent layer and ordered by each process row's first launch-owned CUPTI kernel GPU start. If a process has no launch-owned kernel, the order falls back to its first CUDA runtime call start and then to the process NVTX start.",
        "",
        markdown_table(
            gpu_order_rows,
            [
                "parent_layer_range",
                "forward_id",
                "layer",
                "gpu_order",
                "gpu_start_offset_us",
                "process_id",
                "process_title",
                "CUPTI kernel ms",
                "NVTX CPU ms",
                "process_cupti_pct_in_parent",
                "matched_kernel_families",
                "runtime API calls",
                "kernel instances",
                "gpu_order_basis",
                "validation status",
            ],
        ),
        "",
        "## FX Process Validation",
        "",
        f"- Unexpected kernel-family rows: {len(unexpected)}",
        f"- No-kernel rows: {len(no_kernel)}",
        f"- Partially validated rows: {len(partially_validated)}",
        f"- unresolved rows: {len(unresolved)}",
        "",
        markdown_table(
            (unexpected + no_kernel + unresolved + partially_validated)[:40],
            [
                "CUPTI kernel ms",
                "NVTX CPU ms",
                "phase",
                "forward_id",
                "layer",
                "process_id",
                "fragment_id",
                "expected_kernel_families",
                "matched_kernel_families",
                "notes",
            ],
        ),
        "",
        "## Residual and Unresolved Work",
        "",
        "Residual is parent layer launch-owned time minus the sum of instrumented process fragments in the same traced parent scope. Negative values can appear when asynchronous launch attribution or nested fragments double-count related launch work; inspect the fragment CSV before interpreting them as savings.",
        "",
        markdown_table(
            residuals,
            [
                "parent_layer_range",
                "forward_id",
                "layer",
                "parent_CUPTI kernel ms",
                "process_CUPTI kernel ms",
                "residual_CUPTI kernel ms",
                "parent_NVTX CPU ms",
                "process_NVTX CPU ms",
                "residual_NVTX CPU ms",
            ],
            limit=40,
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `CUPTI kernel ms` is the sum of full GPU kernel durations launched by CUDA runtime calls that started inside the process NVTX CPU range.",
        "- `NVTX CPU ms` is the CPU-side duration of the process range and can include Python/framework/runtime launch overhead.",
        "- Percentages are computed against the containing layer range in the same metric scope, not against whole-request latency.",
        "- `nvtx_gpu_proj_sum` can be used as a diagnostic for overlap, but this report uses launch ownership as the primary attribution relation.",
        "- Cross-function processes such as `output_projection` and `mlp` are aggregated by `aggregation_key`; fragment rows remain in the CSV.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_aggregate_report(
    *,
    path: Path,
    variant: str,
    display_name: str,
    aggregate: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
    process_csv: Path,
) -> None:
    denom_cupti = sum(float(row["parent_CUPTI kernel ms"]) for row in residuals)
    denom_nvtx = sum(float(row["parent_NVTX CPU ms"]) for row in residuals)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregate:
        key = (str(row["aggregation_key"]), str(row["process_title"]))
        entry = grouped.setdefault(
            key,
            {
                "process_id": row["process_id"],
                "process_title": row["process_title"],
                "CUPTI kernel ms": 0.0,
                "NVTX CPU ms": 0.0,
                "kernel instances": 0,
                "runtime API calls": 0,
                "matched_kernel_families": set(),
                "validation status": Counter(),
            },
        )
        entry["CUPTI kernel ms"] += float(row["CUPTI kernel ms"])
        entry["NVTX CPU ms"] += float(row["NVTX CPU ms"])
        entry["kernel instances"] += int(row["kernel instances"])
        entry["runtime API calls"] += int(row["runtime API calls"])
        entry["validation status"][str(row["validation status"])] += 1
        for family in str(row["matched_kernel_families"]).split(","):
            if family.strip():
                entry["matched_kernel_families"].add(family.strip())

    ranked: list[dict[str, Any]] = []
    for entry in grouped.values():
        statuses = list(entry["validation status"].elements())
        ranked.append(
            {
                "CUPTI kernel ms": entry["CUPTI kernel ms"],
                "NVTX CPU ms": entry["NVTX CPU ms"],
                "global_cupti_pct_in_traced_scope": format_pct(entry["CUPTI kernel ms"], denom_cupti),
                "global_nvtx_pct_in_traced_scope": format_pct(entry["NVTX CPU ms"], denom_nvtx),
                "process_id": entry["process_id"],
                "process_title": entry["process_title"],
                "matched_kernel_families": ", ".join(sorted(entry["matched_kernel_families"])),
                "runtime API calls": entry["runtime API calls"],
                "kernel instances": entry["kernel instances"],
                "validation status": combine_status(statuses),
            }
        )
    ranked.sort(key=lambda row: float(row["CUPTI kernel ms"]), reverse=True)

    content = [
        "# SAME_INPUT Process-wise Performance Breakdown",
        "",
        f"## {display_name}",
        "",
        f"- Variant: `{variant}`",
        f"- Aggregated process CSV: `{process_csv}`",
        f"- Denominator: sum of parent layer metrics over the traced representative-layer scope only.",
        f"- Parent CUPTI denominator: {denom_cupti:.3f} ms",
        f"- Parent NVTX denominator: {denom_nvtx:.3f} ms",
        "",
        markdown_table(
            ranked,
            [
                "CUPTI kernel ms",
                "NVTX CPU ms",
                "global_cupti_pct_in_traced_scope",
                "global_nvtx_pct_in_traced_scope",
                "process_id",
                "process_title",
                "matched_kernel_families",
                "runtime API calls",
                "kernel instances",
                "validation status",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--layer-events", type=Path, default=DEFAULT_LAYER_EVENTS)
    parser.add_argument("--variant", default="visipruner-full-eager")
    parser.add_argument("--display-name", default="VisiPruner Full Eager")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DETAIL_JSON)
    parser.add_argument("--process-csv", type=Path, default=DEFAULT_PROCESS_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--aggregate-report", type=Path, default=DEFAULT_AGGREGATE_REPORT)
    args = parser.parse_args()

    if not args.handoff.exists():
        raise SystemExit(f"Missing handoff: {args.handoff}")
    if not args.sqlite.exists():
        raise SystemExit(f"Missing Nsight sqlite trace: {args.sqlite}")

    handoff_rows, handoff_by_fragment = parse_handoff_inventory(args.handoff)
    layer_events = load_layer_event_metadata(args.layer_events)

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row
    try:
        require_tables(
            conn,
            [
                "NVTX_EVENTS",
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "StringIds",
            ],
        )
        process_ranges = [
            rng
            for rng in read_nvtx_ranges(conn, "visprune.fx_process%")
            if parse_process_range(rng.text)
        ]
        if not process_ranges:
            raise SystemExit(
                "Trace contains no process-level NVTX ranges. "
                "Rerun Nsight with FX_PROCESS_PROFILE=on before generating strict process reports."
            )
        layer_ranges = [
            rng for rng in read_nvtx_ranges(conn, "visprune.layer%.%") if LAYER_RE.match(rng.text)
        ]
        runtime_calls = read_runtime_calls(conn)
        kernels = read_kernels(conn)
    finally:
        conn.close()

    kernels_by_correlation: dict[int, list[Kernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.correlation_id is not None:
            kernels_by_correlation[kernel.correlation_id].append(kernel)
    runtime_starts = [call.start for call in runtime_calls]

    parent_cache: dict[tuple[str, int, int], tuple[NvtxRange, LaunchOwnedMetrics]] = {}
    detail_rows: list[dict[str, Any]] = []
    for process_range in process_ranges:
        parsed = parse_process_range(process_range.text)
        if parsed is None:
            continue
        fragment_id = parsed["fragment_id"]
        inventory = handoff_by_fragment.get((parsed["process_id"], fragment_id))
        if inventory is None:
            inventory = handoff_by_fragment.get((parsed["process_id"], "whole"))
        parent = find_parent_range(process_range, layer_ranges, parsed["layer"], parsed["phase"])
        parent_metrics = None
        if parent is not None:
            parent_key = (parent.text, parsed["forward_id"], parsed["layer"])
            if parent_key not in parent_cache:
                parent_cache[parent_key] = (
                    parent,
                    launch_owned_metrics(parent, runtime_calls, runtime_starts, kernels_by_correlation),
                )
            parent_metrics = parent_cache[parent_key][1]
        metrics = launch_owned_metrics(process_range, runtime_calls, runtime_starts, kernels_by_correlation)
        layer_event = layer_events.get((parsed["forward_id"], parsed["layer"]))
        detail_rows.append(
            row_for_range(
                variant=args.variant,
                parsed=parsed,
                nvtx_range=process_range,
                metrics=metrics,
                inventory=inventory,
                parent_range=parent,
                parent_metrics=parent_metrics,
                layer_event=layer_event,
            )
        )

    process_rows = aggregate_rows(detail_rows)
    residuals = residual_rows(detail_rows, parent_cache)

    fieldnames = [
        "variant",
        "phase",
        "forward_id",
        "layer",
        "q_len",
        "kv_len",
        "process_id",
        "process_title",
        "fragment_id",
        "aggregation_key",
        "nvtx_range_name",
        "fx_nodes",
        "fx_op_families",
        "expected_kernel_families",
        "matched_kernel_families",
        "CUPTI kernel ms",
        "NVTX CPU ms",
        "process_cupti_pct_in_parent",
        "process_nvtx_pct_in_parent",
        "parent_layer_range",
        "parent_layer_CUPTI kernel ms",
        "parent_layer_NVTX CPU ms",
        "runtime API calls",
        "kernel instances",
        "dominant kernel family",
        "kernel_family_ms",
        "top_kernel_names",
        "process_nvtx_start_ns",
        "process_nvtx_end_ns",
        "first_runtime_start_ns",
        "first_kernel_start_ns",
        "last_kernel_end_ns",
        "gpu_order_key_ns",
        "gpu_order_basis",
        "attribution method",
        "validation status",
        "notes",
    ]
    write_csv(args.output_csv, detail_rows, fieldnames)
    write_csv(args.process_csv, process_rows, fieldnames)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "sqlite": str(args.sqlite),
                "handoff": str(args.handoff),
                "variant": args.variant,
                "attribution_method": "launch-owned correlationId",
                "process_nvtx_range_count": len(detail_rows),
                "aggregated_process_row_count": len(process_rows),
                "parent_layer_range_count": len(parent_cache),
                "validation_status_counts": dict(Counter(row["validation status"] for row in detail_rows)),
                "detail_csv": str(args.output_csv),
                "process_csv": str(args.process_csv),
                "residuals": residuals,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_report(
        path=args.report,
        variant=args.variant,
        display_name=args.display_name,
        sqlite_path=args.sqlite,
        handoff_path=args.handoff,
        detail_csv=args.output_csv,
        process_csv=args.process_csv,
        rows=detail_rows,
        aggregate=process_rows,
        residuals=residuals,
        handoff_rows=handoff_rows,
    )
    write_aggregate_report(
        path=args.aggregate_report,
        variant=args.variant,
        display_name=args.display_name,
        aggregate=process_rows,
        residuals=residuals,
        process_csv=args.process_csv,
    )

    print(
        json.dumps(
            {
                "process_nvtx_range_count": len(detail_rows),
                "aggregated_process_row_count": len(process_rows),
                "parent_layer_range_count": len(parent_cache),
                "detail_csv": str(args.output_csv),
                "process_csv": str(args.process_csv),
                "report": str(args.report),
                "aggregate_report": str(args.aggregate_report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
