#!/usr/bin/env python3
"""
E1: Summarize — generate all deliverables from parsed nsys/ncu data.

Produces:
    1. e1_nsys_top10_kernels.csv (per-config, from runner)
    2. e1_nsys_kernel_categories.png — stacked bar chart
    3. e1_ncu_per_kernel_bottleneck.csv (from runner)
    4. e1_roofline.png — hotspot kernels on roofline model
    5. e1_dram_bytes.png — DRAM bytes bar chart
    6. e1_l1_l2_l3_stacked.png — cache level time stacked bar
    7. e1_gap_summary.json (from runner)

Usage:
    python e1_summarize.py --summary-json <path> --nsys-dir <path> --output-dir <path>
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)
import e1_analysis as ana

# ── Style constants ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 13,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLOR_MAP = {
    "dense-fa2": "#2ecc71",
    "dense-eager": "#3498db",
    "dense-sdpa": "#9b59b6",
    "visipruner-full": "#e74c3c",
    "visipruner-shallow-only": "#f39c12",
}

CONFIG_DISPLAY = {
    "dense-fa2": "Dense + FA2",
    "dense-eager": "Dense + Eager",
    "dense-sdpa": "Dense + SDPA",
    "visipruner-full": "VisiPruner Full",
    "visipruner-shallow-only": "VisiPruner Shallow",
}


# ── Chart 1: Kernel categories stacked bar ──────────────────────────────

def plot_kernel_categories(nsys_data: dict, output_dir: str):
    """Generate e1_nsys_kernel_categories.png — stacked bar of kernel time by category."""
    configs = sorted(nsys_data.keys())
    all_categories = set()
    config_category_pcts = {}

    for config in configs:
        kernels = nsys_data[config]
        cats = ana.aggregate_by_category(kernels)
        config_category_pcts[config] = {c["category"]: c["time_pct"] for c in cats}
        all_categories.update(c["category"] for c in cats)

    # Sort categories by total contribution across all configs
    cat_total = {}
    for cat in all_categories:
        cat_total[cat] = sum(config_category_pcts.get(c, {}).get(cat, 0)
                            for c in configs)
    sorted_cats = sorted(all_categories, key=lambda x: cat_total[x], reverse=True)
    # Limit to top-10, group rest as "other"
    if len(sorted_cats) > 10:
        main_cats = sorted_cats[:9]
        other_cats = sorted_cats[9:]
    else:
        main_cats = sorted_cats
        other_cats = []

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(configs))
    width = 0.6
    bottoms = np.zeros(len(configs))

    for cat in main_cats:
        values = [config_category_pcts.get(c, {}).get(cat, 0) for c in configs]
        meta = ana.CATEGORY_META.get(cat, ana.CATEGORY_META["other"])
        color = meta["color"]
        label = meta["display"]
        ax.bar(x, values, width, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.3)
        bottoms += np.array(values)

    # Other category
    if other_cats:
        other_values = []
        for c in configs:
            ov = sum(config_category_pcts.get(c, {}).get(oc, 0) for oc in other_cats)
            other_values.append(ov)
        ax.bar(x, other_values, width, bottom=bottoms, label="Other", color="#bdc3c7", edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax.set_ylabel("GPU Time (%)")
    ax.set_title("E1: Kernel Category Breakdown by Config")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 105)

    path = os.path.join(output_dir, "e1_nsys_kernel_categories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Kernel categories chart → {path}", flush=True)


# ── Chart 2: Roofline model ─────────────────────────────────────────────

def plot_roofline(ncu_results: list[dict], output_dir: str, gpu_name: str = "RTX 4090"):
    """Generate e1_roofline.png — hotspot kernels on roofline model.

    Roofline for RTX 4090 (Ada Lovelace):
        Peak FP16 compute (TFLOPS): ~82.6 (with sparsity: ~165.2)
        Peak memory bandwidth (GB/s): ~1008

    X-axis: Arithmetic Intensity (FLOPs/byte)
    Y-axis: Achievable Performance (TFLOPS)
    """
    # RTX 4090 specs
    peak_bw_gbs = 1008.0   # GB/s DRAM bandwidth
    peak_compute_tflops = 82.6  # FP16 TFLOPS (no sparsity)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw roofline ridges
    ai_range = np.logspace(-2, 3, 200)  # 0.01 to 1000 FLOPs/byte
    # Memory-bound ceiling: Performance = AI × Peak_BW
    mem_ceiling = ai_range * peak_bw_gbs / 1000  # convert GB/s → TFLOPS
    # Compute-bound ceiling
    comp_ceiling = np.full_like(ai_range, peak_compute_tflops)

    # The roofline is min(mem_ceiling, comp_ceiling)
    roofline = np.minimum(mem_ceiling, comp_ceiling)
    ridge_x = peak_compute_tflops * 1000 / peak_bw_gbs  # AI at ridge point

    ax.loglog(ai_range, roofline, "k-", linewidth=1.5, label="Roofline", zorder=1)
    ax.axvline(x=ridge_x, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(ridge_x * 1.1, peak_compute_tflops * 0.03,
            f"Ridge AI={ridge_x:.1f} FLOP/byte", fontsize=7, color="gray")

    # Plot each kernel as a point
    config_markers = {"dense-fa2": "o", "dense-eager": "s", "dense-sdpa": "D",
                      "visipruner-full": "^", "visipruner-shallow-only": "v"}

    for r in ncu_results:
        config = r.get("config", "unknown")
        kname = r.get("kernel_name", "unknown")[:30]
        u_compute = r.get("u_compute_pct", 0)
        u_bw = r.get("u_bw_pct", 0)

        # Estimate AI from bottleneck metrics
        # AI = compute_rate / bw_rate = (u_compute * peak_compute) / (u_bw * peak_bw)
        if u_bw > 0:
            ai = (u_compute / 100 * peak_compute_tflops) / max(u_bw / 100 * peak_bw_gbs / 1000, 1e-6)
        else:
            ai = 100.0  # pure compute, push right

        perf = u_compute / 100 * peak_compute_tflops  # achieved TFLOPS
        marker = config_markers.get(config, "x")
        color = COLOR_MAP.get(config, "#333333")

        ax.loglog(ai, perf, marker=marker, color=color, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # Legend for configs
    legend_elements = []
    for config in sorted(config_markers.keys()):
        legend_elements.append(
            plt.Line2D([0], [0], marker=config_markers[config], color="w",
                       markerfacecolor=COLOR_MAP.get(config, "#333"),
                       markersize=8, label=CONFIG_DISPLAY.get(config, config))
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(f"E1: Roofline Model — Hotspot Kernels ({gpu_name})")
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-2, peak_compute_tflops * 2)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate regions
    ax.text(0.05, peak_compute_tflops * 0.5, "Memory\nBound", fontsize=8,
            color="gray", alpha=0.7, ha="center")
    ax.text(ridge_x * 20, peak_compute_tflops * 0.5, "Compute\nBound", fontsize=8,
            color="gray", alpha=0.7, ha="center")

    path = os.path.join(output_dir, "e1_roofline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Roofline chart → {path}", flush=True)


# ── Chart 3: DRAM bytes bar chart ───────────────────────────────────────

def plot_dram_bytes(ncu_results: list[dict], output_dir: str):
    """Generate e1_dram_bytes.png — DRAM read/write bytes per config."""
    configs = sorted(set(r["config"] for r in ncu_results))
    config_dram_read = defaultdict(float)
    config_dram_write = defaultdict(float)

    for r in ncu_results:
        config = r["config"]
        config_dram_read[config] += r.get("dram_bytes", 0) * 0.5  # approximate read/write split
        config_dram_write[config] += r.get("dram_bytes", 0) * 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    width = 0.35

    reads = [config_dram_read[c] / 1e9 for c in configs]  # Convert to GB
    writes = [config_dram_write[c] / 1e9 for c in configs]

    ax.bar(x - width/2, reads, width, label="DRAM Read", color="#3498db", edgecolor="white")
    ax.bar(x + width/2, writes, width, label="DRAM Write", color="#e74c3c", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax.set_ylabel("Data Volume (GB)")
    ax.set_title("E1: DRAM Traffic by Config (Hotspot Kernels)")
    ax.legend()

    path = os.path.join(output_dir, "e1_dram_bytes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] DRAM bytes chart → {path}", flush=True)


# ── Chart 4: L1/L2/L3/DRAM stacked bar ──────────────────────────────────

def plot_cache_levels(ncu_results: list[dict], output_dir: str):
    """Generate e1_l1_l2_l3_stacked.png — cache level data volume per config."""
    configs = sorted(set(r["config"] for r in ncu_results))
    config_l1 = defaultdict(float)
    config_l2 = defaultdict(float)
    config_dram = defaultdict(float)

    for r in ncu_results:
        config = r["config"]
        config_l1[config] += r.get("l1_bytes", 0)
        config_l2[config] += r.get("l2_bytes", 0)
        config_dram[config] += r.get("dram_bytes", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Absolute bytes ---
    x = np.arange(len(configs))
    width = 0.6

    l1_abs = [config_l1[c] / 1e9 for c in configs]
    l2_abs = [config_l2[c] / 1e9 for c in configs]
    dram_abs = [config_dram[c] / 1e9 for c in configs]

    ax1.bar(x, l1_abs, width, label="L1", color="#2ecc71", edgecolor="white")
    ax1.bar(x, l2_abs, width, bottom=l1_abs, label="L2", color="#f39c12", edgecolor="white")
    bottoms_l2 = [a + b for a, b in zip(l1_abs, l2_abs)]
    ax1.bar(x, dram_abs, width, bottom=bottoms_l2, label="DRAM", color="#e74c3c", edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax1.set_ylabel("Data Volume (GB)")
    ax1.set_title("E1: Cache Hierarchy Data Volume (Absolute)")
    ax1.legend()

    # --- Percentage ---
    pcts_l1, pcts_l2, pcts_dram = [], [], []
    for c in configs:
        total = config_l1[c] + config_l2[c] + config_dram[c]
        if total > 0:
            pcts_l1.append(config_l1[c] / total * 100)
            pcts_l2.append(config_l2[c] / total * 100)
            pcts_dram.append(config_dram[c] / total * 100)
        else:
            pcts_l1.append(0)
            pcts_l2.append(0)
            pcts_dram.append(0)

    ax2.bar(x, pcts_l1, width, label="L1", color="#2ecc71", edgecolor="white")
    ax2.bar(x, pcts_l2, width, bottom=pcts_l1, label="L2", color="#f39c12", edgecolor="white")
    bottoms_pct = [a + b for a, b in zip(pcts_l1, pcts_l2)]
    ax2.bar(x, pcts_dram, width, bottom=bottoms_pct, label="DRAM", color="#e74c3c", edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("E1: Cache Hierarchy Data Volume (Relative)")
    ax2.legend()
    ax2.set_ylim(0, 105)

    path = os.path.join(output_dir, "e1_l1_l2_l3_stacked.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Cache levels chart → {path}", flush=True)


# ── Chart 5: Speedup comparison bar ─────────────────────────────────────

def plot_speedup_comparison(summary: dict, output_dir: str):
    """Generate e1_speedup_gap.png — theoretical vs actual speedup comparison."""
    speedups = summary.get("speedups", [])
    if not speedups:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [CONFIG_DISPLAY.get(s["target_config"], s["target_config"]) for s in speedups]
    actuals = [s["actual_speedup"] for s in speedups]
    gaps = [s["gap"] for s in speedups]

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width/2, actuals, width, label="Actual Speedup (nsys GPU time)",
           color="#3498db", edgecolor="white")
    ax.axhline(y=2.17, color="#e74c3c", linestyle="--", linewidth=2, label="Theoretical 2.17×")

    # Annotate gap values
    for i, (actual, gap) in enumerate(zip(actuals, gaps)):
        ax.annotate(f"GAP={gap:.2f}×",
                    xy=(i, actual), xytext=(i + 0.15, actual + 0.05),
                    fontsize=8, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("E1: Actual vs Theoretical Speedup")
    ax.legend()
    ax.set_ylim(0, max(2.5, max(actuals) * 1.2))

    # Add text annotation for the main result
    if speedups:
        main = speedups[0]
        ax.text(0.02, 0.98,
                f"GAP = {main['gap']}× speedup missing\n"
                f"(actual {main['actual_speedup']}× vs theoretical 2.17×)",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    path = os.path.join(output_dir, "e1_speedup_gap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Speedup comparison → {path}", flush=True)


# ── Chart 6: Bottleneck classification summary ──────────────────────────

def plot_bottleneck_summary(ncu_results: list[dict], output_dir: str):
    """Generate e1_bottleneck_summary.png — stacked bar of bottleneck types."""
    configs = sorted(set(r["config"] for r in ncu_results))
    bn_classes = ["compute-bound", "memory-bound", "latency-bound"]

    config_bn = {}
    for c in configs:
        c_results = [r for r in ncu_results if r["config"] == c]
        counts = defaultdict(int)
        for r in c_results:
            counts[r.get("bottleneck_class", "unknown")] += 1
        config_bn[c] = counts

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    width = 0.6

    bn_colors = {"compute-bound": "#e74c3c", "memory-bound": "#3498db",
                 "latency-bound": "#f39c12", "unknown": "#bdc3c7"}
    bottoms = np.zeros(len(configs))

    for bn in bn_classes:
        values = [config_bn[c].get(bn, 0) for c in configs]
        ax.bar(x, values, width, bottom=bottoms, label=bn.title(),
               color=bn_colors[bn], edgecolor="white")
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax.set_ylabel("Number of Hotspot Kernels")
    ax.set_title("E1: Kernel Bottleneck Classification")
    ax.legend()

    path = os.path.join(output_dir, "e1_bottleneck_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Bottleneck summary → {path}", flush=True)


# ── Chart 7: Total GPU time comparison ──────────────────────────────────

def plot_total_gpu_time(summary: dict, output_dir: str):
    """Generate e1_total_gpu_time.png — bar chart of T_total per config."""
    t_total = summary.get("t_total", {})
    if not t_total:
        return

    configs = sorted(t_total.keys())
    times_ms = [t_total[c]["total_time_ms"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLOR_MAP.get(c, "#95a5a6") for c in configs]
    x = np.arange(len(configs))

    bars = ax.bar(x, times_ms, 0.5, color=colors, edgecolor="white")

    # Annotate bars
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times_ms)*0.01,
                f"{t:.1f} ms", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in configs], rotation=15, ha="right")
    ax.set_ylabel("Total GPU Kernel Time (ms)")
    ax.set_title("E1: Total GPU Kernel Time by Config (nsys)")

    path = os.path.join(output_dir, "e1_total_gpu_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[e1_summarize] Total GPU time → {path}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="E1: Summarize and generate deliverables")
    p.add_argument("--summary-json", type=str, required=True,
                   help="Path to e1_gap_summary.json")
    p.add_argument("--nsys-dir", type=str, required=True,
                   help="Directory containing nsys kernel CSVs and e1_nsys_data.json")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for output figures")
    p.add_argument("--gpu-name", type=str, default="RTX 4090",
                   help="GPU name for roofline model")
    p.add_argument("--charts", type=str, nargs="+",
                   default=["all"],
                   choices=["all", "categories", "roofline", "dram", "cache",
                            "speedup", "bottleneck", "gpu_time"],
                   help="Which charts to generate")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load summary
    with open(args.summary_json) as f:
        summary = json.load(f)

    # Load nsys data for category charts
    nsys_data_path = os.path.join(args.nsys_dir, "e1_nsys_data.json")
    nsys_data = {}
    if os.path.exists(nsys_data_path):
        with open(nsys_data_path) as f:
            nsys_data_raw = json.load(f)
        nsys_data = {
            config: [{**k, "total_time_ns": float(k["total_time_ns"])}
                     for k in kernels]
            for config, kernels in nsys_data_raw.items()
        }

    # Load ncu results
    ncu_path = os.path.join(args.nsys_dir, "e1_ncu_results.json")
    ncu_results = []
    if os.path.exists(ncu_path):
        with open(ncu_path) as f:
            ncu_results = json.load(f)

    # Generate charts
    do_all = "all" in args.charts

    if do_all or "categories" in args.charts:
        if nsys_data:
            plot_kernel_categories(nsys_data, args.output_dir)
        else:
            print("[e1_summarize] WARNING: No nsys data, skipping categories chart")

    if do_all or "roofline" in args.charts:
        if ncu_results:
            plot_roofline(ncu_results, args.output_dir, args.gpu_name)
        else:
            print("[e1_summarize] WARNING: No ncu data, skipping roofline chart")

    if do_all or "dram" in args.charts:
        if ncu_results:
            plot_dram_bytes(ncu_results, args.output_dir)
        else:
            print("[e1_summarize] WARNING: No ncu data, skipping DRAM chart")

    if do_all or "cache" in args.charts:
        if ncu_results:
            plot_cache_levels(ncu_results, args.output_dir)
        else:
            print("[e1_summarize] WARNING: No ncu data, skipping cache chart")

    if do_all or "speedup" in args.charts:
        plot_speedup_comparison(summary, args.output_dir)

    if do_all or "bottleneck" in args.charts:
        if ncu_results:
            plot_bottleneck_summary(ncu_results, args.output_dir)
        else:
            print("[e1_summarize] WARNING: No ncu data, skipping bottleneck chart")

    if do_all or "gpu_time" in args.charts:
        plot_total_gpu_time(summary, args.output_dir)

    print(f"\n[e1_summarize] All deliverables generated in {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
