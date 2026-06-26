#!/usr/bin/env python3
"""Generate visualization artifacts for single-request latency breakdown."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EXP_DIR = Path("/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency")
OUTPUT_DIR = EXP_DIR / "output"
VIS_DIR = OUTPUT_DIR / "visualizations"

RUNS = {
    "dense-fa2": {
        "clock": OUTPUT_DIR / "clock_dense_fa2_32tok.json",
        "nsys": OUTPUT_DIR / "nsys_dense_fa2_32tok.json",
        "sqlite": OUTPUT_DIR / "nsys_dense_fa2_32tok.sqlite",
        "kernel_key": "dense_fa2",
    },
    "visipruner-full": {
        "clock": OUTPUT_DIR / "clock_visprune_full_32tok.json",
        "nsys": OUTPUT_DIR / "nsys_visprune_full_32tok.json",
        "sqlite": OUTPUT_DIR / "nsys_visprune_full_32tok.sqlite",
        "kernel_key": "visipruner_full",
    },
}

STAGE_COMPONENTS = [
    ("non_generate_ms", "non-generate"),
    ("vision_encode_project_ms", "vision encode/project"),
    ("prepare_without_vision_ms", "prepare multimodal"),
    ("forward_prefill_ms", "prefill forward"),
    ("forward_decode_sum_ms", "decode forward"),
    ("generate_other_ms", "generate other"),
]

KERNEL_FAMILIES = [
    ("gemv_decode_cublas", "GEMV decode cuBLAS"),
    ("gemm_tensorcore", "Tensor Core GEMM"),
    ("flash_attention", "FlashAttention"),
    ("elementwise_norm_activation", "elementwise/norm/activation"),
    ("copy_gather_cat", "copy/gather/cat"),
    ("selection_reduce_scan", "selection/reduce/scan"),
    ("softmax", "softmax"),
    ("other", "other"),
]


def kernel_family(name: str) -> str:
    lowered = name.lower()
    if "gemvx::kernel" in lowered:
        return "gemv_decode_cublas"
    if "flash::flash" in lowered:
        return "flash_attention"
    if "gemm" in lowered or "cutlass::kernel" in lowered or "cudnn" in lowered:
        return "gemm_tensorcore"
    if (
        "catarraybatchedcopy" in lowered
        or "copy_kernel" in lowered
        or "indexselect" in lowered
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
    ):
        return "elementwise_norm_activation"
    return "other"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage_df() -> pd.DataFrame:
    rows = []
    for run_name, paths in RUNS.items():
        payload = load_json(paths["clock"])
        derived = payload["derived_ms"]
        for key, label in STAGE_COMPONENTS:
            rows.append(
                {
                    "run": run_name,
                    "stage": label,
                    "ms": float(derived.get(key, 0.0)),
                }
            )
        tracker = payload.get("tracker", {})
        rows.append(
            {
                "run": run_name,
                "stage": "value selection (nested)",
                "ms": float(derived.get("value_aware_token_selection_ms", 0.0)),
                "nested": True,
                "selected_visual_tokens": tracker.get("selected_visual_token_counts", []),
                "deep_exit_checks": tracker.get("deep_exit_checks", []),
            }
        )
    return pd.DataFrame(rows)


def build_kernel_df() -> pd.DataFrame:
    summary = load_json(OUTPUT_DIR / "kernel_family_summary.json")
    rows = []
    for run_name, paths in RUNS.items():
        groups = summary[paths["kernel_key"]]["groups"]
        for family, label in KERNEL_FAMILIES:
            rows.append(
                {
                    "run": run_name,
                    "family": label,
                    "ms": float(groups.get(family, {}).get("time_ms", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def load_decode_df() -> pd.DataFrame:
    path = OUTPUT_DIR / "decode_iteration_kernel_breakdown.json"
    payload = load_json(path)
    rows = []
    for row in payload["rows"]:
        groups = row["groups"]
        rows.append(
            {
                "iteration": row["iteration"],
                "range_ms": row["range_ms"],
                "kernel_total_ms": row["kernel_total_ms"],
                "gemv_ms": groups.get("gemv_decode_cublas", {}).get("time_ms", 0.0),
                "elementwise_ms": groups.get("elementwise_norm_activation", {}).get("time_ms", 0.0),
                "copy_ms": groups.get("copy_gather_cat", {}).get("time_ms", 0.0),
            }
        )
    return pd.DataFrame(rows)


def write_data_tables(
    out_dir: Path, stage_df: pd.DataFrame, kernel_df: pd.DataFrame, decode_df: pd.DataFrame
) -> None:
    stage_df.to_csv(out_dir / "latency_stage_breakdown.csv", index=False)
    kernel_df.to_csv(out_dir / "kernel_family_breakdown.csv", index=False)
    decode_df.to_csv(out_dir / "decode_iteration_lines.csv", index=False)


def write_static_png(
    out_dir: Path, stage_df: pd.DataFrame, kernel_df: pd.DataFrame, decode_df: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)

    stage_pivot = (
        stage_df[stage_df.get("nested").fillna(False) != True]
        .pivot(index="run", columns="stage", values="ms")
        .reindex(["dense-fa2", "visipruner-full"])
    )
    stage_pivot[[label for _, label in STAGE_COMPONENTS]].plot(
        kind="barh", stacked=True, ax=axes[0], width=0.7
    )
    axes[0].set_title("Clock request latency breakdown")
    axes[0].set_xlabel("milliseconds")
    axes[0].set_ylabel("")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="stage")

    kernel_pivot = (
        kernel_df.pivot(index="run", columns="family", values="ms")
        .reindex(["dense-fa2", "visipruner-full"])
    )
    kernel_pivot[[label for _, label in KERNEL_FAMILIES]].plot(
        kind="barh", stacked=True, ax=axes[1], width=0.7
    )
    axes[1].set_title("Nsight CUDA kernel family duration")
    axes[1].set_xlabel("milliseconds")
    axes[1].set_ylabel("")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="family")

    axes[2].plot(decode_df["iteration"], decode_df["range_ms"], label="forward_decode NVTX")
    axes[2].plot(decode_df["iteration"], decode_df["kernel_total_ms"], label="CUDA kernels")
    axes[2].plot(decode_df["iteration"], decode_df["gemv_ms"], label="GEMV kernels")
    axes[2].set_title("VisiPrune decode iteration breakdown")
    axes[2].set_xlabel("decode iteration")
    axes[2].set_ylabel("milliseconds")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[2].grid(True, alpha=0.25)

    fig.savefig(out_dir / "single_request_latency_breakdown.png", dpi=180)
    plt.close(fig)


def write_plotly_dashboard(
    out_dir: Path, stage_df: pd.DataFrame, kernel_df: pd.DataFrame, decode_df: pd.DataFrame
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Clock request latency breakdown",
            "Nsight CUDA kernel family duration",
            "VisiPrune decode iteration breakdown",
        ],
        vertical_spacing=0.12,
    )

    runs = ["dense-fa2", "visipruner-full"]
    for _, label in STAGE_COMPONENTS:
        vals = [
            float(stage_df[(stage_df["run"] == run) & (stage_df["stage"] == label)]["ms"].iloc[0])
            for run in runs
        ]
        fig.add_trace(go.Bar(name=label, y=runs, x=vals, orientation="h"), row=1, col=1)

    for _, label in KERNEL_FAMILIES:
        vals = [
            float(kernel_df[(kernel_df["run"] == run) & (kernel_df["family"] == label)]["ms"].iloc[0])
            for run in runs
        ]
        fig.add_trace(go.Bar(name=label, y=runs, x=vals, orientation="h"), row=2, col=1)

    fig.add_trace(
        go.Scatter(
            name="forward_decode NVTX",
            x=decode_df["iteration"],
            y=decode_df["range_ms"],
            mode="lines+markers",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="CUDA kernels",
            x=decode_df["iteration"],
            y=decode_df["kernel_total_ms"],
            mode="lines+markers",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="GEMV kernels",
            x=decode_df["iteration"],
            y=decode_df["gemv_ms"],
            mode="lines+markers",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        barmode="stack",
        height=1050,
        width=1180,
        title="Single-request LLaVA/VisiPrune latency visualization",
        legend_tracegroupgap=8,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="milliseconds", row=1, col=1)
    fig.update_xaxes(title_text="milliseconds", row=2, col=1)
    fig.update_xaxes(title_text="decode iteration", row=3, col=1)
    fig.update_yaxes(title_text="milliseconds", row=3, col=1)
    fig.write_html(
        out_dir / "single_request_latency_dashboard.html",
        include_plotlyjs=True,
        full_html=True,
    )


def fetch_nvtx_ranges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT start, end, text
        FROM NVTX_EVENTS
        WHERE text LIKE 'visprune.%'
          AND end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()


def fetch_request_bounds(conn: sqlite3.Connection) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = 'visprune.request'
          AND end IS NOT NULL
        ORDER BY start
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("Missing visprune.request NVTX range")
    return int(row["start"]), int(row["end"])


def fetch_kernels(conn: sqlite3.Connection, start: int, end: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT k.start, k.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON k.demangledName = s.id
        WHERE k.start >= ?
          AND k.end <= ?
        ORDER BY k.start
        """,
        (start, end),
    ).fetchall()


def write_chrome_trace(out_dir: Path) -> None:
    events: list[dict[str, Any]] = []
    for pid, (run_name, paths) in enumerate(RUNS.items(), start=1):
        conn = sqlite3.connect(paths["sqlite"])
        conn.row_factory = sqlite3.Row
        request_start, request_end = fetch_request_bounds(conn)
        events.append(
            {"ph": "M", "pid": pid, "tid": 1, "name": "process_name", "args": {"name": run_name}}
        )
        events.append(
            {"ph": "M", "pid": pid, "tid": 1, "name": "thread_name", "args": {"name": "NVTX ranges"}}
        )
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "tid": 2,
                "name": "thread_name",
                "args": {"name": "CUDA kernels by family"},
            }
        )

        for row in fetch_nvtx_ranges(conn):
            events.append(
                {
                    "name": row["text"],
                    "cat": "NVTX",
                    "ph": "X",
                    "pid": pid,
                    "tid": 1,
                    "ts": (row["start"] - request_start) / 1000.0,
                    "dur": (row["end"] - row["start"]) / 1000.0,
                    "args": {"duration_ms": (row["end"] - row["start"]) / 1e6},
                }
            )

        for row in fetch_kernels(conn, request_start, request_end):
            fam = kernel_family(row["name"])
            events.append(
                {
                    "name": fam,
                    "cat": "CUDA kernel",
                    "ph": "X",
                    "pid": pid,
                    "tid": 2,
                    "ts": (row["start"] - request_start) / 1000.0,
                    "dur": (row["end"] - row["start"]) / 1000.0,
                    "args": {
                        "duration_ms": (row["end"] - row["start"]) / 1e6,
                        "kernel": row["name"][:240],
                    },
                }
            )
        conn.close()

    trace_path = out_dir / "single_request_latency_trace_chrome.json"
    trace_path.write_text(json.dumps({"traceEvents": events}, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(VIS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_df = build_stage_df()
    kernel_df = build_kernel_df()
    decode_df = load_decode_df()
    write_data_tables(out_dir, stage_df, kernel_df, decode_df)
    write_static_png(out_dir, stage_df, kernel_df, decode_df)
    write_plotly_dashboard(out_dir, stage_df, kernel_df, decode_df)
    write_chrome_trace(out_dir)

    print(f"Visualization output directory: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
