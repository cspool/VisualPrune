#!/usr/bin/env python3
"""Attribute Nsight CUDA kernels to each forward_decode NVTX range."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


DEFAULT_SQLITE = (
    "/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output/"
    "nsys_visprune_full_32tok.sqlite"
)
DEFAULT_OUTPUT = (
    "/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output/"
    "decode_iteration_kernel_breakdown.json"
)


def family(name: str) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=DEFAULT_SQLITE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    db_path = Path(args.sqlite)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    decode_ranges = conn.execute(
        """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = 'visprune.forward_decode'
          AND end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()

    kernels = conn.execute(
        """
        SELECT k.start, k.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON k.demangledName = s.id
        ORDER BY k.start
        """
    ).fetchall()
    conn.close()

    rows = []
    for idx, rng in enumerate(decode_ranges):
        groups: dict[str, dict[str, float | int]] = {}
        for kernel in kernels:
            if kernel["end"] <= rng["start"]:
                continue
            if kernel["start"] >= rng["end"]:
                break
            overlap_start = max(kernel["start"], rng["start"])
            overlap_end = min(kernel["end"], rng["end"])
            if overlap_end <= overlap_start:
                continue
            fam = family(kernel["name"])
            entry = groups.setdefault(fam, {"time_ms": 0.0, "instances": 0})
            entry["time_ms"] += (overlap_end - overlap_start) / 1e6
            entry["instances"] += 1
        total = sum(float(item["time_ms"]) for item in groups.values())
        row = {
            "iteration": idx,
            "range_ms": (rng["end"] - rng["start"]) / 1e6,
            "kernel_total_ms": total,
            "groups": groups,
        }
        rows.append(row)

    payload = {
        "source": str(db_path),
        "decode_iterations": len(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    families = sorted({fam for row in rows for fam in row["groups"]})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "range_ms", "kernel_total_ms"]
            + [f"{fam}_ms" for fam in families],
        )
        writer.writeheader()
        for row in rows:
            flat = {
                "iteration": row["iteration"],
                "range_ms": row["range_ms"],
                "kernel_total_ms": row["kernel_total_ms"],
            }
            for fam in families:
                flat[f"{fam}_ms"] = row["groups"].get(fam, {}).get("time_ms", 0.0)
            writer.writerow(flat)

    print(json.dumps(payload, indent=2))
    print(f"JSON: {out_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
