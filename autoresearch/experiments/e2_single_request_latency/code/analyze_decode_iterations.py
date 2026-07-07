#!/usr/bin/env python3
"""Attribute Nsight CUDA kernels to each forward_decode NVTX range by launch ownership."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


DEFAULT_SQLITE = (
    "/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/"
    "nsys_visprune_full_32tok.sqlite"
)
DEFAULT_OUTPUT = (
    "/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/"
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
        SELECT n.start, n.end
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE COALESCE(s.value, n.text, n.jsonText) = 'visprune.forward_decode'
          AND n.end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()

    kernels = conn.execute(
        """
        SELECT k.start, k.end, k.correlationId, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS s ON k.demangledName = s.id
        ORDER BY k.start
        """
    ).fetchall()
    runtime_calls = conn.execute(
        """
        SELECT r.start, r.end, r.correlationId, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS s ON r.nameId = s.id
        WHERE r.correlationId IS NOT NULL
        ORDER BY r.start
        """
    ).fetchall()
    conn.close()

    kernels_by_correlation: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for kernel in kernels:
        if kernel["correlationId"] is not None:
            kernels_by_correlation[int(kernel["correlationId"])].append(kernel)

    owned_runtime_calls = [
        call
        for call in runtime_calls
        if call["correlationId"] is not None
        and int(call["correlationId"]) in kernels_by_correlation
    ]
    runtime_starts = [int(call["start"]) for call in owned_runtime_calls]

    rows = []
    for idx, rng in enumerate(decode_ranges):
        groups: dict[str, dict[str, float | int]] = {}
        runtime_count = 0
        kernel_instance_count = 0
        seen_correlations: set[int] = set()
        start_idx = bisect_left(runtime_starts, int(rng["start"]))
        for runtime_call in owned_runtime_calls[start_idx:]:
            if runtime_call["start"] >= rng["end"]:
                break
            correlation_id = int(runtime_call["correlationId"])
            if correlation_id in seen_correlations:
                continue
            seen_correlations.add(correlation_id)
            runtime_count += 1
            for kernel in kernels_by_correlation[correlation_id]:
                fam = family(kernel["name"] or "")
                entry = groups.setdefault(fam, {"time_ms": 0.0, "instances": 0})
                entry["time_ms"] += (kernel["end"] - kernel["start"]) / 1e6
                entry["instances"] += 1
                kernel_instance_count += 1
        total = sum(float(item["time_ms"]) for item in groups.values())
        row = {
            "iteration": idx,
            "range_ms": (rng["end"] - rng["start"]) / 1e6,
            "kernel_total_ms": total,
            "attribution_method": "runtime_correlation_id",
            "owned_runtime_api_calls": runtime_count,
            "owned_kernel_instances": kernel_instance_count,
            "groups": groups,
        }
        rows.append(row)

    payload = {
        "source": str(db_path),
        "attribution_method": "runtime_correlation_id",
        "kernel_total_ms_definition": (
            "sum of full CUPTI GPU kernel durations whose CUPTI Runtime API "
            "correlationId matches the kernel correlationId and whose runtime "
            "API call starts inside the forward_decode NVTX CPU range"
        ),
        "kernel_record_count": len(kernels),
        "runtime_record_count": len(runtime_calls),
        "runtime_records_with_kernel_correlation": len(owned_runtime_calls),
        "decode_iterations": len(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    families = sorted({fam for row in rows for fam in row["groups"]})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "range_ms",
                "kernel_total_ms",
                "attribution_method",
                "owned_runtime_api_calls",
                "owned_kernel_instances",
            ]
            + [f"{fam}_ms" for fam in families],
        )
        writer.writeheader()
        for row in rows:
            flat = {
                "iteration": row["iteration"],
                "range_ms": row["range_ms"],
                "kernel_total_ms": row["kernel_total_ms"],
                "attribution_method": row["attribution_method"],
                "owned_runtime_api_calls": row["owned_runtime_api_calls"],
                "owned_kernel_instances": row["owned_kernel_instances"],
            }
            for fam in families:
                flat[f"{fam}_ms"] = row["groups"].get(fam, {}).get("time_ms", 0.0)
            writer.writerow(flat)

    print(json.dumps(payload, indent=2))
    print(f"JSON: {out_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
