#!/usr/bin/env python3
"""Attribute Nsight CUDA kernels to per-layer NVTX ranges by launch ownership."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(r"^visprune\.layer(?P<layer>\d+)\.(?P<phase>prefill|decode)(?:\.(?P<component>attn|mlp))?$")


def kernel_family(name: str) -> str:
    lowered = name.lower()
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
    ):
        return "elementwise_norm_activation"
    return "other"


def read_layer_events(path: str | None) -> dict[tuple[int, str, int], dict[str, str]]:
    if not path:
        return {}
    event_path = Path(path)
    if not event_path.exists():
        return {}
    counters: dict[tuple[int, str], int] = defaultdict(int)
    by_occurrence: dict[tuple[int, str, int], dict[str, str]] = {}
    with event_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["layer_idx"]), row["phase"])
            occurrence = counters[key]
            counters[key] += 1
            by_occurrence[(key[0], key[1], occurrence)] = row
    return by_occurrence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-events", default=None)
    args = parser.parse_args()

    db_path = Path(args.sqlite)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer_metadata = read_layer_events(args.layer_events)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ranges = conn.execute(
        """
        SELECT n.start, n.end, COALESCE(s.value, n.text, n.jsonText) AS text
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON n.textId = s.id
        WHERE COALESCE(s.value, n.text, n.jsonText) LIKE 'visprune.layer%'
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

    occurrence_counters: dict[tuple[int, str, str], int] = defaultdict(int)
    rows = []
    for rng in ranges:
        match = LAYER_RE.match(rng["text"])
        if not match:
            continue
        layer_idx = int(match.group("layer"))
        phase = match.group("phase")
        component = match.group("component") or "total"
        occurrence_key = (layer_idx, phase, component)
        occurrence = occurrence_counters[occurrence_key]
        occurrence_counters[occurrence_key] += 1

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
                family = kernel_family(kernel["name"] or "")
                entry = groups.setdefault(family, {"time_ms": 0.0, "instances": 0})
                entry["time_ms"] += (kernel["end"] - kernel["start"]) / 1e6
                entry["instances"] += 1
                kernel_instance_count += 1

        kernel_total = sum(float(item["time_ms"]) for item in groups.values())
        family_times = {family: float(item["time_ms"]) for family, item in groups.items()}
        dominant_family = max(family_times, key=family_times.get) if family_times else "none"
        metadata = layer_metadata.get((layer_idx, phase, occurrence), {})
        rows.append(
            {
                "range": rng["text"],
                "layer_idx": layer_idx,
                "phase": phase,
                "component": component,
                "occurrence": occurrence,
                "q_len": metadata.get("q_len"),
                "kv_len": metadata.get("kv_len"),
                "workload_type": metadata.get("workload_type"),
                "operator_path": metadata.get("operator_path"),
                "range_ms": (rng["end"] - rng["start"]) / 1e6,
                "kernel_total_ms": kernel_total,
                "attribution_method": "runtime_correlation_id",
                "owned_runtime_api_calls": runtime_count,
                "owned_kernel_instances": kernel_instance_count,
                "dominant_family": dominant_family,
                "families": groups,
            }
        )

    payload = {
        "source": str(db_path),
        "layer_events": args.layer_events,
        "attribution_method": "runtime_correlation_id",
        "kernel_total_ms_definition": (
            "sum of full CUPTI GPU kernel durations whose CUPTI Runtime API "
            "correlationId matches the kernel correlationId and whose runtime "
            "API call starts inside the NVTX CPU range"
        ),
        "kernel_record_count": len(kernels),
        "runtime_record_count": len(runtime_calls),
        "runtime_records_with_kernel_correlation": len(owned_runtime_calls),
        "range_count": len(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    families = sorted({family for row in rows for family in row["families"]})
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "range",
            "layer_idx",
            "phase",
            "component",
            "occurrence",
            "q_len",
            "kv_len",
            "workload_type",
            "range_ms",
            "kernel_total_ms",
            "attribution_method",
            "owned_runtime_api_calls",
            "owned_kernel_instances",
            "dominant_family",
        ] + [f"{family}_ms" for family in families]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name) for name in fieldnames}
            for family in families:
                flat[f"{family}_ms"] = row["families"].get(family, {}).get("time_ms", 0.0)
            writer.writerow(flat)

    print(json.dumps({"range_count": len(rows), "json": str(out_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
