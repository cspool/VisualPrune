#!/usr/bin/env python3
"""Generate a human_draft-style CP/VP-FA layer performance report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def range_dict(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["range"]: row for row in rows if row.get("range")}


def layer_meta(events: list[dict[str, str]]) -> dict[tuple[int, str], list[dict[str, str]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        try:
            key = (int(row["layer_idx"]), row["phase"])
        except Exception:
            continue
        grouped[key].append(row)
    return grouped


def nsys_by_key(rows: list[dict[str, str]]) -> dict[tuple[int, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            key = (int(row["layer_idx"]), row["phase"], row["component"])
        except Exception:
            continue
        grouped[key].append(row)
    return grouped


def mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, 0.0)))
        except Exception:
            pass
    return sum(values) / len(values) if values else None


def dominant_family(rows: list[dict[str, str]]) -> str:
    values_by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if not key.endswith("_ms"):
                continue
            family = key[: -len("_ms")]
            if family in {"range", "kernel_total", "pct_request", "pct_generate"}:
                continue
            try:
                values_by_family[family].append(float(value))
            except Exception:
                pass
    if not values_by_family:
        return "-"
    means = {
        family: sum(values) / len(values)
        for family, values in values_by_family.items()
        if values
    }
    if not means:
        return "-"
    family, value = max(means.items(), key=lambda item: item[1])
    return f"{family} ({value:.3f} ms)"


def clock_value(ranges: dict[str, dict[str, str]], name: str, field: str = "total_ms") -> str:
    row = ranges.get(name)
    return fmt_ms(row.get(field)) if row else "-"


def decode_kv_summary(meta: dict[tuple[int, str], list[dict[str, str]]], layer: int) -> str:
    rows = meta.get((layer, "decode"), [])
    if not rows:
        return "-"
    first = rows[0].get("kv_len", "-")
    last = rows[-1].get("kv_len", "-")
    return f"{first} -> {last}"


def write_prefill_table(
    out: list[str],
    ranges: dict[str, dict[str, str]],
    meta: dict[tuple[int, str], list[dict[str, str]]],
    nsys: dict[tuple[int, str, str], list[dict[str, str]]],
) -> None:
    out.append("## Forward 1: Prefill layer workload and latency")
    out.append("")
    out.append("| layer | q_len | kv_len | workload type | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |")
    out.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for layer in range(32):
        rows = meta.get((layer, "prefill"), [])
        row = rows[0] if rows else {}
        nsys_rows = nsys.get((layer, "prefill", "total"), [])
        out.append(
            "| {layer} | {q} | {kv} | {workload} | {total} | {attn} | {mlp} | {nsys_range} | {nsys_kernel} | {family} |".format(
                layer=layer,
                q=row.get("q_len", "-"),
                kv=row.get("kv_len", "-"),
                workload=row.get("workload_type", "-"),
                total=clock_value(ranges, f"visprune.layer{layer:02d}.prefill"),
                attn=clock_value(ranges, f"visprune.layer{layer:02d}.prefill.attn"),
                mlp=clock_value(ranges, f"visprune.layer{layer:02d}.prefill.mlp"),
                nsys_range=fmt_ms(mean_field(nsys_rows, "range_ms")),
                nsys_kernel=fmt_ms(mean_field(nsys_rows, "kernel_total_ms")),
                family=dominant_family(nsys_rows),
            )
        )
    out.append("")


def write_decode_table(
    out: list[str],
    ranges: dict[str, dict[str, str]],
    meta: dict[tuple[int, str], list[dict[str, str]]],
    nsys: dict[tuple[int, str, str], list[dict[str, str]]],
) -> None:
    out.append("## Decode forwards: per-layer repeated-token workload and latency")
    out.append("")
    out.append("| layer | decode kv_len first -> last | workload type | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |")
    out.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for layer in range(32):
        rows = meta.get((layer, "decode"), [])
        row = rows[0] if rows else {}
        nsys_rows = nsys.get((layer, "decode", "total"), [])
        out.append(
            "| {layer} | {kv} | {workload} | {total} | {attn} | {mlp} | {nsys_range} | {nsys_kernel} | {family} |".format(
                layer=layer,
                kv=decode_kv_summary(meta, layer),
                workload=row.get("workload_type", "-"),
                total=clock_value(ranges, f"visprune.layer{layer:02d}.decode", "mean_ms"),
                attn=clock_value(ranges, f"visprune.layer{layer:02d}.decode.attn", "mean_ms"),
                mlp=clock_value(ranges, f"visprune.layer{layer:02d}.decode.mlp", "mean_ms"),
                nsys_range=fmt_ms(mean_field(nsys_rows, "range_ms")),
                nsys_kernel=fmt_ms(mean_field(nsys_rows, "kernel_total_ms")),
                family=dominant_family(nsys_rows),
            )
        )
    out.append("")


def write_human_draft_style_summary(
    out: list[str],
    meta: dict[tuple[int, str], list[dict[str, str]]],
) -> None:
    out.append("## Human-draft-style workload reading")
    out.append("")
    prefill18 = (meta.get((18, "prefill")) or [{}])[0]
    prefill19 = (meta.get((19, "prefill")) or [{}])[0]
    prefill28 = (meta.get((28, "prefill")) or [{}])[0]
    out.append("forward 1")
    out.append("")
    out.append("- layer 0: full prefill, Triton VP-FA attention applies the layer-0 shallow visual mass fold in-kernel.")
    out.append("- layer 1-5: full prefill, Triton VP-FA attention applies shallow text-to-vision masking in-kernel.")
    out.append("- layer 7-18: full prefill keeps the expanded sequence and computes the last-query pruning proxy where selection is needed.")
    out.append(
        f"- layer 18: observed q_len={prefill18.get('q_len', '-')}, kv_len={prefill18.get('kv_len', '-')}; this is the middle-selection decision point when selected visual tokens are emitted."
    )
    out.append(
        f"- layer 19-27: compact prefill after middle pruning, observed layer19 q_len={prefill19.get('q_len', '-')}, kv_len={prefill19.get('kv_len', '-')}."
    )
    out.append(
        f"- layer 28-31: deep-removed prefill, observed layer28 q_len={prefill28.get('q_len', '-')}, kv_len={prefill28.get('kv_len', '-')}."
    )
    out.append("")
    out.append("forward 2")
    out.append("")
    for layer in [18, 19, 28]:
        rows = meta.get((layer, "decode"), [])
        row = rows[0] if rows else {}
        out.append(
            f"- layer {layer}: q_len={row.get('q_len', '-')}, kv_len={row.get('kv_len', '-')}, workload={row.get('workload_type', '-')}."
        )
    out.append("")
    out.append("forward 32")
    out.append("")
    for layer in [18, 19, 28]:
        rows = meta.get((layer, "decode"), [])
        row = rows[-1] if rows else {}
        out.append(
            f"- layer {layer}: q_len={row.get('q_len', '-')}, kv_len={row.get('kv_len', '-')}, workload={row.get('workload_type', '-')}."
        )
    out.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-json", required=True)
    parser.add_argument("--clock-ranges", required=True)
    parser.add_argument("--layer-events", required=True)
    parser.add_argument("--nsys-layer-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--human-draft", default="/workspace/VisiPrune/workload_analysis/human_draft.md")
    args = parser.parse_args()

    clock = read_json(args.clock_json)
    ranges = range_dict(read_csv(args.clock_ranges))
    events = read_csv(args.layer_events)
    nsys_rows = read_csv(args.nsys_layer_csv)
    meta = layer_meta(events)
    nsys = nsys_by_key(nsys_rows)

    out: list[str] = []
    out.append("# Triton VP-FA Single-Request Layer Performance Report")
    out.append("")
    out.append("This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.")
    out.append("")
    out.append("## Data sources")
    out.append("")
    out.append(f"- clock json: `{args.clock_json}`")
    out.append(f"- clock ranges: `{args.clock_ranges}`")
    out.append(f"- layer events: `{args.layer_events}`")
    out.append(f"- Nsight layer kernels: `{args.nsys_layer_csv}`")
    out.append("- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.")
    out.append(f"- human draft reference: `{args.human_draft}`")
    out.append("")
    out.append("## End-to-end clock summary")
    out.append("")
    derived = clock.get("derived_ms", {})
    out.append("| metric | ms |")
    out.append("|---|---:|")
    for key in [
        "request_total_ms",
        "generate_total_ms",
        "prepare_multimodal_ms",
        "vision_encode_project_ms",
        "forward_prefill_ms",
        "forward_decode_sum_ms",
        "value_aware_token_selection_ms",
    ]:
        out.append(f"| {key} | {fmt_ms(derived.get(key))} |")
    out.append("")
    write_human_draft_style_summary(out, meta)
    write_prefill_table(out, ranges, meta, nsys)
    write_decode_table(out, ranges, meta, nsys)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
