#!/usr/bin/env python3
"""Generate a human_draft-style per-layer performance report for one run."""

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


def compact_layers(layers: list[int]) -> str:
    if not layers:
        return "-"
    layers = sorted(set(layers))
    groups: list[tuple[int, int]] = []
    start = prev = layers[0]
    for layer in layers[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        groups.append((start, prev))
        start = prev = layer
    groups.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)


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


def resolved_backend_name(clock: dict[str, Any]) -> str:
    backend = clock.get("resolved_visipruner_decode_backend") or {}
    if isinstance(backend, dict):
        return str(backend.get("selected") or "")
    return ""


def display_config_name(clock: dict[str, Any], resolved_backend: str) -> str:
    config = str(clock.get("config", "-"))
    if config == "visipruner-full-fa2" and resolved_backend == "vp_fa":
        return "visipruner-full-vp-fa (legacy output config: visipruner-full-fa2)"
    return config


def display_requested_backend(clock: dict[str, Any], resolved_backend: str) -> str:
    requested = str(clock.get("visipruner_decode_backend", "-"))
    if requested == "auto" and resolved_backend == "vp_fa":
        return "vp-fa (legacy output backend request: auto)"
    if requested == "fa2" and resolved_backend == "vp_fa":
        return "vp-fa (legacy output backend request: fa2)"
    return requested


def display_workload(row: dict[str, str], resolved_backend: str) -> str:
    workload = row.get("workload_type", "-")
    if resolved_backend == "vp_fa" and workload.startswith("eager_visipruner_"):
        return workload.replace("eager_visipruner_", "triton_vpfa_", 1)
    return workload


def display_operator(row: dict[str, str], resolved_backend: str) -> str:
    operator = row.get("operator_path", "-")
    if resolved_backend != "vp_fa" or row.get("phase") != "prefill":
        return operator
    operator = operator.replace(
        "eager QK^T/softmax/AV attention",
        "Triton causal VP-FA prefill attention",
    )
    operator = operator.replace(
        "shallow VisiPruner post-softmax edits",
        "shallow VisiPruner post-softmax edits in-kernel",
    )
    operator = operator.replace(
        "last-query pruning proxy/score computation",
        "Triton last-query pruning proxy where needed",
    )
    return operator


def clock_value(
    ranges: dict[str, dict[str, str]],
    name: str,
    field: str = "total_ms",
) -> str:
    row = ranges.get(name)
    return fmt_ms(row.get(field)) if row else "-"


def first_row(meta: dict[tuple[int, str], list[dict[str, str]]], layer: int, phase: str) -> dict[str, str]:
    return (meta.get((layer, phase)) or [{}])[0]


def last_row(meta: dict[tuple[int, str], list[dict[str, str]]], layer: int, phase: str) -> dict[str, str]:
    return (meta.get((layer, phase)) or [{}])[-1]


def grouped_layer_lines(rows: list[dict[str, str]], resolved_backend: str) -> list[str]:
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        try:
            layer = int(row["layer_idx"])
        except Exception:
            continue
        key = (
            row.get("q_len", "-"),
            row.get("kv_len", "-"),
            display_workload(row, resolved_backend),
            display_operator(row, resolved_backend),
        )
        groups[key].append(layer)
    lines = []
    for (q_len, kv_len, workload, operator), layers in sorted(
        groups.items(), key=lambda item: min(item[1])
    ):
        lines.append(
            f"- layer {compact_layers(layers)}: q_len={q_len}, kv_len={kv_len}, "
            f"workload={workload}; operator={operator}."
        )
    return lines


def decode_forward_count(meta: dict[tuple[int, str], list[dict[str, str]]]) -> int:
    counts = [len(rows) for (layer, phase), rows in meta.items() if phase == "decode"]
    return (max(counts) + 1) if counts else 1


def write_human_draft_style_summary(
    out: list[str],
    meta: dict[tuple[int, str], list[dict[str, str]]],
    resolved_backend: str,
) -> None:
    out.append("## Human-draft-style workload reading")
    out.append("")
    prefill_rows = [first_row(meta, layer, "prefill") for layer in range(32)]
    prefill_rows = [row for row in prefill_rows if row]
    out.append("forward 1")
    out.append("")
    out.extend(grouped_layer_lines(prefill_rows, resolved_backend) or ["- no prefill layer events found."])
    out.append("")

    decode_first = [first_row(meta, layer, "decode") for layer in range(32)]
    decode_first = [row for row in decode_first if row]
    out.append("forward 2")
    out.append("")
    out.extend(grouped_layer_lines(decode_first, resolved_backend) or ["- no decode layer events found."])
    out.append("")

    last_forward = decode_forward_count(meta)
    decode_last = [last_row(meta, layer, "decode") for layer in range(32)]
    decode_last = [row for row in decode_last if row]
    out.append(f"forward {last_forward}")
    out.append("")
    out.extend(grouped_layer_lines(decode_last, resolved_backend) or ["- no decode layer events found."])
    out.append("")


def write_prefill_table(
    out: list[str],
    ranges: dict[str, dict[str, str]],
    meta: dict[tuple[int, str], list[dict[str, str]]],
    nsys: dict[tuple[int, str, str], list[dict[str, str]]],
    resolved_backend: str,
) -> None:
    out.append("## Forward 1: Prefill layer workload and latency")
    out.append("")
    out.append("| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |")
    out.append("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
    for layer in range(32):
        row = first_row(meta, layer, "prefill")
        nsys_rows = nsys.get((layer, "prefill", "total"), [])
        out.append(
            "| {layer} | {q} | {kv} | {workload} | {operator} | {total} | {attn} | {mlp} | {nsys_range} | {nsys_kernel} | {family} |".format(
                layer=layer,
                q=row.get("q_len", "-"),
                kv=row.get("kv_len", "-"),
                workload=display_workload(row, resolved_backend),
                operator=display_operator(row, resolved_backend),
                total=clock_value(ranges, f"visprune.layer{layer:02d}.prefill"),
                attn=clock_value(ranges, f"visprune.layer{layer:02d}.prefill.attn"),
                mlp=clock_value(ranges, f"visprune.layer{layer:02d}.prefill.mlp"),
                nsys_range=fmt_ms(mean_field(nsys_rows, "range_ms")),
                nsys_kernel=fmt_ms(mean_field(nsys_rows, "kernel_total_ms")),
                family=dominant_family(nsys_rows),
            )
        )
    out.append("")


def decode_kv_summary(meta: dict[tuple[int, str], list[dict[str, str]]], layer: int) -> str:
    rows = meta.get((layer, "decode"), [])
    if not rows:
        return "-"
    return f"{rows[0].get('kv_len', '-')} -> {rows[-1].get('kv_len', '-')}"


def write_decode_table(
    out: list[str],
    ranges: dict[str, dict[str, str]],
    meta: dict[tuple[int, str], list[dict[str, str]]],
    nsys: dict[tuple[int, str, str], list[dict[str, str]]],
    resolved_backend: str,
) -> None:
    out.append("## Decode forwards: per-layer repeated-token workload and latency")
    out.append("")
    out.append("| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |")
    out.append("|---:|---|---|---|---:|---:|---:|---:|---:|---|")
    for layer in range(32):
        row = first_row(meta, layer, "decode")
        nsys_rows = nsys.get((layer, "decode", "total"), [])
        out.append(
            "| {layer} | {kv} | {workload} | {operator} | {total} | {attn} | {mlp} | {nsys_range} | {nsys_kernel} | {family} |".format(
                layer=layer,
                kv=decode_kv_summary(meta, layer),
                workload=display_workload(row, resolved_backend),
                operator=display_operator(row, resolved_backend),
                total=clock_value(ranges, f"visprune.layer{layer:02d}.decode", "mean_ms"),
                attn=clock_value(ranges, f"visprune.layer{layer:02d}.decode.attn", "mean_ms"),
                mlp=clock_value(ranges, f"visprune.layer{layer:02d}.decode.mlp", "mean_ms"),
                nsys_range=fmt_ms(mean_field(nsys_rows, "range_ms")),
                nsys_kernel=fmt_ms(mean_field(nsys_rows, "kernel_total_ms")),
                family=dominant_family(nsys_rows),
            )
        )
    out.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-json", required=True)
    parser.add_argument("--clock-ranges", required=True)
    parser.add_argument("--layer-events", required=True)
    parser.add_argument("--nsys-layer-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--human-draft", default="/workspace/VisiPrune/workload_analysis/human_draft.md")
    args = parser.parse_args()

    clock = read_json(args.clock_json)
    ranges = range_dict(read_csv(args.clock_ranges))
    events = read_csv(args.layer_events)
    nsys_rows = read_csv(args.nsys_layer_csv)
    meta = layer_meta(events)
    nsys = nsys_by_key(nsys_rows)
    resolved_backend = resolved_backend_name(clock)

    title = args.title or f"{clock.get('config', 'unknown')} Layer Performance Report"
    out: list[str] = []
    out.append(f"# {title}")
    out.append("")
    out.append("This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.")
    out.append("")
    out.append("## Run metadata")
    out.append("")
    out.append(f"- config: `{display_config_name(clock, resolved_backend)}`")
    out.append(f"- description: {clock.get('description', '-')}")
    out.append(f"- max_new_tokens: `{clock.get('max_new_tokens', '-')}`")
    out.append(f"- use_flash_attn: `{clock.get('use_flash_attn', '-')}`")
    out.append(f"- use_visipruner: `{clock.get('use_visipruner', '-')}`")
    out.append(f"- visipruner_decode_backend: `{display_requested_backend(clock, resolved_backend)}`")
    out.append(f"- resolved_visipruner_decode_backend: `{resolved_backend or '-'}`")
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
    write_human_draft_style_summary(out, meta, resolved_backend)
    write_prefill_table(out, ranges, meta, nsys, resolved_backend)
    write_decode_table(out, ranges, meta, nsys, resolved_backend)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
