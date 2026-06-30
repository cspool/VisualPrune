from __future__ import annotations

import json
import re
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from dispatch_io import iter_arg_tensors, output_tensor_shape, parse_json_field, summarize_ops, tensor_ids_from_row, tensor_shape


def parse_event_id(event_id: str) -> dict[str, int | str | None]:
    match = re.fullmatch(r"input(\d+)_layer(\d+)", event_id)
    if not match:
        return {"input_id": None, "layer_id": None, "event_id": event_id}
    return {
        "input_id": int(match.group(1)),
        "layer_id": int(match.group(2)),
        "event_id": event_id,
    }


def _first_int(rows: list[dict[str, str]], key: str) -> int | None:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _first_text(rows: list[dict[str, str]], key: str) -> str | None:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _row_index(row: dict[str, str]) -> int:
    try:
        return int(row.get("event_op_index") or 0)
    except ValueError:
        return 0


def _row_global_index(row: dict[str, str]) -> int:
    try:
        return int(row.get("global_op_index") or 0)
    except ValueError:
        return 0


def _producer_record(row: dict[str, str], tensor_id: str) -> dict[str, Any]:
    return {
        "tensor_id": tensor_id,
        "event_detail_id": row.get("event_detail_id", ""),
        "event_id": row.get("event_id", ""),
        "event_op_index": _row_index(row),
        "global_op_index": _row_global_index(row),
        "op_name": row.get("op_name", ""),
        "op_schema": row.get("op_schema", ""),
        "module_path": row.get("module_path", ""),
        "module_relative_path": row.get("module_relative_path", ""),
    }


def build_tensor_dataflow(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Build observed producer-consumer edges from dispatch tensor ids only."""
    sorted_rows = sorted(rows, key=_row_index)
    producers: dict[str, dict[str, Any]] = {}
    ops: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    external_inputs: list[str] = []
    consumed: list[str] = []
    produced: list[str] = []

    for row in sorted_rows:
        input_ids = tensor_ids_from_row(row, "input_tensor_ids")
        output_ids = tensor_ids_from_row(row, "output_tensor_ids")
        op_dependencies: list[dict[str, Any]] = []
        op_external_inputs: list[str] = []
        op_index = _row_index(row)
        op_name = row.get("op_name", "")
        for tensor_id in input_ids:
            consumed.append(tensor_id)
            producer = producers.get(tensor_id)
            if producer is None:
                external_inputs.append(tensor_id)
                op_external_inputs.append(tensor_id)
                continue
            edge = {
                "tensor_id": tensor_id,
                "producer_event_detail_id": producer.get("event_detail_id", ""),
                "producer_event_id": producer.get("event_id", ""),
                "producer_event_op_index": producer.get("event_op_index", 0),
                "producer_global_op_index": producer.get("global_op_index", 0),
                "producer_op_name": producer.get("op_name", ""),
                "consumer_event_detail_id": row.get("event_detail_id", ""),
                "consumer_event_id": row.get("event_id", ""),
                "consumer_event_op_index": op_index,
                "consumer_global_op_index": _row_global_index(row),
                "consumer_op_name": op_name,
            }
            op_dependencies.append(edge)
            edges.append(edge)
        ops.append({
            "event_detail_id": row.get("event_detail_id", ""),
            "event_op_index": op_index,
            "global_op_index": _row_global_index(row),
            "op_name": op_name,
            "module_path": row.get("module_path", ""),
            "module_relative_path": row.get("module_relative_path", ""),
            "input_tensor_ids": input_ids,
            "output_tensor_ids": output_ids,
            "external_input_tensor_ids": unique_preserve(op_external_inputs),
            "data_dependencies": op_dependencies,
        })
        for tensor_id in output_ids:
            produced.append(tensor_id)
            producers[tensor_id] = _producer_record(row, tensor_id)

    consumed_set = set(consumed)
    return {
        "op_count": len(sorted_rows),
        "edge_count": len(edges),
        "external_input_tensor_ids": unique_preserve(external_inputs),
        "produced_tensor_ids": unique_preserve(produced),
        "final_output_tensor_ids": [tensor_id for tensor_id in unique_preserve(produced) if tensor_id not in consumed_set],
        "ops": ops,
        "edges": edges,
    }


def build_stage_tensor_io(core_evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage_io: dict[str, dict[str, Any]] = {}
    for stage, evidence in core_evidence.items():
        rows = evidence.get("evidence_ops", []) if isinstance(evidence, dict) else []
        produced_so_far: list[str] = []
        consumed: list[str] = []
        produced: list[str] = []
        dependencies: list[dict[str, Any]] = []
        for item in rows:
            input_ids = [str(value) for value in item.get("input_tensor_ids", [])]
            output_ids = [str(value) for value in item.get("output_tensor_ids", [])]
            consumed.extend(input_ids)
            for tensor_id in input_ids:
                if tensor_id in produced_so_far:
                    dependencies.append({
                        "tensor_id": tensor_id,
                        "consumer_event_op_index": item.get("event_op_index", 0),
                        "consumer_op_name": item.get("op_name", ""),
                    })
            produced.extend(output_ids)
            produced_so_far.extend(output_ids)
        produced_set = set(produced)
        consumed_set = set(consumed)
        stage_io[stage] = {
            "input_tensor_ids": [tensor_id for tensor_id in unique_preserve(consumed) if tensor_id not in produced_set],
            "output_tensor_ids": [tensor_id for tensor_id in unique_preserve(produced) if tensor_id not in consumed_set],
            "internal_tensor_ids": [tensor_id for tensor_id in unique_preserve(produced) if tensor_id in consumed_set],
            "evidence_input_tensor_ids": unique_preserve(consumed),
            "evidence_output_tensor_ids": unique_preserve(produced),
            "data_dependencies": dependencies,
        }
    return stage_io


def build_dispatch_op_coverage(
    rows: list[dict[str, str]],
    module_split: list[dict[str, Any]],
    tensor_dataflow: dict[str, Any],
    core_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Account for every dispatch op row in process and split-subprocess evidence."""
    module_by_index: dict[int, dict[str, Any]] = {}
    duplicate_module_indices: list[int] = []
    for module in module_split:
        for raw_index in module.get("event_op_indices", []):
            index = int(raw_index)
            if index in module_by_index:
                duplicate_module_indices.append(index)
            module_by_index[index] = module

    dataflow_by_index = {
        int(item.get("event_op_index") or 0): item
        for item in tensor_dataflow.get("ops", [])
    }

    stages_by_index: dict[int, list[str]] = {}
    for stage, evidence in core_evidence.items():
        evidence_rows = evidence.get("evidence_ops", []) if isinstance(evidence, dict) else []
        for item in evidence_rows:
            index = int(item.get("event_op_index") or 0)
            if index <= 0:
                continue
            stages_by_index.setdefault(index, []).append(str(stage))

    ops: list[dict[str, Any]] = []
    seen_indices: list[int] = []
    missing_from_module_split: list[int] = []
    missing_from_tensor_dataflow: list[int] = []

    for row in sorted(rows, key=_row_index):
        index = _row_index(row)
        seen_indices.append(index)
        module = module_by_index.get(index)
        dataflow_op = dataflow_by_index.get(index)
        input_tensor_ids = tensor_ids_from_row(row, "input_tensor_ids")
        output_tensor_ids = tensor_ids_from_row(row, "output_tensor_ids")
        if module is None:
            missing_from_module_split.append(index)
        if dataflow_op is None:
            missing_from_tensor_dataflow.append(index)
        stage_evidence = unique_preserve(stages_by_index.get(index, []))
        ops.append({
            "event_detail_id": row.get("event_detail_id", ""),
            "event_op_index": index,
            "global_op_index": _row_global_index(row),
            "op_name": row.get("op_name", ""),
            "op_schema": row.get("op_schema", ""),
            "module_path": row.get("module_path", ""),
            "module_relative_path": row.get("module_relative_path", ""),
            "runtime_subprocess": (module or {}).get("module_path", row.get("module_path", "")),
            "module_split_covered": module is not None,
            "tensor_dataflow_covered": dataflow_op is not None,
            "stage_evidence": stage_evidence,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": output_tensor_ids,
        })

    expected_indices = list(range(1, len(rows) + 1))
    covered_once = sorted(index for index in seen_indices if seen_indices.count(index) == 1)
    duplicate_row_indices = sorted(index for index, count in Counter(seen_indices).items() if count > 1)
    missing_row_indices = [index for index in expected_indices if index not in set(seen_indices)]
    return {
        "op_count": len(rows),
        "covered_op_count": len(ops),
        "expected_event_op_indices": expected_indices,
        "covered_event_op_indices": seen_indices,
        "covered_once_event_op_indices": covered_once,
        "missing_event_op_indices": missing_row_indices,
        "duplicate_event_op_indices": duplicate_row_indices,
        "missing_from_module_split": missing_from_module_split,
        "duplicate_module_split_indices": sorted(set(duplicate_module_indices)),
        "missing_from_tensor_dataflow": missing_from_tensor_dataflow,
        "ops": ops,
    }


def infer_original_dimensions(rows: list[dict[str, str]]) -> dict[str, Any]:
    seq = _first_int(rows, "q_len")
    kv_len = _first_int(rows, "kv_len")
    sequence_dims = {dim for dim in (seq, kv_len) if dim is not None}
    hidden = None
    heads = None
    head_dim = None
    ffn = None
    hidden_candidates: Counter[int] = Counter()
    head_candidates: Counter[int] = Counter()

    all_shapes: list[list[int]] = []
    for row in rows:
        output_shape = output_tensor_shape(row)
        if output_shape:
            all_shapes.append(output_shape)
        for arg in iter_arg_tensors(row):
            shape = tensor_shape(arg)
            if shape:
                all_shapes.append(shape)

    for shape in all_shapes:
        if len(shape) == 3 and shape[0] == 1:
            if shape[-1] >= 512 and shape[-1] not in sequence_dims:
                hidden_candidates[shape[-1]] += 1
            if seq is None and shape[1] > 1:
                seq = shape[1]
        if len(shape) == 4 and shape[0] == 1:
            if shape[-1] > 1 and shape[-1] <= 256 and shape[-2] != shape[-1]:
                head_dim = shape[-1]
            for dim in shape[1:3]:
                if 1 < dim <= 128:
                    head_candidates[dim] += 1

    if hidden_candidates:
        hidden = hidden_candidates.most_common(1)[0][0]
        larger = [dim for dim in hidden_candidates if dim > hidden]
        if larger:
            ffn = max(larger)

    if hidden is not None and heads is not None and head_dim is None and hidden % heads == 0:
        head_dim = hidden // heads
    if hidden is not None and head_dim is not None and hidden % head_dim == 0:
        heads = hidden // head_dim
    elif head_candidates:
        heads = head_candidates.most_common(1)[0][0]

    visual_start, visual_end, tail_start = infer_attention_boundaries(rows, seq or kv_len)
    return {
        "seq": seq,
        "kv_len": kv_len,
        "hidden": hidden,
        "heads": heads,
        "head_dim": head_dim,
        "ffn": ffn,
        "visual_start": visual_start,
        "visual_end": visual_end,
        "tail_start": tail_start,
    }


def infer_attention_boundaries(rows: list[dict[str, str]], seq: int | None) -> tuple[int | None, int | None, int | None]:
    if not any(row.get("op_name") in {"fill_.Tensor", "copy_.default", "sum.dim_IntList"} for row in rows):
        return None, None, None

    starts_dim2: Counter[int] = Counter()
    starts_dim3: Counter[int] = Counter()
    ends_dim3: Counter[int] = Counter()
    for row in rows:
        if row.get("op_name") != "slice.Tensor":
            continue
        args = parse_json_field(row.get("args", ""))
        if not isinstance(args, list) or len(args) < 4:
            continue
        dim = args[1]
        start = args[2]
        end = args[3]
        if not isinstance(dim, int) or not isinstance(start, int):
            continue
        if dim == 2 and start > 0:
            starts_dim2[start] += 1
        if dim == 3:
            if start > 0:
                starts_dim3[start] += 1
            if isinstance(end, int) and end > 0 and end < 9_000_000_000_000_000_000:
                ends_dim3[end] += 1

    tail_start = max(starts_dim2) if starts_dim2 else None
    visual_start = min(starts_dim3) if starts_dim3 else None
    visual_end_candidates = [value for value in ends_dim3 if visual_start is None or value > visual_start]
    visual_end = min(visual_end_candidates) if visual_end_candidates else tail_start
    if visual_end is None and seq is not None:
        visual_end = seq
    return visual_start, visual_end, tail_start


def infer_small_config(original: dict[str, Any], small_seq: int, small_hidden: int, small_heads: int, small_ffn: int) -> dict[str, int]:
    original_q = original.get("seq")
    original_kv = original.get("kv_len")
    if isinstance(original_q, int) and isinstance(original_kv, int) and original_q != original_kv:
        q_seq = 1 if original_q == 1 else min(small_seq, max(1, original_q))
        kv_seq = max(q_seq, small_seq)
    else:
        q_seq = small_seq
        kv_seq = small_seq

    prefix_tokens = max(1, small_seq // 5)
    tail_tokens = max(1, small_seq // 5)
    visual_start = prefix_tokens
    visual_end = max(visual_start + 1, small_seq - tail_tokens)
    tail_start = visual_end

    if small_hidden % small_heads != 0:
        raise ValueError(f"small_hidden={small_hidden} must be divisible by small_heads={small_heads}")
    return {
        "seq": small_seq,
        "q_seq": q_seq,
        "kv_seq": kv_seq,
        "hidden": small_hidden,
        "heads": small_heads,
        "head_dim": small_hidden // small_heads,
        "visual_start": visual_start,
        "visual_end": visual_end,
        "tail_start": tail_start,
        "ffn": small_ffn,
    }


def build_layer_summary(rows: list[dict[str, str]], event_id: str, small_config: dict[str, int]) -> dict[str, Any]:
    parsed_event = parse_event_id(event_id)
    original = infer_original_dimensions(rows)
    return {
        **parsed_event,
        "row_count": len(rows),
        "phase": _first_text(rows, "phase"),
        "priority": _first_text(rows, "priority"),
        "token_state": _first_text(rows, "token_state"),
        "visipruner_role": _first_text(rows, "visipruner_role"),
        "q_len": _first_int(rows, "q_len"),
        "kv_len": _first_int(rows, "kv_len"),
        "past_len": _first_int(rows, "past_len"),
        "original_dimensions": original,
        "small_config": small_config,
        "op_counts": summarize_ops(rows),
    }


def build_module_split(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        module_path = row.get("module_path") or "<no_module_context>"
        group = groups.get(module_path)
        event_op_index = int(row.get("event_op_index") or 0)
        op_name = row.get("op_name", "")
        input_tensor_ids = tensor_ids_from_row(row, "input_tensor_ids")
        output_tensor_ids = tensor_ids_from_row(row, "output_tensor_ids")
        if group is None:
            group = {
                "module_path": module_path,
                "module_relative_path": row.get("module_relative_path", ""),
                "module_type": row.get("module_type", ""),
                "module_class": row.get("module_class", ""),
                "module_forward_file": row.get("module_forward_file", ""),
                "module_forward_lineno": row.get("module_forward_lineno", ""),
                "first_event_op_index": event_op_index,
                "last_event_op_index": event_op_index,
                "op_count": 0,
                "op_counts": Counter(),
                "event_op_indices": [],
                "input_tensor_ids": [],
                "output_tensor_ids": [],
            }
            groups[module_path] = group
        group["op_count"] += 1
        group["op_counts"][op_name] += 1
        group["event_op_indices"].append(event_op_index)
        group["input_tensor_ids"].extend(input_tensor_ids)
        group["output_tensor_ids"].extend(output_tensor_ids)
        if event_op_index:
            first = group.get("first_event_op_index") or event_op_index
            group["first_event_op_index"] = min(int(first), event_op_index)
            group["last_event_op_index"] = max(int(group.get("last_event_op_index") or event_op_index), event_op_index)

    result: list[dict[str, Any]] = []
    for group in groups.values():
        op_counts = dict(sorted(group["op_counts"].items(), key=lambda item: (-item[1], item[0])))
        event_op_indices = sorted({int(value) for value in group["event_op_indices"] if int(value) > 0})
        input_tensor_ids = unique_preserve([str(value) for value in group["input_tensor_ids"]])
        output_tensor_ids = unique_preserve([str(value) for value in group["output_tensor_ids"]])
        output_set = set(output_tensor_ids)
        input_set = set(input_tensor_ids)
        result.append({
            "module_path": group["module_path"],
            "module_relative_path": group["module_relative_path"],
            "module_type": group["module_type"],
            "module_class": group["module_class"],
            "module_forward_file": group["module_forward_file"],
            "module_forward_lineno": group["module_forward_lineno"],
            "first_event_op_index": group["first_event_op_index"],
            "last_event_op_index": group["last_event_op_index"],
            "op_count": group["op_count"],
            "op_counts": op_counts,
            "event_op_indices": event_op_indices,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": output_tensor_ids,
            "external_input_tensor_ids": [tensor_id for tensor_id in input_tensor_ids if tensor_id not in output_set],
            "module_output_tensor_ids": [tensor_id for tensor_id in output_tensor_ids if tensor_id not in input_set],
        })
    return sorted(result, key=lambda item: (int(item.get("first_event_op_index") or 0), str(item.get("module_path") or "")))


def write_tensor_dataflow_outputs(analysis_dir: Path, event_id: str, dataflow: dict[str, Any]) -> dict[str, str]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "tensor_dataflow.json"
    csv_path = analysis_dir / "tensor_dataflow_edges.csv"
    md_path = analysis_dir / "tensor_dataflow.md"

    json_path.write_text(json.dumps(dataflow, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "tensor_id",
        "producer_event_op_index",
        "producer_op_name",
        "consumer_event_op_index",
        "consumer_op_name",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for edge in dataflow.get("edges", []):
            writer.writerow({field: edge.get(field, "") for field in csv_fields})

    lines = [
        f"# {event_id} Tensor Dataflow",
        "",
        "This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.",
        "",
        f"- ops: `{dataflow.get('op_count', 0)}`",
        f"- observed producer-consumer edges: `{dataflow.get('edge_count', 0)}`",
        f"- external input tensor ids: `{len(dataflow.get('external_input_tensor_ids', []))}`",
        f"- produced tensor ids: `{len(dataflow.get('produced_tensor_ids', []))}`",
        f"- final output tensor ids: `{len(dataflow.get('final_output_tensor_ids', []))}`",
        "",
        "## First Observed Edges",
        "",
    ]
    for edge in dataflow.get("edges", [])[:40]:
        lines.append(
            "- `{tensor}`: `#{pidx} {pop}` -> `#{cidx} {cop}`".format(
                tensor=edge.get("tensor_id", ""),
                pidx=edge.get("producer_event_op_index", ""),
                pop=edge.get("producer_op_name", ""),
                cidx=edge.get("consumer_event_op_index", ""),
                cop=edge.get("consumer_op_name", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "tensor_dataflow_json": str(json_path),
        "tensor_dataflow_edges_csv": str(csv_path),
        "tensor_dataflow_markdown": str(md_path),
    }


def write_dispatch_op_coverage_outputs(analysis_dir: Path, event_id: str, coverage: dict[str, Any]) -> dict[str, str]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "dispatch_op_coverage.json"
    csv_path = analysis_dir / "dispatch_op_coverage.csv"
    md_path = analysis_dir / "dispatch_op_coverage.md"

    json_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "event_op_index",
        "global_op_index",
        "op_name",
        "op_schema",
        "module_path",
        "module_relative_path",
        "runtime_subprocess",
        "module_split_covered",
        "tensor_dataflow_covered",
        "stage_evidence",
        "input_tensor_ids",
        "output_tensor_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in coverage.get("ops", []):
            row = {field: item.get(field, "") for field in csv_fields}
            for key in ["stage_evidence", "input_tensor_ids", "output_tensor_ids"]:
                row[key] = json.dumps(item.get(key, []), separators=(",", ":"))
            writer.writerow(row)

    lines = [
        f"# {event_id} Dispatch Op Coverage",
        "",
        "This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.",
        "",
        f"- ops in dispatch rows: `{coverage.get('op_count', 0)}`",
        f"- ops listed in coverage: `{coverage.get('covered_op_count', 0)}`",
        f"- missing event_op_index values: `{coverage.get('missing_event_op_indices', [])}`",
        f"- duplicate event_op_index values: `{coverage.get('duplicate_event_op_indices', [])}`",
        f"- missing from module_split: `{coverage.get('missing_from_module_split', [])}`",
        f"- missing from tensor_dataflow: `{coverage.get('missing_from_tensor_dataflow', [])}`",
        "",
        "| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for item in coverage.get("ops", []):
        lines.append(
            "| {idx} | `{op}` | `{subprocess}` | `{module}` | `{dataflow}` | `{stages}` | `{inputs}` | `{outputs}` |".format(
                idx=item.get("event_op_index", ""),
                op=item.get("op_name", ""),
                subprocess=item.get("runtime_subprocess", ""),
                module=item.get("module_split_covered", ""),
                dataflow=item.get("tensor_dataflow_covered", ""),
                stages=", ".join(item.get("stage_evidence", [])),
                inputs=", ".join(item.get("input_tensor_ids", [])[:6]),
                outputs=", ".join(item.get("output_tensor_ids", [])[:6]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dispatch_op_coverage_json": str(json_path),
        "dispatch_op_coverage_csv": str(csv_path),
        "dispatch_op_coverage_markdown": str(md_path),
    }


def write_module_split_outputs(analysis_dir: Path, event_id: str, module_split: list[dict[str, Any]]) -> dict[str, str]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "module_split.json"
    csv_path = analysis_dir / "module_split.csv"
    md_path = analysis_dir / "module_split.md"

    json_path.write_text(json.dumps(module_split, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "module_path",
        "module_relative_path",
        "module_type",
        "module_class",
        "module_forward_file",
        "module_forward_lineno",
        "first_event_op_index",
        "last_event_op_index",
        "op_count",
        "event_op_indices",
        "op_counts",
        "input_tensor_ids",
        "output_tensor_ids",
        "external_input_tensor_ids",
        "module_output_tensor_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in module_split:
            row = {field: item.get(field, "") for field in csv_fields}
            row["event_op_indices"] = json.dumps(item.get("event_op_indices", []), separators=(",", ":"))
            row["op_counts"] = json.dumps(item.get("op_counts", {}), sort_keys=True)
            for key in ["input_tensor_ids", "output_tensor_ids", "external_input_tensor_ids", "module_output_tensor_ids"]:
                row[key] = json.dumps(item.get(key, []), separators=(",", ":"))
            writer.writerow(row)

    lines = [
        f"# {event_id} Runtime Module Split",
        "",
        "This split is derived directly from sampled `module_*` columns in this layer's dispatch rows.",
        "",
        "| First op | Last op | Ops | Op indices | Module | Type | Tensor ID inputs | Tensor ID outputs | Forward source | Top ATen ops |",
        "|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for item in module_split:
        top_ops = ", ".join(
            f"`{name}` x{count}"
            for name, count in list((item.get("op_counts") or {}).items())[:6]
        )
        source = f"{item.get('module_forward_file', '')}:{item.get('module_forward_lineno', '')}"
        indices = ",".join(str(value) for value in item.get("event_op_indices", []))
        lines.append(
            "| {first} | {last} | {count} | `{indices}` | `{module}` | `{typ}` | `{inputs}` | `{outputs}` | `{source}` | {ops} |".format(
                first=item.get("first_event_op_index", ""),
                last=item.get("last_event_op_index", ""),
                count=item.get("op_count", ""),
                indices=indices,
                module=item.get("module_path", ""),
                typ=item.get("module_type", ""),
                inputs=", ".join(item.get("external_input_tensor_ids", [])[:8]),
                outputs=", ".join(item.get("module_output_tensor_ids", [])[:8]),
                source=source,
                ops=top_ops,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "module_split_json": str(json_path),
        "module_split_csv": str(csv_path),
        "module_split_markdown": str(md_path),
    }


def write_process_markdown(path: Path, summary: dict[str, Any]) -> None:
    cfg = summary["small_config"]
    original = summary["original_dimensions"]
    features = summary.get("dispatch_features") or {}
    expected_stages = features.get("expected_stages") or [
        "input_rmsnorm",
        "qkv_projection",
        "rope",
        "attention",
        "attention_output",
        "mlp",
    ]
    stage_descriptions = {
        "input_rmsnorm": "Input RMSNorm: cast/square/mean/rsqrt/multiply weight.",
        "qkv_projection": "Q/K/V projection: linear projections and head split.",
        "rope": "RoPE: gather cos/sin by position ids and rotate half channels.",
        "kv_cache_concat": "K/V cache concat: append decode token K/V to past K/V cache.",
        "attention": "Attention: q @ k^T, scale, add mask, softmax.",
        "visual_adjust": "Visual attention adjustment: apply the dispatch-proven visual clear/fold variant.",
        "visipruner_similarity_check": "VisiPrune similarity/check: cosine-similarity probe or check sub-process.",
        "attention_output": "Attention output: attention @ value, merge heads, output projection, residual add.",
        "mlp": "MLP: post RMSNorm, SiLU gate, up projection, gated product, down projection, final residual.",
    }
    lines = [
        f"# {summary['event_id']} Dispatch Analysis",
        "",
        "## Source",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- phase: `{summary.get('phase')}`",
        f"- token_state: `{summary.get('token_state')}`",
        f"- role: `{summary.get('visipruner_role')}`",
        "",
        "## Original Dimensions Inferred From Dispatch",
        "",
        "```json",
        json.dumps(original, indent=2),
        "```",
        "",
        "## Small Tensor Config",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## Reconstructed Compute Stages",
        "",
    ]
    for index, stage in enumerate(expected_stages, start=1):
        lines.append(f"{index}. `{stage}`: {stage_descriptions.get(stage, 'dispatch-derived stage.')}")
    tensor_dataflow = summary.get("tensor_dataflow") or {}
    if tensor_dataflow:
        lines.extend([
            "",
            "## Tensor ID Data Dependencies",
            "",
            "Data dependencies are derived from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.",
            "",
            f"- tensor-id producer-consumer edges: `{tensor_dataflow.get('edge_count')}`",
            f"- external input tensor ids: `{tensor_dataflow.get('external_input_tensor_ids', [])}`",
            f"- final output tensor ids: `{tensor_dataflow.get('final_output_tensor_ids', [])}`",
        ])
    dispatch_op_coverage = summary.get("dispatch_op_coverage") or {}
    if dispatch_op_coverage:
        lines.extend([
            "",
            "## Dispatch Op Coverage",
            "",
            "Every dispatch op row must be listed in `dispatch_op_coverage.*` and covered by runtime module split plus tensor-id dataflow.",
            "",
            f"- ops in dispatch rows: `{dispatch_op_coverage.get('op_count')}`",
            f"- ops listed in coverage: `{dispatch_op_coverage.get('covered_op_count')}`",
            f"- missing event_op_index values: `{dispatch_op_coverage.get('missing_event_op_indices', [])}`",
            f"- missing from module_split: `{dispatch_op_coverage.get('missing_from_module_split', [])}`",
            f"- missing from tensor_dataflow: `{dispatch_op_coverage.get('missing_from_tensor_dataflow', [])}`",
        ])
    lines.extend(["", "## Top Ops", ""])
    for name, count in sorted(summary["op_counts"].items(), key=lambda item: (-item[1], item[0]))[:25]:
        lines.append(f"- `{name}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
