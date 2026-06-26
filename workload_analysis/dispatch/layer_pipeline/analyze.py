from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dispatch_io import iter_arg_tensors, output_tensor_shape, parse_json_field, summarize_ops, tensor_shape


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
    lines.extend(["", "## Top Ops", ""])
    for name, count in sorted(summary["op_counts"].items(), key=lambda item: (-item[1], item[0]))[:25]:
        lines.append(f"- `{name}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
