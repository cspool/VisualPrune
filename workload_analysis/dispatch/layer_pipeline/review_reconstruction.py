#!/usr/bin/env python3
"""Review reconstructed layer flows against the real dispatch trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dispatch_io import iter_arg_tensors, output_tensor_shape, parse_json_field, read_dispatch_rows, tensor_ids_from_row, tensor_shape


WORKLOAD_ROOT = Path(__file__).resolve().parents[2]
DISPATCH_ROOT = WORKLOAD_ROOT / "dispatch"
DEFAULT_SOURCE_CSV = DISPATCH_ROOT / "profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv"
DEFAULT_LAYER_DIR = DISPATCH_ROOT / "visualize"
DEFAULT_CODE_DIR = DEFAULT_LAYER_DIR
DEFAULT_REVIEW_DIR = DISPATCH_ROOT / "visualize/reconstruction_review"

STAGE_CODE_FILES = {
    "input_rmsnorm": "rmsnorm.py",
    "qkv_projection": "qkv_projection.py",
    "rope": "rope.py",
    "kv_cache_concat": "kv_cache.py",
    "attention": "attention.py",
    "visual_adjust": "visual_adjust.py",
    "visipruner_similarity_check": "visipruner_similarity.py",
    "attention_output": "attention_output.py",
    "mlp": "mlp.py",
    "full_flow": "run_full_flow.py",
}

PROCESS_CODE_FILES = {
    "dispatch_reconstruction": "dispatch_reconstructed.py",
    "toy_tensor_compute": "toy_tensor_compute.py",
    "layer_profile": "layer_profile.json",
    "process_index": "process_index.md",
}

BASE_EXPECTED_STAGES = [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention",
    "attention_output",
    "mlp",
]


def stage_from_onnx_name(filename: str) -> str:
    return re.sub(r"^\d+_", "", Path(filename).stem)


def _first(rows: list[dict[str, str]], key: str) -> str | None:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_int(rows: list[dict[str, str]], key: str) -> int | None:
    value = _first(rows, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _all_tensor_shapes(rows: list[dict[str, str]]) -> list[list[int]]:
    shapes: list[list[int]] = []
    for row in rows:
        output_shape = output_tensor_shape(row)
        if output_shape:
            shapes.append(output_shape)
        for arg in iter_arg_tensors(row):
            shape = tensor_shape(arg)
            if shape:
                shapes.append(shape)
    return shapes


def infer_dimensions_from_dispatch(rows: list[dict[str, str]]) -> dict[str, int | None]:
    q_len = _first_int(rows, "q_len")
    kv_len = _first_int(rows, "kv_len")
    sequence_dims = {dim for dim in (q_len, kv_len) if dim is not None}
    hidden: int | None = None
    ffn: int | None = None
    head_dim: int | None = None
    hidden_candidates: Counter[int] = Counter()
    head_candidates: Counter[int] = Counter()

    shapes = _all_tensor_shapes(rows)
    for shape in shapes:
        if len(shape) == 3 and shape[0] == 1:
            last = shape[-1]
            if last >= 512 and last not in sequence_dims:
                hidden_candidates[last] += 1
        if len(shape) == 4 and shape[0] == 1:
            if 1 < shape[-1] <= 256:
                head_dim = shape[-1]
            for dim in shape[1:3]:
                if 1 < dim <= 128:
                    head_candidates[dim] += 1

    if hidden_candidates:
        hidden = hidden_candidates.most_common(1)[0][0]
        larger = [dim for dim in hidden_candidates if dim > hidden]
        if larger:
            ffn = max(larger)

    heads = None
    if hidden is not None and head_dim is not None and hidden % head_dim == 0:
        heads = hidden // head_dim
    elif head_candidates:
        heads = head_candidates.most_common(1)[0][0]

    return {
        "q_len": q_len,
        "kv_len": kv_len,
        "hidden": hidden,
        "heads": heads,
        "head_dim": head_dim,
        "ffn": ffn,
    }


def _has_output_shape(rows: list[dict[str, str]], op_name: str, rank: int | None = None) -> bool:
    for row in rows:
        if row.get("op_name") != op_name:
            continue
        shape = output_tensor_shape(row)
        if shape is not None and (rank is None or len(shape) == rank):
            return True
    return False


def infer_dispatch_features(rows: list[dict[str, str]]) -> dict[str, Any]:
    ops = Counter(row.get("op_name", "") for row in rows)
    phase = _first(rows, "phase")
    role = _first(rows, "visipruner_role") or ""
    token_state = _first(rows, "token_state") or ""
    q_len = _first_int(rows, "q_len")
    kv_len = _first_int(rows, "kv_len")

    visual_adjust_kind = None
    if ops["copy_.default"] and ops["sum.dim_IntList"]:
        visual_adjust_kind = "fold_tail_visual_mass_and_clear_region"
    elif ops["fill_.Tensor"]:
        visual_adjust_kind = "clear_visual_region"

    prune_probe_kind = None
    if ops["cosine_similarity.default"]:
        if "deep_check" in role or ops["arange.start"]:
            prune_probe_kind = "deep_exit_similarity_check"
        elif "middle_select" in role:
            prune_probe_kind = "middle_selection_similarity_check"
        else:
            prune_probe_kind = "middle_probe_similarity_check"

    has_cache_concat = bool(phase == "decode" and ops["cat.default"] >= 2 and q_len == 1 and kv_len and kv_len > 1)
    has_rope = bool(ops["index.Tensor"] and ops["slice.Tensor"] and ops["neg.default"] and ops["cat.default"])
    has_attention = bool(ops["matmul.default"] >= 2 and ops["softmax.int"])
    has_mlp = bool(ops["silu.default"] and ops["linear.default"] >= 7)

    expected_stages = list(BASE_EXPECTED_STAGES)
    if has_cache_concat:
        expected_stages.insert(expected_stages.index("attention"), "kv_cache_concat")
    if visual_adjust_kind:
        expected_stages.insert(expected_stages.index("attention_output"), "visual_adjust")
    if prune_probe_kind:
        expected_stages.insert(expected_stages.index("attention_output"), "visipruner_similarity_check")

    return {
        "phase": phase,
        "role": role,
        "token_state": token_state,
        "q_len": q_len,
        "kv_len": kv_len,
        "op_counts": dict(ops),
        "has_rope": has_rope,
        "has_attention": has_attention,
        "has_mlp": has_mlp,
        "has_cache_concat": has_cache_concat,
        "visual_adjust_kind": visual_adjust_kind,
        "prune_probe_kind": prune_probe_kind,
        "expected_stages": expected_stages,
    }


def generated_onnx_stages(layer_dir: Path) -> list[str]:
    manifest_path = layer_dir / "onnx/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [stage_from_onnx_name(str(item["path"])) for item in manifest]
    onnx_dir = layer_dir / "onnx"
    return [stage_from_onnx_name(path.name) for path in sorted(onnx_dir.glob("*.onnx"))]


def generated_onnx_details(layer_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = layer_dir / "onnx/manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details: dict[str, dict[str, Any]] = {}
    for item in manifest:
        stage = stage_from_onnx_name(str(item["path"]))
        details[stage] = item
    return details


def _attention_is_rectangular_when_needed(
    generated_details: dict[str, dict[str, Any]],
    features: dict[str, Any],
) -> bool:
    if features["q_len"] == features["kv_len"]:
        return True
    detail = generated_details.get("attention")
    if not detail:
        return False
    outputs = detail.get("outputs", {})
    if not isinstance(outputs, dict):
        return False
    raw_shape = outputs.get("raw_scores")
    if not isinstance(raw_shape, list) or len(raw_shape) < 3:
        return False
    return raw_shape[-2] != raw_shape[-1]


def generated_code_files(code_dir: Path, event_id: str) -> dict[str, bool]:
    flow_dir = code_dir / event_id / "torch_flow"
    return {stage: (flow_dir / filename).exists() for stage, filename in STAGE_CODE_FILES.items()}


def generated_process_files(code_dir: Path, event_id: str) -> dict[str, bool]:
    flow_dir = code_dir / event_id / "torch_flow"
    return {name: (flow_dir / filename).exists() for name, filename in PROCESS_CODE_FILES.items()}


def _generated_config_value(code_dir: Path, event_id: str, name: str) -> str | None:
    config_path = code_dir / event_id / "torch_flow/config.py"
    if not config_path.exists():
        return None
    pattern = re.compile(rf"^{re.escape(name)}\s*=\s*(.+)$", flags=re.MULTILINE)
    match = pattern.search(config_path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def _output_value(row: dict[str, str]) -> Any:
    return parse_json_field(row.get("outputs", ""))


def _output_evidence(row: dict[str, str]) -> str:
    value = _output_value(row)
    if isinstance(value, dict) and value.get("type") == "Tensor":
        return f"shape={value.get('shape')}, dtype={value.get('dtype')}"
    return repr(value)


def _compact_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "event_op_index": int(row.get("event_op_index") or 0),
        "op_name": row.get("op_name"),
        "output": _output_evidence(row),
        "input_tensor_ids": tensor_ids_from_row(row, "input_tensor_ids"),
        "output_tensor_ids": tensor_ids_from_row(row, "output_tensor_ids"),
    }


def _select_rows(
    rows: list[dict[str, str]],
    op_names: set[str],
    limit: int = 12,
    predicate: Any | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("op_name") not in op_names:
            continue
        if predicate is not None and not predicate(row):
            continue
        selected.append(_compact_row(row))
        if len(selected) >= limit:
            break
    return selected


def _shape(row: dict[str, str]) -> list[int] | None:
    return output_tensor_shape(row)


def _has_shape(row: dict[str, str], rank: int | None = None, dim2: int | None = None, dim3: int | None = None) -> bool:
    shape = _shape(row)
    if shape is None:
        return False
    if rank is not None and len(shape) != rank:
        return False
    if dim2 is not None and (len(shape) <= 2 or shape[2] != dim2):
        return False
    if dim3 is not None and (len(shape) <= 3 or shape[3] != dim3):
        return False
    return True


def stage_evidence(stage: str, rows: list[dict[str, str]], features: dict[str, Any], dims: dict[str, int | None]) -> dict[str, Any]:
    q_len = features["q_len"]
    kv_len = features["kv_len"]
    head_dim = dims["head_dim"]

    if stage == "input_rmsnorm":
        ops = {"to.dtype", "pow.Tensor_Scalar", "mean.dim", "add.Tensor", "rsqrt.default", "mul.Tensor"}
        evidence = _select_rows(rows[:12], ops, limit=8)
        summary = "RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence."
    elif stage == "qkv_projection":
        ops = {"linear.default", "view.default", "transpose.int"}
        evidence = _select_rows(rows, ops, limit=9, predicate=lambda row: int(row.get("event_op_index") or 0) <= 17)
        summary = "Q/K/V evidence is three hidden-size linear projections followed by view/transpose head split."
    elif stage == "rope":
        ops = {"index.Tensor", "unsqueeze.default", "slice.Tensor", "neg.default", "cat.default", "mul.Tensor", "add.Tensor"}
        evidence = _select_rows(rows, ops, limit=14, predicate=lambda row: 18 <= int(row.get("event_op_index") or 0) <= 46)
        summary = "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
    elif stage == "kv_cache_concat":
        ops = {"cat.default"}
        evidence = _select_rows(
            rows,
            ops,
            limit=4,
            predicate=lambda row: _has_shape(row, rank=4, dim2=kv_len),
        )
        summary = "Decode cache evidence is K/V cat outputs whose sequence dimension equals dispatch kv_len."
    elif stage == "attention":
        ops = {"transpose.int", "matmul.default", "div.Tensor", "add.Tensor", "softmax.int", "dropout.default"}
        evidence = _select_rows(
            rows,
            ops,
            limit=12,
            predicate=lambda row: (
                row.get("op_name") in {"softmax.int", "dropout.default", "div.Tensor"}
                or _has_shape(row, rank=4, dim2=q_len)
                or _has_shape(row, rank=4, dim3=kv_len)
            ),
        )
        summary = "Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores."
    elif stage == "visual_adjust":
        if features["visual_adjust_kind"] is None:
            evidence = []
        else:
            ops = {"sum.dim_IntList", "fill_.Tensor", "copy_.default", "slice.Tensor", "select.int", "lift_fresh.default"}
            evidence = _select_rows(
                rows,
                ops,
                limit=12,
                predicate=lambda row: row.get("op_name") in {"sum.dim_IntList", "fill_.Tensor", "copy_.default", "lift_fresh.default"} or int(row.get("event_op_index") or 0) >= 52,
            )
        summary = f"Visual-adjust evidence kind: {features['visual_adjust_kind']}."
    elif stage == "visipruner_similarity_check":
        ops = {"cosine_similarity.default", "any.default", "arange.start", "sub.Tensor", "gt.Scalar", "is_nonzero.default"}
        evidence = _select_rows(rows, ops, limit=12)
        summary = f"VisiPrune check evidence kind: {features['prune_probe_kind']}."
    elif stage == "attention_output":
        ops = {"matmul.default", "transpose.int", "contiguous.default", "reshape.default", "linear.default", "add.Tensor"}
        evidence = _select_rows(
            rows,
            ops,
            limit=12,
            predicate=lambda row: (
                int(row.get("event_op_index") or 0) >= 50
                and (
                    _has_shape(row, rank=4, dim2=q_len, dim3=head_dim)
                    or _has_shape(row, rank=3, dim2=dims["hidden"])
                    or row.get("op_name") in {"contiguous.default", "reshape.default"}
                )
            ),
        )
        summary = "Attention-output evidence is attn @ V, transpose/reshape, output projection, and residual add."
    elif stage == "mlp":
        ops = {"to.dtype", "pow.Tensor_Scalar", "mean.dim", "rsqrt.default", "linear.default", "silu.default", "mul.Tensor", "add.Tensor"}
        evidence = _select_rows(rows[-24:], ops, limit=14)
        summary = "MLP evidence is post-attention RMSNorm, gate/up linear, SiLU, gated product, down linear, residual add."
    else:
        evidence = []
        summary = "No dispatch evidence rule is defined for this stage."

    return {
        "stage": stage,
        "summary": summary,
        "evidence_ops": evidence,
        "dispatch_supported": bool(evidence),
    }


def compare_reconstruction(event_id: str, rows: list[dict[str, str]], layer_dir: Path, code_dir: Path) -> dict[str, Any]:
    dims = infer_dimensions_from_dispatch(rows)
    features = infer_dispatch_features(rows)
    expected = features["expected_stages"]
    generated = generated_onnx_stages(layer_dir)
    generated_details = generated_onnx_details(layer_dir)
    generated_without_full = [stage for stage in generated if stage != "full_flow"]

    missing = [stage for stage in expected if stage not in generated]
    extra = [
        stage
        for stage in generated
        if stage != "full_flow" and stage not in expected
    ]
    notes: list[str] = []
    if features["has_cache_concat"] and "kv_cache_concat" in missing:
        notes.append("decode dispatch concatenates past K/V cache before attention; current ONNX has no explicit cache-concat stage")
    if features["prune_probe_kind"] and "visipruner_similarity_check" in missing:
        notes.append(f"dispatch includes {features['prune_probe_kind']}; current ONNX has no cosine-similarity/check stage")
    if "visual_adjust" in extra:
        notes.append("current ONNX includes visual_adjust but dispatch for this layer has no fill/copy visual attention adjustment")
    if features["visual_adjust_kind"] == "clear_visual_region" and "visual_adjust" in generated:
        generated_kind = _generated_config_value(code_dir, event_id, "VISUAL_ADJUST_KIND")
        if generated_kind != repr("clear_visual_region"):
            notes.append("dispatch only clears a visual region; generated visual_adjust also folds tail visual mass")
    if features["q_len"] != features["kv_len"] and "attention" in generated:
        if not _attention_is_rectangular_when_needed(generated_details, features):
            notes.append("dispatch attention is rectangular q_len x kv_len; generated attention stage is square small-seq attention")

    status = "match"
    if missing or extra or notes:
        status = "needs_revision"

    split_alignment = "aligned" if generated_without_full == expected else "misaligned"
    process_alignment = "aligned" if status == "match" else "misaligned"
    compared_stages = list(dict.fromkeys(expected + generated_without_full))
    core_evidence = {
        stage: stage_evidence(stage, rows, features, dims)
        for stage in compared_stages
    }

    result = {
        "event_id": event_id,
        "row_count": len(rows),
        "dimensions": dims,
        "features": features,
        "generated_stages": generated,
        "generated_stage_details": generated_details,
        "generated_code_files": generated_code_files(code_dir, event_id),
        "generated_process_files": generated_process_files(code_dir, event_id),
        "split_alignment": split_alignment,
        "process_alignment": process_alignment,
        "core_evidence": core_evidence,
        "missing_expected_stages": missing,
        "extra_generated_stages": extra,
        "notes": notes,
        "torch_flow_dir": str(code_dir / event_id / "torch_flow"),
        "layer_dir": str(layer_dir),
        "status": status,
    }
    return result


def write_csv(path: Path, reviews: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "status",
        "split_alignment",
        "process_alignment",
        "phase",
        "role",
        "token_state",
        "q_len",
        "kv_len",
        "hidden",
        "heads",
        "head_dim",
        "ffn",
        "expected_stages",
        "generated_stages",
        "missing_expected_stages",
        "extra_generated_stages",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            features = review["features"]
            dims = review["dimensions"]
            writer.writerow(
                {
                    "event_id": review["event_id"],
                    "status": review["status"],
                    "split_alignment": review["split_alignment"],
                    "process_alignment": review["process_alignment"],
                    "phase": features["phase"],
                    "role": features["role"],
                    "token_state": features["token_state"],
                    "q_len": features["q_len"],
                    "kv_len": features["kv_len"],
                    "hidden": dims["hidden"],
                    "heads": dims["heads"],
                    "head_dim": dims["head_dim"],
                    "ffn": dims["ffn"],
                    "expected_stages": " | ".join(features["expected_stages"]),
                    "generated_stages": " | ".join(review["generated_stages"]),
                    "missing_expected_stages": " | ".join(review["missing_expected_stages"]),
                    "extra_generated_stages": " | ".join(review["extra_generated_stages"]),
                    "notes": " | ".join(review["notes"]),
                }
            )


def write_markdown(path: Path, reviews: list[dict[str, Any]]) -> None:
    counts = Counter(review["status"] for review in reviews)
    split_counts = Counter(review["split_alignment"] for review in reviews)
    process_counts = Counter(review["process_alignment"] for review in reviews)
    lines = [
        "# Dispatch Reconstruction Review",
        "",
        "This review compares each layer's generated small-tensor ONNX stages against the real TorchDispatch op trace.",
        "",
        "## Summary",
        "",
        f"- layers reviewed: `{len(reviews)}`",
        f"- matching reconstructions: `{counts.get('match', 0)}`",
        f"- needs revision: `{counts.get('needs_revision', 0)}`",
        f"- split aligned: `{split_counts.get('aligned', 0)}`",
        f"- process aligned: `{process_counts.get('aligned', 0)}`",
        "",
        "## Key Findings",
        "",
        "- The base transformer-block stages are present for every layer: RMSNorm, Q/K/V projection, RoPE, attention, attention output, and MLP.",
        "- The previous generated flow is not layer-specific enough for every dispatch trace.",
        "- Decode layers need an explicit K/V cache concatenation stage because dispatch attention has `q_len=1` and `kv_len>1`.",
        "- Middle/deep VisiPrune probe layers need cosine-similarity/check stages when dispatch contains `aten::cosine_similarity`.",
        "- Some layers have no visual attention adjustment, while the generated ONNX still includes `05_visual_adjust.onnx`.",
        "- `input1_layer5` has a clear-only visual adjustment, whereas the generated visual-adjust stage also folds tail visual mass.",
        "",
        "## Per-Layer Review",
        "",
    ]
    for review in reviews:
        features = review["features"]
        dims = review["dimensions"]
        lines.extend(
            [
                f"### {review['event_id']}",
                "",
                f"- status: `{review['status']}`",
                f"- split_alignment: `{review['split_alignment']}`",
                f"- process_alignment: `{review['process_alignment']}`",
                f"- dispatch: phase=`{features['phase']}`, role=`{features['role']}`, token_state=`{features['token_state']}`, q_len=`{features['q_len']}`, kv_len=`{features['kv_len']}`",
                f"- dimensions: hidden=`{dims['hidden']}`, heads=`{dims['heads']}`, head_dim=`{dims['head_dim']}`, ffn=`{dims['ffn']}`",
                f"- expected stages: `{', '.join(features['expected_stages'])}`",
                f"- generated stages: `{', '.join(review['generated_stages'])}`",
            ]
        )
        if review["missing_expected_stages"]:
            lines.append(f"- missing expected stages: `{', '.join(review['missing_expected_stages'])}`")
        if review["extra_generated_stages"]:
            lines.append(f"- extra generated stages: `{', '.join(review['extra_generated_stages'])}`")
        if review["notes"]:
            lines.append(f"- notes: {'; '.join(review['notes'])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_generated_detail(stage: str, review: dict[str, Any]) -> str:
    detail = review["generated_stage_details"].get(stage)
    if not detail:
        return "not generated"
    return f"{Path(detail['path']).name}: inputs={detail.get('inputs')}, outputs={detail.get('outputs')}"


def review_to_markdown(review: dict[str, Any]) -> str:
    features = review["features"]
    dims = review["dimensions"]
    lines = [
        f"# {review['event_id']} Reconstruction Review",
        "",
        "This file reviews the generated small-tensor reconstruction against this layer's real dispatch trace.",
        "",
        "## Dispatch Evidence",
        "",
        f"- rows: `{review['row_count']}`",
        f"- phase: `{features['phase']}`",
        f"- role: `{features['role']}`",
        f"- token_state: `{features['token_state']}`",
        f"- q_len: `{features['q_len']}`",
        f"- kv_len: `{features['kv_len']}`",
        f"- hidden: `{dims['hidden']}`",
        f"- heads: `{dims['heads']}`",
        f"- head_dim: `{dims['head_dim']}`",
        f"- ffn: `{dims['ffn']}`",
        "",
        "## Dispatch-Derived Expected Stages",
        "",
    ]
    lines.extend(f"{index}. `{stage}`" for index, stage in enumerate(features["expected_stages"], start=1))
    lines.extend(
        [
            "",
            "## Generated ONNX Stages",
            "",
        ]
    )
    lines.extend(f"- `{stage}`: {_format_generated_detail(stage, review)}" for stage in review["generated_stages"])
    lines.extend(
        [
            "",
            "## Generated Process Code",
            "",
        ]
    )
    for name, exists in review["generated_process_files"].items():
        lines.append(f"- `{name}`: `{exists}`")
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            f"- split_alignment: `{review['split_alignment']}`",
            f"- process_alignment: `{review['process_alignment']}`",
            "",
            "## Verdict",
            "",
            f"- status: `{review['status']}`",
        ]
    )
    if review["missing_expected_stages"]:
        lines.append(f"- missing expected stages: `{', '.join(review['missing_expected_stages'])}`")
    if review["extra_generated_stages"]:
        lines.append(f"- extra generated stages: `{', '.join(review['extra_generated_stages'])}`")
    if review["notes"]:
        lines.append(f"- notes: {'; '.join(review['notes'])}")
    lines.extend(
        [
            "",
            "## Core Dispatch Evidence",
            "",
        ]
    )
    for stage, evidence in review["core_evidence"].items():
        lines.extend(
            [
                f"### `{stage}`",
                "",
                f"- dispatch_supported: `{evidence['dispatch_supported']}`",
                f"- summary: {evidence['summary']}",
            ]
        )
        if evidence["evidence_ops"]:
            lines.append("- evidence ops:")
            for item in evidence["evidence_ops"]:
                lines.append(
                    f"  - `#{item['event_op_index']} {item['op_name']}` -> {item['output']}"
                )
        else:
            lines.append("- evidence ops: none")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--layer-dir", type=Path, default=DEFAULT_LAYER_DIR)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_dispatch_rows(args.source_csv)
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)

    event_ids = sorted(path.name for path in args.layer_dir.iterdir() if path.is_dir() and "_layer" in path.name)
    reviews = [
        compare_reconstruction(
            event_id=event_id,
            rows=sorted(by_event[event_id], key=lambda row: int(row.get("event_op_index") or 0)),
            layer_dir=args.layer_dir / event_id,
            code_dir=args.code_dir,
        )
        for event_id in event_ids
    ]

    args.review_dir.mkdir(parents=True, exist_ok=True)
    for review in reviews:
        per_layer_dir = args.layer_dir / review["event_id"] / "dispatch_review"
        per_layer_dir.mkdir(parents=True, exist_ok=True)
        per_layer_path = per_layer_dir / "reconstruction_review.json"
        per_layer_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        per_layer_md = per_layer_dir / "reconstruction_review.md"
        per_layer_text = review_to_markdown(review)
        per_layer_md.write_text(per_layer_text, encoding="utf-8")
        per_layer_audit_json = per_layer_dir / "alignment_audit.json"
        per_layer_audit_json.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        per_layer_audit_md = per_layer_dir / "alignment_audit.md"
        per_layer_audit_md.write_text(per_layer_text, encoding="utf-8")

    (args.review_dir / "review.json").write_text(json.dumps(reviews, indent=2) + "\n", encoding="utf-8")
    (args.review_dir / "alignment_audit.json").write_text(json.dumps(reviews, indent=2) + "\n", encoding="utf-8")
    write_csv(args.review_dir / "review.csv", reviews)
    write_csv(args.review_dir / "alignment_audit.csv", reviews)
    write_markdown(args.review_dir / "review.md", reviews)
    write_markdown(args.review_dir / "alignment_audit.md", reviews)
    status_counts = Counter(review["status"] for review in reviews)
    print(json.dumps({"reviewed": len(reviews), "status_counts": dict(status_counts), "review_dir": str(args.review_dir)}, indent=2))


if __name__ == "__main__":
    main()
