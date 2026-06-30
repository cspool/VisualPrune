#!/usr/bin/env python3
"""Audit per-layer dispatch reconstruction artifacts against layer-local evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import runpy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze import build_dispatch_op_coverage, build_module_split, build_stage_tensor_io, build_tensor_dataflow
from dispatch_io import filter_event, read_dispatch_rows, summarize_ops
from review_reconstruction import infer_dimensions_from_dispatch, infer_dispatch_features, stage_evidence, stage_from_onnx_name


WORKLOAD_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WORKLOAD_ROOT.parent
DISPATCH_ROOT = WORKLOAD_ROOT / "dispatch"
DEFAULT_SOURCE_CSV = DISPATCH_ROOT / "profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv"
DEFAULT_LAYER_DIR = DISPATCH_ROOT / "visualize"
DEFAULT_AUDIT_DIR = DISPATCH_ROOT / "visualize/reconstruction_audit"

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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def add_issue(issues: list[dict[str, str]], code: str, detail: str, severity: str = "error") -> None:
    issues.append({"severity": severity, "code": code, "detail": detail})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_stages(onnx_manifest: list[dict[str, Any]]) -> list[str]:
    return [stage_from_onnx_name(str(item["path"])) for item in onnx_manifest]


def check_shape(
    issues: list[dict[str, str]],
    shapes: dict[str, Any],
    name: str,
    expected: list[int],
    context: str,
) -> None:
    actual = shapes.get(name)
    if actual != expected:
        add_issue(issues, "shape_mismatch", f"{context}.{name}: expected {expected}, got {actual}")


def expected_onnx_shapes(stage: str, cfg: dict[str, int], visual_kind: str | None) -> dict[str, dict[str, list[int]]]:
    q_seq = cfg["q_seq"]
    kv_seq = cfg["kv_seq"]
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    head_dim = cfg["head_dim"]
    ffn = cfg["ffn"]
    visual_tokens = cfg["visual_end"] - cfg["visual_start"]
    tail_q = max(0, q_seq - cfg["tail_start"])
    cleared_q = max(0, q_seq - cfg["visual_start"])
    cleared_k = max(0, cfg["visual_end"] - cfg["visual_start"])

    by_stage = {
        "input_rmsnorm": {
            "inputs": {"hidden_states": [q_seq, hidden]},
            "outputs": {"x_norm": [q_seq, hidden], "variance": [q_seq, 1], "inv_rms": [q_seq, 1]},
        },
        "qkv_projection": {
            "inputs": {"x_norm": [q_seq, hidden]},
            "outputs": {
                "q_heads": [heads, q_seq, head_dim],
                "k_heads": [heads, q_seq, head_dim],
                "v_heads": [heads, q_seq, head_dim],
            },
        },
        "rope": {
            "inputs": {
                "q_heads": [heads, q_seq, head_dim],
                "k_heads": [heads, q_seq, head_dim],
                "position_ids": [q_seq],
            },
            "outputs": {"q_rope": [heads, q_seq, head_dim], "k_current_rope": [heads, q_seq, head_dim]},
        },
        "kv_cache_concat": {
            "inputs": {
                "k_current_rope": [heads, q_seq, head_dim],
                "v_current": [heads, q_seq, head_dim],
                "past_k": [heads, kv_seq - q_seq, head_dim],
                "past_v": [heads, kv_seq - q_seq, head_dim],
            },
            "outputs": {"k_heads": [heads, kv_seq, head_dim], "v_heads": [heads, kv_seq, head_dim]},
        },
        "attention": {
            "inputs": {
                "q_rope": [heads, q_seq, head_dim],
                "k_heads": [heads, kv_seq, head_dim],
                "attention_mask": [1, q_seq, kv_seq],
            },
            "outputs": {
                "raw_scores": [heads, q_seq, kv_seq],
                "masked_scores": [heads, q_seq, kv_seq],
                "attn": [heads, q_seq, kv_seq],
            },
        },
        "attention_output": {
            "inputs": {
                "adjusted_attn": [heads, q_seq, kv_seq],
                "v_heads": [heads, kv_seq, head_dim],
                "hidden_states": [q_seq, hidden],
            },
            "outputs": {"context": [q_seq, hidden], "attn_out": [q_seq, hidden], "after_attn": [q_seq, hidden]},
        },
        "mlp": {
            "inputs": {"after_attn": [q_seq, hidden]},
            "outputs": {"gated": [q_seq, ffn], "ffn_out": [q_seq, hidden], "output": [q_seq, hidden]},
        },
        "full_flow": {
            "inputs": {"hidden_states": [q_seq, hidden], "position_ids": [q_seq], "attention_mask": [1, q_seq, kv_seq]},
            "outputs": {"output": [q_seq, hidden]},
        },
    }
    if kv_seq != q_seq:
        by_stage["full_flow"]["inputs"]["past_k"] = [heads, kv_seq - q_seq, head_dim]
        by_stage["full_flow"]["inputs"]["past_v"] = [heads, kv_seq - q_seq, head_dim]
    if visual_kind == "fold_tail_visual_mass_and_clear_region":
        by_stage["visual_adjust"] = {
            "inputs": {"attn": [heads, q_seq, kv_seq]},
            "outputs": {
                "adjusted_attn": [heads, q_seq, kv_seq],
                "tail_visual_sum": [heads, tail_q],
                "cleared_visual_region": [heads, cleared_q, cleared_k],
            },
        }
    elif visual_kind == "clear_visual_region":
        by_stage["visual_adjust"] = {
            "inputs": {"attn": [heads, q_seq, kv_seq]},
            "outputs": {
                "adjusted_attn": [heads, q_seq, kv_seq],
                "cleared_visual_region": [heads, cleared_q, cleared_k],
            },
        }
    by_stage["visipruner_similarity_check"] = {
        "inputs": {"hidden_states": [q_seq, hidden]},
        "outputs": {"similarity": [max(0, q_seq - 1)], "any_close": []},
    }
    return by_stage


def parse_config(flow_dir: Path) -> dict[str, Any]:
    data = runpy.run_path(str(flow_dir / "config.py"))
    cfg = data["CFG"]
    return {
        "flow_config": {
            "seq": cfg.seq,
            "q_seq": cfg.q_seq,
            "kv_seq": cfg.kv_seq,
            "hidden": cfg.hidden,
            "heads": cfg.heads,
            "head_dim": cfg.head_dim,
            "visual_start": cfg.visual_start,
            "visual_end": cfg.visual_end,
            "tail_start": cfg.tail_start,
            "ffn": cfg.ffn,
        },
        "expected_stages": list(data["EXPECTED_STAGES"]),
        "visual_adjust_kind": data.get("VISUAL_ADJUST_KIND"),
        "prune_probe_kind": data.get("PRUNE_PROBE_KIND"),
    }


def assert_path_exists(issues: list[dict[str, str]], path_value: str | Path, code: str) -> Path:
    path = resolve_artifact_path(path_value)
    if not path.exists():
        add_issue(issues, code, f"missing artifact: {path}")
    return path


def audit_event(event_id: str, source_rows: list[dict[str, str]], layer_root: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    event_dir = layer_root / event_id
    review_dir = event_dir / "dispatch_review"
    flow_dir = event_dir / "torch_flow"
    onnx_dir = event_dir / "onnx"

    rows = filter_event(source_rows, event_id)
    if not rows:
        add_issue(issues, "missing_source_rows", "no rows in source dispatch CSV")
    event_op_indices = [int(row.get("event_op_index") or 0) for row in rows]
    expected_indices = list(range(1, len(rows) + 1))
    if event_op_indices != expected_indices:
        add_issue(issues, "event_op_index_not_contiguous", f"expected {expected_indices[:3]}...{expected_indices[-3:] if expected_indices else []}, got first/last {event_op_indices[:3]}...{event_op_indices[-3:] if event_op_indices else []}")

    features = infer_dispatch_features(rows)
    dims = infer_dimensions_from_dispatch(rows)
    expected_stages = list(features["expected_stages"])
    expected_with_full = expected_stages + ["full_flow"]

    for path in [
        event_dir / "dispatch_ops.csv",
        review_dir / "dispatch_ops.csv",
        review_dir / "summary.json",
        review_dir / "tensor_compute_process.md",
        review_dir / "tensor_dataflow.json",
        review_dir / "tensor_dataflow_edges.csv",
        review_dir / "tensor_dataflow.md",
        review_dir / "dispatch_op_coverage.json",
        review_dir / "dispatch_op_coverage.csv",
        review_dir / "dispatch_op_coverage.md",
        review_dir / "process_manifest.json",
        review_dir / "process_code_index.md",
        review_dir / "module_split.json",
        review_dir / "module_split.csv",
        review_dir / "module_split.md",
        review_dir / "onnx_code_index.md",
        review_dir / "onnx_code_map.json",
        flow_dir / "config.py",
        flow_dir / "layer_profile.json",
        flow_dir / "process_index.md",
        onnx_dir / "manifest.json",
        event_dir / "layer_manifest.json",
    ]:
        if not path.exists():
            add_issue(issues, "missing_required_artifact", f"missing {path}")

    layer_csv_rows = read_csv_rows(event_dir / "dispatch_ops.csv") if (event_dir / "dispatch_ops.csv").exists() else []
    if len(layer_csv_rows) != len(rows):
        add_issue(issues, "layer_dispatch_ops_row_count_mismatch", f"{event_dir / 'dispatch_ops.csv'} has {len(layer_csv_rows)} rows, source has {len(rows)}")

    review_rows = read_csv_rows(review_dir / "dispatch_ops.csv") if (review_dir / "dispatch_ops.csv").exists() else []
    if compact_json(review_rows) != compact_json(rows):
        add_issue(issues, "review_dispatch_ops_not_equal_source", "dispatch_review/dispatch_ops.csv differs from source rows for this event")

    expected_dataflow = build_tensor_dataflow(rows)
    tensor_dataflow = load_json(review_dir / "tensor_dataflow.json") if (review_dir / "tensor_dataflow.json").exists() else {}
    if compact_json(tensor_dataflow) != compact_json(expected_dataflow):
        add_issue(issues, "tensor_dataflow_json_mismatch", "tensor_dataflow.json is not reproducible from input_tensor_ids/output_tensor_ids")
    tensor_edges_rows = read_csv_rows(review_dir / "tensor_dataflow_edges.csv") if (review_dir / "tensor_dataflow_edges.csv").exists() else []
    if len(tensor_edges_rows) != len(expected_dataflow.get("edges", [])):
        add_issue(issues, "tensor_dataflow_edges_csv_count_mismatch", f"tensor_dataflow_edges.csv rows {len(tensor_edges_rows)} != expected edges {len(expected_dataflow.get('edges', []))}")

    summary = load_json(review_dir / "summary.json") if (review_dir / "summary.json").exists() else {}
    summary_fields = {
        "event_id": event_id,
        "row_count": len(rows),
        "phase": rows[0].get("phase") if rows else None,
        "token_state": rows[0].get("token_state") if rows else None,
        "visipruner_role": rows[0].get("visipruner_role") if rows else None,
        "q_len": int(rows[0].get("q_len") or 0) if rows else None,
        "kv_len": int(rows[0].get("kv_len") or 0) if rows else None,
        "past_len": int(rows[0].get("past_len") or 0) if rows else None,
    }
    for key, expected in summary_fields.items():
        if summary.get(key) != expected:
            add_issue(issues, "summary_field_mismatch", f"{key}: expected {expected!r}, got {summary.get(key)!r}")
    if summary.get("op_counts") != summarize_ops(rows):
        add_issue(issues, "summary_op_counts_mismatch", "summary op_counts does not match dispatch rows")
    if (summary.get("dispatch_features") or {}).get("expected_stages") != expected_stages:
        add_issue(issues, "summary_expected_stages_mismatch", f"expected {expected_stages}, got {(summary.get('dispatch_features') or {}).get('expected_stages')}")
    if (summary.get("dispatch_features") or {}).get("visual_adjust_kind") != features.get("visual_adjust_kind"):
        add_issue(issues, "summary_visual_kind_mismatch", f"expected {features.get('visual_adjust_kind')}, got {(summary.get('dispatch_features') or {}).get('visual_adjust_kind')}")

    module_expected = build_module_split(rows)
    module_split = load_json(review_dir / "module_split.json") if (review_dir / "module_split.json").exists() else []
    if compact_json(module_split) != compact_json(module_expected):
        add_issue(issues, "module_split_json_mismatch", "module_split.json is not reproducible from dispatch rows")
    covered = sorted(index for item in module_split for index in item.get("event_op_indices", []))
    if covered != expected_indices:
        add_issue(issues, "module_split_coverage_mismatch", f"module_split covers {len(covered)} op indices, expected {len(expected_indices)}")
    if sum(int(item.get("op_count") or 0) for item in module_split) != len(rows):
        add_issue(issues, "module_split_op_count_mismatch", "sum(module_split.op_count) does not match dispatch row count")
    module_csv_rows = read_csv_rows(review_dir / "module_split.csv") if (review_dir / "module_split.csv").exists() else []
    if len(module_csv_rows) != len(module_split):
        add_issue(issues, "module_split_csv_row_count_mismatch", f"module_split.csv rows {len(module_csv_rows)} != json entries {len(module_split)}")
    if module_csv_rows and "event_op_indices" not in module_csv_rows[0]:
        add_issue(issues, "module_split_csv_missing_event_op_indices", "module_split.csv must expose exact event_op_indices for each runtime subprocess")
    module_md = (review_dir / "module_split.md").read_text(encoding="utf-8") if (review_dir / "module_split.md").exists() else ""
    for item in module_split:
        if item.get("module_path") and str(item["module_path"]) not in module_md:
            add_issue(issues, "module_split_md_missing_module", f"{item['module_path']} missing from module_split.md")

    expected_core_evidence = {
        stage: stage_evidence(stage, rows, features, dims)
        for stage in expected_stages
    }
    expected_stage_tensor_io = build_stage_tensor_io(expected_core_evidence)
    expected_op_coverage = build_dispatch_op_coverage(
        rows=rows,
        module_split=module_expected,
        tensor_dataflow=expected_dataflow,
        core_evidence=expected_core_evidence,
    )
    dispatch_op_coverage = load_json(review_dir / "dispatch_op_coverage.json") if (review_dir / "dispatch_op_coverage.json").exists() else {}
    if compact_json(dispatch_op_coverage) != compact_json(expected_op_coverage):
        add_issue(issues, "dispatch_op_coverage_json_mismatch", "dispatch_op_coverage.json is not reproducible from dispatch rows, module split, tensor dataflow, and stage evidence")
    coverage_csv_rows = read_csv_rows(review_dir / "dispatch_op_coverage.csv") if (review_dir / "dispatch_op_coverage.csv").exists() else []
    if len(coverage_csv_rows) != len(rows):
        add_issue(issues, "dispatch_op_coverage_csv_row_count_mismatch", f"dispatch_op_coverage.csv rows {len(coverage_csv_rows)} != dispatch rows {len(rows)}")
    coverage_csv_indices: list[int] = []
    for row in coverage_csv_rows:
        try:
            coverage_csv_indices.append(int(row.get("event_op_index") or 0))
        except ValueError:
            coverage_csv_indices.append(0)
    if coverage_csv_indices != expected_indices:
        add_issue(issues, "dispatch_op_coverage_csv_indices_mismatch", f"coverage csv indices {coverage_csv_indices[:5]}... != expected {expected_indices[:5]}...")
    if expected_op_coverage.get("covered_event_op_indices") != expected_indices:
        add_issue(issues, "dispatch_op_coverage_indices_mismatch", "dispatch op coverage does not list each event_op_index exactly once")
    for key in ["missing_event_op_indices", "duplicate_event_op_indices", "missing_from_module_split", "missing_from_tensor_dataflow"]:
        if expected_op_coverage.get(key):
            add_issue(issues, f"dispatch_op_coverage_{key}", f"{key}: {expected_op_coverage.get(key)}")
    coverage_summary_expected = {
        "op_count": expected_op_coverage.get("op_count", 0),
        "covered_op_count": expected_op_coverage.get("covered_op_count", 0),
        "missing_event_op_indices": expected_op_coverage.get("missing_event_op_indices", []),
        "missing_from_module_split": expected_op_coverage.get("missing_from_module_split", []),
        "missing_from_tensor_dataflow": expected_op_coverage.get("missing_from_tensor_dataflow", []),
    }
    if summary.get("dispatch_op_coverage") != coverage_summary_expected:
        add_issue(issues, "summary_dispatch_op_coverage_mismatch", "summary dispatch_op_coverage does not match recomputed full op coverage")

    layer_profile = load_json(flow_dir / "layer_profile.json") if (flow_dir / "layer_profile.json").exists() else {}
    profile_summary = layer_profile.get("summary") or {}
    if compact_json(profile_summary) != compact_json(summary):
        add_issue(issues, "layer_profile_summary_mismatch", "torch_flow/layer_profile.json summary differs from dispatch_review/summary.json")
    if (layer_profile.get("dispatch_features") or {}).get("expected_stages") != expected_stages:
        add_issue(issues, "layer_profile_expected_stages_mismatch", "layer_profile dispatch_features expected_stages mismatch")

    if layer_profile.get("stage_tensor_io") != expected_stage_tensor_io:
        add_issue(issues, "layer_profile_stage_tensor_io_mismatch", "layer_profile stage_tensor_io is not reproducible from dispatch tensor ids")
    profile_dataflow_summary = layer_profile.get("tensor_dataflow_summary") or {}
    if profile_dataflow_summary.get("edge_count") != expected_dataflow.get("edge_count"):
        add_issue(issues, "layer_profile_tensor_dataflow_edge_count_mismatch", "layer_profile tensor_dataflow_summary.edge_count mismatch")
    if layer_profile.get("dispatch_op_coverage_summary") != coverage_summary_expected:
        add_issue(issues, "layer_profile_dispatch_op_coverage_summary_mismatch", "layer_profile dispatch_op_coverage_summary mismatch")

    process_manifest = load_json(review_dir / "process_manifest.json") if (review_dir / "process_manifest.json").exists() else {}
    if process_manifest.get("expected_stages") != expected_stages:
        add_issue(issues, "process_manifest_expected_stages_mismatch", f"expected {expected_stages}, got {process_manifest.get('expected_stages')}")
    if process_manifest.get("stage_tensor_io") != expected_stage_tensor_io:
        add_issue(issues, "process_manifest_stage_tensor_io_mismatch", "process_manifest stage_tensor_io is not reproducible from dispatch tensor ids")
    process_dataflow = process_manifest.get("tensor_dataflow") or {}
    if process_dataflow.get("edge_count") != expected_dataflow.get("edge_count"):
        add_issue(issues, "process_manifest_tensor_dataflow_edge_count_mismatch", "process_manifest tensor_dataflow.edge_count mismatch")
    for key in ["tensor_dataflow_json", "tensor_dataflow_edges_csv", "tensor_dataflow_markdown"]:
        if key in process_dataflow:
            assert_path_exists(issues, process_dataflow[key], f"process_manifest_missing_{key}")
    process_coverage = process_manifest.get("dispatch_op_coverage") or {}
    if (process_coverage.get("summary") or {}) != coverage_summary_expected:
        add_issue(issues, "process_manifest_dispatch_op_coverage_summary_mismatch", "process_manifest dispatch_op_coverage summary mismatch")
    if len(process_coverage.get("ops", [])) != len(rows):
        add_issue(issues, "process_manifest_dispatch_op_coverage_ops_mismatch", "process_manifest dispatch_op_coverage ops length mismatch")
    for key in ["dispatch_op_coverage_json", "dispatch_op_coverage_csv", "dispatch_op_coverage_markdown"]:
        if key in process_coverage:
            assert_path_exists(issues, process_coverage[key], f"process_manifest_missing_{key}")
    for key in ["dispatch_reconstruction", "toy_tensor_compute", "layer_profile", "process_index"]:
        if key in process_manifest:
            assert_path_exists(issues, process_manifest[key], f"process_manifest_missing_{key}")
    for stage in expected_with_full:
        file_name = STAGE_CODE_FILES.get(stage)
        if file_name and not (flow_dir / file_name).exists():
            add_issue(issues, "missing_stage_code_file", f"{stage}: missing {flow_dir / file_name}")

    process_md = (review_dir / "tensor_compute_process.md").read_text(encoding="utf-8") if (review_dir / "tensor_compute_process.md").exists() else ""
    for heading in ["## Original Dimensions Inferred From Dispatch", "## Small Tensor Config", "## Reconstructed Compute Stages", "## Dispatch Op Coverage"]:
        if heading not in process_md:
            add_issue(issues, "process_markdown_missing_section", f"{heading} missing from tensor_compute_process.md")
    for stage in expected_stages:
        if f"`{stage}`" not in process_md:
            add_issue(issues, "process_markdown_missing_stage", f"{stage} missing from tensor_compute_process.md")
    process_index_md = (flow_dir / "process_index.md").read_text(encoding="utf-8") if (flow_dir / "process_index.md").exists() else ""
    if "## Complete Dispatch Op Coverage" not in process_index_md:
        add_issue(issues, "process_index_missing_complete_op_coverage", "process_index.md must include complete dispatch op coverage")
    for index in expected_indices:
        if f"| {index} |" not in process_index_md:
            add_issue(issues, "process_index_missing_op_index", f"process_index.md missing op index {index}")

    config = parse_config(flow_dir) if (flow_dir / "config.py").exists() else {}
    small_config = summary.get("small_config") or {}
    if config.get("flow_config") != small_config:
        add_issue(issues, "config_small_shape_mismatch", f"config FlowConfig {config.get('flow_config')} != summary small_config {small_config}")
    if config.get("expected_stages") != expected_stages:
        add_issue(issues, "config_expected_stages_mismatch", f"expected {expected_stages}, got {config.get('expected_stages')}")
    if config.get("visual_adjust_kind") != features.get("visual_adjust_kind"):
        add_issue(issues, "config_visual_kind_mismatch", f"expected {features.get('visual_adjust_kind')}, got {config.get('visual_adjust_kind')}")
    if config.get("prune_probe_kind") != features.get("prune_probe_kind"):
        add_issue(issues, "config_prune_probe_kind_mismatch", f"expected {features.get('prune_probe_kind')}, got {config.get('prune_probe_kind')}")

    onnx_manifest = load_json(onnx_dir / "manifest.json") if (onnx_dir / "manifest.json").exists() else []
    stages = manifest_stages(onnx_manifest)
    if stages != expected_with_full:
        add_issue(issues, "onnx_stage_order_mismatch", f"expected {expected_with_full}, got {stages}")
    onnx_by_stage = {stage: item for stage, item in zip(stages, onnx_manifest)}
    shape_expectations = expected_onnx_shapes(
        stage="",
        cfg=small_config,
        visual_kind=features.get("visual_adjust_kind"),
    ) if small_config else {}
    for stage in stages:
        if stage not in shape_expectations:
            add_issue(issues, "unexpected_onnx_stage", f"{stage} has no shape expectation")
            continue
        item = onnx_by_stage[stage]
        expected = shape_expectations[stage]
        for name, shape in expected.get("inputs", {}).items():
            check_shape(issues, item.get("inputs", {}), name, shape, f"{stage}.inputs")
        for name, shape in expected.get("outputs", {}).items():
            check_shape(issues, item.get("outputs", {}), name, shape, f"{stage}.outputs")
        extra_outputs = sorted(set((item.get("outputs") or {})) - set(expected.get("outputs", {})))
        if extra_outputs:
            add_issue(issues, "onnx_stage_extra_outputs", f"{stage}: extra outputs {extra_outputs}")
    if features.get("visual_adjust_kind") == "clear_visual_region":
        visual_outputs = (onnx_by_stage.get("visual_adjust") or {}).get("outputs", {})
        if "tail_visual_sum" in visual_outputs:
            add_issue(issues, "clear_visual_adjust_has_tail_sum_output", "clear-only visual_adjust exposes tail_visual_sum despite no sum.dim_IntList dispatch evidence")

    onnx_code_map = load_json(review_dir / "onnx_code_map.json") if (review_dir / "onnx_code_map.json").exists() else {}
    entries = onnx_code_map.get("entries") if isinstance(onnx_code_map, dict) else []
    if len(entries) != len(onnx_manifest):
        add_issue(issues, "onnx_code_map_entry_count_mismatch", f"entries {len(entries)} != onnx manifest {len(onnx_manifest)}")
    if isinstance(onnx_code_map, dict) and onnx_code_map.get("stage_tensor_io") != expected_stage_tensor_io:
        add_issue(issues, "onnx_code_map_stage_tensor_io_mismatch", "onnx_code_map stage_tensor_io is not reproducible from dispatch tensor ids")
    entry_stages = [entry.get("stage") for entry in entries]
    if entry_stages != stages:
        add_issue(issues, "onnx_code_map_stage_order_mismatch", f"entries {entry_stages} != manifest {stages}")
    for entry in entries:
        expected_io = expected_stage_tensor_io.get(str(entry.get("stage")), {
            "input_tensor_ids": [],
            "output_tensor_ids": [],
            "data_dependencies": [],
        })
        if entry.get("dispatch_input_tensor_ids", []) != expected_io.get("input_tensor_ids", []):
            add_issue(issues, "onnx_code_entry_input_tensor_ids_mismatch", f"{entry.get('stage')}: dispatch_input_tensor_ids mismatch")
        if entry.get("dispatch_output_tensor_ids", []) != expected_io.get("output_tensor_ids", []):
            add_issue(issues, "onnx_code_entry_output_tensor_ids_mismatch", f"{entry.get('stage')}: dispatch_output_tensor_ids mismatch")
        for key in ["code_review_markdown", "onnx_sidecar_markdown"]:
            if entry.get(key):
                assert_path_exists(issues, entry[key], f"missing_{key}")
        for key in ["primary_code_files", "support_code_files"]:
            for path_value in entry.get(key, []):
                assert_path_exists(issues, path_value, f"missing_{key}")
    for item in onnx_manifest:
        assert_path_exists(issues, item["path"], "missing_onnx_file")

    layer_manifest = load_json(event_dir / "layer_manifest.json") if (event_dir / "layer_manifest.json").exists() else {}
    dispatch_review = layer_manifest.get("dispatch_review") or {}
    for key in [
        "dispatch_ops",
        "summary",
        "process",
        "process_manifest",
        "process_code_index",
        "module_split_json",
        "module_split_csv",
        "module_split_markdown",
        "tensor_dataflow_json",
        "tensor_dataflow_edges_csv",
        "tensor_dataflow_markdown",
        "dispatch_op_coverage_json",
        "dispatch_op_coverage_csv",
        "dispatch_op_coverage_markdown",
        "onnx_code_index",
        "onnx_code_map",
    ]:
        if key not in dispatch_review:
            add_issue(issues, "layer_manifest_missing_dispatch_review_key", f"{key} missing")
        else:
            assert_path_exists(issues, dispatch_review[key], f"layer_manifest_missing_{key}")

    status = "pass" if not any(issue["severity"] == "error" for issue in issues) else "fail"
    return {
        "event_id": event_id,
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
        "row_count": len(rows),
        "expected_stages": expected_stages,
        "onnx_stages": stages,
        "phase": features.get("phase"),
        "role": features.get("role"),
        "token_state": features.get("token_state"),
        "q_len": features.get("q_len"),
        "kv_len": features.get("kv_len"),
        "visual_adjust_kind": features.get("visual_adjust_kind"),
        "prune_probe_kind": features.get("prune_probe_kind"),
        "dimensions": dims,
        "small_config": small_config,
        "module_count": len(module_split),
    }


def write_csv(path: Path, audits: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id",
        "status",
        "issue_count",
        "row_count",
        "module_count",
        "phase",
        "role",
        "token_state",
        "q_len",
        "kv_len",
        "visual_adjust_kind",
        "prune_probe_kind",
        "expected_stages",
        "onnx_stages",
        "issue_codes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            writer.writerow({
                "event_id": audit["event_id"],
                "status": audit["status"],
                "issue_count": audit["issue_count"],
                "row_count": audit["row_count"],
                "module_count": audit["module_count"],
                "phase": audit["phase"],
                "role": audit["role"],
                "token_state": audit["token_state"],
                "q_len": audit["q_len"],
                "kv_len": audit["kv_len"],
                "visual_adjust_kind": audit["visual_adjust_kind"],
                "prune_probe_kind": audit["prune_probe_kind"],
                "expected_stages": " | ".join(audit["expected_stages"]),
                "onnx_stages": " | ".join(audit["onnx_stages"]),
                "issue_codes": " | ".join(issue["code"] for issue in audit["issues"]),
            })


def write_markdown(path: Path, audits: list[dict[str, Any]]) -> None:
    counts = Counter(audit["status"] for audit in audits)
    issue_counts = Counter(issue["code"] for audit in audits for issue in audit["issues"])
    lines = [
        "# Layer Reconstruction Audit",
        "",
        "This audit re-checks each layer against its own dispatch rows, generated process files, small-shape ONNX manifests, and runtime module split.",
        "",
        "## Summary",
        "",
        f"- layers audited: `{len(audits)}`",
        f"- pass: `{counts.get('pass', 0)}`",
        f"- fail: `{counts.get('fail', 0)}`",
        "",
        "## Issue Counts",
        "",
    ]
    if issue_counts:
        for code, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Per-Layer Results", ""])
    for audit in audits:
        lines.extend([
            f"### {audit['event_id']}",
            "",
            f"- status: `{audit['status']}`",
            f"- rows/modules: `{audit['row_count']}` / `{audit['module_count']}`",
            f"- dispatch: phase=`{audit['phase']}`, role=`{audit['role']}`, q_len=`{audit['q_len']}`, kv_len=`{audit['kv_len']}`",
            f"- expected stages: `{', '.join(audit['expected_stages'])}`",
            f"- onnx stages: `{', '.join(audit['onnx_stages'])}`",
        ])
        if audit["issues"]:
            lines.append("- issues:")
            for issue in audit["issues"]:
                lines.append(f"  - `{issue['code']}`: {issue['detail']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--layer-dir", type=Path, default=DEFAULT_LAYER_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--layers", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_dispatch_rows(args.source_csv)
    if args.layers:
        event_ids = args.layers
    else:
        event_ids = sorted(path.name for path in args.layer_dir.iterdir() if path.is_dir() and "_layer" in path.name)
    audits = [audit_event(event_id, source_rows, args.layer_dir) for event_id in event_ids]
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "layer_audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    write_csv(args.audit_dir / "layer_audit.csv", audits)
    write_markdown(args.audit_dir / "layer_audit.md", audits)
    print(json.dumps({
        "audited": len(audits),
        "status_counts": dict(Counter(audit["status"] for audit in audits)),
        "issue_counts": dict(Counter(issue["code"] for audit in audits for issue in audit["issues"])),
        "audit_dir": str(args.audit_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
