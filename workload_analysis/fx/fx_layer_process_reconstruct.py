#!/usr/bin/env python3
"""Reconstruct readable FX layer processes from saved GraphModule graphs.

This script intentionally consumes ``fx_graph_module.pt`` and walks
``GraphModule.graph.nodes`` directly. It does not read ``fx_nodes.json``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx


STAGE_ORDER = [
    "inputs",
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention_scores",
    "attention_output",
    "visual_process",
    "output_projection",
    "post_attention_rmsnorm",
    "mlp",
    "layer_output",
    "other",
]

STAGE_TITLES = {
    "inputs": "Runtime FX inputs",
    "input_rmsnorm": "Input RMSNorm",
    "qkv_projection": "Q/K/V projection and head reshape",
    "rope": "RoPE position embedding",
    "attention_scores": "QK scores, mask, softmax",
    "attention_output": "Attention-weighted V and hidden reshape",
    "visual_process": "Visual-related value-aware process",
    "output_projection": "Attention output projection and residual",
    "post_attention_rmsnorm": "Post-attention RMSNorm",
    "mlp": "MLP and final residual",
    "layer_output": "Layer output",
    "other": "Other FX nodes",
}

SETUP_TARGETS = {
    "aten.expand.default",
    "aten.view.default",
    "aten.transpose.int",
    "aten._unsafe_view.default",
}


def compact_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def target_to_str(target: Any) -> str:
    text = str(target)
    if text.startswith("aten."):
        return text
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        module = str(target.__module__)
        name = str(target.__name__)
        if module == "torch._ops.aten":
            return f"aten.{name}"
        if module and module != "builtins":
            return f"{module}.{name}"
        return name
    return text


def ensure_torch_graphmodule_load_compat() -> None:
    """Patch a PyTorch 2.12 GraphModule pickle load path if needed.

    This environment has ``torch._higher_order_ops.triton_kernel_wrap`` on disk,
    but it is not always attached as an attribute on ``torch._higher_order_ops``.
    FX GraphModule deserialization reaches it through that attribute path.
    """
    try:
        higher_order_ops = torch._higher_order_ops
    except AttributeError:
        return
    if hasattr(higher_order_ops, "triton_kernel_wrap"):
        return
    try:
        module = importlib.import_module("torch._higher_order_ops.triton_kernel_wrap")
    except Exception:
        return
    setattr(higher_order_ops, "triton_kernel_wrap", module)


def load_graph_module(path: Path) -> fx.GraphModule:
    ensure_torch_graphmodule_load_compat()
    try:
        graph_module = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graph_module = torch.load(path, map_location="cpu")
    if not isinstance(graph_module, fx.GraphModule):
        raise TypeError(f"Expected torch.fx.GraphModule in {path}, got {type(graph_module)!r}")
    return graph_module


def iter_node_refs(value: Any) -> Iterable[fx.Node]:
    if isinstance(value, fx.Node):
        yield value
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from iter_node_refs(item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from iter_node_refs(item)


def node_arg_names(node: fx.Node) -> list[str]:
    return [ref.name for ref in iter_node_refs((node.args, node.kwargs))]


def describe_tensor(value: Any, depth: int = 0) -> Any:
    if torch.is_tensor(value):
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, tuple):
        return {"tuple": [describe_tensor(item, depth + 1) for item in value]}
    if isinstance(value, list):
        return [describe_tensor(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        return {str(key): describe_tensor(item, depth + 1) for key, item in value.items()}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in shape] if shape is not None else None,
            "dtype": str(getattr(value, "dtype", "")),
        }
    return type(value).__name__


def node_meta_summary(node: fx.Node) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "tensor_meta" in node.meta:
        tensor_meta = node.meta["tensor_meta"]
        shape = getattr(tensor_meta, "shape", None)
        dtype = getattr(tensor_meta, "dtype", None)
        if shape is not None:
            summary["shape"] = [int(item) for item in shape]
        if dtype is not None:
            summary["dtype"] = str(dtype).replace("torch.", "")
    if "val" in node.meta:
        summary["val"] = describe_tensor(node.meta["val"])
        if "shape" not in summary and torch.is_tensor(node.meta["val"]):
            summary["shape"] = [int(item) for item in node.meta["val"].shape]
            summary["dtype"] = str(node.meta["val"].dtype).replace("torch.", "")
    return summary


def output_expr(value: Any) -> str:
    if isinstance(value, fx.Node):
        return value.name
    if isinstance(value, tuple):
        return "(" + ", ".join(output_expr(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(output_expr(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key!r}: {output_expr(item)}" for key, item in value.items()) + "}"
    return repr(value)


def render_node_line(node: fx.Node) -> str:
    if node.op == "placeholder":
        return f"# placeholder {node.name}"
    if node.op == "get_attr":
        return f"{node.name} = self.{node.target}"
    if node.op == "output":
        return f"return {output_expr(node.args[0] if node.args else None)}"
    args = ", ".join(output_expr(item) for item in node.args)
    kwargs = ", ".join(f"{key}={output_expr(item)}" for key, item in node.kwargs.items())
    joined_args = ", ".join(item for item in [args, kwargs] if item)
    if node.op == "call_function":
        return f"{node.name} = {target_to_str(node.target)}({joined_args})"
    if node.op == "call_method":
        return f"{node.name} = {node.target}({joined_args})"
    if node.op == "call_module":
        return f"{node.name} = self.{node.target}({joined_args})"
    return f"{node.name} = <{node.op}:{target_to_str(node.target)}>({joined_args})"


def index_by_node(nodes: list[fx.Node]) -> dict[fx.Node, int]:
    return {node: index for index, node in enumerate(nodes)}


def is_target(node: fx.Node, target: str) -> bool:
    return target_to_str(node.target) == target


def direct_input_node(node: fx.Node, position: int = 0) -> fx.Node | None:
    refs = list(iter_node_refs(node.args))
    if position >= len(refs):
        return None
    return refs[position]


def get_attr_feeding_t_before(nodes: list[fx.Node], before_index: int) -> int | None:
    node_to_index = index_by_node(nodes)
    candidates: list[int] = []
    for index, node in enumerate(nodes[:before_index]):
        if not is_target(node, "aten.t.default"):
            continue
        ref = direct_input_node(node, 0)
        if ref is None or ref.op != "get_attr":
            continue
        candidates.append(node_to_index[ref])
    return min(candidates) if candidates else None


def get_attr_for_mm_weight(mm_node: fx.Node, node_to_index: dict[fx.Node, int]) -> int | None:
    refs = list(iter_node_refs(mm_node.args))
    if len(refs) < 2:
        return None
    weight_arg = refs[1]
    if is_target(weight_arg, "aten.t.default"):
        param = direct_input_node(weight_arg, 0)
        if param is not None and param.op == "get_attr":
            return node_to_index[param]
    if weight_arg.op == "get_attr":
        return node_to_index[weight_arg]
    return None


def first_index_after(nodes: list[fx.Node], start: int, targets: set[str]) -> int | None:
    for index, node in enumerate(nodes[start:], start=start):
        if target_to_str(node.target) in targets:
            return index
    return None


def last_index_before(nodes: list[fx.Node], end: int, target: str) -> int | None:
    for index in range(end - 1, -1, -1):
        if is_target(nodes[index], target):
            return index
    return None


def projection_input_source(mm_node: fx.Node) -> fx.Node | None:
    first_arg = direct_input_node(mm_node, 0)
    if first_arg is None:
        return None
    if target_to_str(first_arg.target) in {"aten.view.default", "aten._unsafe_view.default"}:
        inner = direct_input_node(first_arg, 0)
        if inner is not None:
            return inner
    return first_arg


def build_stage_ranges(nodes: list[fx.Node]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    if not nodes:
        return ranges

    node_to_index = index_by_node(nodes)
    output_indices = [idx for idx, node in enumerate(nodes) if node.op == "output"]
    output_idx = output_indices[0] if output_indices else len(nodes) - 1

    placeholder_indices = [idx for idx, node in enumerate(nodes) if node.op == "placeholder"]
    if placeholder_indices:
        ranges["inputs"] = (min(placeholder_indices), max(placeholder_indices))

    mm_indices = [idx for idx, node in enumerate(nodes) if is_target(node, "aten.mm.default")]
    bmm_indices = [idx for idx, node in enumerate(nodes) if is_target(node, "aten.bmm.default")]

    first_mm_idx = mm_indices[0] if mm_indices else None
    first_bmm_idx = bmm_indices[0] if bmm_indices else None
    second_bmm_idx = bmm_indices[1] if len(bmm_indices) > 1 else None

    if first_mm_idx is None:
        ranges["other"] = (max(placeholder_indices, default=-1) + 1, output_idx - 1)
        ranges["layer_output"] = (output_idx, output_idx)
        return ranges

    qkv_start = get_attr_feeding_t_before(nodes, first_bmm_idx or len(nodes))
    if qkv_start is None:
        qkv_start = max(max(placeholder_indices, default=-1) + 1, first_mm_idx - 3)

    norm_start = max(placeholder_indices, default=-1) + 1
    if norm_start <= qkv_start - 1:
        ranges["input_rmsnorm"] = (norm_start, qkv_start - 1)

    rope_start = None
    if first_bmm_idx is not None:
        first_index_tensor_idx = None
        for index in range(qkv_start, first_bmm_idx):
            node = nodes[index]
            if is_target(node, "aten.index.Tensor"):
                first_index_tensor_idx = index
                break
        if first_index_tensor_idx is not None:
            candidate_indices = [first_index_tensor_idx]
            for ref in iter_node_refs(nodes[first_index_tensor_idx].args):
                if ref.op == "get_attr":
                    candidate_indices.append(node_to_index[ref])
            rope_start = min(candidate_indices)
        else:
            for index in range(qkv_start, first_bmm_idx):
                node = nodes[index]
                if node.op == "get_attr" and str(node.target).startswith("_tensor_constant"):
                    rope_start = index
                    break
    if rope_start is None:
        rope_start = first_bmm_idx if first_bmm_idx is not None else output_idx

    if qkv_start <= rope_start - 1:
        ranges["qkv_projection"] = (qkv_start, rope_start - 1)

    if first_bmm_idx is None:
        if rope_start <= output_idx - 1:
            ranges["other"] = (rope_start, output_idx - 1)
        ranges["layer_output"] = (output_idx, output_idx)
        return ranges

    last_rope_add = last_index_before(nodes, first_bmm_idx, "aten.add.Tensor")
    if last_rope_add is not None and last_rope_add >= rope_start:
        rope_end = last_rope_add
        attention_scores_start = rope_end + 1
    else:
        rope_end = first_bmm_idx - 1
        attention_scores_start = first_bmm_idx
    if rope_start <= rope_end:
        ranges["rope"] = (rope_start, rope_end)

    if second_bmm_idx is None:
        ranges["attention_scores"] = (attention_scores_start, output_idx - 1)
        ranges["layer_output"] = (output_idx, output_idx)
        return ranges

    attention_output_start = None
    for index in range(second_bmm_idx - 1, first_bmm_idx, -1):
        node = nodes[index]
        if target_to_str(node.target) in {
            "aten.clone.default",
            "aten._to_copy.default",
            "aten._softmax.default",
        }:
            attention_output_start = index + 1
            break
    if attention_output_start is None:
        attention_output_start = first_index_after(nodes, first_bmm_idx + 1, SETUP_TARGETS) or second_bmm_idx

    if attention_scores_start <= attention_output_start - 1:
        ranges["attention_scores"] = (attention_scores_start, attention_output_start - 1)

    mm_after_second_bmm = [idx for idx in mm_indices if idx > second_bmm_idx]
    output_proj_mm_idx = mm_after_second_bmm[0] if mm_after_second_bmm else None

    if output_proj_mm_idx is None:
        ranges["attention_output"] = (attention_output_start, output_idx - 1)
        ranges["layer_output"] = (output_idx, output_idx)
        return ranges

    output_proj_get_attr = get_attr_for_mm_weight(nodes[output_proj_mm_idx], node_to_index)
    output_proj_first_arg = direct_input_node(nodes[output_proj_mm_idx], 0)
    output_proj_first_arg_idx = node_to_index.get(output_proj_first_arg) if output_proj_first_arg is not None else None
    output_proj_start = min(
        index
        for index in [output_proj_get_attr, output_proj_first_arg_idx, output_proj_mm_idx]
        if index is not None
    )

    proj_source = projection_input_source(nodes[output_proj_mm_idx])
    attention_output_end = node_to_index.get(proj_source, output_proj_start - 1) if proj_source else output_proj_start - 1
    if attention_output_end >= output_proj_start:
        attention_output_end = output_proj_start - 1

    if attention_output_start <= attention_output_end:
        ranges["attention_output"] = (attention_output_start, attention_output_end)
    if attention_output_end + 1 <= output_proj_start - 1:
        ranges["visual_process"] = (attention_output_end + 1, output_proj_start - 1)

    output_proj_end = None
    for index in range(output_proj_mm_idx + 1, output_idx):
        if is_target(nodes[index], "aten.add.Tensor"):
            output_proj_end = index
            break
    if output_proj_end is None:
        output_proj_end = output_proj_mm_idx
    ranges["output_projection"] = (output_proj_start, output_proj_end)

    remaining_mm = [idx for idx in mm_indices if idx > output_proj_mm_idx]
    mlp_first_mm_idx = remaining_mm[0] if remaining_mm else None
    mlp_start = None
    if mlp_first_mm_idx is not None:
        mlp_start = get_attr_for_mm_weight(nodes[mlp_first_mm_idx], node_to_index)
    if mlp_start is None and mlp_first_mm_idx is not None:
        mlp_start = mlp_first_mm_idx

    if mlp_start is not None:
        if output_proj_end + 1 <= mlp_start - 1:
            ranges["post_attention_rmsnorm"] = (output_proj_end + 1, mlp_start - 1)
        if mlp_start <= output_idx - 1:
            ranges["mlp"] = (mlp_start, output_idx - 1)
    elif output_proj_end + 1 <= output_idx - 1:
        ranges["post_attention_rmsnorm"] = (output_proj_end + 1, output_idx - 1)

    ranges["layer_output"] = (output_idx, output_idx)
    return ranges


def stage_for_index(index: int, ranges: Mapping[str, tuple[int, int]]) -> str:
    for stage in STAGE_ORDER:
        span = ranges.get(stage)
        if span is None:
            continue
        if span[0] <= index <= span[1]:
            return stage
    return "other"


def serialize_graph_nodes(nodes: list[fx.Node], stage_map: Mapping[int, str]) -> list[dict[str, Any]]:
    rows = []
    for index, node in enumerate(nodes):
        meta = node_meta_summary(node)
        rows.append(
            {
                "index": index,
                "name": node.name,
                "op": node.op,
                "target": target_to_str(node.target),
                "process_stage": stage_map.get(index, "other"),
                "args": node_arg_names(node),
                "users": sorted(user.name for user in node.users),
                "meta": meta,
            }
        )
    return rows


def summarize_stages(nodes: list[fx.Node], ranges: Mapping[str, tuple[int, int]], stage_map: Mapping[int, str]) -> list[dict[str, Any]]:
    node_to_index = index_by_node(nodes)
    summaries: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        indices = [index for index in range(len(nodes)) if stage_map.get(index) == stage]
        if not indices:
            continue
        stage_set = set(indices)
        external_inputs = sorted(
            {
                ref.name
                for index in indices
                for ref in iter_node_refs((nodes[index].args, nodes[index].kwargs))
                if node_to_index.get(ref) not in stage_set
            }
        )
        external_outputs = sorted(
            {
                nodes[index].name
                for index in indices
                for user in nodes[index].users
                if node_to_index.get(user) not in stage_set
            }
        )
        summaries.append(
            {
                "stage": stage,
                "title": STAGE_TITLES.get(stage, stage),
                "start_index": min(indices),
                "end_index": max(indices),
                "node_count": len(indices),
                "external_inputs": external_inputs,
                "external_outputs": external_outputs,
                "nodes": [nodes[index].name for index in indices],
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "process_stage",
        "name",
        "op",
        "target",
        "args",
        "users",
        "shape",
        "dtype",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            writer.writerow(
                {
                    "index": row["index"],
                    "process_stage": row["process_stage"],
                    "name": row["name"],
                    "op": row["op"],
                    "target": row["target"],
                    "args": " ".join(row["args"]),
                    "users": " ".join(row["users"]),
                    "shape": compact_json(meta.get("shape")) if meta.get("shape") is not None else "",
                    "dtype": meta.get("dtype", ""),
                }
            )


def render_markdown(
    trace_dir: Path,
    graph_module_pt: Path,
    stages: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    graph_module: fx.GraphModule,
) -> str:
    lines = [
        "# FX Layer Process Reconstruction",
        "",
        f"Trace directory: `{trace_dir}`",
        f"GraphModule: `{graph_module_pt.name}`",
        "",
        "Source: this file is reconstructed by loading `fx_graph_module.pt` and iterating `GraphModule.graph.nodes`.",
        "FX provides the graph DAG and node metadata; the process labels below are reconstruction labels over that DAG.",
        "",
        "## Stage Summary",
        "",
        "| stage | node range | node count | external inputs | external outputs |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for stage in stages:
        inputs = ", ".join(f"`{item}`" for item in stage["external_inputs"]) or "-"
        outputs = ", ".join(f"`{item}`" for item in stage["external_outputs"]) or "-"
        lines.append(
            f"| {stage['title']} | {stage['start_index']}-{stage['end_index']} | "
            f"{stage['node_count']} | {inputs} | {outputs} |"
        )
    lines.extend(["", "## Process Code", ""])
    for stage in stages:
        stage_nodes = [node for node in nodes if node["process_stage"] == stage["stage"]]
        lines.extend([f"### {stage['title']}", "", "```python"])
        for node in stage_nodes:
            fx_node = next(item for item in graph_module.graph.nodes if item.name == node["name"])
            lines.append(render_node_line(fx_node))
        lines.extend(["```", ""])
    lines.extend(
        [
            "## Node Table",
            "",
            "| index | stage | name | op | target | args | users |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for node in nodes:
        args = ", ".join(f"`{item}`" for item in node["args"]) or "-"
        users = ", ".join(f"`{item}`" for item in node["users"]) or "-"
        lines.append(
            f"| {node['index']} | `{node['process_stage']}` | `{node['name']}` | "
            f"`{node['op']}` | `{node['target']}` | {args} | {users} |"
        )
    return "\n".join(lines) + "\n"


def reconstruct_one(trace_dir: Path, graph_module_pt: Path | None = None) -> dict[str, Any]:
    if graph_module_pt is None:
        graph_module_pt = trace_dir / "fx_graph_module.pt"
    graph_module = load_graph_module(graph_module_pt)
    graph_nodes = list(graph_module.graph.nodes)
    ranges = build_stage_ranges(graph_nodes)
    stage_map = {index: stage_for_index(index, ranges) for index in range(len(graph_nodes))}
    nodes = serialize_graph_nodes(graph_nodes, stage_map)
    stages = summarize_stages(graph_nodes, ranges, stage_map)

    payload = {
        "trace_dir": str(trace_dir),
        "graph_module_pt": str(graph_module_pt),
        "source": "GraphModule.graph",
        "node_count": len(nodes),
        "stage_count": len(stages),
        "stages": stages,
        "nodes": nodes,
    }

    json_path = trace_dir / "fx_process_reconstruction.json"
    md_path = trace_dir / "fx_process_reconstruction.md"
    csv_path = trace_dir / "fx_process_nodes.csv"
    json_path.write_text(compact_json(payload) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(trace_dir, graph_module_pt, stages, nodes, graph_module), encoding="utf-8")
    write_csv(csv_path, nodes)

    return {
        "trace_dir": str(trace_dir),
        "graph_module_pt": str(graph_module_pt),
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "node_count": len(nodes),
        "stage_count": len(stages),
        "status": "ok",
    }


def discover_graph_modules(root: Path) -> list[Path]:
    return sorted(root.rglob("fx_graph_module.pt"))


def write_manifest(root: Path, results: list[dict[str, Any]]) -> None:
    json_path = root / "fx_process_reconstruction_manifest.json"
    csv_path = root / "fx_process_reconstruction_manifest.csv"
    json_path.write_text(compact_json({"processed": len(results), "results": results}) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["trace_dir", "graph_module_pt", "json", "markdown", "csv", "node_count", "stage_count", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Directory containing fx_graph_module.pt, or recursive trace root.")
    parser.add_argument("--graph-module", help="Explicit fx_graph_module.pt path. Overrides <trace-dir>/fx_graph_module.pt.")
    parser.add_argument("--recursive", action="store_true", help="Process every fx_graph_module.pt under --trace-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    if args.recursive:
        graph_modules = discover_graph_modules(trace_dir)
        if not graph_modules:
            raise FileNotFoundError(f"No fx_graph_module.pt files found under {trace_dir}")
        results = []
        for graph_module_pt in graph_modules:
            event_dir = graph_module_pt.parent
            try:
                results.append(reconstruct_one(event_dir, graph_module_pt))
            except Exception as exc:
                results.append(
                    {
                        "trace_dir": str(event_dir),
                        "graph_module_pt": str(graph_module_pt),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
        write_manifest(trace_dir, results)
        print(compact_json({"processed": len(results), "results": results}))
        return

    graph_module_pt = Path(args.graph_module) if args.graph_module else trace_dir / "fx_graph_module.pt"
    result = reconstruct_one(trace_dir, graph_module_pt)
    print(compact_json(result))


if __name__ == "__main__":
    main()
