#!/usr/bin/env python3
"""Create Zetane/ONNX and TensorHue visualizations for input1_layer0."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import tensorhue
from onnx import TensorProto, helper, numpy_helper
from rich.console import Console


WORKLOAD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_DIR = WORKLOAD_DIR / "dispatch/visualize/input1_layer0"

ZETANE_COMPAT_IR_VERSION = 6
ZETANE_COMPAT_OPSET = 11


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    dtype: str
    device: str = ""
    requires_grad: bool = False

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TensorLayout":
        return cls(
            shape=tuple(int(dim) for dim in record.get("shape", [])),
            dtype=str(record.get("dtype", "")),
            device=str(record.get("device", "")),
            requires_grad=bool(record.get("requires_grad", False)),
        )

    def key(self) -> tuple[tuple[int, ...], str, str, bool]:
        return (self.shape, self.dtype, self.device, self.requires_grad)


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return safe or "unnamed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def flatten_tensors(value: Any, path: str = "value") -> list[tuple[str, TensorLayout]]:
    if isinstance(value, dict) and value.get("type") == "Tensor":
        return [(path, TensorLayout.from_record(value))]
    if isinstance(value, list):
        out: list[tuple[str, TensorLayout]] = []
        for idx, item in enumerate(value):
            out.extend(flatten_tensors(item, f"{path}_{idx}"))
        return out
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(flatten_tensors(item, f"{path}_{sanitize_name(str(key))}"))
        return out
    return []


def tensorproto_dtype(dtype: str) -> int:
    mapping = {
        "float16": TensorProto.FLOAT16,
        "float32": TensorProto.FLOAT,
        "float": TensorProto.FLOAT,
        "int64": TensorProto.INT64,
        "int32": TensorProto.INT32,
        "bool": TensorProto.BOOL,
    }
    return mapping.get(dtype, TensorProto.FLOAT)


def value_info(name: str, layout: TensorLayout) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, tensorproto_dtype(layout.dtype), list(layout.shape))


def compact(value: Any, max_chars: int = 400) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def constant_of_shape_value(dtype: str) -> onnx.TensorProto:
    tensor_dtype = tensorproto_dtype(dtype)
    if tensor_dtype == TensorProto.FLOAT16:
        return numpy_helper.from_array(np.array([0], dtype=np.float16), name="value")
    if tensor_dtype == TensorProto.FLOAT:
        return numpy_helper.from_array(np.array([0], dtype=np.float32), name="value")
    if tensor_dtype == TensorProto.INT64:
        return numpy_helper.from_array(np.array([0], dtype=np.int64), name="value")
    if tensor_dtype == TensorProto.INT32:
        return numpy_helper.from_array(np.array([0], dtype=np.int32), name="value")
    if tensor_dtype == TensorProto.BOOL:
        return numpy_helper.from_array(np.array([False], dtype=np.bool_), name="value")
    return numpy_helper.from_array(np.array([0], dtype=np.float32), name="value")


def scalar_constant(name: str, value: Any) -> onnx.NodeProto:
    if isinstance(value, bool):
        tensor = numpy_helper.from_array(np.array(value, dtype=np.bool_), name=f"{name}_value")
    elif isinstance(value, int):
        tensor = numpy_helper.from_array(np.array(value, dtype=np.int64), name=f"{name}_value")
    elif isinstance(value, float):
        tensor = numpy_helper.from_array(np.array(value, dtype=np.float32), name=f"{name}_value")
    else:
        tensor = numpy_helper.from_array(np.array(0, dtype=np.float32), name=f"{name}_value")
    return helper.make_node("Constant", inputs=[], outputs=[name], value=tensor)


def make_zetane_onnx(rows: list[dict[str, str]], output_path: Path) -> dict[str, Any]:
    """Build a conservative standard-ONNX graph for Zetane Viewer.

    Zetane Viewer can reject valid ONNX files that use custom domains or newer
    IR/opset versions. This graph intentionally uses only standard
    ConstantOfShape/Constant nodes, opset 11, and IR v6. Each dispatch op is
    represented by one or more standard nodes whose names/doc strings preserve
    the original ATen op and recorded tensor layout.
    """
    graph_outputs: dict[str, onnx.ValueInfoProto] = {}
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    for row in rows:
        op_index = int(row.get("event_op_index") or len(nodes) + 1)
        op_name = row["op_name"]
        node_prefix = f"op_{op_index:03d}_{sanitize_name(op_name)}"
        outputs = safe_json_loads(row.get("outputs", "null"))
        output_layouts = flatten_tensors(outputs, "outputs")
        if output_layouts:
            for out_idx, (_path, layout) in enumerate(output_layouts):
                shape_name = f"{node_prefix}_out{out_idx}_shape"
                output_name = f"{node_prefix}_out{out_idx}"
                initializers.append(
                    numpy_helper.from_array(
                        np.array(layout.shape, dtype=np.int64),
                        name=shape_name,
                    )
                )
                info = value_info(output_name, layout)
                graph_outputs[output_name] = info
                node = helper.make_node(
                    "ConstantOfShape",
                    inputs=[shape_name],
                    outputs=[output_name],
                    name=f"{node_prefix}_layout_out{out_idx}",
                    value=constant_of_shape_value(layout.dtype),
                )
                node.doc_string = (
                    f"ATen dispatch op: {op_name}\n"
                    f"schema: {row.get('op_schema', '')}\n"
                    f"recorded output shape: {list(layout.shape)}\n"
                    f"recorded dtype/device: {layout.dtype}/{layout.device}\n"
                    f"q_len={row.get('q_len', '')}, kv_len={row.get('kv_len', '')}, "
                    f"token_state={row.get('token_state', '')}"
                )
                nodes.append(node)
        else:
            output_name = f"{node_prefix}_scalar"
            node = scalar_constant(output_name, outputs)
            node.name = f"{node_prefix}_scalar"
            node.doc_string = (
                f"ATen dispatch op: {op_name}\n"
                f"schema: {row.get('op_schema', '')}\n"
                f"recorded non-tensor output: {compact(outputs)}"
            )
            nodes.append(node)
            graph_outputs[output_name] = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [])

    graph = helper.make_graph(
        nodes=nodes,
        name="input1_layer0_dispatch_tensor_process",
        inputs=[],
        outputs=list(graph_outputs.values()),
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="VisiPrune workload_analysis",
        opset_imports=[helper.make_operatorsetid("", ZETANE_COMPAT_OPSET)],
    )
    model.ir_version = ZETANE_COMPAT_IR_VERSION
    model.doc_string = (
        "Zetane-compatible standard ONNX graph generated from input1_layer0 dispatch_ops.csv. "
        "Node names/doc strings preserve ATen dispatch op names; ConstantOfShape outputs preserve tensor layouts."
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return {
        "onnx_path": str(output_path),
        "node_count": len(nodes),
        "graph_input_count": 0,
        "graph_output_count": len(graph_outputs),
        "opset": ZETANE_COMPAT_OPSET,
        "ir_version": ZETANE_COMPAT_IR_VERSION,
        "custom_domain": "",
        "compatibility": "standard ONNX ConstantOfShape/Constant nodes; no custom op domain",
    }


def make_zetane_sequence_onnx(rows: list[dict[str, str]], output_path: Path) -> dict[str, Any]:
    """Build a maximal-compatibility ONNX graph as a standard Identity chain."""
    nodes: list[onnx.NodeProto] = []
    value_infos: list[onnx.ValueInfoProto] = []
    previous = "dispatch_sequence_input"
    for row in rows:
        op_index = int(row.get("event_op_index") or len(nodes) + 1)
        op_name = row["op_name"]
        output = f"seq_{op_index:03d}_{sanitize_name(op_name)}"
        node = helper.make_node(
            "Identity",
            inputs=[previous],
            outputs=[output],
            name=f"op_{op_index:03d}_{sanitize_name(op_name)}",
        )
        node.doc_string = (
            f"ATen dispatch op: {op_name}\n"
            f"schema: {row.get('op_schema', '')}\n"
            f"recorded outputs: {compact(safe_json_loads(row.get('outputs', 'null')))}\n"
            f"q_len={row.get('q_len', '')}, kv_len={row.get('kv_len', '')}, "
            f"token_state={row.get('token_state', '')}"
        )
        nodes.append(node)
        value_infos.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [1]))
        previous = output

    graph = helper.make_graph(
        nodes=nodes,
        name="input1_layer0_dispatch_op_sequence",
        inputs=[helper.make_tensor_value_info("dispatch_sequence_input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1])],
        value_info=value_infos[:-1],
    )
    model = helper.make_model(
        graph,
        producer_name="VisiPrune workload_analysis",
        opset_imports=[helper.make_operatorsetid("", ZETANE_COMPAT_OPSET)],
    )
    model.ir_version = ZETANE_COMPAT_IR_VERSION
    model.doc_string = (
        "Maximal-compatibility ONNX graph for Zetane Viewer. "
        "A single dummy tensor flows through 91 standard Identity nodes; "
        "node names/doc strings preserve the input1_layer0 ATen dispatch process."
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return {
        "onnx_path": str(output_path),
        "input_name": "dispatch_sequence_input",
        "input_shape": [1],
        "input_dtype": "float32",
        "node_count": len(nodes),
        "graph_input_count": 1,
        "graph_output_count": 1,
        "opset": ZETANE_COMPAT_OPSET,
        "ir_version": ZETANE_COMPAT_IR_VERSION,
        "custom_domain": "",
        "compatibility": "standard ONNX Identity chain with one dummy input",
    }


def make_zetane_feature_map_sequence_onnx(rows: list[dict[str, str]], output_path: Path) -> dict[str, Any]:
    """Build a Zetane-friendly Identity chain with a 4D NCHW feature-map input."""
    feature_shape = [1, 1, len(rows), 6]
    nodes: list[onnx.NodeProto] = []
    value_infos: list[onnx.ValueInfoProto] = []
    previous = "dispatch_feature_map_input"
    for row in rows:
        op_index = int(row.get("event_op_index") or len(nodes) + 1)
        op_name = row["op_name"]
        output = f"feature_seq_{op_index:03d}_{sanitize_name(op_name)}"
        node = helper.make_node(
            "Identity",
            inputs=[previous],
            outputs=[output],
            name=f"op_{op_index:03d}_{sanitize_name(op_name)}",
        )
        node.doc_string = (
            f"ATen dispatch op: {op_name}\n"
            f"schema: {row.get('op_schema', '')}\n"
            f"recorded outputs: {compact(safe_json_loads(row.get('outputs', 'null')))}\n"
            f"feature map input rows are dispatch ops; columns are normalized layout features."
        )
        nodes.append(node)
        value_infos.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, feature_shape))
        previous = output

    graph = helper.make_graph(
        nodes=nodes,
        name="input1_layer0_dispatch_feature_map_sequence",
        inputs=[helper.make_tensor_value_info("dispatch_feature_map_input", TensorProto.FLOAT, feature_shape)],
        outputs=[helper.make_tensor_value_info(previous, TensorProto.FLOAT, feature_shape)],
        value_info=value_infos[:-1],
    )
    model = helper.make_model(
        graph,
        producer_name="VisiPrune workload_analysis",
        opset_imports=[helper.make_operatorsetid("", ZETANE_COMPAT_OPSET)],
    )
    model.ir_version = ZETANE_COMPAT_IR_VERSION
    model.doc_string = (
        "Zetane-friendly feature-map ONNX graph for input1_layer0. "
        "A [1,1,91,6] feature map flows through 91 standard Identity nodes; "
        "node names/doc strings preserve the ATen dispatch process."
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return {
        "onnx_path": str(output_path),
        "input_name": "dispatch_feature_map_input",
        "input_shape": feature_shape,
        "input_dtype": "float32",
        "node_count": len(nodes),
        "graph_input_count": 1,
        "graph_output_count": 1,
        "opset": ZETANE_COMPAT_OPSET,
        "ir_version": ZETANE_COMPAT_IR_VERSION,
        "custom_domain": "",
        "compatibility": "standard ONNX Identity chain with one 4D feature-map input",
    }


def make_zetane_dummy_inputs(output_dir: Path) -> dict[str, str]:
    """Write single-input files for Zetane's custom input loader."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy = np.array([0.0], dtype=np.float32)
    npy_path = output_dir / "dispatch_sequence_input.npy"
    npz_path = output_dir / "dispatch_sequence_input.npz"
    np.save(npy_path, dummy)
    np.savez(npz_path, dispatch_sequence_input=dummy)
    return {
        "npy": str(npy_path),
        "npz": str(npz_path),
        "input_name": "dispatch_sequence_input",
        "shape": "[1]",
        "dtype": "float32",
    }


def make_zetane_feature_map_inputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, str]:
    """Write 4D feature-map input files for Zetane's custom input loader."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_map = shape_heatmap(rows).reshape(1, 1, len(rows), 6).astype(np.float32)
    npy_path = output_dir / "dispatch_feature_map_input.npy"
    npz_path = output_dir / "dispatch_feature_map_input.npz"
    np.save(npy_path, feature_map)
    np.savez(npz_path, dispatch_feature_map_input=feature_map)
    return {
        "npy": str(npy_path),
        "npz": str(npz_path),
        "input_name": "dispatch_feature_map_input",
        "shape": f"{list(feature_map.shape)}",
        "dtype": "float32",
    }


def shape_heatmap(rows: list[dict[str, str]]) -> np.ndarray:
    matrix = np.zeros((len(rows), 6), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        args = safe_json_loads(row.get("args", "null"))
        outputs = safe_json_loads(row.get("outputs", "null"))
        arg_layouts = [layout for _path, layout in flatten_tensors(args, "args")]
        out_layouts = [layout for _path, layout in flatten_tensors(outputs, "outputs")]
        out_numel = max((np.prod(layout.shape, dtype=np.float64) for layout in out_layouts), default=1.0)
        in_numel = max((np.prod(layout.shape, dtype=np.float64) for layout in arg_layouts), default=1.0)
        out_rank = max((len(layout.shape) for layout in out_layouts), default=0)
        in_rank = max((len(layout.shape) for layout in arg_layouts), default=0)
        matrix[row_idx] = np.array(
            [
                row_idx + 1,
                in_rank,
                out_rank,
                np.log10(in_numel + 1.0),
                np.log10(out_numel + 1.0),
                float(len(out_layouts)),
            ],
            dtype=np.float32,
        )
    # Normalize columns for display while preserving relative op-by-op changes.
    mins = matrix.min(axis=0, keepdims=True)
    maxs = matrix.max(axis=0, keepdims=True)
    return (matrix - mins) / np.maximum(maxs - mins, 1e-6)


def op_family_heatmap(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    families = sorted({row["op_name"].split(".", 1)[0] for row in rows})
    index = {name: idx for idx, name in enumerate(families)}
    matrix = np.zeros((len(rows), len(families)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, index[row["op_name"].split(".", 1)[0]]] = 1.0
    return matrix, families


def capture_tensorhue(tensor: np.ndarray, force_color: bool = False, **kwargs: Any) -> str:
    buffer = io.StringIO()
    viz_module = importlib.import_module("tensorhue.viz")
    original_console = viz_module.Console

    def make_console(*_args: Any, **_kwargs: Any) -> Console:
        return Console(
            file=buffer,
            force_terminal=force_color,
            color_system="truecolor" if force_color else None,
            width=120,
            log_path=False,
            record=False,
        )

    viz_module.Console = make_console
    try:
        with contextlib.redirect_stdout(buffer):
            tensorhue.viz(tensor, **kwargs)
    finally:
        viz_module.Console = original_console
    return buffer.getvalue()


def make_tensorhue_outputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_matrix = shape_heatmap(rows)
    family_matrix, families = op_family_heatmap(rows)

    np.save(output_dir / "tensorhue_shape_matrix.npy", shape_matrix)
    np.save(output_dir / "tensorhue_op_family_matrix.npy", family_matrix)
    (output_dir / "tensorhue_op_family_labels.json").write_text(
        json.dumps(families, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    shape_text = capture_tensorhue(shape_matrix, legend=True, scale=1)
    family_text = capture_tensorhue(family_matrix, legend=True, scale=1)
    shape_ansi = capture_tensorhue(shape_matrix, force_color=True, legend=True, scale=1)
    family_ansi = capture_tensorhue(family_matrix, force_color=True, legend=True, scale=1)
    (output_dir / "tensorhue_shape_matrix.txt").write_text(shape_text, encoding="utf-8")
    (output_dir / "tensorhue_op_family_matrix.txt").write_text(family_text, encoding="utf-8")
    (output_dir / "tensorhue_shape_matrix.ansi").write_text(shape_ansi, encoding="utf-8")
    (output_dir / "tensorhue_op_family_matrix.ansi").write_text(family_ansi, encoding="utf-8")

    op_counts = Counter(row["op_name"] for row in rows)
    summary = {
        "shape_matrix": "tensorhue_shape_matrix.npy",
        "shape_matrix_text": "tensorhue_shape_matrix.txt",
        "shape_matrix_ansi": "tensorhue_shape_matrix.ansi",
        "op_family_matrix": "tensorhue_op_family_matrix.npy",
        "op_family_matrix_text": "tensorhue_op_family_matrix.txt",
        "op_family_matrix_ansi": "tensorhue_op_family_matrix.ansi",
        "op_family_labels": "tensorhue_op_family_labels.json",
        "shape_matrix_columns": [
            "op_index",
            "max_input_rank",
            "max_output_rank",
            "log10_max_input_numel",
            "log10_max_output_numel",
            "tensor_output_count",
        ],
        "op_count": len(rows),
        "unique_op_count": len(op_counts),
        "top_ops": op_counts.most_common(20),
    }
    (output_dir / "tensorhue_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_readme(
    event_dir: Path,
    zetane: dict[str, Any],
    zetane_sequence: dict[str, Any],
    zetane_feature_map_sequence: dict[str, Any],
    zetane_inputs: dict[str, str],
    zetane_feature_map_inputs: dict[str, str],
    tensorhue_summary: dict[str, Any],
) -> None:
    lines = [
        "# input1_layer0 External-Tool Visualizations",
        "",
        "This directory contains visualizations generated from `dispatch_ops.csv` using the cloned external tools:",
        "",
        "- Zetane Viewer repository: `../../../../external/zetane-viewer`",
        "- TensorHue repository/package: `../../../../external/TensorHue`",
        "",
        "## Zetane Viewer",
        "",
        "Open this ONNX file with Zetane Viewer using `Import ONNX Model`:",
        "",
        "- `zetane/input1_layer0_dispatch_tensor_process.onnx`",
        "",
        "This is a standard-ONNX compatibility graph using no custom op domain.",
        f"It contains `{zetane['node_count']}` dispatch op nodes.",
        f"It is written as IR `{zetane['ir_version']}` / opset `{zetane['opset']}`.",
        "",
        "If Zetane still rejects that layout graph, use the simpler fallback sequence graph:",
        "",
        "- `zetane/input1_layer0_dispatch_op_sequence.onnx`",
        "",
        "The fallback graph uses one dummy input and standard `Identity` nodes only.",
        f"It contains `{zetane_sequence['node_count']}` dispatch op nodes.",
        "",
        "For Zetane feature-map input handling, prefer this 4D feature-map fallback:",
        "",
        "- `zetane/input1_layer0_dispatch_feature_map_sequence.onnx`",
        "",
        "It uses input `dispatch_feature_map_input`, shape `[1, 1, 91, 6]`, dtype `float32`.",
        "Rows are the 91 dispatch ops; columns are normalized layout features.",
        "",
        "When Zetane asks for the feature-map input, load either:",
        "",
        "- `zetane/dispatch_feature_map_input.npy`",
        "- `zetane/dispatch_feature_map_input.npz`",
        "",
        "When Zetane asks for an input for the fallback graph, load either of these files:",
        "",
        "- `zetane/dispatch_sequence_input.npy`",
        "- `zetane/dispatch_sequence_input.npz`",
        "",
        "The input is named `dispatch_sequence_input`, shape `[1]`, dtype `float32`.",
        "",
        "## TensorHue",
        "",
        "TensorHue outputs are terminal heatmap snapshots over the 91-op dispatch process:",
        "",
        "- `tensorhue/tensorhue_shape_matrix.txt`: normalized per-op tensor layout features.",
        "- `tensorhue/tensorhue_op_family_matrix.txt`: one-hot op-family sequence.",
        "- `tensorhue/tensorhue_shape_matrix.ansi`: colored TensorHue terminal snapshot; view with `less -R`.",
        "- `tensorhue/tensorhue_op_family_matrix.ansi`: colored TensorHue terminal snapshot; view with `less -R`.",
        "- `tensorhue/tensorhue_shape_matrix.npy`: numeric source for the shape heatmap.",
        "- `tensorhue/tensorhue_op_family_matrix.npy`: numeric source for the op-family heatmap.",
        "",
        "Shape matrix columns:",
        "",
    ]
    lines.extend(f"- `{col}`" for col in tensorhue_summary["shape_matrix_columns"])
    lines.extend(
        [
            "",
            "Regenerate:",
            "",
            "```bash",
            "python /workspace/VisiPrune/workload_analysis/dispatch/tools/visualize_input1_layer0_with_external_tools.py",
            "```",
            "",
        ]
    )
    (event_dir / "external_tool_visualizations.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", default=str(DEFAULT_EVENT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_dir = Path(args.event_dir)
    csv_path = event_dir / "dispatch_ops.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing input CSV: {csv_path}")

    rows = read_csv(csv_path)
    zetane_dir = event_dir / "zetane"
    tensorhue_dir = event_dir / "tensorhue"
    zetane = make_zetane_onnx(rows, zetane_dir / "input1_layer0_dispatch_tensor_process.onnx")
    zetane_sequence = make_zetane_sequence_onnx(rows, zetane_dir / "input1_layer0_dispatch_op_sequence.onnx")
    zetane_feature_map_sequence = make_zetane_feature_map_sequence_onnx(
        rows, zetane_dir / "input1_layer0_dispatch_feature_map_sequence.onnx"
    )
    zetane_inputs = make_zetane_dummy_inputs(zetane_dir)
    zetane_feature_map_inputs = make_zetane_feature_map_inputs(rows, zetane_dir)
    tensorhue_summary = make_tensorhue_outputs(rows, tensorhue_dir)
    write_readme(
        event_dir,
        zetane,
        zetane_sequence,
        zetane_feature_map_sequence,
        zetane_inputs,
        zetane_feature_map_inputs,
        tensorhue_summary,
    )
    print(
        json.dumps(
            {
                "event_dir": str(event_dir),
                "dispatch_ops": str(csv_path),
                "zetane": zetane,
                "zetane_sequence": zetane_sequence,
                "zetane_feature_map_sequence": zetane_feature_map_sequence,
                "zetane_inputs": zetane_inputs,
                "zetane_feature_map_inputs": zetane_feature_map_inputs,
                "tensorhue": tensorhue_summary,
                "readme": str(event_dir / "external_tool_visualizations.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
