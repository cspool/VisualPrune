#!/usr/bin/env python3
"""Generate toy ONNX files for input1_layer0.

This model is a small executable surrogate derived from the input1_layer0
dispatch process. Batch is fixed to 1 and intentionally removed from top-level
inputs/outputs. The script writes both a three-input ONNX and a static ONNX
with those inputs embedded into the graph.

hidden_states:   [16, 32]
position_ids:    [16]
attention_mask:  [1, 16, 16]

Only the final `output` is exposed as an ONNX graph output. Intermediate tensors
remain internal computation nodes and are not exported as viewer outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F


WORKLOAD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_DIR = WORKLOAD_DIR / "dispatch/visualize/input1_layer0"


class UnbatchedToyInput1Layer0(nn.Module):
    seq = 16
    hidden = 32
    heads = 4
    head_dim = 8
    half_dim = 4
    visual_start = 3
    visual_end = 13
    tail_start = 13
    ffn = 64

    def __init__(self, seed: int = 7) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def param(*shape: int, scale: float = 0.08) -> nn.Parameter:
            return nn.Parameter(torch.randn(*shape, generator=generator) * scale)

        self.input_norm_weight = nn.Parameter(torch.ones(self.hidden))
        self.post_norm_weight = nn.Parameter(torch.ones(self.hidden))
        self.q_weight = param(self.hidden, self.hidden)
        self.k_weight = param(self.hidden, self.hidden)
        self.v_weight = param(self.hidden, self.hidden)
        self.o_weight = param(self.hidden, self.hidden)
        self.gate_weight = param(self.ffn, self.hidden)
        self.up_weight = param(self.ffn, self.hidden)
        self.down_weight = param(self.hidden, self.ffn)

        positions = torch.arange(self.seq, dtype=torch.float32).unsqueeze(1)
        freqs = torch.arange(self.head_dim, dtype=torch.float32).unsqueeze(0)
        angles = positions / (10000.0 ** (freqs / self.head_dim))
        self.register_buffer("cos_cached", torch.cos(angles), persistent=True)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=True)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + 1e-5)
        return x * inv_rms * weight, variance, inv_rms

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.seq, self.heads, self.head_dim).transpose(0, 1)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        left = x[..., : self.half_dim]
        right = x[..., self.half_dim :]
        return torch.cat((-right, left), dim=-1)

    def shallow_full_visual_attention_adjust(self, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tail_to_visual = attn[:, self.tail_start :, self.visual_start : self.visual_end]
        tail_visual_sum = tail_to_visual.sum(dim=-1)
        adjusted = attn.clone()
        adjusted[:, self.visual_start :, self.visual_start : self.visual_end] = 0.0
        adjusted[:, self.tail_start :, self.visual_start] = tail_visual_sum
        return adjusted, tail_to_visual, tail_visual_sum

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        residual0 = hidden_states
        x_norm, input_variance, input_inv_rms = self.rms_norm(hidden_states, self.input_norm_weight)

        q_linear = F.linear(x_norm, self.q_weight)
        k_linear = F.linear(x_norm, self.k_weight)
        v_linear = F.linear(x_norm, self.v_weight)
        q = self.split_heads(q_linear)
        k = self.split_heads(k_linear)
        v = self.split_heads(v_linear)

        cos = self.cos_cached.index_select(0, position_ids.reshape(-1))
        sin = self.sin_cached.index_select(0, position_ids.reshape(-1))
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q_rope = q * cos + self.rotate_half(q) * sin
        k_rope = k * cos + self.rotate_half(k) * sin

        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)
        adjusted_attn, tail_to_visual, tail_visual_sum = self.shallow_full_visual_attention_adjust(attn)

        context_heads = torch.matmul(adjusted_attn, v)
        context = context_heads.transpose(0, 1).contiguous().view(self.seq, self.hidden)
        attn_out = F.linear(context, self.o_weight)
        after_attn = residual0 + attn_out

        mlp_in, post_variance, post_inv_rms = self.rms_norm(after_attn, self.post_norm_weight)
        gate = F.silu(F.linear(mlp_in, self.gate_weight))
        up = F.linear(mlp_in, self.up_weight)
        gated = gate * up
        ffn_out = F.linear(gated, self.down_weight)
        output = after_attn + ffn_out

        return output


OUTPUT_NAMES = ["output"]


SLIM_ADD_WRAPPER_OUTPUTS = {
    # Input RMSNorm.
    "/Mul_output_0",
    "/ReduceMean_output_0",
    "/Div_output_0",
    "/Mul_2_output_0",
    # Q/K/V projections and their transposed weights.
    "/Transpose_output_0",
    "/MatMul_output_0",
    "/Transpose_1_output_0",
    "/MatMul_1_output_0",
    "/Transpose_2_output_0",
    "/MatMul_2_output_0",
    # Head split and RoPE inputs.
    "/Transpose_3_output_0",
    "/Transpose_4_output_0",
    "/Transpose_5_output_0",
    "/Gather_output_0",
    "/Gather_1_output_0",
    "/Unsqueeze_output_0",
    "/Unsqueeze_1_output_0",
    # RoPE outputs and attention score path.
    "/Add_1_output_0",
    "/Add_2_output_0",
    "/Transpose_6_output_0",
    "/MatMul_3_output_0",
    "/Div_1_output_0",
    "/Add_3_output_0",
    "/Softmax_output_0",
    # Shallow full visual attention adjustment.
    "/Slice_5_output_0",
    "/ReduceSum_output_0",
    "/ScatterND_output_0",
    "/ScatterND_1_output_0",
    # Attention output projection and residual.
    "/MatMul_4_output_0",
    "/Transpose_7_output_0",
    "/Reshape_6_output_0",
    "/Transpose_8_output_0",
    "/MatMul_5_output_0",
    "/Add_8_output_0",
    # Post-attention RMSNorm and MLP.
    "/ReduceMean_1_output_0",
    "/Div_2_output_0",
    "/Mul_15_output_0",
    "/Transpose_9_output_0",
    "/MatMul_6_output_0",
    "/Sigmoid_output_0",
    "/Mul_16_output_0",
    "/Transpose_10_output_0",
    "/MatMul_7_output_0",
    "/Mul_17_output_0",
    "/Transpose_11_output_0",
    "/MatMul_8_output_0",
    "output",
}


def build_inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = UnbatchedToyInput1Layer0
    hidden_states = torch.randn(model.seq, model.hidden, generator=generator)
    position_ids = torch.arange(model.seq, dtype=torch.long)
    future = torch.triu(torch.ones(model.seq, model.seq), diagonal=1)
    attention_mask = future.masked_fill(future > 0, -10000.0).reshape(1, model.seq, model.seq)
    return hidden_states, position_ids, attention_mask


def save_inputs(out_dir: Path, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, str]:
    hidden_states, position_ids, attention_mask = inputs
    paths = {
        "hidden_states": out_dir / "hidden_states.npy",
        "position_ids": out_dir / "position_ids.npy",
        "attention_mask": out_dir / "attention_mask.npy",
    }
    np.save(paths["hidden_states"], hidden_states.numpy().astype(np.float32))
    np.save(paths["position_ids"], position_ids.numpy().astype(np.int64))
    np.save(paths["attention_mask"], attention_mask.numpy().astype(np.float32))
    npz_path = out_dir / "input1_layer0_inputs.npz"
    reversed_npz_path = out_dir / "input1_layer0_inputs_reversed.npz"
    np.savez(
        npz_path,
        hidden_states.numpy().astype(np.float32),
        position_ids.numpy().astype(np.int64),
        attention_mask.numpy().astype(np.float32),
    )
    np.savez(
        reversed_npz_path,
        attention_mask.numpy().astype(np.float32),
        position_ids.numpy().astype(np.int64),
        hidden_states.numpy().astype(np.float32),
    )
    return {
        "hidden_states_npy": str(paths["hidden_states"]),
        "position_ids_npy": str(paths["position_ids"]),
        "attention_mask_npy": str(paths["attention_mask"]),
        "inputs_npz": str(npz_path),
        "inputs_npz_reversed": str(reversed_npz_path),
    }


def write_static_onnx(
    dynamic_model: onnx.ModelProto,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    static_path: Path,
) -> onnx.ModelProto:
    static_model = onnx.ModelProto()
    static_model.CopyFrom(dynamic_model)

    input_arrays = {
        "hidden_states": inputs[0].numpy().astype(np.float32),
        "position_ids": inputs[1].numpy().astype(np.int64),
        "attention_mask": inputs[2].numpy().astype(np.float32),
    }
    existing_initializers = {initializer.name for initializer in static_model.graph.initializer}
    for name, value in input_arrays.items():
        if name in existing_initializers:
            raise ValueError(f"Cannot embed input {name!r}; initializer already exists")
        static_model.graph.initializer.append(numpy_helper.from_array(value, name=name))

    embedded_input_names = set(input_arrays)
    remaining_inputs = [value_info for value_info in static_model.graph.input if value_info.name not in embedded_input_names]
    del static_model.graph.input[:]
    static_model.graph.input.extend(remaining_inputs)

    onnx.checker.check_model(static_model)
    onnx.save(static_model, static_path)
    return static_model


def sanitize_onnx_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return cleaned or "unnamed"


def make_unique_name(base_name: str, used_names: set[str]) -> str:
    name = base_name
    suffix = 1
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def collect_node_output_values(
    static_model: onnx.ModelProto,
    include_existing_graph_outputs: bool = False,
) -> dict[str, np.ndarray]:
    eval_model = onnx.ModelProto()
    eval_model.CopyFrom(static_model)

    existing_outputs = {output.name for output in eval_model.graph.output}
    target_outputs: list[str] = []
    for node in eval_model.graph.node:
        for output_name in node.output:
            if not output_name:
                continue
            if output_name not in existing_outputs:
                eval_model.graph.output.append(helper.make_tensor_value_info(output_name, onnx.TensorProto.UNDEFINED, None))
                existing_outputs.add(output_name)
            elif not include_existing_graph_outputs:
                continue
            target_outputs.append(output_name)

    evaluator = ReferenceEvaluator(eval_model)
    output_names = [output.name for output in eval_model.graph.output]
    output_values = evaluator.run(None, {})
    values_by_name = dict(zip(output_names, output_values))
    return {
        name: np.asarray(values_by_name[name])
        for name in target_outputs
        if name in values_by_name
    }


def make_value_info_from_array(name: str, value: np.ndarray) -> onnx.ValueInfoProto:
    tensor = numpy_helper.from_array(np.asarray(value), name=name)
    return helper.make_tensor_value_info(name, tensor.data_type, list(value.shape))


def write_activation_constant_onnx(
    static_model: onnx.ModelProto,
    activation_values: dict[str, np.ndarray],
    activation_path: Path,
) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    activation_model = onnx.ModelProto()
    activation_model.CopyFrom(static_model)

    used_value_names = {value.name for value in activation_model.graph.input}
    used_value_names.update(value.name for value in activation_model.graph.output)
    used_value_names.update(value.name for value in activation_model.graph.value_info)
    used_value_names.update(initializer.name for initializer in activation_model.graph.initializer)
    for node in activation_model.graph.node:
        used_value_names.update(output for output in node.output if output)

    original_outputs = {output.name for output in activation_model.graph.output}
    new_nodes = []
    mapping: list[dict[str, object]] = []
    for node_index, node in enumerate(activation_model.graph.node):
        new_nodes.append(node)
        for output_index, original_output in enumerate(node.output):
            if not original_output or original_output in original_outputs or original_output not in activation_values:
                continue

            value = activation_values[original_output]
            safe_output = sanitize_onnx_name(original_output)
            safe_node = sanitize_onnx_name(node.name or f"node_{node_index}")
            const_output = make_unique_name(f"viz_const_{node_index:03d}_{safe_output}", used_value_names)
            const_node_name = f"viz_const_after_{node_index:03d}_{safe_node}_out{output_index}"
            const_tensor = numpy_helper.from_array(value, name=f"{const_output}_value")
            const_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[const_output],
                name=const_node_name,
                value=const_tensor,
            )
            const_node.doc_string = (
                f"Precomputed value for original tensor {original_output!r}; "
                f"producer node index={node_index}, name={node.name!r}, op_type={node.op_type}. "
                "This node is not connected to the main computation chain."
            )
            new_nodes.append(const_node)
            activation_model.graph.output.append(make_value_info_from_array(const_output, value))
            mapping.append(
                {
                    "constant_output": const_output,
                    "original_tensor": original_output,
                    "producer_node_index": node_index,
                    "producer_node_name": node.name,
                    "producer_op_type": node.op_type,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )

    del activation_model.graph.node[:]
    activation_model.graph.node.extend(new_nodes)
    onnx.checker.check_model(activation_model)
    onnx.save(activation_model, activation_path)
    return activation_model, mapping


def write_value_flow_constant_onnx(
    static_model: onnx.ModelProto,
    node_output_values: dict[str, np.ndarray],
    value_flow_path: Path,
) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    value_flow_model = onnx.ModelProto()
    value_flow_model.CopyFrom(static_model)

    used_value_names = {value.name for value in value_flow_model.graph.input}
    used_value_names.update(value.name for value in value_flow_model.graph.output)
    used_value_names.update(value.name for value in value_flow_model.graph.value_info)
    used_value_names.update(initializer.name for initializer in value_flow_model.graph.initializer)
    for node in value_flow_model.graph.node:
        used_value_names.update(output for output in node.output if output)

    exposed_outputs = {output.name for output in value_flow_model.graph.output}
    new_nodes = []
    mapping: list[dict[str, object]] = []
    for node_index, node in enumerate(value_flow_model.graph.node):
        original_outputs = list(node.output)
        safe_node = sanitize_onnx_name(node.name or f"node_{node_index}")
        replacements: list[tuple[int, str, str, str, np.ndarray]] = []

        for output_index, original_output in enumerate(original_outputs):
            if not original_output or original_output not in node_output_values:
                continue
            value = node_output_values[original_output]
            safe_output = sanitize_onnx_name(original_output)
            raw_output = make_unique_name(f"raw_{node_index:03d}_{safe_output}", used_value_names)
            const_output = make_unique_name(f"flow_const_{node_index:03d}_{safe_output}", used_value_names)
            node.output[output_index] = raw_output
            replacements.append((output_index, original_output, raw_output, const_output, value))

        new_nodes.append(node)

        for output_index, original_output, raw_output, const_output, value in replacements:
            const_tensor = numpy_helper.from_array(value, name=f"{const_output}_value")
            const_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[const_output],
                name=f"flow_const_after_{node_index:03d}_{safe_node}_out{output_index}",
                value=const_tensor,
            )
            const_node.doc_string = (
                f"Precomputed value replacing original tensor {original_output!r}; "
                f"original producer raw output is {raw_output!r}. "
                "Downstream nodes consume the Identity output with the original tensor name."
            )
            identity_node = helper.make_node(
                "Identity",
                inputs=[const_output],
                outputs=[original_output],
                name=f"flow_identity_restore_{node_index:03d}_{safe_node}_out{output_index}",
            )
            new_nodes.extend([const_node, identity_node])

            if original_output not in exposed_outputs:
                value_flow_model.graph.output.append(make_value_info_from_array(original_output, value))
                exposed_outputs.add(original_output)

            mapping.append(
                {
                    "flow_tensor": original_output,
                    "constant_output": const_output,
                    "raw_original_node_output": raw_output,
                    "producer_node_index": node_index,
                    "producer_node_name": node.name,
                    "producer_op_type": node.op_type,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )

    del value_flow_model.graph.node[:]
    value_flow_model.graph.node.extend(new_nodes)
    onnx.checker.check_model(value_flow_model)
    onnx.save(value_flow_model, value_flow_path)
    return value_flow_model, mapping


def write_add_value_wrapper_onnx(
    static_model: onnx.ModelProto,
    node_output_values: dict[str, np.ndarray],
    add_wrapper_path: Path,
    target_output_names: set[str] | None = None,
) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    add_model = onnx.ModelProto()
    add_model.CopyFrom(static_model)

    used_value_names = {value.name for value in add_model.graph.input}
    used_value_names.update(value.name for value in add_model.graph.output)
    used_value_names.update(value.name for value in add_model.graph.value_info)
    used_value_names.update(initializer.name for initializer in add_model.graph.initializer)
    for node in add_model.graph.node:
        used_value_names.update(output for output in node.output if output)

    exposed_outputs = {output.name for output in add_model.graph.output}
    new_nodes = []
    mapping: list[dict[str, object]] = []
    for node_index, node in enumerate(add_model.graph.node):
        original_outputs = list(node.output)
        safe_node = sanitize_onnx_name(node.name or f"node_{node_index}")
        replacements: list[tuple[int, str, str, str, np.ndarray]] = []

        for output_index, original_output in enumerate(original_outputs):
            if not original_output or original_output not in node_output_values:
                continue
            if target_output_names is not None and original_output not in target_output_names:
                continue
            value = node_output_values[original_output]
            safe_output = sanitize_onnx_name(original_output)
            raw_output = make_unique_name(f"raw_add_{node_index:03d}_{safe_output}", used_value_names)
            const_output = make_unique_name(f"add_const_{node_index:03d}_{safe_output}", used_value_names)
            node.output[output_index] = raw_output
            replacements.append((output_index, original_output, raw_output, const_output, value))

        new_nodes.append(node)

        for output_index, original_output, raw_output, const_output, value in replacements:
            const_tensor = numpy_helper.from_array(value, name=f"{const_output}_value")
            const_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[const_output],
                name=f"add_const_after_{node_index:03d}_{safe_node}_out{output_index}",
                value=const_tensor,
            )
            const_node.doc_string = (
                f"Precomputed true value for original tensor {original_output!r}; "
                f"raw producer output is {raw_output!r}."
            )
            new_nodes.append(const_node)

            if np.asarray(value).dtype == np.bool_:
                condition_output = make_unique_name(f"add_bool_cond_{node_index:03d}_{sanitize_onnx_name(original_output)}", used_value_names)
                equal_node = helper.make_node(
                    "Equal",
                    inputs=[raw_output, raw_output],
                    outputs=[condition_output],
                    name=f"add_bool_equal_{node_index:03d}_{safe_node}_out{output_index}",
                )
                where_node = helper.make_node(
                    "Where",
                    inputs=[condition_output, const_output, raw_output],
                    outputs=[original_output],
                    name=f"add_bool_restore_{node_index:03d}_{safe_node}_out{output_index}",
                )
                wrapper_kind = "Where"
                new_nodes.extend([equal_node, where_node])
            else:
                delta_output = make_unique_name(f"add_delta_{node_index:03d}_{sanitize_onnx_name(original_output)}", used_value_names)
                sub_node = helper.make_node(
                    "Sub",
                    inputs=[const_output, raw_output],
                    outputs=[delta_output],
                    name=f"add_delta_after_{node_index:03d}_{safe_node}_out{output_index}",
                )
                add_node = helper.make_node(
                    "Add",
                    inputs=[raw_output, delta_output],
                    outputs=[original_output],
                    name=f"add_wrapper_restore_{node_index:03d}_{safe_node}_out{output_index}",
                )
                wrapper_kind = "Sub+Add"
                new_nodes.extend([sub_node, add_node])

            if original_output not in exposed_outputs:
                add_model.graph.output.append(make_value_info_from_array(original_output, value))
                exposed_outputs.add(original_output)

            mapping.append(
                {
                    "flow_tensor": original_output,
                    "constant_output": const_output,
                    "raw_original_node_output": raw_output,
                    "producer_node_index": node_index,
                    "producer_node_name": node.name,
                    "producer_op_type": node.op_type,
                    "wrapper_kind": wrapper_kind,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )

    del add_model.graph.node[:]
    add_model.graph.node.extend(new_nodes)
    onnx.checker.check_model(add_model)
    onnx.save(add_model, add_wrapper_path)
    return add_model, mapping


def export_onnx(out_dir: Path, opset: int = 11, clean: bool = True) -> dict[str, object]:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    model = UnbatchedToyInput1Layer0(seed=7).eval()
    inputs = build_inputs(seed=11)
    with torch.no_grad():
        output = model(*inputs)

    onnx_path = out_dir / "input1_layer0_unbatched_toy.onnx"
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        export_params=True,
        external_data=False,
        opset_version=opset,
        do_constant_folding=False,
        dynamo=False,
        input_names=["hidden_states", "position_ids", "attention_mask"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=None,
    )
    loaded = onnx.load(onnx_path)
    onnx.checker.check_model(loaded)

    static_onnx_path = out_dir / "input1_layer0_static.onnx"
    static_loaded = write_static_onnx(loaded, inputs, static_onnx_path)
    activation_values = collect_node_output_values(static_loaded)
    activation_onnx_path = out_dir / "input1_layer0_static_with_activation_constants.onnx"
    activation_loaded, activation_mapping = write_activation_constant_onnx(
        static_loaded,
        activation_values,
        activation_onnx_path,
    )
    activation_mapping_path = out_dir / "activation_constant_mapping.json"
    activation_mapping_path.write_text(
        json.dumps(activation_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    value_flow_values = collect_node_output_values(static_loaded, include_existing_graph_outputs=True)
    value_flow_onnx_path = out_dir / "input1_layer0_static_value_flow.onnx"
    value_flow_loaded, value_flow_mapping = write_value_flow_constant_onnx(
        static_loaded,
        value_flow_values,
        value_flow_onnx_path,
    )
    value_flow_mapping_path = out_dir / "value_flow_constant_mapping.json"
    value_flow_mapping_path.write_text(
        json.dumps(value_flow_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    add_wrapper_onnx_path = out_dir / "input1_layer0_static_add_wrapped.onnx"
    add_wrapper_loaded, add_wrapper_mapping = write_add_value_wrapper_onnx(
        static_loaded,
        value_flow_values,
        add_wrapper_onnx_path,
    )
    add_wrapper_mapping_path = out_dir / "add_wrapper_mapping.json"
    add_wrapper_mapping_path.write_text(
        json.dumps(add_wrapper_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    slim_add_wrapper_onnx_path = out_dir / "input1_layer0_static_add_wrapped_slim.onnx"
    slim_add_wrapper_loaded, slim_add_wrapper_mapping = write_add_value_wrapper_onnx(
        static_loaded,
        value_flow_values,
        slim_add_wrapper_onnx_path,
        target_output_names=SLIM_ADD_WRAPPER_OUTPUTS,
    )
    slim_add_wrapper_mapping_path = out_dir / "add_wrapper_slim_mapping.json"
    slim_add_wrapper_mapping_path.write_text(
        json.dumps(slim_add_wrapper_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    input_paths = save_inputs(out_dir, inputs)
    metadata = {
        "onnx_path": str(onnx_path),
        "static_onnx_path": str(static_onnx_path),
        "activation_constant_onnx_path": str(activation_onnx_path),
        "activation_constant_mapping_path": str(activation_mapping_path),
        "value_flow_onnx_path": str(value_flow_onnx_path),
        "value_flow_mapping_path": str(value_flow_mapping_path),
        "add_wrapper_onnx_path": str(add_wrapper_onnx_path),
        "add_wrapper_mapping_path": str(add_wrapper_mapping_path),
        "slim_add_wrapper_onnx_path": str(slim_add_wrapper_onnx_path),
        "slim_add_wrapper_mapping_path": str(slim_add_wrapper_mapping_path),
        "opset": opset,
        "batch_dimension": "removed; fixed to 1 by construction",
        "external_input_policy": "three ONNX inputs: hidden_states, position_ids, attention_mask",
        "static_external_input_policy": "no ONNX graph inputs; hidden_states, position_ids, and attention_mask are embedded initializers",
        "activation_constant_policy": (
            "static model plus one Constant node after each original internal node output; "
            "these constants are graph outputs and do not feed the main computation chain"
        ),
        "value_flow_policy": (
            "static model where each original node output is renamed to raw_* and immediately "
            "replaced by Constant->Identity using the precomputed value; downstream nodes consume "
            "the restored original tensor name"
        ),
        "add_wrapper_policy": (
            "static model where each original node output is renamed to raw_add_* and immediately "
            "wrapped by Constant(value), Sub(value, raw), Add(raw, delta) so downstream nodes consume "
            "the true precomputed value while retaining raw-node dependency"
        ),
        "slim_add_wrapper_policy": (
            "smaller add-wrapper model for viewers that crash on the full wrapper graph; "
            "only key semantic compute tensors and their main input tensors are wrapped"
        ),
        "toy_dimensions": {
            "seq": UnbatchedToyInput1Layer0.seq,
            "hidden": UnbatchedToyInput1Layer0.hidden,
            "heads": UnbatchedToyInput1Layer0.heads,
            "head_dim": UnbatchedToyInput1Layer0.head_dim,
            "visual_start": UnbatchedToyInput1Layer0.visual_start,
            "visual_end": UnbatchedToyInput1Layer0.visual_end,
            "tail_start": UnbatchedToyInput1Layer0.tail_start,
            "ffn": UnbatchedToyInput1Layer0.ffn,
        },
        "input_shapes": {
            "hidden_states": list(inputs[0].shape),
            "position_ids": list(inputs[1].shape),
            "attention_mask": list(inputs[2].shape),
        },
        "output_shapes": {
            "output": list(output.shape),
        },
        "static_graph_inputs": [graph_input.name for graph_input in static_loaded.graph.input],
        "activation_constant_graph_inputs": [graph_input.name for graph_input in activation_loaded.graph.input],
        "activation_constant_output_count": len(activation_mapping),
        "value_flow_graph_inputs": [graph_input.name for graph_input in value_flow_loaded.graph.input],
        "value_flow_output_count": len(value_flow_loaded.graph.output),
        "value_flow_constant_count": len(value_flow_mapping),
        "add_wrapper_graph_inputs": [graph_input.name for graph_input in add_wrapper_loaded.graph.input],
        "add_wrapper_output_count": len(add_wrapper_loaded.graph.output),
        "add_wrapper_count": len(add_wrapper_mapping),
        "slim_add_wrapper_graph_inputs": [graph_input.name for graph_input in slim_add_wrapper_loaded.graph.input],
        "slim_add_wrapper_output_count": len(slim_add_wrapper_loaded.graph.output),
        "slim_add_wrapper_count": len(slim_add_wrapper_mapping),
        "input_files": input_paths,
        "source_event": "input1_layer0",
        "source_compute_process": str(DEFAULT_EVENT_DIR / "tensor_compute_process.md"),
        "note": (
            "Three-input toy ONNX. Top-level batch dimension is removed; inputs "
            "and output represent batch=1 tensors directly. Only final output is exposed."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(out_dir, metadata)
    return metadata


def write_readme(out_dir: Path, metadata: dict[str, object]) -> None:
    lines = [
        "# input1_layer0 Unbatched Toy ONNX",
        "",
        "This is the final Zetane-facing ONNX for `input1_layer0`.",
        "It includes a static no-input model and a three-input model.",
        "The fixed batch dimension has been removed from top-level examples.",
        "",
        "## Files",
        "",
        "- `input1_layer0_unbatched_toy.onnx`: executable toy ONNX.",
        "- `input1_layer0_static.onnx`: static ONNX with all three inputs embedded.",
        "- `input1_layer0_static_with_activation_constants.onnx`: static ONNX with precomputed node-output constants inserted after producer nodes.",
        "- `activation_constant_mapping.json`: maps each injected constant output to its original producer node and tensor.",
        "- `input1_layer0_static_value_flow.onnx`: static ONNX where precomputed constants replace each node output in the downstream data flow.",
        "- `value_flow_constant_mapping.json`: maps each value-flow constant to the original node output it replaces.",
        "- `input1_layer0_static_add_wrapped.onnx`: static ONNX where each node output is wrapped by a true-value constant and an Add-based restore node.",
        "- `add_wrapper_mapping.json`: maps each Add wrapper to its original producer node and tensor.",
        "- `input1_layer0_static_add_wrapped_slim.onnx`: smaller Add-wrapper ONNX with wrappers only on key compute tensors.",
        "- `add_wrapper_slim_mapping.json`: maps each slim Add wrapper to its original producer node and tensor.",
        "- `input1_layer0_inputs.npz`: all inputs in ONNX input order.",
        "- `input1_layer0_inputs_reversed.npz`: same inputs in reverse order for Zetane input-order testing.",
        "- `hidden_states.npy`, `position_ids.npy`, `attention_mask.npy`: individual input files.",
        "- `metadata.json`: shape and generation metadata.",
        "",
        "## ONNX Inputs",
        "",
    ]
    for name, shape in metadata["input_shapes"].items():
        lines.append(f"- `{name}`: `{shape}`")
    lines.extend(
        [
            "",
            "## Static ONNX",
            "",
            "For viewers that cannot load external inputs, import this no-input model:",
            "",
            "```text",
            str(Path(metadata["static_onnx_path"])),
            "```",
            "",
            "`hidden_states`, `position_ids`, and `attention_mask` are embedded as initializers in this static model.",
            "",
            "## Static ONNX With Activation Constants",
            "",
            "For no-Pro inspection of precomputed intermediate tensors, import this model:",
            "",
            "```text",
            str(Path(metadata["activation_constant_onnx_path"])),
            "```",
            "",
            "It keeps the main static computation graph and inserts one `Constant` node after each original internal node output.",
            "Those constants are graph outputs and do not feed the main computation chain.",
            f"Injected activation constants: `{metadata['activation_constant_output_count']}`.",
            "Use `activation_constant_mapping.json` to map a `viz_const_*` output back to the original node output.",
            "",
            "## Static ONNX With Constant Value Flow",
            "",
            "For viewers that place tensors only along the main computation flow, import this model:",
            "",
            "```text",
            str(Path(metadata["value_flow_onnx_path"])),
            "```",
            "",
            "Each original node output is renamed to `raw_*`, then a precomputed `Constant -> Identity` restores the original tensor name.",
            "Downstream nodes consume the restored constant value, so the constants participate in the ONNX data flow.",
            f"Value-flow constants: `{metadata['value_flow_constant_count']}`.",
            "Use `value_flow_constant_mapping.json` to map each replacement back to its original producer node.",
            "",
            "## Static ONNX With Add Wrappers",
            "",
            "For viewers that need constants and original compute nodes in the same main data flow, import this model:",
            "",
            "```text",
            str(Path(metadata["add_wrapper_onnx_path"])),
            "```",
            "",
            "Each original node output is renamed to `raw_add_*`, then restored through a true-value constant wrapper.",
            "Numeric tensors use `Constant(value) -> Sub(value, raw) -> Add(raw, delta)`, so downstream tensors equal the precomputed true values while still depending on the original compute node.",
            "Boolean tensors use `Where` because ONNX `Add` does not accept bool tensors.",
            f"Add wrappers: `{metadata['add_wrapper_count']}`.",
            "Use `add_wrapper_mapping.json` to map each wrapper back to its original producer node.",
            "",
            "## Slim Add-Wrapper ONNX",
            "",
            "If the full Add-wrapper ONNX is too large for a viewer, import this reduced model:",
            "",
            "```text",
            str(Path(metadata["slim_add_wrapper_onnx_path"])),
            "```",
            "",
            "It wraps only key semantic tensors: RMSNorm, Q/K/V, RoPE, attention score/mask/softmax, attention adjustment, context, residual, and MLP tensors.",
            f"Slim Add wrappers: `{metadata['slim_add_wrapper_count']}`.",
            "Use `add_wrapper_slim_mapping.json` to map each wrapper back to its original producer node.",
            "",
            "## Viewer Use",
            "",
            "Import this model in Zetane:",
            "",
            "```text",
            str(Path(metadata["onnx_path"])),
            "```",
            "",
            "When it asks for inputs, load `input1_layer0_inputs.npz` if multi-input NPZ is accepted.",
            "The NPZ uses Zetane-style ordered keys: `arr_0=hidden_states`, `arr_1=position_ids`, `arr_2=attention_mask`.",
            "To test reverse NPZ mapping, load `input1_layer0_inputs_reversed.npz`: `arr_0=attention_mask`, `arr_1=position_ids`, `arr_2=hidden_states`.",
            "Otherwise load the three `.npy` files by input name:",
            "",
            "```text",
            str(Path(metadata["input_files"]["hidden_states_npy"])),
            str(Path(metadata["input_files"]["position_ids_npy"])),
            str(Path(metadata["input_files"]["attention_mask_npy"])),
            "```",
            "",
            "The ONNX graph exposes only the final `output`. Intermediate tensors remain internal nodes.",
            "",
        ]
    )
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", default=str(DEFAULT_EVENT_DIR))
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--no-clean", action="store_true", help="Do not delete the existing output directory first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_dir = Path(args.event_dir)
    out_dir = event_dir / "zetane"
    metadata = export_onnx(out_dir, opset=args.opset, clean=not args.no_clean)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
