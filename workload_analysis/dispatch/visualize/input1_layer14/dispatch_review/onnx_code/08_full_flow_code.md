# input1_layer14 `08_full_flow.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/onnx/08_full_flow.onnx`
- Stage: `full_flow`
- Stage title: Full Layer Flow
- ONNX nodes: `117`
- ONNX initializers: `10`

### ONNX Inputs

- `hidden_states`: `[16, 32]`
- `position_ids`: `[16]`
- `attention_mask`: `[1, 16, 16]`

### ONNX Outputs

- `output`: `[16, 32]`

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/torch_flow/export_stage_onnx.py::FullFlowStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/torch_flow/run_full_flow.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/torch_flow/export_stage_onnx.py`

## Code Explanation

Aggregates the split small-tensor stages into one end-to-end layer flow for whole-layer visualization.

## Review Comments

- This page binds `08_full_flow.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- `full_flow` is a convenience aggregate that calls the split stage implementations; consult the individual stage pages for per-stage dispatch evidence.

## Dispatch Evidence Notes

- No direct dispatch evidence rows were assigned to this stage.

## Export Wrapper Source

```python
class FullFlowStage(_FullFlowBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(hidden_states, position_ids, attention_mask)
```

## Primary Source: `run_full_flow.py`

```python
#!/usr/bin/env python3
"""Run the small executable tensor flow for one dispatch layer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from attention import scaled_dot_product_attention
from attention_output import attention_output
from config import CFG, EXPECTED_STAGES, FlowConfig
from init_data import build_inputs, build_rope_cache, build_weights
from kv_cache import concat_kv_cache
from mlp import gated_mlp
from qkv_projection import project_qkv
from report import log_section, log_tensor, log_tensors
from rmsnorm import rms_norm
from rope import apply_rope
from visipruner_similarity import visipruner_similarity_check
from visual_adjust import shallow_full_visual_attention_adjust


def run_flow(cfg: FlowConfig = CFG, verbose: bool = True) -> dict[str, torch.Tensor]:
    inputs = build_inputs(cfg)
    weights = build_weights(cfg)
    rope_cache = build_rope_cache(cfg)
    tensors: dict[str, torch.Tensor] = {}
    tensors.update(inputs)

    if verbose:
        log_section("0. inputs")
        log_tensor("hidden_states", inputs["hidden_states"])
        log_tensor("position_ids", inputs["position_ids"])
        log_tensor("attention_mask", inputs["attention_mask"], "causal mask: 0 or -10000")
        if "kv_cache_concat" in EXPECTED_STAGES:
            log_tensor("past_k", inputs["past_k"])
            log_tensor("past_v", inputs["past_v"])

    input_norm = rms_norm(inputs["hidden_states"], weights["input_norm_weight"], cfg, "input_norm")
    tensors.update(input_norm)
    if verbose:
        log_tensors(
            "1. input RMSNorm",
            {
                "variance": input_norm["input_norm_variance"],
                "inv_rms": input_norm["input_norm_inv_rms"],
                "x_norm": input_norm["input_norm_output"],
            },
        )

    qkv = project_qkv(input_norm["input_norm_output"], weights, cfg)
    tensors.update(qkv)
    tensors["k_heads_current"] = qkv["k_heads"]
    tensors["v_heads_current"] = qkv["v_heads"]
    if verbose:
        log_tensors(
            "2. Q/K/V projection + head split",
            {
                "q_linear": qkv["q_linear"],
                "k_linear": qkv["k_linear"],
                "v_linear": qkv["v_linear"],
                "q_heads": qkv["q_heads"],
                "k_heads_current": tensors["k_heads_current"],
                "v_heads_current": tensors["v_heads_current"],
            },
        )

    rope = apply_rope(qkv["q_heads"], qkv["k_heads"], inputs["position_ids"], rope_cache, cfg)
    tensors.update(
        {
            **rope,
            "k_current_rope": rope["k_rope"],
            "v_current": qkv["v_heads"],
        }
    )
    if "kv_cache_concat" in EXPECTED_STAGES:
        kv = concat_kv_cache(rope["k_rope"], qkv["v_heads"], inputs["past_k"], inputs["past_v"])
    else:
        kv = {"k_heads": rope["k_rope"], "v_heads": qkv["v_heads"]}
    tensors.update(kv)
    if verbose:
        log_tensors(
            "3. RoPE and optional K/V cache concat",
            {
                "q_rope": rope["q_rope"],
                "k_current_rope": rope["k_rope"],
                "k_heads": kv["k_heads"],
                "v_heads": kv["v_heads"],
            },
        )

    attention = scaled_dot_product_attention(
        rope["q_rope"],
        kv["k_heads"],
        inputs["attention_mask"],
        cfg,
    )
    tensors.update(attention)
    if verbose:
        log_tensors(
            "4. scaled dot-product attention",
            {
                "raw_scores": attention["raw_scores"],
                "masked_scores": attention["masked_scores"],
                "attn": attention["attn"],
            },
        )

    if "visual_adjust" in EXPECTED_STAGES:
        adjusted = shallow_full_visual_attention_adjust(attention["attn"], cfg)
        tensors.update(adjusted)
        attn_for_output = adjusted["adjusted_attn"]
        if verbose:
            log_tensors(
                "5. visual attention adjustment",
                {
                    "tail_visual_sum": adjusted["tail_visual_sum"],
                    "cleared_visual_region": adjusted["cleared_visual_region"],
                    "adjusted_attn": adjusted["adjusted_attn"],
                },
            )
    else:
        tensors["adjusted_attn"] = attention["attn"]
        attn_for_output = attention["attn"]

    if "visipruner_similarity_check" in EXPECTED_STAGES:
        similarity = visipruner_similarity_check(inputs["hidden_states"])
        tensors.update(similarity)
        if verbose:
            log_tensors(
                "5. VisiPrune similarity/check",
                {
                    "similarity": similarity["similarity"],
                    "above_threshold": similarity["above_threshold"],
                    "any_close": similarity["any_close"],
                },
            )

    attn_out = attention_output(
        attn_for_output,
        kv["v_heads"],
        inputs["hidden_states"],
        weights,
        cfg,
    )
    tensors.update(attn_out)
    if verbose:
        log_tensors(
            "6. attention value matmul + output projection + residual",
            {
                "context_heads": attn_out["context_heads"],
                "context": attn_out["context"],
                "attn_out": attn_out["attn_out"],
                "after_attn": attn_out["after_attn"],
            },
        )

    mlp = gated_mlp(attn_out["after_attn"], weights, cfg)
    tensors.update(mlp)
    if verbose:
        log_tensors(
            "7. post-attention RMSNorm + gated MLP + final residual",
            {
                "post_norm_variance": mlp["post_norm_variance"],
                "post_norm_output": mlp["post_norm_output"],
                "gate_linear": mlp["gate_linear"],
                "gate_silu": mlp["gate_silu"],
                "up": mlp["up"],
                "gated": mlp["gated"],
                "ffn_out": mlp["ffn_out"],
                "output": mlp["output"],
            },
        )

    return tensors


def save_npz(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }
    np.savez(path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Run without printing tensor summaries.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional path to save all tensors as an NPZ.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with torch.no_grad():
        tensors = run_flow(verbose=not args.quiet)
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        save_npz(args.save_npz, tensors)
        print(f"saved {len(tensors)} tensors to {args.save_npz}")


if __name__ == "__main__":
    main()
```
