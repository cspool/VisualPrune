#!/usr/bin/env python3
"""Export one small ONNX per dispatch-derived computation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import onnx
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from attention import scaled_dot_product_attention
from attention_output import attention_output
from config import CFG, EXPECTED_STAGES, FlowConfig
from init_data import build_inputs, build_rope_cache, build_weights
from kv_cache import concat_kv_cache
from mlp import gated_mlp
from qkv_projection import project_qkv
from rmsnorm import rms_norm
from rope import apply_rope
from run_full_flow import run_flow
from visipruner_similarity import visipruner_similarity_check
from visual_adjust import shallow_full_visual_attention_adjust


class InputRMSNormStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_norm_weight", weights["input_norm_weight"])

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = rms_norm(hidden_states, self.input_norm_weight, self.cfg, "input_norm")
        return out["input_norm_output"], out["input_norm_variance"], out["input_norm_inv_rms"]


class QKVProjectionStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("q_weight", weights["q_weight"])
        self.register_buffer("k_weight", weights["k_weight"])
        self.register_buffer("v_weight", weights["v_weight"])

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = {
            "q_weight": self.q_weight,
            "k_weight": self.k_weight,
            "v_weight": self.v_weight,
        }
        out = project_qkv(x_norm, weights, self.cfg)
        return out["q_heads"], out["k_heads"], out["v_heads"]


class RoPEStage(nn.Module):
    def __init__(self, cfg: FlowConfig, rope_cache: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("cos_cached", rope_cache["cos_cached"])
        self.register_buffer("sin_cached", rope_cache["sin_cached"])

    def forward(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_cache = {
            "cos_cached": self.cos_cached,
            "sin_cached": self.sin_cached,
        }
        out = apply_rope(q_heads, k_heads, position_ids, rope_cache, self.cfg)
        return out["q_rope"], out["k_rope"]


class KVCacheConcatStage(nn.Module):
    def forward(
        self,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = concat_kv_cache(k_current, v_current, past_k, past_v)
        return out["k_heads"], out["v_heads"]


class AttentionStage(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = scaled_dot_product_attention(q_rope, k_rope, attention_mask, self.cfg)
        return out["raw_scores"], out["masked_scores"], out["attn"]


class VisualAdjustStage(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = shallow_full_visual_attention_adjust(attn, self.cfg)
        return out["adjusted_attn"], out["tail_visual_sum"], out["cleared_visual_region"]


class VisiPrunerSimilarityCheckStage(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = visipruner_similarity_check(hidden_states)
        return out["similarity"], out["any_close"].to(torch.float32)


class AttentionOutputStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("o_weight", weights["o_weight"])

    def forward(
        self,
        adjusted_attn: torch.Tensor,
        v_heads: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = attention_output(
            adjusted_attn,
            v_heads,
            hidden_states,
            {"o_weight": self.o_weight},
            self.cfg,
        )
        return out["context"], out["attn_out"], out["after_attn"]


class MLPStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("post_norm_weight", weights["post_norm_weight"])
        self.register_buffer("gate_weight", weights["gate_weight"])
        self.register_buffer("up_weight", weights["up_weight"])
        self.register_buffer("down_weight", weights["down_weight"])

    def forward(self, after_attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = {
            "post_norm_weight": self.post_norm_weight,
            "gate_weight": self.gate_weight,
            "up_weight": self.up_weight,
            "down_weight": self.down_weight,
        }
        out = gated_mlp(after_attn, weights, self.cfg)
        return out["gated"], out["ffn_out"], out["output"]


class _FullFlowBase(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor], rope_cache: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        for name, tensor in weights.items():
            self.register_buffer(name, tensor)
        self.register_buffer("cos_cached", rope_cache["cos_cached"])
        self.register_buffer("sin_cached", rope_cache["sin_cached"])

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = {
            "input_norm_weight": self.input_norm_weight,
            "post_norm_weight": self.post_norm_weight,
            "q_weight": self.q_weight,
            "k_weight": self.k_weight,
            "v_weight": self.v_weight,
            "o_weight": self.o_weight,
            "gate_weight": self.gate_weight,
            "up_weight": self.up_weight,
            "down_weight": self.down_weight,
        }
        rope_cache = {"cos_cached": self.cos_cached, "sin_cached": self.sin_cached}
        input_norm = rms_norm(hidden_states, weights["input_norm_weight"], self.cfg, "input_norm")
        qkv = project_qkv(input_norm["input_norm_output"], weights, self.cfg)
        rope = apply_rope(qkv["q_heads"], qkv["k_heads"], position_ids, rope_cache, self.cfg)
        if "kv_cache_concat" in EXPECTED_STAGES:
            if past_k is None or past_v is None:
                raise ValueError("decode full-flow export requires past_k and past_v")
            kv = concat_kv_cache(rope["k_rope"], qkv["v_heads"], past_k, past_v)
        else:
            kv = {"k_heads": rope["k_rope"], "v_heads": qkv["v_heads"]}
        attn = scaled_dot_product_attention(rope["q_rope"], kv["k_heads"], attention_mask, self.cfg)
        if "visual_adjust" in EXPECTED_STAGES:
            adjusted = shallow_full_visual_attention_adjust(attn["attn"], self.cfg)
            attn_for_output = adjusted["adjusted_attn"]
        else:
            attn_for_output = attn["attn"]
        if "visipruner_similarity_check" in EXPECTED_STAGES:
            _ = visipruner_similarity_check(hidden_states)
        attn_out = attention_output(attn_for_output, kv["v_heads"], hidden_states, weights, self.cfg)
        mlp = gated_mlp(attn_out["after_attn"], weights, self.cfg)
        return mlp["output"]


class FullFlowStage(_FullFlowBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(hidden_states, position_ids, attention_mask)


class DecodeFullFlowStage(_FullFlowBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(hidden_states, position_ids, attention_mask, past_k, past_v)


def shape_of(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def export_one(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        outputs = model(*args)
    if isinstance(outputs, torch.Tensor):
        outputs_tuple = (outputs,)
    else:
        outputs_tuple = tuple(outputs)

    torch.onnx.export(
        model,
        args,
        path,
        export_params=True,
        external_data=False,
        opset_version=opset,
        do_constant_folding=False,
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )
    loaded = onnx.load(path)
    onnx.checker.check_model(loaded)
    return {
        "path": str(path),
        "inputs": {name: shape_of(tensor) for name, tensor in zip(input_names, args)},
        "outputs": {name: shape_of(tensor) for name, tensor in zip(output_names, outputs_tuple)},
        "nodes": len(loaded.graph.node),
        "initializers": len(loaded.graph.initializer),
    }


def _stage_specs(out_dir: Path, cfg: FlowConfig, opset: int) -> list[tuple[str, nn.Module, tuple[torch.Tensor, ...], list[str], list[str]]]:
    _ = out_dir, opset
    inputs = build_inputs(cfg)
    weights = build_weights(cfg)
    rope_cache = build_rope_cache(cfg)
    with torch.no_grad():
        tensors = run_flow(cfg, verbose=False)

    specs: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...], list[str], list[str]]] = {
        "input_rmsnorm": (
            InputRMSNormStage(cfg, weights),
            (inputs["hidden_states"],),
            ["hidden_states"],
            ["x_norm", "variance", "inv_rms"],
        ),
        "qkv_projection": (
            QKVProjectionStage(cfg, weights),
            (tensors["input_norm_output"],),
            ["x_norm"],
            ["q_heads", "k_heads", "v_heads"],
        ),
        "rope": (
            RoPEStage(cfg, rope_cache),
            (tensors["q_heads"], tensors["k_heads_current"], inputs["position_ids"]),
            ["q_heads", "k_heads", "position_ids"],
            ["q_rope", "k_current_rope"],
        ),
        "kv_cache_concat": (
            KVCacheConcatStage(),
            (tensors["k_current_rope"], tensors["v_current"], inputs["past_k"], inputs["past_v"]),
            ["k_current_rope", "v_current", "past_k", "past_v"],
            ["k_heads", "v_heads"],
        ),
        "attention": (
            AttentionStage(cfg),
            (tensors["q_rope"], tensors["k_heads"], inputs["attention_mask"]),
            ["q_rope", "k_heads", "attention_mask"],
            ["raw_scores", "masked_scores", "attn"],
        ),
        "visual_adjust": (
            VisualAdjustStage(cfg),
            (tensors["attn"],),
            ["attn"],
            ["adjusted_attn", "tail_visual_sum", "cleared_visual_region"],
        ),
        "visipruner_similarity_check": (
            VisiPrunerSimilarityCheckStage(),
            (inputs["hidden_states"],),
            ["hidden_states"],
            ["similarity", "any_close"],
        ),
        "attention_output": (
            AttentionOutputStage(cfg, weights),
            (tensors["adjusted_attn"], tensors["v_heads"], inputs["hidden_states"]),
            ["adjusted_attn", "v_heads", "hidden_states"],
            ["context", "attn_out", "after_attn"],
        ),
        "mlp": (
            MLPStage(cfg, weights),
            (tensors["after_attn"],),
            ["after_attn"],
            ["gated", "ffn_out", "output"],
        ),
    }
    if "kv_cache_concat" in EXPECTED_STAGES:
        specs["full_flow"] = (
            DecodeFullFlowStage(cfg, weights, rope_cache),
            (inputs["hidden_states"], inputs["position_ids"], inputs["attention_mask"], inputs["past_k"], inputs["past_v"]),
            ["hidden_states", "position_ids", "attention_mask", "past_k", "past_v"],
            ["output"],
        )
    else:
        specs["full_flow"] = (
            FullFlowStage(cfg, weights, rope_cache),
            (inputs["hidden_states"], inputs["position_ids"], inputs["attention_mask"]),
            ["hidden_states", "position_ids", "attention_mask"],
            ["output"],
        )

    stages = [*EXPECTED_STAGES, "full_flow"]
    result = []
    for index, stage in enumerate(stages, start=1):
        model, args, input_names, output_names = specs[stage]
        result.append((f"{index:02d}_{stage}.onnx", model, args, input_names, output_names))
    return result


def export_stages(out_dir: Path, cfg: FlowConfig, opset: int) -> list[dict[str, object]]:
    manifest = []
    for filename, model, stage_args, input_names, output_names in _stage_specs(out_dir, cfg, opset):
        manifest.append(
            export_one(
                model=model,
                args=stage_args,
                path=out_dir / filename,
                input_names=input_names,
                output_names=output_names,
                opset=opset,
            )
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "onnx",
        help="Directory for per-stage ONNX files.",
    )
    parser.add_argument("--opset", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_stages(args.out_dir, CFG, args.opset)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "manifest": str(manifest_path), "stages": manifest}, indent=2))


if __name__ == "__main__":
    main()
