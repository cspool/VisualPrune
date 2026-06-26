#!/usr/bin/env python3
"""Torch reconstruction scaffold for input2_layer19, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input2_layer19'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 59,
  "q_len": 1
}""")
DISPATCH_FEATURES = json.loads(r"""{
  "expected_stages": [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "kv_cache_concat",
    "attention",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": true,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 59,
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 4,
    "div.Tensor": 1,
    "dropout.default": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 2,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 2,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "phase": "decode",
  "prune_probe_kind": null,
  "q_len": 1,
  "role": "decode_prune_effect",
  "token_state": "middle_pruned_cache",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 15,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 17,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 39,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 46,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 49,
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 59], dtype=float16"
      },
      {
"event_op_index": 50,
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 59], dtype=float16"
      },
      {
"event_op_index": 51,
"op_name": "div.Tensor",
"output": "shape=[1, 32, 1, 59], dtype=float16"
      },
      {
"event_op_index": 52,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 59], dtype=float16"
      },
      {
"event_op_index": 53,
"op_name": "softmax.int",
"output": "shape=[1, 32, 1, 59], dtype=float32"
      },
      {
"event_op_index": 55,
"op_name": "dropout.default",
"output": "shape=[1, 32, 1, 59], dtype=float16"
      },
      {
"event_op_index": 56,
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      }
    ],
    "stage": "attention",
    "summary": "Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores."
  },
  "attention_output": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 56,
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 58,
"op_name": "reshape.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 61,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 62,
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 75,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 76,
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16"
      }
    ],
    "stage": "attention_output",
    "summary": "Attention-output evidence is attn @ V, transpose/reshape, output projection, and residual add."
  },
  "input_rmsnorm": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 1,
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 2,
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 3,
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 4,
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 5,
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 6,
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 7,
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 8,
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16"
      }
    ],
    "stage": "input_rmsnorm",
    "summary": "RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence."
  },
  "kv_cache_concat": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 47,
"op_name": "cat.default",
"output": "shape=[1, 32, 59, 128], dtype=float16"
      },
      {
"event_op_index": 48,
"op_name": "cat.default",
"output": "shape=[1, 32, 59, 128], dtype=float16"
      }
    ],
    "stage": "kv_cache_concat",
    "summary": "Decode cache evidence is K/V cat outputs whose sequence dimension equals dispatch kv_len."
  },
  "mlp": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 54,
"op_name": "to.dtype",
"output": "shape=[1, 32, 1, 59], dtype=float16"
      },
      {
"event_op_index": 61,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 62,
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 63,
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 64,
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 65,
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 66,
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 67,
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32"
      },
      {
"event_op_index": 68,
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32"
      },
      {
"event_op_index": 69,
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 70,
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 71,
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16"
      },
      {
"event_op_index": 72,
"op_name": "silu.default",
"output": "shape=[1, 1, 11008], dtype=float16"
      },
      {
"event_op_index": 73,
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16"
      }
    ],
    "stage": "mlp",
    "summary": "MLP evidence is post-attention RMSNorm, gate/up linear, SiLU, gated product, down linear, residual add."
  },
  "qkv_projection": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 9,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 10,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 11,
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16"
      },
      {
"event_op_index": 12,
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16"
      },
      {
"event_op_index": 13,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 14,
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16"
      },
      {
"event_op_index": 15,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 16,
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16"
      },
      {
"event_op_index": 17,
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      }
    ],
    "stage": "qkv_projection",
    "summary": "Q/K/V evidence is three hidden-size linear projections followed by view/transpose head split."
  },
  "rope": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 20,
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64"
      },
      {
"event_op_index": 24,
"op_name": "slice.Tensor",
"output": "shape=[625, 128], dtype=float16"
      },
      {
"event_op_index": 27,
"op_name": "slice.Tensor",
"output": "shape=[625, 128], dtype=float16"
      },
      {
"event_op_index": 29,
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16"
      },
      {
"event_op_index": 30,
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16"
      },
      {
"event_op_index": 31,
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16"
      },
      {
"event_op_index": 32,
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16"
      },
      {
"event_op_index": 33,
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 34,
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16"
      },
      {
"event_op_index": 35,
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16"
      },
      {
"event_op_index": 36,
"op_name": "neg.default",
"output": "shape=[1, 32, 1, 64], dtype=float16"
      },
      {
"event_op_index": 37,
"op_name": "cat.default",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 38,
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      },
      {
"event_op_index": 39,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16"
      }
    ],
    "stage": "rope",
    "summary": "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
  }
}""")
SUMMARY = json.loads(r"""{
  "dispatch_features": {
    "expected_stages": [
      "input_rmsnorm",
      "qkv_projection",
      "rope",
      "kv_cache_concat",
      "attention",
      "attention_output",
      "mlp"
    ],
    "has_attention": true,
    "has_cache_concat": true,
    "has_mlp": true,
    "has_rope": true,
    "kv_len": 59,
    "op_counts": {
      "add.Tensor": 8,
      "cat.default": 4,
      "div.Tensor": 1,
      "dropout.default": 1,
      "gt.Scalar": 2,
      "index.Tensor": 2,
      "is_nonzero.default": 2,
      "item.default": 2,
      "linear.default": 7,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 9,
      "neg.default": 2,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 2,
      "silu.default": 1,
      "slice.Tensor": 6,
      "softmax.int": 1,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 2,
      "view.default": 3
    },
    "phase": "decode",
    "prune_probe_kind": null,
    "q_len": 1,
    "role": "decode_prune_effect",
    "token_state": "middle_pruned_cache",
    "visual_adjust_kind": null
  },
  "event_id": "input2_layer19",
  "input_id": 2,
  "kv_len": 59,
  "layer_id": 19,
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 4,
    "div.Tensor": 1,
    "dropout.default": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 2,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 2,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "original_dimensions": {
    "ffn": 11008,
    "head_dim": 128,
    "heads": 32,
    "hidden": 4096,
    "kv_len": 59,
    "seq": 1,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 58,
  "phase": "decode",
  "priority": "P3",
  "q_len": 1,
  "row_count": 76,
  "small_config": {
    "ffn": 64,
    "head_dim": 8,
    "heads": 4,
    "hidden": 32,
    "kv_seq": 16,
    "q_seq": 1,
    "seq": 16,
    "tail_start": 13,
    "visual_end": 13,
    "visual_start": 3
  },
  "token_state": "middle_pruned_cache",
  "visipruner_role": "decode_prune_effect"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [1, 4096] -> normalized [1, 4096]
# - qkv_projection: Q/K/V projection: [1, 4096] -> [32, 1, 128]
# - rope: see dispatch evidence for exact tensor roles
# - kv_cache_concat: decode cache concat: current K/V plus past cache -> [32, 59, 128]
# - attention: attention scores: [32, 1, 59]
# - attention_output: attention output: [32, 1, 128] -> [1, 4096]
# - mlp: MLP: [1, 4096] -> [1, 4096]


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> dict[str, torch.Tensor]:
    x_float = hidden_states.to(torch.float32)
    squared = x_float.pow(2)
    variance = squared.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    output = (x_float * inv_rms).to(hidden_states.dtype) * weight
    return {
        "squared": squared,
        "variance": variance,
        "inv_rms": inv_rms,
        "output": output,
    }


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return x.view(x.shape[0], heads, head_dim).transpose(0, 1)


def qkv_projection(x_norm: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    heads = int(ORIGINAL_DIMS["heads"])
    head_dim = int(ORIGINAL_DIMS["head_dim"])
    q_linear = F.linear(x_norm, weights["q_weight"])
    k_linear = F.linear(x_norm, weights["k_weight"])
    v_linear = F.linear(x_norm, weights["v_weight"])
    return {
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_heads": split_heads(q_linear, heads, head_dim),
        "k_heads_current": split_heads(k_linear, heads, head_dim),
        "v_heads_current": split_heads(v_linear, heads, head_dim),
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(heads_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    cos_for_pos = cos.index_select(0, position_ids).unsqueeze(0)
    sin_for_pos = sin.index_select(0, position_ids).unsqueeze(0)
    return (heads_tensor * cos_for_pos) + (rotate_half(heads_tensor) * sin_for_pos)


def kv_cache_concat(
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "kv_cache_concat" not in EXPECTED_STAGES:
        return k_current, v_current
    if past_k is None or past_v is None:
        raise ValueError("dispatch expects kv_cache_concat; provide past_k and past_v")
    return torch.cat((past_k, k_current), dim=-2), torch.cat((past_v, v_current), dim=-2)


def attention(q_rope: torch.Tensor, k_rope: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    raw_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scaled_scores = raw_scores / (q_rope.shape[-1] ** 0.5)
    masked_scores = scaled_scores + attention_mask
    attn = torch.softmax(masked_scores, dim=-1)
    return {
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "masked_scores": masked_scores,
        "attn": attn,
    }


def visual_adjust(attn: torch.Tensor, visual_start: int, visual_end: int, tail_start: int) -> dict[str, torch.Tensor]:
    kind = DISPATCH_FEATURES.get("visual_adjust_kind")
    adjusted = attn.clone()
    if kind is None:
        return {"adjusted_attn": adjusted}
    adjusted[..., visual_start:, visual_start:visual_end] = 0
    if kind == "fold_tail_visual_mass_and_clear_region":
        tail_visual_sum = attn[..., tail_start:, visual_start:visual_end].sum(dim=-1)
        adjusted[..., tail_start:, visual_start] = tail_visual_sum
        return {"adjusted_attn": adjusted, "tail_visual_sum": tail_visual_sum}
    return {"adjusted_attn": adjusted}


def visipruner_similarity_check(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, torch.Tensor]:
    similarity = F.cosine_similarity(reference, candidate, dim=-1)
    threshold = torch.tensor(0.9, dtype=similarity.dtype, device=similarity.device)
    return {"similarity": similarity, "any_close": torch.any(similarity > threshold)}


def attention_output(
    attn: torch.Tensor,
    v_heads: torch.Tensor,
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    context_heads = torch.matmul(attn, v_heads)
    context = context_heads.transpose(0, 1).contiguous().reshape(hidden_states.shape[0], -1)
    attn_out = F.linear(context, weights["o_weight"])
    after_attn = hidden_states + attn_out
    return {"context": context, "attn_out": attn_out, "after_attn": after_attn}


def mlp(after_attn: torch.Tensor, weights: dict[str, torch.Tensor], post_norm_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    post_norm = rms_norm(after_attn, post_norm_weight)
    gate = F.linear(post_norm["output"], weights["gate_weight"])
    up = F.linear(post_norm["output"], weights["up_weight"])
    gated = F.silu(gate) * up
    ffn_out = F.linear(gated, weights["down_weight"])
    output = after_attn + ffn_out
    return {"post_norm": post_norm["output"], "gated": gated, "ffn_out": ffn_out, "output": output}


def expected_stage_evidence() -> dict[str, object]:
    return {stage: CORE_EVIDENCE.get(stage, {}) for stage in EXPECTED_STAGES}
