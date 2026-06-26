#!/usr/bin/env python3
"""Runnable small-shape toy tensor process for input1_layer24.

The code keeps the dispatch-derived stage order and optional stages, but
shrinks tensor sizes for inspection and ONNX/debugger-friendly execution.
Batch dimension is intentionally omitted because the traced batch is fixed
to one.
"""

from __future__ import annotations

import argparse
import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer24'
SMALL_CONFIG = json.loads(r"""{
  "ffn": 64,
  "head_dim": 8,
  "heads": 4,
  "hidden": 32,
  "kv_seq": 16,
  "q_seq": 16,
  "seq": 16,
  "tail_start": 13,
  "visual_end": 13,
  "visual_start": 3
}""")
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 58,
  "q_len": 58
}""")
DISPATCH_FEATURES = json.loads(r"""{
  "expected_stages": [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention",
    "visipruner_similarity_check",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": false,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 58,
  "op_counts": {
    "add.Tensor": 10,
    "any.default": 1,
    "arange.start": 1,
    "cat.default": 2,
    "contiguous.default": 2,
    "cosine_similarity.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 3,
    "is_nonzero.default": 3,
    "item.default": 3,
    "linear.default": 7,
    "lt.Scalar": 1,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 11,
    "neg.default": 2,
    "permute.default": 1,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 6,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "squeeze.dim": 1,
    "sub.Tensor": 2,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 5,
    "view.default": 4
  },
  "phase": "prefill",
  "prune_probe_kind": "deep_exit_similarity_check",
  "q_len": 58,
  "role": "deep_check",
  "token_state": "middle_pruned",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 15,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 17,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 39,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 46,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 47,
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 58], dtype=float16"
      },
      {
"event_op_index": 48,
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 58], dtype=float16"
      },
      {
"event_op_index": 49,
"op_name": "div.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16"
      },
      {
"event_op_index": 50,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16"
      },
      {
"event_op_index": 51,
"op_name": "softmax.int",
"output": "shape=[1, 32, 58, 58], dtype=float32"
      },
      {
"event_op_index": 53,
"op_name": "dropout.default",
"output": "shape=[1, 32, 58, 58], dtype=float16"
      },
      {
"event_op_index": 54,
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      }
    ],
    "stage": "attention",
    "summary": "Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores."
  },
  "attention_output": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 54,
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 56,
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16"
      },
      {
"event_op_index": 57,
"op_name": "reshape.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 73,
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16"
      },
      {
"event_op_index": 85,
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 86,
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 99,
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 100,
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16"
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
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 2,
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 3,
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 4,
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 5,
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 6,
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 7,
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 8,
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16"
      }
    ],
    "stage": "input_rmsnorm",
    "summary": "RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence."
  },
  "mlp": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 85,
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 86,
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 87,
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 88,
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 89,
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 90,
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 91,
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32"
      },
      {
"event_op_index": 92,
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32"
      },
      {
"event_op_index": 93,
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 94,
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 95,
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16"
      },
      {
"event_op_index": 96,
"op_name": "silu.default",
"output": "shape=[1, 58, 11008], dtype=float16"
      },
      {
"event_op_index": 97,
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16"
      },
      {
"event_op_index": 98,
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 11008], dtype=float16"
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
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 10,
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 11,
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16"
      },
      {
"event_op_index": 12,
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16"
      },
      {
"event_op_index": 13,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 14,
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16"
      },
      {
"event_op_index": 15,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 16,
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16"
      },
      {
"event_op_index": 17,
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16"
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
"output": "shape=[624, 128], dtype=float16"
      },
      {
"event_op_index": 27,
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16"
      },
      {
"event_op_index": 29,
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16"
      },
      {
"event_op_index": 30,
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16"
      },
      {
"event_op_index": 31,
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16"
      },
      {
"event_op_index": 32,
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16"
      },
      {
"event_op_index": 33,
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 34,
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16"
      },
      {
"event_op_index": 35,
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16"
      },
      {
"event_op_index": 36,
"op_name": "neg.default",
"output": "shape=[1, 32, 58, 64], dtype=float16"
      },
      {
"event_op_index": 37,
"op_name": "cat.default",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 38,
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      },
      {
"event_op_index": 39,
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16"
      }
    ],
    "stage": "rope",
    "summary": "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
  },
  "visipruner_similarity_check": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 21,
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool"
      },
      {
"event_op_index": 22,
"op_name": "is_nonzero.default",
"output": "False"
      },
      {
"event_op_index": 58,
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool"
      },
      {
"event_op_index": 59,
"op_name": "is_nonzero.default",
"output": "True"
      },
      {
"event_op_index": 64,
"op_name": "sub.Tensor",
"output": "shape=[], dtype=int64"
      },
      {
"event_op_index": 67,
"op_name": "is_nonzero.default",
"output": "True"
      },
      {
"event_op_index": 75,
"op_name": "arange.start",
"output": "shape=[10], dtype=int64"
      },
      {
"event_op_index": 78,
"op_name": "sub.Tensor",
"output": "shape=[1, 10, 4096], dtype=float16"
      },
      {
"event_op_index": 80,
"op_name": "cosine_similarity.default",
"output": "shape=[1, 10], dtype=float16"
      },
      {
"event_op_index": 83,
"op_name": "any.default",
"output": "shape=[], dtype=bool"
      }
    ],
    "stage": "visipruner_similarity_check",
    "summary": "VisiPrune check evidence kind: deep_exit_similarity_check."
  }
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]


def _cfg(name: str) -> int:
    return int(SMALL_CONFIG[name])


def _randn(generator: torch.Generator, *shape: int, scale: float = 0.08) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale


def q_seq() -> int:
    return _cfg("q_seq")


def kv_seq() -> int:
    return _cfg("kv_seq")


def make_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    q = q_seq()
    kv = kv_seq()
    hidden_states = torch.randn(q, _cfg("hidden"), generator=generator, dtype=torch.float32)
    position_ids = torch.arange(max(kv - q, 0), kv, dtype=torch.long)
    key_positions = torch.arange(kv, dtype=torch.long)
    future = key_positions.unsqueeze(0) > position_ids.unsqueeze(1)
    attention_mask = future.to(torch.float32).masked_fill(future, -10000.0)
    past_tokens = max(kv - q, 0)
    past_k = _randn(generator, _cfg("heads"), past_tokens, _cfg("head_dim")) if past_tokens else torch.empty(_cfg("heads"), 0, _cfg("head_dim"))
    past_v = _randn(generator, _cfg("heads"), past_tokens, _cfg("head_dim")) if past_tokens else torch.empty(_cfg("heads"), 0, _cfg("head_dim"))
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_k": past_k,
        "past_v": past_v,
    }


def make_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    return {
        "input_norm_weight": torch.ones(_cfg("hidden"), dtype=torch.float32),
        "post_norm_weight": torch.ones(_cfg("hidden"), dtype=torch.float32),
        "q_weight": _randn(generator, _cfg("hidden"), _cfg("hidden")),
        "k_weight": _randn(generator, _cfg("hidden"), _cfg("hidden")),
        "v_weight": _randn(generator, _cfg("hidden"), _cfg("hidden")),
        "o_weight": _randn(generator, _cfg("hidden"), _cfg("hidden")),
        "gate_weight": _randn(generator, _cfg("ffn"), _cfg("hidden")),
        "up_weight": _randn(generator, _cfg("ffn"), _cfg("hidden")),
        "down_weight": _randn(generator, _cfg("hidden"), _cfg("ffn")),
    }


def rope_cache() -> dict[str, torch.Tensor]:
    positions = torch.arange(kv_seq(), dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(_cfg("head_dim"), dtype=torch.float32).unsqueeze(0)
    angles = positions / (10000.0 ** (freqs / _cfg("head_dim")))
    return {"cos": torch.cos(angles), "sin": torch.sin(angles)}


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> dict[str, torch.Tensor]:
    x_float = hidden_states.to(torch.float32)
    squared = x_float.pow(2)
    variance = squared.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    output = (x_float * inv_rms).to(hidden_states.dtype) * weight
    return {"squared": squared, "variance": variance, "inv_rms": inv_rms, "output": output}


def split_heads(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], _cfg("heads"), _cfg("head_dim")).transpose(0, 1)


def project_qkv(x_norm: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    q_linear = F.linear(x_norm, weights["q_weight"])
    k_linear = F.linear(x_norm, weights["k_weight"])
    v_linear = F.linear(x_norm, weights["v_weight"])
    return {
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_heads": split_heads(q_linear),
        "k_heads_current": split_heads(k_linear),
        "v_heads_current": split_heads(v_linear),
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_to_heads(heads_tensor: torch.Tensor, positions: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    cos = cache["cos"].index_select(0, positions).unsqueeze(0)
    sin = cache["sin"].index_select(0, positions).unsqueeze(0)
    return (heads_tensor * cos) + (rotate_half(heads_tensor) * sin)


def kv_cache_concat(k_current: torch.Tensor, v_current: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "kv_cache_concat" not in EXPECTED_STAGES:
        return {"k_heads": k_current, "v_heads": v_current}
    return {
        "k_heads": torch.cat((inputs["past_k"], k_current), dim=-2),
        "v_heads": torch.cat((inputs["past_v"], v_current), dim=-2),
    }


def attention(q_rope: torch.Tensor, k_rope: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    raw_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scaled_scores = raw_scores / (_cfg("head_dim") ** 0.5)
    masked_scores = scaled_scores + attention_mask
    attn = torch.softmax(masked_scores, dim=-1)
    return {"raw_scores": raw_scores, "scaled_scores": scaled_scores, "masked_scores": masked_scores, "attn": attn}


def visual_adjust(attn: torch.Tensor) -> dict[str, torch.Tensor]:
    kind = DISPATCH_FEATURES.get("visual_adjust_kind")
    adjusted = attn.clone()
    if kind is None:
        return {"adjusted_attn": adjusted}
    q_visual_start = min(_cfg("visual_start"), adjusted.shape[-2])
    q_tail_start = min(_cfg("tail_start"), adjusted.shape[-2])
    k_visual_start = min(_cfg("visual_start"), adjusted.shape[-1])
    k_visual_end = min(_cfg("visual_end"), adjusted.shape[-1])
    if q_visual_start < adjusted.shape[-2] and k_visual_start < k_visual_end:
        adjusted[..., q_visual_start:, k_visual_start:k_visual_end] = 0.0
    result = {"adjusted_attn": adjusted}
    if kind == "fold_tail_visual_mass_and_clear_region" and q_tail_start < adjusted.shape[-2] and k_visual_start < k_visual_end:
        tail_visual_sum = attn[..., q_tail_start:, k_visual_start:k_visual_end].sum(dim=-1)
        adjusted[..., q_tail_start:, k_visual_start] = tail_visual_sum
        result["tail_visual_sum"] = tail_visual_sum
    return result


def visipruner_similarity_check(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    if "visipruner_similarity_check" not in EXPECTED_STAGES:
        empty = torch.empty(0, dtype=hidden_states.dtype)
        return {"similarity": empty, "any_close": torch.tensor(False)}
    if hidden_states.shape[0] < 2:
        reference = hidden_states
        candidate = hidden_states
    else:
        reference = hidden_states[:-1]
        candidate = hidden_states[1:]
    similarity = F.cosine_similarity(reference, candidate, dim=-1)
    return {"similarity": similarity, "any_close": torch.any(similarity > 0.9)}


def attention_output(attn: torch.Tensor, v_heads: torch.Tensor, hidden_states: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    context_heads = torch.matmul(attn, v_heads)
    context = context_heads.transpose(0, 1).contiguous().reshape(hidden_states.shape[0], _cfg("hidden"))
    attn_out = F.linear(context, weights["o_weight"])
    after_attn = hidden_states + attn_out
    return {"context_heads": context_heads, "context": context, "attn_out": attn_out, "after_attn": after_attn}


def gated_mlp(after_attn: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    post_norm = rms_norm(after_attn, weights["post_norm_weight"])
    gate = F.linear(post_norm["output"], weights["gate_weight"])
    up = F.linear(post_norm["output"], weights["up_weight"])
    gated = F.silu(gate) * up
    ffn_out = F.linear(gated, weights["down_weight"])
    output = after_attn + ffn_out
    return {"post_norm_output": post_norm["output"], "gate": gate, "up": up, "gated": gated, "ffn_out": ffn_out, "output": output}


def run_toy_flow() -> dict[str, torch.Tensor]:
    inputs = make_inputs()
    weights = make_weights()
    cache = rope_cache()
    tensors: dict[str, torch.Tensor] = dict(inputs)
    input_norm = rms_norm(inputs["hidden_states"], weights["input_norm_weight"])
    tensors.update({"input_norm_output": input_norm["output"], "input_norm_variance": input_norm["variance"]})
    qkv = project_qkv(input_norm["output"], weights)
    tensors.update(qkv)
    q_rope = apply_rope_to_heads(qkv["q_heads"], inputs["position_ids"], cache)
    k_current_rope = apply_rope_to_heads(qkv["k_heads_current"], inputs["position_ids"], cache)
    tensors.update({"q_rope": q_rope, "k_current_rope": k_current_rope})
    kv = kv_cache_concat(k_current_rope, qkv["v_heads_current"], inputs)
    tensors.update(kv)
    attn = attention(q_rope, kv["k_heads"], inputs["attention_mask"])
    tensors.update(attn)
    if "visual_adjust" in EXPECTED_STAGES:
        adjusted = visual_adjust(attn["attn"])
        tensors.update(adjusted)
        attn_for_output = adjusted["adjusted_attn"]
    else:
        tensors["adjusted_attn"] = attn["attn"]
        attn_for_output = attn["attn"]
    if "visipruner_similarity_check" in EXPECTED_STAGES:
        tensors.update(visipruner_similarity_check(inputs["hidden_states"]))
    attn_out = attention_output(attn_for_output, kv["v_heads"], inputs["hidden_states"], weights)
    tensors.update(attn_out)
    mlp_out = gated_mlp(attn_out["after_attn"], weights)
    tensors.update(mlp_out)
    return tensors


def _describe(name: str, tensor: torch.Tensor) -> str:
    return f"{name}: shape={tuple(tensor.shape)} dtype={str(tensor.dtype).replace('torch.', '')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    with torch.no_grad():
        tensors = run_toy_flow()
    if not args.quiet:
        print(f"event_id={EVENT_ID}")
        print(f"expected_stages={', '.join(EXPECTED_STAGES)}")
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                print(_describe(name, tensor))


if __name__ == "__main__":
    main()
