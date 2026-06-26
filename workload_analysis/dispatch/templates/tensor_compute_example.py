#!/usr/bin/env python3
"""Shape-level tensor process for input1_layer0.

This example is built from dispatch_ops.csv layouts. It allocates small metadata
objects instead of CUDA tensors, so it can run anywhere and prints the inferred
layer computation sequence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: str
    note: str = ""

    def __str__(self) -> str:
        note = f"  # {self.note}" if self.note else ""
        return f"{self.name:<24} {str(self.shape):<22} {self.dtype}{note}"


def emit(step: str, *layouts: TensorLayout) -> None:
    print(f"\n{step}")
    for layout in layouts:
        print(f"  {layout}")


def main() -> None:
    bsz, seq, hidden = 1, 624, 4096
    heads, head_dim, half_dim = 32, 128, 64
    visual_start, visual_end, tail_start = 35, 611, 611
    visual_tokens = visual_end - visual_start
    tail_tokens = seq - tail_start
    ffn_dim = 11008

    x = TensorLayout("x", (bsz, seq, hidden), "float16", "input hidden_states")
    emit("0. input", x)

    x_f32 = TensorLayout("x_f32", (bsz, seq, hidden), "float32")
    variance = TensorLayout("variance", (bsz, seq, 1), "float32")
    inv_rms = TensorLayout("inv_rms", (bsz, seq, 1), "float32")
    x_norm = TensorLayout("x_norm", (bsz, seq, hidden), "float16")
    emit("1. input RMSNorm", x_f32, variance, inv_rms, x_norm)

    q = TensorLayout("q", (bsz, heads, seq, head_dim), "float16")
    k = TensorLayout("k", (bsz, heads, seq, head_dim), "float16")
    v = TensorLayout("v", (bsz, heads, seq, head_dim), "float16")
    emit("2. Q/K/V projection + view/transpose", q, k, v)

    cos = TensorLayout("cos", (bsz, 1, seq, head_dim), "float16")
    sin = TensorLayout("sin", (bsz, 1, seq, head_dim), "float16")
    q_rot = TensorLayout("q_rot", (bsz, heads, seq, head_dim), "float16", "cat(-q[...,64:], q[...,:64])")
    k_rot = TensorLayout("k_rot", (bsz, heads, seq, head_dim), "float16", "cat(-k[...,64:], k[...,:64])")
    q_rope = TensorLayout("q_rope", (bsz, heads, seq, head_dim), "float16")
    k_rope = TensorLayout("k_rope", (bsz, heads, seq, head_dim), "float16")
    emit("3. RoPE gather and apply", cos, sin, q_rot, k_rot, q_rope, k_rope)

    k_t = TensorLayout("k_t", (bsz, heads, head_dim, seq), "float16")
    scores = TensorLayout("scores", (bsz, heads, seq, seq), "float16")
    attn = TensorLayout("attn", (bsz, heads, seq, seq), "float16")
    emit("4. scaled dot-product attention", k_t, scores, attn)

    tail_to_all = TensorLayout("tail_to_all", (bsz, heads, tail_tokens, seq), "float16")
    tail_to_visual = TensorLayout("tail_to_visual", (bsz, heads, tail_tokens, visual_tokens), "float16")
    tail_visual_sum = TensorLayout("tail_visual_sum", (bsz, heads, tail_tokens), "float16")
    non_text_to_all = TensorLayout("non_text_to_all", (bsz, heads, seq - visual_start, seq), "float16")
    non_text_to_visual = TensorLayout("non_text_to_visual", (bsz, heads, seq - visual_start, visual_tokens), "float16")
    first_visual_slot = TensorLayout("first_visual_slot", (bsz, heads, tail_tokens), "float16")
    emit(
        "5. inferred shallow/full-visual attention adjustment",
        tail_to_all,
        tail_to_visual,
        tail_visual_sum,
        non_text_to_all,
        non_text_to_visual,
        first_visual_slot,
    )

    context_heads = TensorLayout("context_heads", (bsz, heads, seq, head_dim), "float16")
    context = TensorLayout("context", (bsz, seq, hidden), "float16")
    attn_out = TensorLayout("attn_out", (bsz, seq, hidden), "float16")
    after_attn = TensorLayout("after_attn", (bsz, seq, hidden), "float16")
    emit("6. attention value matmul + output projection + residual", context_heads, context, attn_out, after_attn)

    mlp_norm = TensorLayout("mlp_norm", (bsz, seq, hidden), "float16")
    gate = TensorLayout("gate", (bsz, seq, ffn_dim), "float16")
    up = TensorLayout("up", (bsz, seq, ffn_dim), "float16")
    gated = TensorLayout("gated", (bsz, seq, ffn_dim), "float16")
    ffn_out = TensorLayout("ffn_out", (bsz, seq, hidden), "float16")
    output = TensorLayout("output", (bsz, seq, hidden), "float16")
    emit("7. post-attention RMSNorm + gated MLP + final residual", mlp_norm, gate, up, gated, ffn_out, output)

    print(
        "\nformula:\n"
        "  output = after_attn + down_proj(silu(gate_proj(rms_norm(after_attn))) * "
        "up_proj(rms_norm(after_attn)))\n"
        "  where after_attn = x + o_proj((adjusted_attn @ v).transpose(1,2).reshape(1,624,4096))"
    )


if __name__ == "__main__":
    main()
