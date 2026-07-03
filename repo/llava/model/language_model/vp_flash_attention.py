"""Triton VP-FA prefill helpers for VisiPruner inference.

This module is the integration boundary for VisiPruner-aware prefill attention.
The public helper returns the same pre-o_proj attention output shape as
LlamaAttention and, when requested, the last-query attention weights needed by
value-aware token selection.

The current experiment direction deliberately does not depend on a patched
FlashAttention build. Full VP-FA prefill uses Triton kernels for shallow
VisiPruner edits and middle/deep dense causal attention. This keeps the
experiment self-contained while preserving the current shallow Triton path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability is environment-specific.
    triton = None
    tl = None


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent to transformers.models.llama.modeling_llama.repeat_kv."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def needs_official_shallow_path(
    *,
    pruning_mode: list[str] | tuple[str, ...] | set[str] | None,
    layer_idx: int,
    shallow_mid_layer: int,
    q_len: int,
    num_images: int,
) -> bool:
    return (
        "shallow" in (pruning_mode or [])
        and layer_idx < shallow_mid_layer
        and q_len > 611
        and num_images > 0
    )


def _apply_official_shallow_visipruner(
    attn_weights: torch.Tensor,
    *,
    layer_idx: int,
    model_size: str,
    vis_end_index: int,
    vis_half_index: int,
) -> torch.Tensor:
    """Apply official VisiPruner post-softmax shallow edits in-place."""
    if layer_idx == 0 and model_size == "7b":
        sum_vision_attn_weights = attn_weights[:, :, vis_end_index:, 35:vis_end_index].sum(dim=-1)
        attn_weights[:, :, 35:, 35:vis_end_index] = 0
        attn_weights[:, :, vis_end_index:, 35] = sum_vision_attn_weights
    elif layer_idx == 0 and model_size == "13b":
        attn_weights[:, :, 35:vis_end_index, 35:vis_end_index] = 0
        attn_weights[:, :, vis_end_index:, 35:vis_half_index] = 0
    elif 0 < layer_idx:
        attn_weights[:, :, vis_end_index:, 35:vis_end_index] = 0
    return attn_weights


def _official_attention_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    scale: float,
    num_key_value_groups: int,
    dropout_p: float,
    training: bool,
    layer_idx: int,
    model_size: str,
    vis_end_index: int,
    vis_half_index: int,
    apply_shallow: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact official VisiPruner attention path for shallow layers."""
    bsz, _, q_len, _ = query_states.shape
    key_states_full = _repeat_kv(key_states, num_key_value_groups)
    value_states_full = _repeat_kv(value_states, num_key_value_groups)
    kv_seq_len = key_states_full.shape[-2]

    attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) * scale
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :q_len, :kv_seq_len]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if apply_shallow:
        attn_weights = _apply_official_shallow_visipruner(
            attn_weights,
            layer_idx=layer_idx,
            model_size=model_size,
            vis_end_index=vis_end_index,
            vis_half_index=vis_half_index,
        )
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    attn_output = torch.matmul(attn_weights, value_states_full)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    return attn_output, attn_weights


if triton is not None:
    @triton.jit
    def _vp_fa_prefill_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        q_len: tl.constexpr,
        kv_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        scale: tl.constexpr,
        layer_idx: tl.constexpr,
        model_kind: tl.constexpr,
        vis_end_index: tl.constexpr,
        vis_half_index: tl.constexpr,
        apply_shallow: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        head_mask = offs_d < head_dim

        kv_head = pid_h // num_key_value_groups

        q_base = ((pid_b * num_heads + pid_h) * q_len) * head_dim
        k_base = ((pid_b * num_kv_heads + kv_head) * kv_len) * head_dim
        v_base = k_base
        out_base = ((pid_b * num_heads + pid_h) * q_len) * head_dim

        q = tl.load(
            q_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
            mask=(offs_m[:, None] < q_len) & head_mask[None, :],
            other=0.0,
        )

        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, block_d), tl.float32)
        visual_mass = tl.zeros((block_m,), tl.float32)

        for start_n in range(0, kv_len, block_n):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + k_base + cols[:, None] * head_dim + offs_d[None, :],
                mask=(cols[:, None] < kv_len) & head_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            causal_mask = cols[None, :] <= offs_m[:, None]
            valid_mask = (offs_m[:, None] < q_len) & (cols[None, :] < kv_len) & causal_mask
            scores = tl.where(valid_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            p_acc = p

            if apply_shallow:
                # Official VisiPruner shallow edits are applied after softmax
                # and do not renormalize rows. l_i tracks the original softmax
                # denominator, while p_acc controls the mass used by P @ V.
                visual_cols = (cols >= 35) & (cols < vis_end_index)
                visual_half_cols = (cols >= 35) & (cols < vis_half_index)
                q_visual_or_text = offs_m >= 35
                q_text_after_image = offs_m >= vis_end_index

                if layer_idx == 0 and model_kind == 7:
                    drop_mask = q_visual_or_text[:, None] & visual_cols[None, :]
                    p_acc = tl.where(drop_mask, 0.0, p_acc)
                    visual_mass = visual_mass * alpha + tl.sum(
                        tl.where(q_text_after_image[:, None] & visual_cols[None, :], p, 0.0),
                        axis=1,
                    )
                elif layer_idx == 0 and model_kind == 13:
                    drop_image_rows = (
                        (offs_m[:, None] >= 35)
                        & (offs_m[:, None] < vis_end_index)
                        & visual_cols[None, :]
                    )
                    drop_text_rows = q_text_after_image[:, None] & visual_half_cols[None, :]
                    p_acc = tl.where(drop_image_rows | drop_text_rows, 0.0, p_acc)
                    visual_mass = visual_mass * alpha
                else:
                    drop_mask = q_text_after_image[:, None] & visual_cols[None, :]
                    p_acc = tl.where(drop_mask, 0.0, p_acc)
                    visual_mass = visual_mass * alpha
            else:
                visual_mass = visual_mass * alpha

            v = tl.load(
                v_ptr + v_base + cols[:, None] * head_dim + offs_d[None, :],
                mask=(cols[:, None] < kv_len) & head_mask[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p_acc.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        if apply_shallow and layer_idx == 0 and model_kind == 7:
            v35 = tl.load(
                v_ptr + v_base + 35 * head_dim + offs_d,
                mask=head_mask,
                other=0.0,
            )
            acc = acc + visual_mass[:, None] * v35[None, :]

        acc = acc / l_i[:, None]
        tl.store(
            out_ptr + out_base + offs_m[:, None] * head_dim + offs_d[None, :],
            acc,
            mask=(offs_m[:, None] < q_len) & head_mask[None, :],
        )


def _vp_fa_prefill_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    scale: float,
    num_key_value_groups: int,
    layer_idx: int,
    model_size: str,
    vis_end_index: int,
    vis_half_index: int,
    apply_shallow: bool,
) -> torch.Tensor:
    if triton is None or not query_states.is_cuda:
        raise RuntimeError("Triton VP-FA prefill kernel is unavailable.")

    bsz, num_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()
    out = torch.empty_like(q)

    block_m = 16
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise RuntimeError(f"Unsupported VP-FA head_dim: {head_dim}")

    grid = (triton.cdiv(q_len, block_m), num_heads, bsz)
    _vp_fa_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        q_len,
        key_states.shape[2],
        head_dim,
        num_heads,
        num_kv_heads,
        num_key_value_groups,
        scale,
        layer_idx,
        7 if model_size == "7b" else 13,
        vis_end_index,
        vis_half_index,
        apply_shallow,
        block_m,
        block_n,
        block_d,
        num_warps=4,
    )
    return out.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)


if triton is not None:
    @triton.jit
    def _vp_fa_last_query_weights_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        weights_ptr,
        out_ptr,
        q_len: tl.constexpr,
        kv_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        scale: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        head_mask = offs_d < head_dim
        kv_head = pid_h // num_key_value_groups

        q_base = ((pid_b * num_heads + pid_h) * q_len + (q_len - 1)) * head_dim
        k_base = ((pid_b * num_kv_heads + kv_head) * kv_len) * head_dim
        v_base = k_base
        weights_base = (pid_b * num_heads + pid_h) * kv_len
        out_base = (pid_b * num_heads + pid_h) * head_dim

        q = tl.load(q_ptr + q_base + offs_d, mask=head_mask, other=0.0)
        k = tl.load(
            k_ptr + k_base + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=(offs_n[:, None] < kv_len) & head_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(offs_n < kv_len, scores, -float("inf"))
        scores = scores - tl.max(scores, axis=0)
        weights = tl.exp(scores)
        weights = weights / tl.sum(weights, axis=0)
        v = tl.load(
            v_ptr + v_base + offs_n[:, None] * head_dim + offs_d[None, :],
            mask=(offs_n[:, None] < kv_len) & head_mask[None, :],
            other=0.0,
        )
        acc = tl.sum(weights[:, None] * v.to(tl.float32), axis=0)
        tl.store(
            weights_ptr + weights_base + offs_n,
            weights,
            mask=offs_n < kv_len,
        )
        tl.store(
            out_ptr + out_base + offs_d,
            acc,
            mask=head_mask,
        )


def _last_query_weights_and_output_vp_fa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    scale: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not query_states.is_cuda:
        raise RuntimeError("Triton VP-FA last-query kernel is unavailable.")

    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]
    block_n = triton.next_power_of_2(kv_len)
    block_d = triton.next_power_of_2(head_dim)
    if block_n > 2048 or block_d > 256:
        raise RuntimeError(f"Unsupported VP-FA last-query shape: kv_len={kv_len}, head_dim={head_dim}")

    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()
    weights = torch.empty(
        (bsz, num_heads, 1, kv_len),
        device=query_states.device,
        dtype=query_states.dtype,
    )
    out_heads = torch.empty(
        (bsz, num_heads, head_dim),
        device=query_states.device,
        dtype=query_states.dtype,
    )

    _vp_fa_last_query_weights_kernel[(num_heads, bsz)](
        q,
        k,
        v,
        weights,
        out_heads,
        q_len,
        kv_len,
        head_dim,
        num_heads,
        key_states.shape[1],
        num_key_value_groups,
        scale,
        block_n,
        block_d,
        num_warps=8,
    )

    attn_output_last = out_heads.reshape(bsz, 1, num_heads * head_dim)
    return weights, attn_output_last


def _last_query_weights_and_output_torch(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    scale: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states_full = _repeat_kv(key_states, num_key_value_groups)
    value_states_full = _repeat_kv(value_states, num_key_value_groups)
    kv_seq_len = key_states_full.shape[-2]
    pruning_weights = torch.matmul(
        query_states[:, :, -1:, :], key_states_full.transpose(2, 3)
    ) * scale
    if attention_mask is not None and attention_mask.dim() == 4:
        pruning_weights = pruning_weights + attention_mask[:, :, -1:, :kv_seq_len]
    pruning_weights = F.softmax(pruning_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output_last = torch.matmul(pruning_weights, value_states_full)
    attn_output_last = attn_output_last.transpose(1, 2).contiguous()
    attn_output_last = attn_output_last.reshape(query_states.shape[0], 1, -1)
    return pruning_weights, attn_output_last


def _can_ignore_last_query_mask(attention_mask: Optional[torch.Tensor], kv_seq_len: int) -> bool:
    if attention_mask is None:
        return True
    if attention_mask.dim() != 4:
        return False
    last_query_mask = attention_mask[:, :, -1, :kv_seq_len]
    return bool(torch.all(last_query_mask == 0).item())


def _last_query_weights_and_output(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    scale: float,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _can_ignore_last_query_mask(attention_mask, key_states.shape[2]):
        try:
            return _last_query_weights_and_output_vp_fa(
                query_states,
                key_states,
                value_states,
                scale=scale,
                num_key_value_groups=num_key_value_groups,
            )
        except RuntimeError:
            pass
    return _last_query_weights_and_output_torch(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        scale=scale,
        num_key_value_groups=num_key_value_groups,
    )


def vp_flash_attn_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    layer_idx: int,
    model_size: str,
    pruning_mode: list[str] | tuple[str, ...] | set[str] | None,
    shallow_mid_layer: int,
    num_images: int,
    vis_end_index: int,
    vis_half_index: int,
    num_key_value_groups: int,
    need_pruning_weights: bool,
    is_causal: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run VisiPruner-aware prefill attention.

    Inputs use LlamaAttention layout: query is [B, Hq, Q, D], key/value are
    [B, Hkv, K, D]. The returned attention output is [B, Q, hidden_size], before
    o_proj. When requested, pruning weights are [B, Hq, 1, K] and the
    last-query output is [B, 1, hidden_size].
    """
    _, _, q_len, head_dim = query_states.shape
    scale = 1.0 / math.sqrt(head_dim)
    if torch.is_tensor(num_images):
        num_images = int(num_images.item())
    if torch.is_tensor(vis_end_index):
        vis_end_index = int(vis_end_index.item())
    if torch.is_tensor(vis_half_index):
        vis_half_index = int(vis_half_index.item())

    use_official_shallow = needs_official_shallow_path(
        pruning_mode=pruning_mode,
        layer_idx=layer_idx,
        shallow_mid_layer=shallow_mid_layer,
        q_len=q_len,
        num_images=num_images,
    )
    if training or dropout_p != 0.0:
        raise RuntimeError("Triton VP-FA prefill is inference-only and requires dropout_p=0.")
    if not is_causal:
        raise RuntimeError("Triton VP-FA prefill currently supports causal prefill only.")

    attn_output = _vp_fa_prefill_triton(
        query_states,
        key_states,
        value_states,
        scale=scale,
        num_key_value_groups=num_key_value_groups,
        layer_idx=layer_idx,
        model_size=model_size,
        vis_end_index=vis_end_index,
        vis_half_index=vis_half_index,
        apply_shallow=use_official_shallow,
    )

    pruning_weights = None
    selection_attn_output_last = None
    if need_pruning_weights:
        pruning_weights, selection_attn_output_last = _last_query_weights_and_output(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scale=scale,
            num_key_value_groups=num_key_value_groups,
        )
    return attn_output, pruning_weights, selection_attn_output_last
