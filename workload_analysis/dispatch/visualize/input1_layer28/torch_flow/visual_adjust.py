from __future__ import annotations

import torch

from config import FlowConfig, VISUAL_ADJUST_KIND


def shallow_full_visual_attention_adjust(attn: torch.Tensor, cfg: FlowConfig) -> dict[str, torch.Tensor]:
    q_visual_start = min(cfg.visual_start, attn.shape[-2])
    q_tail_start = min(cfg.tail_start, attn.shape[-2])
    k_visual_start = min(cfg.visual_start, attn.shape[-1])
    k_visual_end = min(cfg.visual_end, attn.shape[-1])

    tail_to_all = attn[:, q_tail_start:, :]
    tail_to_visual = attn[:, q_tail_start:, k_visual_start:k_visual_end]
    tail_visual_sum = tail_to_visual.sum(dim=-1)
    non_text_to_all = attn[:, q_visual_start:, :]
    non_text_to_visual_before = attn[:, q_visual_start:, k_visual_start:k_visual_end]

    adjusted = attn.clone()
    if q_visual_start < adjusted.shape[-2] and k_visual_start < k_visual_end:
        adjusted[:, q_visual_start:, k_visual_start:k_visual_end] = 0.0
    cleared_visual_region = adjusted[:, q_visual_start:, k_visual_start:k_visual_end]
    if (
        VISUAL_ADJUST_KIND == "fold_tail_visual_mass_and_clear_region"
        and q_tail_start < adjusted.shape[-2]
        and k_visual_start < k_visual_end
    ):
        adjusted[:, q_tail_start:, k_visual_start] = tail_visual_sum
    copied_first_visual_slot = adjusted[:, q_tail_start:, k_visual_start] if k_visual_start < adjusted.shape[-1] else torch.empty(0, dtype=attn.dtype)

    return {
        "tail_to_all": tail_to_all,
        "tail_to_visual": tail_to_visual,
        "tail_visual_sum": tail_visual_sum,
        "non_text_to_all": non_text_to_all,
        "non_text_to_visual_before": non_text_to_visual_before,
        "cleared_visual_region": cleared_visual_region,
        "copied_first_visual_slot": copied_first_visual_slot,
        "adjusted_attn": adjusted,
    }
