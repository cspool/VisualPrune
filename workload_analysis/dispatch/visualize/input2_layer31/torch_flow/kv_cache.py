from __future__ import annotations

import torch


def concat_kv_cache(
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    k_heads = torch.cat((past_k, k_current), dim=-2)
    v_heads = torch.cat((past_v, v_current), dim=-2)
    return {
        "k_heads": k_heads,
        "v_heads": v_heads,
    }
