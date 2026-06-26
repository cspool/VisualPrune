from __future__ import annotations

import torch
import torch.nn.functional as F

from config import FlowConfig


def split_heads(x: torch.Tensor, cfg: FlowConfig) -> torch.Tensor:
    return x.view(x.shape[0], cfg.heads, cfg.head_dim).transpose(0, 1)


def project_qkv(x_norm: torch.Tensor, weights: dict[str, torch.Tensor], cfg: FlowConfig) -> dict[str, torch.Tensor]:
    q_linear = F.linear(x_norm, weights["q_weight"])
    k_linear = F.linear(x_norm, weights["k_weight"])
    v_linear = F.linear(x_norm, weights["v_weight"])
    q = split_heads(q_linear, cfg)
    k = split_heads(k_linear, cfg)
    v = split_heads(v_linear, cfg)
    return {
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_heads": q,
        "k_heads": k,
        "v_heads": v,
    }
