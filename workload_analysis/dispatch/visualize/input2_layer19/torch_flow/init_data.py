from __future__ import annotations

import torch

from config import FlowConfig


def _randn(generator: torch.Generator, *shape: int, scale: float = 0.08) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale


def build_inputs(cfg: FlowConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.input_seed)
    hidden_states = torch.randn(cfg.q_seq, cfg.hidden, generator=generator, dtype=torch.float32)
    position_ids = torch.arange(cfg.kv_seq - cfg.q_seq, cfg.kv_seq, dtype=torch.long)
    key_positions = torch.arange(cfg.kv_seq, dtype=torch.long)
    future = key_positions.unsqueeze(0) > position_ids.unsqueeze(1)
    attention_mask = future.to(torch.float32).masked_fill(future, -10000.0).reshape(1, cfg.q_seq, cfg.kv_seq)
    past_tokens = max(cfg.kv_seq - cfg.q_seq, 0)
    past_k = _randn(generator, cfg.heads, past_tokens, cfg.head_dim) if past_tokens else torch.empty(cfg.heads, 0, cfg.head_dim)
    past_v = _randn(generator, cfg.heads, past_tokens, cfg.head_dim) if past_tokens else torch.empty(cfg.heads, 0, cfg.head_dim)
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_k": past_k,
        "past_v": past_v,
    }


def build_weights(cfg: FlowConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.weight_seed)
    return {
        "input_norm_weight": torch.ones(cfg.hidden, dtype=torch.float32),
        "post_norm_weight": torch.ones(cfg.hidden, dtype=torch.float32),
        "q_weight": _randn(generator, cfg.hidden, cfg.hidden),
        "k_weight": _randn(generator, cfg.hidden, cfg.hidden),
        "v_weight": _randn(generator, cfg.hidden, cfg.hidden),
        "o_weight": _randn(generator, cfg.hidden, cfg.hidden),
        "gate_weight": _randn(generator, cfg.ffn, cfg.hidden),
        "up_weight": _randn(generator, cfg.ffn, cfg.hidden),
        "down_weight": _randn(generator, cfg.hidden, cfg.ffn),
    }


def build_rope_cache(cfg: FlowConfig) -> dict[str, torch.Tensor]:
    positions = torch.arange(cfg.kv_seq, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(cfg.head_dim, dtype=torch.float32).unsqueeze(0)
    angles = positions / (10000.0 ** (freqs / cfg.head_dim))
    return {
        "cos_cached": torch.cos(angles),
        "sin_cached": torch.sin(angles),
    }
