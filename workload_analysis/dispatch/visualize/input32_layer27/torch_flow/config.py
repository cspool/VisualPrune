from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowConfig:
    seq: int = 16
    q_seq: int = 1
    kv_seq: int = 16
    hidden: int = 32
    heads: int = 4
    head_dim: int = 8
    visual_start: int = 3
    visual_end: int = 13
    tail_start: int = 13
    ffn: int = 64
    eps: float = 1e-5
    weight_seed: int = 7
    input_seed: int = 11
    event_id: str = 'input32_layer27'

    @property
    def half_dim(self) -> int:
        return self.head_dim // 2

    @property
    def visual_tokens(self) -> int:
        return self.visual_end - self.visual_start

    @property
    def tail_tokens(self) -> int:
        return self.seq - self.tail_start


EXPECTED_STAGES = ('input_rmsnorm', 'qkv_projection', 'rope', 'kv_cache_concat', 'attention', 'attention_output', 'mlp')
VISUAL_ADJUST_KIND = None
PRUNE_PROBE_KIND = None

CFG = FlowConfig()
