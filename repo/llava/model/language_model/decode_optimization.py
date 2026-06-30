"""Decode backend selection for VisiPruner.

The optimized FA2 path is implemented in
custom_modeling_llama_decode_optimized.py. This module deliberately contains no
runtime model monkey-patching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import warnings

from transformers.utils import is_flash_attn_2_available


VALID_VISIPRUNER_DECODE_BACKENDS = ("off", "eager", "fa2", "flash_attention_2", "vp_fa", "vp-fa", "auto")


@dataclass(frozen=True)
class VisiPrunerDecodeBackend:
    requested: str
    selected: str
    use_flash_attn: bool
    use_optimized_modeling: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "use_flash_attn": self.use_flash_attn,
            "use_optimized_modeling": self.use_optimized_modeling,
            "reason": self.reason,
        }


def _normalize_backend(backend: Optional[str]) -> Optional[str]:
    if backend is None:
        return None
    normalized = backend.strip().lower().replace("-", "_")
    aliases = {
        "flash_attention_2": "fa2",
        "flash_attention2": "fa2",
        "flash_attn_2": "fa2",
        "flash_attn2": "fa2",
        "dense_fa2": "fa2",
        "vp-fa": "vp_fa",
        "vp_flash_attention": "vp_fa",
        "vp_flash_attn": "vp_fa",
        "manual": "eager",
        "none": "off",
        "false": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"off", "eager", "fa2", "vp_fa", "auto"}:
        valid = ", ".join(VALID_VISIPRUNER_DECODE_BACKENDS)
        raise ValueError(f"Unsupported VisiPruner decode backend: {backend!r}. Valid values: {valid}.")
    return normalized


def resolve_visipruner_decode_backend(
    backend: Optional[str],
    *,
    use_visipruner: bool,
    use_flash_attn: bool,
) -> VisiPrunerDecodeBackend:
    """Resolve the requested VisiPruner backend before model loading."""
    requested = _normalize_backend(backend)
    if not use_visipruner:
        return VisiPrunerDecodeBackend(
            requested=requested or "off",
            selected="off",
            use_flash_attn=use_flash_attn,
            use_optimized_modeling=False,
            reason="VisiPruner is disabled.",
        )

    if requested is None:
        requested = "fa2" if use_flash_attn else "eager"

    if requested == "off":
        return VisiPrunerDecodeBackend(
            requested="off",
            selected="off",
            use_flash_attn=use_flash_attn,
            use_optimized_modeling=False,
            reason="Decode backend override disabled.",
        )

    if requested == "eager":
        return VisiPrunerDecodeBackend(
            requested="eager",
            selected="eager",
            use_flash_attn=False,
            use_optimized_modeling=False,
            reason="Using the original eager VisiPruner implementation.",
        )

    if requested == "vp_fa" and bool(is_flash_attn_2_available()):
        return VisiPrunerDecodeBackend(
            requested=requested,
            selected="vp_fa",
            use_flash_attn=True,
            use_optimized_modeling=True,
            reason="Using copied optimized VisiPruner modeling with VP-FA prefill/decode.",
        )

    if requested in {"fa2", "auto"} and bool(is_flash_attn_2_available()):
        return VisiPrunerDecodeBackend(
            requested=requested,
            selected="vp_fa" if requested == "auto" else "fa2",
            use_flash_attn=True,
            use_optimized_modeling=True,
            reason="Using copied optimized VisiPruner modeling with VP-FA/FA2 decode.",
        )

    warnings.warn(
        "FlashAttention2 is not available; falling back to the original eager "
        "VisiPruner implementation.",
        RuntimeWarning,
    )
    return VisiPrunerDecodeBackend(
        requested=requested,
        selected="eager",
        use_flash_attn=False,
        use_optimized_modeling=False,
        reason="FlashAttention2 unavailable; fell back to eager.",
    )
