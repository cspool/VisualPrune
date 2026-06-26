# input1_layer5 `05_visual_adjust.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/onnx/05_visual_adjust.onnx`
- Stage: `visual_adjust`
- Stage title: Visual Attention Adjustment
- ONNX nodes: `81`
- ONNX initializers: `0`

### ONNX Inputs

- `attn`: `[4, 16, 16]`

### ONNX Outputs

- `adjusted_attn`: `[4, 16, 16]`
- `tail_visual_sum`: `[4, 3]`
- `cleared_visual_region`: `[4, 13, 10]`

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py::VisualAdjustStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/visual_adjust.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py`

## Code Explanation

Models the VisiPrune visual-token attention adjustment stage when dispatch evidence shows visual-region clearing or tail-mass folding.

## Review Comments

- This page binds `05_visual_adjust.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#55 lift_fresh.default` -> shape=[], dtype=float16
- `#57 slice.Tensor` -> shape=[1, 32, 13, 624], dtype=float16
- `#59 slice.Tensor` -> shape=[1, 32, 13, 576], dtype=float16
- `#60 fill_.Tensor` -> shape=[1, 32, 13, 576], dtype=float16

## Export Wrapper Source

```python
class VisualAdjustStage(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = shallow_full_visual_attention_adjust(attn, self.cfg)
        return out["adjusted_attn"], out["tail_visual_sum"], out["cleared_visual_region"]
```

## Primary Source: `visual_adjust.py`

```python
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
```
