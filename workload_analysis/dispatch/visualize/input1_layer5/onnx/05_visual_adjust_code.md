# input1_layer5 `05_visual_adjust.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/onnx/05_visual_adjust.onnx`
- Stage: `visual_adjust`
- Stage title: Visual Attention Adjustment
- ONNX nodes: `70`
- ONNX initializers: `0`

### ONNX Inputs

- `attn`: `[4, 16, 16]`

### ONNX Outputs

- `adjusted_attn`: `[4, 16, 16]`
- `cleared_visual_region`: `[4, 13, 10]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00000147']`
- Dispatch output tensor ids: `[]`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000151', 'consumer_event_op_index': 59, 'consumer_op_name': 'slice.Tensor'}, {'tensor_id': 't00000152', 'consumer_event_op_index': 60, 'consumer_op_name': 'fill_.Tensor'}, {'tensor_id': 't00000149', 'consumer_event_op_index': 60, 'consumer_op_name': 'fill_.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py::VisualAdjustStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/visual_adjust.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py`

## Code Explanation

Models the VisiPrune visual-token attention adjustment stage when dispatch evidence shows visual-region clearing or tail-mass folding.

## Review Comments

- This page binds `05_visual_adjust.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#55 lift_fresh.default` inputs=`['t00000149']` outputs=`['t00000149']` -> shape=[], dtype=float16
- `#57 slice.Tensor` inputs=`['t00000147']` outputs=`['t00000151']` -> shape=[1, 32, 13, 624], dtype=float16
- `#59 slice.Tensor` inputs=`['t00000151']` outputs=`['t00000152']` -> shape=[1, 32, 13, 576], dtype=float16
- `#60 fill_.Tensor` inputs=`['t00000152', 't00000149']` outputs=`['t00000152']` -> shape=[1, 32, 13, 576], dtype=float16

## Export Wrapper Source

```python
class VisualAdjustStage(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, attn: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = shallow_full_visual_attention_adjust(attn, self.cfg)
        if VISUAL_ADJUST_KIND == "fold_tail_visual_mass_and_clear_region":
            return out["adjusted_attn"], out["tail_visual_sum"], out["cleared_visual_region"]
        return out["adjusted_attn"], out["cleared_visual_region"]
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

    non_text_to_all = attn[:, q_visual_start:, :]
    non_text_to_visual_before = attn[:, q_visual_start:, k_visual_start:k_visual_end]

    adjusted = attn.clone()
    if q_visual_start < adjusted.shape[-2] and k_visual_start < k_visual_end:
        adjusted[:, q_visual_start:, k_visual_start:k_visual_end] = 0.0
    cleared_visual_region = adjusted[:, q_visual_start:, k_visual_start:k_visual_end]

    result = {
        "non_text_to_all": non_text_to_all,
        "non_text_to_visual_before": non_text_to_visual_before,
        "cleared_visual_region": cleared_visual_region,
        "adjusted_attn": adjusted,
    }
    if (
        VISUAL_ADJUST_KIND == "fold_tail_visual_mass_and_clear_region"
        and q_tail_start < adjusted.shape[-2]
        and k_visual_start < k_visual_end
    ):
        tail_to_all = attn[:, q_tail_start:, :]
        tail_to_visual = attn[:, q_tail_start:, k_visual_start:k_visual_end]
        tail_visual_sum = tail_to_visual.sum(dim=-1)
        adjusted[:, q_tail_start:, k_visual_start] = tail_visual_sum
        copied_first_visual_slot = adjusted[:, q_tail_start:, k_visual_start] if k_visual_start < adjusted.shape[-1] else torch.empty(0, dtype=attn.dtype)
        result.update({
            "tail_to_all": tail_to_all,
            "tail_to_visual": tail_to_visual,
            "tail_visual_sum": tail_visual_sum,
            "copied_first_visual_slot": copied_first_visual_slot,
        })

    return result
```
