# input1_layer5 `04_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/onnx/04_attention.onnx`
- Stage: `attention`
- Stage title: Scaled Dot-Product Attention
- ONNX nodes: `6`
- ONNX initializers: `0`

### ONNX Inputs

- `q_rope`: `[4, 16, 8]`
- `k_heads`: `[4, 16, 8]`
- `attention_mask`: `[1, 16, 16]`

### ONNX Outputs

- `raw_scores`: `[4, 16, 16]`
- `masked_scores`: `[4, 16, 16]`
- `attn`: `[4, 16, 16]`

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/attention.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `04_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` -> shape=[1, 32, 624, 624], dtype=float32
- `#61 dropout.default` -> shape=[1, 32, 624, 624], dtype=float16
- `#62 matmul.default` -> shape=[1, 32, 624, 128], dtype=float16

## Export Wrapper Source

```python
class AttentionStage(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = scaled_dot_product_attention(q_rope, k_rope, attention_mask, self.cfg)
        return out["raw_scores"], out["masked_scores"], out["attn"]
```

## Primary Source: `attention.py`

```python
from __future__ import annotations

import torch

from config import FlowConfig


def scaled_dot_product_attention(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: FlowConfig,
) -> dict[str, torch.Tensor]:
    k_t = k_rope.transpose(-2, -1)
    raw_scores = torch.matmul(q_rope, k_t)
    scaled_scores = raw_scores / (cfg.head_dim ** 0.5)
    masked_scores = scaled_scores + attention_mask
    attn = torch.softmax(masked_scores, dim=-1)
    return {
        "k_transposed": k_t,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "masked_scores": masked_scores,
        "attn": attn,
    }
```
