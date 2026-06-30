# input2_layer28 `05_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer28/onnx/05_attention.onnx`
- Stage: `attention`
- Stage title: Scaled Dot-Product Attention
- ONNX nodes: `6`
- ONNX initializers: `0`

### ONNX Inputs

- `q_rope`: `[4, 1, 8]`
- `k_heads`: `[4, 16, 8]`
- `attention_mask`: `[1, 1, 16]`

### ONNX Outputs

- `raw_scores`: `[4, 1, 16]`
- `masked_scores`: `[4, 1, 16]`
- `attn`: `[4, 1, 16]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00002687', 't00002689', 't00002691', 't00002703', 't00002708', 't00002710', 't00002715', 't00002717', 't00002722', 't00002718']`
- Dispatch output tensor ids: `['t00002688', 't00002690', 't00002692', 't00002716', 't00002724', 't00002726']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002709', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002719', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002720', 'consumer_event_op_index': 51, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00002721', 'consumer_event_op_index': 52, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002723', 'consumer_event_op_index': 53, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00002725', 'consumer_event_op_index': 56, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `05_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00002687']` outputs=`['t00002688']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002689']` outputs=`['t00002690']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002691']` outputs=`['t00002692']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002703', 't00002708']` outputs=`['t00002709']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002710', 't00002715']` outputs=`['t00002716']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002717']` outputs=`['t00002719']` -> shape=[1, 32, 128, 49], dtype=float16
- `#50 matmul.default` inputs=`['t00002709', 't00002719']` outputs=`['t00002720']` -> shape=[1, 32, 1, 49], dtype=float16
- `#51 div.Tensor` inputs=`['t00002720']` outputs=`['t00002721']` -> shape=[1, 32, 1, 49], dtype=float16
- `#52 add.Tensor` inputs=`['t00002721', 't00002722']` outputs=`['t00002723']` -> shape=[1, 32, 1, 49], dtype=float16
- `#53 softmax.int` inputs=`['t00002723']` outputs=`['t00002724']` -> shape=[1, 32, 1, 49], dtype=float32
- `#55 dropout.default` inputs=`['t00002725']` outputs=`['t00002725']` -> shape=[1, 32, 1, 49], dtype=float16
- `#56 matmul.default` inputs=`['t00002725', 't00002718']` outputs=`['t00002726']` -> shape=[1, 32, 1, 128], dtype=float16

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
