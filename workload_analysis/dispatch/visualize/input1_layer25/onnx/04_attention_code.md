# input1_layer25 `04_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer25/onnx/04_attention.onnx`
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

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00002083', 't00002085', 't00002087', 't00002101', 't00002106', 't00002108', 't00002113', 't00001505']`
- Dispatch output tensor ids: `['t00002084', 't00002086', 't00002119', 't00002121']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002114', 'consumer_event_op_index': 47, 'consumer_op_name': 'transpose.int'}, {'tensor_id': 't00002107', 'consumer_event_op_index': 48, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002115', 'consumer_event_op_index': 48, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002116', 'consumer_event_op_index': 49, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00002117', 'consumer_event_op_index': 50, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002118', 'consumer_event_op_index': 51, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00002120', 'consumer_event_op_index': 54, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002088', 'consumer_event_op_index': 54, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `04_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00002083']` outputs=`['t00002084']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002085']` outputs=`['t00002086']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002087']` outputs=`['t00002088']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002101', 't00002106']` outputs=`['t00002107']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002108', 't00002113']` outputs=`['t00002114']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002114']` outputs=`['t00002115']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00002107', 't00002115']` outputs=`['t00002116']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00002116']` outputs=`['t00002117']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00002117', 't00001505']` outputs=`['t00002118']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00002118']` outputs=`['t00002119']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00002120']` outputs=`['t00002120']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00002120', 't00002088']` outputs=`['t00002121']` -> shape=[1, 32, 58, 128], dtype=float16

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
