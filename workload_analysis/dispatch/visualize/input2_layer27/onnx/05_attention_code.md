# input2_layer27 `05_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer27/onnx/05_attention.onnx`
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

- Dispatch input tensor ids: `['t00002617', 't00002619', 't00002621', 't00002633', 't00002638', 't00002640', 't00002645', 't00002647', 't00002652', 't00002648']`
- Dispatch output tensor ids: `['t00002618', 't00002620', 't00002622', 't00002646', 't00002654', 't00002656']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002639', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002649', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002650', 'consumer_event_op_index': 51, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00002651', 'consumer_event_op_index': 52, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002653', 'consumer_event_op_index': 53, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00002655', 'consumer_event_op_index': 56, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `05_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00002617']` outputs=`['t00002618']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002619']` outputs=`['t00002620']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002621']` outputs=`['t00002622']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002633', 't00002638']` outputs=`['t00002639']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002640', 't00002645']` outputs=`['t00002646']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002647']` outputs=`['t00002649']` -> shape=[1, 32, 128, 59], dtype=float16
- `#50 matmul.default` inputs=`['t00002639', 't00002649']` outputs=`['t00002650']` -> shape=[1, 32, 1, 59], dtype=float16
- `#51 div.Tensor` inputs=`['t00002650']` outputs=`['t00002651']` -> shape=[1, 32, 1, 59], dtype=float16
- `#52 add.Tensor` inputs=`['t00002651', 't00002652']` outputs=`['t00002653']` -> shape=[1, 32, 1, 59], dtype=float16
- `#53 softmax.int` inputs=`['t00002653']` outputs=`['t00002654']` -> shape=[1, 32, 1, 59], dtype=float32
- `#55 dropout.default` inputs=`['t00002655']` outputs=`['t00002655']` -> shape=[1, 32, 1, 59], dtype=float16
- `#56 matmul.default` inputs=`['t00002655', 't00002648']` outputs=`['t00002656']` -> shape=[1, 32, 1, 128], dtype=float16

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
