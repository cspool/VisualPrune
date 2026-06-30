# input1_layer11 `04_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer11/onnx/04_attention.onnx`
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

- Dispatch input tensor ids: `['t00000669', 't00000671', 't00000673', 't00000687', 't00000692', 't00000694', 't00000699', 't00000053']`
- Dispatch output tensor ids: `['t00000670', 't00000672', 't00000705', 't00000707']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000700', 'consumer_event_op_index': 47, 'consumer_op_name': 'transpose.int'}, {'tensor_id': 't00000693', 'consumer_event_op_index': 48, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00000701', 'consumer_event_op_index': 48, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00000702', 'consumer_event_op_index': 49, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00000703', 'consumer_event_op_index': 50, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000704', 'consumer_event_op_index': 51, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00000706', 'consumer_event_op_index': 54, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00000674', 'consumer_event_op_index': 54, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `04_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00000669']` outputs=`['t00000670']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000671']` outputs=`['t00000672']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000673']` outputs=`['t00000674']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000687', 't00000692']` outputs=`['t00000693']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000694', 't00000699']` outputs=`['t00000700']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000700']` outputs=`['t00000701']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000693', 't00000701']` outputs=`['t00000702']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000702']` outputs=`['t00000703']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000703', 't00000053']` outputs=`['t00000704']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000704']` outputs=`['t00000705']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000706']` outputs=`['t00000706']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000706', 't00000674']` outputs=`['t00000707']` -> shape=[1, 32, 624, 128], dtype=float16

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
