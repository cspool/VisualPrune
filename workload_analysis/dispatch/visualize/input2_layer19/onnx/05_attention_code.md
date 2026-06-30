# input2_layer19 `05_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer19/onnx/05_attention.onnx`
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

- Dispatch input tensor ids: `['t00002546', 't00002548', 't00002550', 't00002562', 't00002567', 't00002569', 't00002574', 't00002576', 't00002581', 't00002577']`
- Dispatch output tensor ids: `['t00002547', 't00002549', 't00002551', 't00002575', 't00002583', 't00002585']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002568', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002578', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002579', 'consumer_event_op_index': 51, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00002580', 'consumer_event_op_index': 52, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002582', 'consumer_event_op_index': 53, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00002584', 'consumer_event_op_index': 56, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `05_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00002546']` outputs=`['t00002547']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002548']` outputs=`['t00002549']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002550']` outputs=`['t00002551']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002562', 't00002567']` outputs=`['t00002568']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002569', 't00002574']` outputs=`['t00002575']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002576']` outputs=`['t00002578']` -> shape=[1, 32, 128, 59], dtype=float16
- `#50 matmul.default` inputs=`['t00002568', 't00002578']` outputs=`['t00002579']` -> shape=[1, 32, 1, 59], dtype=float16
- `#51 div.Tensor` inputs=`['t00002579']` outputs=`['t00002580']` -> shape=[1, 32, 1, 59], dtype=float16
- `#52 add.Tensor` inputs=`['t00002580', 't00002581']` outputs=`['t00002582']` -> shape=[1, 32, 1, 59], dtype=float16
- `#53 softmax.int` inputs=`['t00002582']` outputs=`['t00002583']` -> shape=[1, 32, 1, 59], dtype=float32
- `#55 dropout.default` inputs=`['t00002584']` outputs=`['t00002584']` -> shape=[1, 32, 1, 59], dtype=float16
- `#56 matmul.default` inputs=`['t00002584', 't00002577']` outputs=`['t00002585']` -> shape=[1, 32, 1, 128], dtype=float16

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
