# input2_layer31 `05_attention.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer31/onnx/05_attention.onnx`
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

- Dispatch input tensor ids: `['t00002762', 't00002764', 't00002766', 't00002780', 't00002785', 't00002787', 't00002792', 't00002795', 't00002801', 't00002797']`
- Dispatch output tensor ids: `['t00002763', 't00002765', 't00002767', 't00002793', 't00002803', 't00002805']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002786', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002798', 'consumer_event_op_index': 50, 'consumer_op_name': 'matmul.default'}, {'tensor_id': 't00002799', 'consumer_event_op_index': 51, 'consumer_op_name': 'div.Tensor'}, {'tensor_id': 't00002800', 'consumer_event_op_index': 52, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002802', 'consumer_event_op_index': 53, 'consumer_op_name': 'softmax.int'}, {'tensor_id': 't00002804', 'consumer_event_op_index': 56, 'consumer_op_name': 'matmul.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/export_stage_onnx.py::AttentionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/attention.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes rectangular or square attention scores, applies the mask, and materializes softmax attention probabilities.

## Review Comments

- This page binds `05_attention.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#13 transpose.int` inputs=`['t00002762']` outputs=`['t00002763']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002764']` outputs=`['t00002765']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002766']` outputs=`['t00002767']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002780', 't00002785']` outputs=`['t00002786']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002787', 't00002792']` outputs=`['t00002793']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002795']` outputs=`['t00002798']` -> shape=[1, 32, 128, 49], dtype=float16
- `#50 matmul.default` inputs=`['t00002786', 't00002798']` outputs=`['t00002799']` -> shape=[1, 32, 1, 49], dtype=float16
- `#51 div.Tensor` inputs=`['t00002799']` outputs=`['t00002800']` -> shape=[1, 32, 1, 49], dtype=float16
- `#52 add.Tensor` inputs=`['t00002800', 't00002801']` outputs=`['t00002802']` -> shape=[1, 32, 1, 49], dtype=float16
- `#53 softmax.int` inputs=`['t00002802']` outputs=`['t00002803']` -> shape=[1, 32, 1, 49], dtype=float32
- `#55 dropout.default` inputs=`['t00002804']` outputs=`['t00002804']` -> shape=[1, 32, 1, 49], dtype=float16
- `#56 matmul.default` inputs=`['t00002804', 't00002797']` outputs=`['t00002805']` -> shape=[1, 32, 1, 128], dtype=float16

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
