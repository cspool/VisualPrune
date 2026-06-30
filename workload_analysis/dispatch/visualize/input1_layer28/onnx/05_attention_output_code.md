# input1_layer28 `05_attention_output.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer28/onnx/05_attention_output.onnx`
- Stage: `attention_output`
- Stage title: Attention Output
- ONNX nodes: `11`
- ONNX initializers: `1`

### ONNX Inputs

- `adjusted_attn`: `[4, 16, 16]`
- `v_heads`: `[4, 16, 8]`
- `hidden_states`: `[16, 32]`

### ONNX Outputs

- `context`: `[16, 32]`
- `attn_out`: `[16, 32]`
- `after_attn`: `[16, 32]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00002429', 't00002395', 't00002431', 't00002442', 't00002374', 't00002459', 't00002460']`
- Dispatch output tensor ids: `['t00002430', 't00002462']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002432', 'consumer_event_op_index': 57, 'consumer_op_name': 'reshape.default'}, {'tensor_id': 't00002433', 'consumer_event_op_index': 68, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00002443', 'consumer_event_op_index': 69, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002444', 'consumer_event_op_index': 83, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002461', 'consumer_event_op_index': 83, 'consumer_op_name': 'add.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/export_stage_onnx.py::AttentionOutputStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/attention_output.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/export_stage_onnx.py`

## Code Explanation

Multiplies attention probabilities by value heads, merges heads back to hidden size, applies output projection, and adds the residual.

## Review Comments

- This page binds `05_attention_output.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#54 matmul.default` inputs=`['t00002429', 't00002395']` outputs=`['t00002430']` -> shape=[1, 32, 48, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002431']` outputs=`['t00002432']` -> shape=[1, 48, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002432']` outputs=`['t00002433']` -> shape=[1, 48, 4096], dtype=float16
- `#68 linear.default` inputs=`['t00002433', 't00002442']` outputs=`['t00002443']` -> shape=[1, 48, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00002374', 't00002443']` outputs=`['t00002444']` -> shape=[1, 48, 4096], dtype=float16
- `#82 linear.default` inputs=`['t00002459', 't00002460']` outputs=`['t00002461']` -> shape=[1, 48, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00002444', 't00002461']` outputs=`['t00002462']` -> shape=[1, 48, 4096], dtype=float16

## Export Wrapper Source

```python
class AttentionOutputStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("o_weight", weights["o_weight"])

    def forward(
        self,
        adjusted_attn: torch.Tensor,
        v_heads: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = attention_output(
            adjusted_attn,
            v_heads,
            hidden_states,
            {"o_weight": self.o_weight},
            self.cfg,
        )
        return out["context"], out["attn_out"], out["after_attn"]
```

## Primary Source: `attention_output.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import FlowConfig


def attention_output(
    adjusted_attn: torch.Tensor,
    v_heads: torch.Tensor,
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cfg: FlowConfig,
) -> dict[str, torch.Tensor]:
    context_heads = torch.matmul(adjusted_attn, v_heads)
    context_transposed = context_heads.transpose(0, 1).contiguous()
    context = context_transposed.view(context_heads.shape[-2], cfg.hidden)
    attn_out = F.linear(context, weights["o_weight"])
    after_attn = hidden_states + attn_out
    return {
        "context_heads": context_heads,
        "context_transposed": context_transposed,
        "context": context,
        "attn_out": attn_out,
        "after_attn": after_attn,
    }
```
