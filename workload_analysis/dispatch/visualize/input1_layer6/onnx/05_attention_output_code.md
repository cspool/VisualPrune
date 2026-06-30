# input1_layer6 `05_attention_output.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer6/onnx/05_attention_output.onnx`
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

- Dispatch input tensor ids: `['t00000231', 't00000199', 't00000233', 't00000237', 't00000178', 't00000254', 't00000255']`
- Dispatch output tensor ids: `['t00000232', 't00000257']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000234', 'consumer_event_op_index': 57, 'consumer_op_name': 'reshape.default'}, {'tensor_id': 't00000235', 'consumer_event_op_index': 60, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00000238', 'consumer_event_op_index': 61, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000239', 'consumer_event_op_index': 75, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000256', 'consumer_event_op_index': 75, 'consumer_op_name': 'add.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/export_stage_onnx.py::AttentionOutputStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/attention_output.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/export_stage_onnx.py`

## Code Explanation

Multiplies attention probabilities by value heads, merges heads back to hidden size, applies output projection, and adds the residual.

## Review Comments

- This page binds `05_attention_output.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#54 matmul.default` inputs=`['t00000231', 't00000199']` outputs=`['t00000232']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000233']` outputs=`['t00000234']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000234']` outputs=`['t00000235']` -> shape=[1, 624, 4096], dtype=float16
- `#60 linear.default` inputs=`['t00000235', 't00000237']` outputs=`['t00000238']` -> shape=[1, 624, 4096], dtype=float16
- `#61 add.Tensor` inputs=`['t00000178', 't00000238']` outputs=`['t00000239']` -> shape=[1, 624, 4096], dtype=float16
- `#74 linear.default` inputs=`['t00000254', 't00000255']` outputs=`['t00000256']` -> shape=[1, 624, 4096], dtype=float16
- `#75 add.Tensor` inputs=`['t00000239', 't00000256']` outputs=`['t00000257']` -> shape=[1, 624, 4096], dtype=float16

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
