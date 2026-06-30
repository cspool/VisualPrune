# input1_layer0 `06_attention_output.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer0/onnx/06_attention_output.onnx`
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

- Dispatch input tensor ids: `['t00000056', 't00000022', 't00000069', 't00000073', 't00000001', 't00000090', 't00000091']`
- Dispatch output tensor ids: `['t00000068', 't00000093']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000070', 'consumer_event_op_index': 73, 'consumer_op_name': 'reshape.default'}, {'tensor_id': 't00000071', 'consumer_event_op_index': 76, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00000074', 'consumer_event_op_index': 77, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000075', 'consumer_event_op_index': 91, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000092', 'consumer_event_op_index': 91, 'consumer_op_name': 'add.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/export_stage_onnx.py::AttentionOutputStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/attention_output.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/export_stage_onnx.py`

## Code Explanation

Multiplies attention probabilities by value heads, merges heads back to hidden size, applies output projection, and adds the residual.

## Review Comments

- This page binds `06_attention_output.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#70 matmul.default` inputs=`['t00000056', 't00000022']` outputs=`['t00000068']` -> shape=[1, 32, 624, 128], dtype=float16
- `#72 contiguous.default` inputs=`['t00000069']` outputs=`['t00000070']` -> shape=[1, 624, 32, 128], dtype=float16
- `#73 reshape.default` inputs=`['t00000070']` outputs=`['t00000071']` -> shape=[1, 624, 4096], dtype=float16
- `#76 linear.default` inputs=`['t00000071', 't00000073']` outputs=`['t00000074']` -> shape=[1, 624, 4096], dtype=float16
- `#77 add.Tensor` inputs=`['t00000001', 't00000074']` outputs=`['t00000075']` -> shape=[1, 624, 4096], dtype=float16
- `#90 linear.default` inputs=`['t00000090', 't00000091']` outputs=`['t00000092']` -> shape=[1, 624, 4096], dtype=float16
- `#91 add.Tensor` inputs=`['t00000075', 't00000092']` outputs=`['t00000093']` -> shape=[1, 624, 4096], dtype=float16

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
