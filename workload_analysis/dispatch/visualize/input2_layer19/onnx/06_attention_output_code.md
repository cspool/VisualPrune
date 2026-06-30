# input2_layer19 `06_attention_output.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer19/onnx/06_attention_output.onnx`
- Stage: `attention_output`
- Stage title: Attention Output
- ONNX nodes: `11`
- ONNX initializers: `1`

### ONNX Inputs

- `adjusted_attn`: `[4, 1, 16]`
- `v_heads`: `[4, 16, 8]`
- `hidden_states`: `[1, 32]`

### ONNX Outputs

- `context`: `[1, 32]`
- `attn_out`: `[1, 32]`
- `after_attn`: `[1, 32]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00002584', 't00002577', 't00002586', 't00001537', 't00002534', 't00002602', 't00001555']`
- Dispatch output tensor ids: `['t00002585', 't00002604']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002587', 'consumer_event_op_index': 61, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00002589', 'consumer_event_op_index': 62, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002590', 'consumer_event_op_index': 76, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002603', 'consumer_event_op_index': 76, 'consumer_op_name': 'add.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py::AttentionOutputStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/attention_output.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py`

## Code Explanation

Multiplies attention probabilities by value heads, merges heads back to hidden size, applies output projection, and adds the residual.

## Review Comments

- This page binds `06_attention_output.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#56 matmul.default` inputs=`['t00002584', 't00002577']` outputs=`['t00002585']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002586']` outputs=`['t00002587']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002587', 't00001537']` outputs=`['t00002589']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002534', 't00002589']` outputs=`['t00002590']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002602', 't00001555']` outputs=`['t00002603']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002590', 't00002603']` outputs=`['t00002604']` -> shape=[1, 1, 4096], dtype=float16

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
