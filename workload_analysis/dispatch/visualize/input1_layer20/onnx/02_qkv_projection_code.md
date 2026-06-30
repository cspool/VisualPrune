# input1_layer20 `02_qkv_projection.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer20/onnx/02_qkv_projection.onnx`
- Stage: `qkv_projection`
- Stage title: Q/K/V Projection
- ONNX nodes: `29`
- ONNX initializers: `3`

### ONNX Inputs

- `x_norm`: `[16, 32]`

### ONNX Outputs

- `q_heads`: `[4, 16, 8]`
- `k_heads`: `[4, 16, 8]`
- `v_heads`: `[4, 16, 8]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00001566', 't00001567', 't00001569', 't00001571']`
- Dispatch output tensor ids: `['t00001574', 't00001576', 't00001578']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00001568', 'consumer_event_op_index': 12, 'consumer_op_name': 'view.default'}, {'tensor_id': 't00001573', 'consumer_event_op_index': 13, 'consumer_op_name': 'transpose.int'}, {'tensor_id': 't00001570', 'consumer_event_op_index': 14, 'consumer_op_name': 'view.default'}, {'tensor_id': 't00001575', 'consumer_event_op_index': 15, 'consumer_op_name': 'transpose.int'}, {'tensor_id': 't00001572', 'consumer_event_op_index': 16, 'consumer_op_name': 'view.default'}, {'tensor_id': 't00001577', 'consumer_event_op_index': 17, 'consumer_op_name': 'transpose.int'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/export_stage_onnx.py::QKVProjectionStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/qkv_projection.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/export_stage_onnx.py`

## Code Explanation

Applies the query/key/value linear projections and reshapes the projected hidden states into attention heads.

## Review Comments

- This page binds `02_qkv_projection.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#9 linear.default` inputs=`['t00001566', 't00001567']` outputs=`['t00001568']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001566', 't00001569']` outputs=`['t00001570']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001566', 't00001571']` outputs=`['t00001572']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001568']` outputs=`['t00001573']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001573']` outputs=`['t00001574']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001570']` outputs=`['t00001575']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001575']` outputs=`['t00001576']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001572']` outputs=`['t00001577']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001577']` outputs=`['t00001578']` -> shape=[1, 32, 58, 128], dtype=float16

## Export Wrapper Source

```python
class QKVProjectionStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("q_weight", weights["q_weight"])
        self.register_buffer("k_weight", weights["k_weight"])
        self.register_buffer("v_weight", weights["v_weight"])

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = {
            "q_weight": self.q_weight,
            "k_weight": self.k_weight,
            "v_weight": self.v_weight,
        }
        out = project_qkv(x_norm, weights, self.cfg)
        return out["q_heads"], out["k_heads"], out["v_heads"]
```

## Primary Source: `qkv_projection.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import FlowConfig


def split_heads(x: torch.Tensor, cfg: FlowConfig) -> torch.Tensor:
    return x.view(x.shape[0], cfg.heads, cfg.head_dim).transpose(0, 1)


def project_qkv(x_norm: torch.Tensor, weights: dict[str, torch.Tensor], cfg: FlowConfig) -> dict[str, torch.Tensor]:
    q_linear = F.linear(x_norm, weights["q_weight"])
    k_linear = F.linear(x_norm, weights["k_weight"])
    v_linear = F.linear(x_norm, weights["v_weight"])
    q = split_heads(q_linear, cfg)
    k = split_heads(k_linear, cfg)
    v = split_heads(v_linear, cfg)
    return {
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_heads": q,
        "k_heads": k,
        "v_heads": v,
    }
```
