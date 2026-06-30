# input1_layer10 `01_input_rmsnorm.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer10/onnx/01_input_rmsnorm.onnx`
- Stage: `input_rmsnorm`
- Stage title: Input RMSNorm
- ONNX nodes: `10`
- ONNX initializers: `1`

### ONNX Inputs

- `hidden_states`: `[16, 32]`

### ONNX Outputs

- `x_norm`: `[16, 32]`
- `variance`: `[16, 1]`
- `inv_rms`: `[16, 1]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00000554', 't00000562']`
- Dispatch output tensor ids: `['t00000563']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000555', 'consumer_event_op_index': 2, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00000556', 'consumer_event_op_index': 3, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00000557', 'consumer_event_op_index': 4, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000558', 'consumer_event_op_index': 5, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00000555', 'consumer_event_op_index': 6, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000559', 'consumer_event_op_index': 6, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000560', 'consumer_event_op_index': 7, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00000561', 'consumer_event_op_index': 8, 'consumer_op_name': 'mul.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/export_stage_onnx.py::InputRMSNormStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/export_stage_onnx.py`

## Code Explanation

Normalizes the incoming hidden states with RMSNorm and exposes variance/inverse-RMS tensors for inspection.

## Review Comments

- This page binds `01_input_rmsnorm.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#1 to.dtype` inputs=`['t00000554']` outputs=`['t00000555']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000555']` outputs=`['t00000556']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000556']` outputs=`['t00000557']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000557']` outputs=`['t00000558']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000558']` outputs=`['t00000559']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000555', 't00000559']` outputs=`['t00000560']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000560']` outputs=`['t00000561']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000562', 't00000561']` outputs=`['t00000563']` -> shape=[1, 624, 4096], dtype=float16

## Export Wrapper Source

```python
class InputRMSNormStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_norm_weight", weights["input_norm_weight"])

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = rms_norm(hidden_states, self.input_norm_weight, self.cfg, "input_norm")
        return out["input_norm_output"], out["input_norm_variance"], out["input_norm_inv_rms"]
```

## Primary Source: `rmsnorm.py`

```python
from __future__ import annotations

import torch

from config import FlowConfig


def rms_norm(x: torch.Tensor, weight: torch.Tensor, cfg: FlowConfig, prefix: str) -> dict[str, torch.Tensor]:
    x_f32 = x.to(torch.float32)
    squared = x_f32 * x_f32
    variance = squared.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + cfg.eps)
    normalized = x_f32 * inv_rms
    output = normalized * weight
    return {
        f"{prefix}_x_f32": x_f32,
        f"{prefix}_squared": squared,
        f"{prefix}_variance": variance,
        f"{prefix}_inv_rms": inv_rms,
        f"{prefix}_normalized": normalized,
        f"{prefix}_output": output,
    }
```
