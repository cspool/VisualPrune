# input1_layer26 `01_input_rmsnorm.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer26/onnx/01_input_rmsnorm.onnx`
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

- Dispatch input tensor ids: `['t00002169', 't00002177']`
- Dispatch output tensor ids: `['t00002178']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002170', 'consumer_event_op_index': 2, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00002171', 'consumer_event_op_index': 3, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00002172', 'consumer_event_op_index': 4, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002173', 'consumer_event_op_index': 5, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00002170', 'consumer_event_op_index': 6, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002174', 'consumer_event_op_index': 6, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002175', 'consumer_event_op_index': 7, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002176', 'consumer_event_op_index': 8, 'consumer_op_name': 'mul.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/export_stage_onnx.py::InputRMSNormStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/export_stage_onnx.py`

## Code Explanation

Normalizes the incoming hidden states with RMSNorm and exposes variance/inverse-RMS tensors for inspection.

## Review Comments

- This page binds `01_input_rmsnorm.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#1 to.dtype` inputs=`['t00002169']` outputs=`['t00002170']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002170']` outputs=`['t00002171']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002171']` outputs=`['t00002172']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002172']` outputs=`['t00002173']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002173']` outputs=`['t00002174']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002170', 't00002174']` outputs=`['t00002175']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002175']` outputs=`['t00002176']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002177', 't00002176']` outputs=`['t00002178']` -> shape=[1, 58, 4096], dtype=float16

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
