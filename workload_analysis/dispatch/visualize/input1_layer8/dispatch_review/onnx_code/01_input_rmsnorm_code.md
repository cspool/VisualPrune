# input1_layer8 `01_input_rmsnorm.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/onnx/01_input_rmsnorm.onnx`
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

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/torch_flow/export_stage_onnx.py::InputRMSNormStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/torch_flow/rmsnorm.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer8/torch_flow/export_stage_onnx.py`

## Code Explanation

Normalizes the incoming hidden states with RMSNorm and exposes variance/inverse-RMS tensors for inspection.

## Review Comments

- This page binds `01_input_rmsnorm.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#1 to.dtype` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` -> shape=[1, 624, 4096], dtype=float16

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
