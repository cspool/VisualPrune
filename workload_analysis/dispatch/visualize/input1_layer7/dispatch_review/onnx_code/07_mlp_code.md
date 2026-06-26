# input1_layer7 `07_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/onnx/07_mlp.onnx`
- Stage: `mlp`
- Stage title: MLP
- ONNX nodes: `20`
- ONNX initializers: `4`

### ONNX Inputs

- `after_attn`: `[16, 32]`

### ONNX Outputs

- `gated`: `[16, 64]`
- `ffn_out`: `[16, 32]`
- `output`: `[16, 32]`

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/mlp.py`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/rmsnorm.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer7/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `07_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#82 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` -> shape=[1, 624, 11008], dtype=float16

## Export Wrapper Source

```python
class MLPStage(nn.Module):
    def __init__(self, cfg: FlowConfig, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("post_norm_weight", weights["post_norm_weight"])
        self.register_buffer("gate_weight", weights["gate_weight"])
        self.register_buffer("up_weight", weights["up_weight"])
        self.register_buffer("down_weight", weights["down_weight"])

    def forward(self, after_attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = {
            "post_norm_weight": self.post_norm_weight,
            "gate_weight": self.gate_weight,
            "up_weight": self.up_weight,
            "down_weight": self.down_weight,
        }
        out = gated_mlp(after_attn, weights, self.cfg)
        return out["gated"], out["ffn_out"], out["output"]
```

## Primary Source: `mlp.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import FlowConfig
from rmsnorm import rms_norm


def gated_mlp(after_attn: torch.Tensor, weights: dict[str, torch.Tensor], cfg: FlowConfig) -> dict[str, torch.Tensor]:
    norm = rms_norm(after_attn, weights["post_norm_weight"], cfg, "post_norm")
    mlp_in = norm["post_norm_output"]
    gate_linear = F.linear(mlp_in, weights["gate_weight"])
    gate = F.silu(gate_linear)
    up = F.linear(mlp_in, weights["up_weight"])
    gated = gate * up
    ffn_out = F.linear(gated, weights["down_weight"])
    output = after_attn + ffn_out
    return {
        **norm,
        "gate_linear": gate_linear,
        "gate_silu": gate,
        "up": up,
        "gated": gated,
        "ffn_out": ffn_out,
        "output": output,
    }
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
