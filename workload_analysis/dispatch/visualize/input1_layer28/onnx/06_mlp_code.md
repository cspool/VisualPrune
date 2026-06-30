# input1_layer28 `06_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer28/onnx/06_mlp.onnx`
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

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00002436', 't00000057', 't00002439', 't00002433', 't00002442', 't00002374', 't00002452', 't00002454']`
- Dispatch output tensor ids: `['t00002437', 't00002438', 't00002440', 't00002455']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002443', 'consumer_event_op_index': 69, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002444', 'consumer_event_op_index': 70, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002445', 'consumer_event_op_index': 71, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00002446', 'consumer_event_op_index': 72, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00002447', 'consumer_event_op_index': 73, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002448', 'consumer_event_op_index': 74, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00002445', 'consumer_event_op_index': 75, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002449', 'consumer_event_op_index': 75, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002450', 'consumer_event_op_index': 76, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002451', 'consumer_event_op_index': 77, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002453', 'consumer_event_op_index': 78, 'consumer_op_name': 'linear.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `06_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#62 add.Tensor` inputs=`['t00002436']` outputs=`['t00002437']` -> shape=[], dtype=int64
- `#63 mul.Tensor` inputs=`['t00000057']` outputs=`['t00002438']` -> shape=[], dtype=int64
- `#65 add.Tensor` inputs=`['t00002439']` outputs=`['t00002440']` -> shape=[], dtype=int64
- `#68 linear.default` inputs=`['t00002433', 't00002442']` outputs=`['t00002443']` -> shape=[1, 48, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00002374', 't00002443']` outputs=`['t00002444']` -> shape=[1, 48, 4096], dtype=float16
- `#70 to.dtype` inputs=`['t00002444']` outputs=`['t00002445']` -> shape=[1, 48, 4096], dtype=float32
- `#71 pow.Tensor_Scalar` inputs=`['t00002445']` outputs=`['t00002446']` -> shape=[1, 48, 4096], dtype=float32
- `#72 mean.dim` inputs=`['t00002446']` outputs=`['t00002447']` -> shape=[1, 48, 1], dtype=float32
- `#73 add.Tensor` inputs=`['t00002447']` outputs=`['t00002448']` -> shape=[1, 48, 1], dtype=float32
- `#74 rsqrt.default` inputs=`['t00002448']` outputs=`['t00002449']` -> shape=[1, 48, 1], dtype=float32
- `#75 mul.Tensor` inputs=`['t00002445', 't00002449']` outputs=`['t00002450']` -> shape=[1, 48, 4096], dtype=float32
- `#76 to.dtype` inputs=`['t00002450']` outputs=`['t00002451']` -> shape=[1, 48, 4096], dtype=float16
- `#77 mul.Tensor` inputs=`['t00002452', 't00002451']` outputs=`['t00002453']` -> shape=[1, 48, 4096], dtype=float16
- `#78 linear.default` inputs=`['t00002453', 't00002454']` outputs=`['t00002455']` -> shape=[1, 48, 11008], dtype=float16

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
