# input1_layer27 `07_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer27/onnx/07_mlp.onnx`
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

- Dispatch input tensor ids: `['t00002328', 't00002353', 't00002271', 't00002363', 't00002365', 't00002368']`
- Dispatch output tensor ids: `['t00002370']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002354', 'consumer_event_op_index': 86, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002355', 'consumer_event_op_index': 87, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002356', 'consumer_event_op_index': 88, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00002357', 'consumer_event_op_index': 89, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00002358', 'consumer_event_op_index': 90, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002359', 'consumer_event_op_index': 91, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00002356', 'consumer_event_op_index': 92, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002360', 'consumer_event_op_index': 92, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002361', 'consumer_event_op_index': 93, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002362', 'consumer_event_op_index': 94, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002364', 'consumer_event_op_index': 95, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00002366', 'consumer_event_op_index': 96, 'consumer_op_name': 'silu.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `07_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#85 linear.default` inputs=`['t00002328', 't00002353']` outputs=`['t00002354']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002271', 't00002354']` outputs=`['t00002355']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00002355']` outputs=`['t00002356']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00002356']` outputs=`['t00002357']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00002357']` outputs=`['t00002358']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00002358']` outputs=`['t00002359']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00002359']` outputs=`['t00002360']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00002356', 't00002360']` outputs=`['t00002361']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00002361']` outputs=`['t00002362']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00002363', 't00002362']` outputs=`['t00002364']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00002364', 't00002365']` outputs=`['t00002366']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00002366']` outputs=`['t00002367']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00002364', 't00002368']` outputs=`['t00002369']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00002367', 't00002369']` outputs=`['t00002370']` -> shape=[1, 58, 11008], dtype=float16

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
