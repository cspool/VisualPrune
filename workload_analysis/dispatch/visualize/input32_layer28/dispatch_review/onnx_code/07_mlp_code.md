# input32_layer28 `07_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/onnx/07_mlp.onnx`
- Stage: `mlp`
- Stage title: MLP
- ONNX nodes: `20`
- ONNX initializers: `4`

### ONNX Inputs

- `after_attn`: `[1, 32]`

### ONNX Outputs

- `gated`: `[1, 64]`
- `ffn_out`: `[1, 32]`
- `output`: `[1, 32]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00003099', 't00003103', 't00002442', 't00003048', 't00002452', 't00002454', 't00002457']`
- Dispatch output tensor ids: `['t00003100', 't00003116', 't00003117']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00003105', 'consumer_event_op_index': 62, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00003106', 'consumer_event_op_index': 63, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00003107', 'consumer_event_op_index': 64, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00003108', 'consumer_event_op_index': 65, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00003109', 'consumer_event_op_index': 66, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00003110', 'consumer_event_op_index': 67, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00003107', 'consumer_event_op_index': 68, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00003111', 'consumer_event_op_index': 68, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00003112', 'consumer_event_op_index': 69, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00003113', 'consumer_event_op_index': 70, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00003114', 'consumer_event_op_index': 71, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00003115', 'consumer_event_op_index': 72, 'consumer_op_name': 'silu.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `07_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#54 to.dtype` inputs=`['t00003099']` outputs=`['t00003100']` -> shape=[1, 32, 1, 79], dtype=float16
- `#61 linear.default` inputs=`['t00003103', 't00002442']` outputs=`['t00003105']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00003048', 't00003105']` outputs=`['t00003106']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00003106']` outputs=`['t00003107']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00003107']` outputs=`['t00003108']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00003108']` outputs=`['t00003109']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00003109']` outputs=`['t00003110']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00003110']` outputs=`['t00003111']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00003107', 't00003111']` outputs=`['t00003112']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00003112']` outputs=`['t00003113']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002452', 't00003113']` outputs=`['t00003114']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00003114', 't00002454']` outputs=`['t00003115']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00003115']` outputs=`['t00003116']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00003114', 't00002457']` outputs=`['t00003117']` -> shape=[1, 1, 11008], dtype=float16

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
