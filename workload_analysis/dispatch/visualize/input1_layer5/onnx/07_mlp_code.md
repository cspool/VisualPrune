# input1_layer5 `07_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/onnx/07_mlp.onnx`
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

- Dispatch input tensor ids: `['t00000156', 't00000158', 't00000094', 't00000168', 't00000170', 't00000173']`
- Dispatch output tensor ids: `['t00000175']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000159', 'consumer_event_op_index': 69, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000160', 'consumer_event_op_index': 70, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00000161', 'consumer_event_op_index': 71, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00000162', 'consumer_event_op_index': 72, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00000163', 'consumer_event_op_index': 73, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000164', 'consumer_event_op_index': 74, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00000161', 'consumer_event_op_index': 75, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000165', 'consumer_event_op_index': 75, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000166', 'consumer_event_op_index': 76, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00000167', 'consumer_event_op_index': 77, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000169', 'consumer_event_op_index': 78, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00000171', 'consumer_event_op_index': 79, 'consumer_op_name': 'silu.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `07_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#68 linear.default` inputs=`['t00000156', 't00000158']` outputs=`['t00000159']` -> shape=[1, 624, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00000094', 't00000159']` outputs=`['t00000160']` -> shape=[1, 624, 4096], dtype=float16
- `#70 to.dtype` inputs=`['t00000160']` outputs=`['t00000161']` -> shape=[1, 624, 4096], dtype=float32
- `#71 pow.Tensor_Scalar` inputs=`['t00000161']` outputs=`['t00000162']` -> shape=[1, 624, 4096], dtype=float32
- `#72 mean.dim` inputs=`['t00000162']` outputs=`['t00000163']` -> shape=[1, 624, 1], dtype=float32
- `#73 add.Tensor` inputs=`['t00000163']` outputs=`['t00000164']` -> shape=[1, 624, 1], dtype=float32
- `#74 rsqrt.default` inputs=`['t00000164']` outputs=`['t00000165']` -> shape=[1, 624, 1], dtype=float32
- `#75 mul.Tensor` inputs=`['t00000161', 't00000165']` outputs=`['t00000166']` -> shape=[1, 624, 4096], dtype=float32
- `#76 to.dtype` inputs=`['t00000166']` outputs=`['t00000167']` -> shape=[1, 624, 4096], dtype=float16
- `#77 mul.Tensor` inputs=`['t00000168', 't00000167']` outputs=`['t00000169']` -> shape=[1, 624, 4096], dtype=float16
- `#78 linear.default` inputs=`['t00000169', 't00000170']` outputs=`['t00000171']` -> shape=[1, 624, 11008], dtype=float16
- `#79 silu.default` inputs=`['t00000171']` outputs=`['t00000172']` -> shape=[1, 624, 11008], dtype=float16
- `#80 linear.default` inputs=`['t00000169', 't00000173']` outputs=`['t00000174']` -> shape=[1, 624, 11008], dtype=float16
- `#81 mul.Tensor` inputs=`['t00000172', 't00000174']` outputs=`['t00000175']` -> shape=[1, 624, 11008], dtype=float16

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
