# input1_layer6 `06_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer6/onnx/06_mlp.onnx`
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

- Dispatch input tensor ids: `['t00000230', 't00000235', 't00000237', 't00000178', 't00000247', 't00000249', 't00000252']`
- Dispatch output tensor ids: `['t00000231', 't00000251', 't00000253']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000238', 'consumer_event_op_index': 61, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000239', 'consumer_event_op_index': 62, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00000240', 'consumer_event_op_index': 63, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00000241', 'consumer_event_op_index': 64, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00000242', 'consumer_event_op_index': 65, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00000243', 'consumer_event_op_index': 66, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00000240', 'consumer_event_op_index': 67, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000244', 'consumer_event_op_index': 67, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000245', 'consumer_event_op_index': 68, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00000246', 'consumer_event_op_index': 69, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00000248', 'consumer_event_op_index': 70, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00000250', 'consumer_event_op_index': 71, 'consumer_op_name': 'silu.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `06_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#52 to.dtype` inputs=`['t00000230']` outputs=`['t00000231']` -> shape=[1, 32, 624, 624], dtype=float16
- `#60 linear.default` inputs=`['t00000235', 't00000237']` outputs=`['t00000238']` -> shape=[1, 624, 4096], dtype=float16
- `#61 add.Tensor` inputs=`['t00000178', 't00000238']` outputs=`['t00000239']` -> shape=[1, 624, 4096], dtype=float16
- `#62 to.dtype` inputs=`['t00000239']` outputs=`['t00000240']` -> shape=[1, 624, 4096], dtype=float32
- `#63 pow.Tensor_Scalar` inputs=`['t00000240']` outputs=`['t00000241']` -> shape=[1, 624, 4096], dtype=float32
- `#64 mean.dim` inputs=`['t00000241']` outputs=`['t00000242']` -> shape=[1, 624, 1], dtype=float32
- `#65 add.Tensor` inputs=`['t00000242']` outputs=`['t00000243']` -> shape=[1, 624, 1], dtype=float32
- `#66 rsqrt.default` inputs=`['t00000243']` outputs=`['t00000244']` -> shape=[1, 624, 1], dtype=float32
- `#67 mul.Tensor` inputs=`['t00000240', 't00000244']` outputs=`['t00000245']` -> shape=[1, 624, 4096], dtype=float32
- `#68 to.dtype` inputs=`['t00000245']` outputs=`['t00000246']` -> shape=[1, 624, 4096], dtype=float16
- `#69 mul.Tensor` inputs=`['t00000247', 't00000246']` outputs=`['t00000248']` -> shape=[1, 624, 4096], dtype=float16
- `#70 linear.default` inputs=`['t00000248', 't00000249']` outputs=`['t00000250']` -> shape=[1, 624, 11008], dtype=float16
- `#71 silu.default` inputs=`['t00000250']` outputs=`['t00000251']` -> shape=[1, 624, 11008], dtype=float16
- `#72 linear.default` inputs=`['t00000248', 't00000252']` outputs=`['t00000253']` -> shape=[1, 624, 11008], dtype=float16

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
