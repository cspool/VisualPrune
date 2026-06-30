# input2_layer19 `07_mlp.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input2_layer19/onnx/07_mlp.onnx`
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

- Dispatch input tensor ids: `['t00002583', 't00002587', 't00001537', 't00002534', 't00001547', 't00001549', 't00001552']`
- Dispatch output tensor ids: `['t00002584', 't00002600', 't00002601']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00002589', 'consumer_event_op_index': 62, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002590', 'consumer_event_op_index': 63, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002591', 'consumer_event_op_index': 64, 'consumer_op_name': 'pow.Tensor_Scalar'}, {'tensor_id': 't00002592', 'consumer_event_op_index': 65, 'consumer_op_name': 'mean.dim'}, {'tensor_id': 't00002593', 'consumer_event_op_index': 66, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002594', 'consumer_event_op_index': 67, 'consumer_op_name': 'rsqrt.default'}, {'tensor_id': 't00002591', 'consumer_event_op_index': 68, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002595', 'consumer_event_op_index': 68, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002596', 'consumer_event_op_index': 69, 'consumer_op_name': 'to.dtype'}, {'tensor_id': 't00002597', 'consumer_event_op_index': 70, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002598', 'consumer_event_op_index': 71, 'consumer_op_name': 'linear.default'}, {'tensor_id': 't00002599', 'consumer_event_op_index': 72, 'consumer_op_name': 'silu.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py::MLPStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/mlp.py`
- Primary implementation: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/rmsnorm.py`
- Support files: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/export_stage_onnx.py`

## Code Explanation

Runs post-attention RMSNorm, gated SiLU MLP, down projection, and residual addition.

## Review Comments

- This page binds `07_mlp.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#54 to.dtype` inputs=`['t00002583']` outputs=`['t00002584']` -> shape=[1, 32, 1, 59], dtype=float16
- `#61 linear.default` inputs=`['t00002587', 't00001537']` outputs=`['t00002589']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002534', 't00002589']` outputs=`['t00002590']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002590']` outputs=`['t00002591']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002591']` outputs=`['t00002592']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002592']` outputs=`['t00002593']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002593']` outputs=`['t00002594']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002594']` outputs=`['t00002595']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002591', 't00002595']` outputs=`['t00002596']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002596']` outputs=`['t00002597']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00001547', 't00002597']` outputs=`['t00002598']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002598', 't00001549']` outputs=`['t00002599']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002599']` outputs=`['t00002600']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002598', 't00001552']` outputs=`['t00002601']` -> shape=[1, 1, 11008], dtype=float16

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
