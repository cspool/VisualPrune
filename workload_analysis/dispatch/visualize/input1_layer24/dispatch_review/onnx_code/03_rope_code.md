# input1_layer24 `03_rope.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer24/onnx/03_rope.onnx`
- Stage: `rope`
- Stage title: RoPE
- ONNX nodes: `41`
- ONNX initializers: `2`

### ONNX Inputs

- `q_heads`: `[4, 16, 8]`
- `k_heads`: `[4, 16, 8]`
- `position_ids`: `[16]`

### ONNX Outputs

- `q_rope`: `[4, 16, 8]`
- `k_current_rope`: `[4, 16, 8]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00001988', 't00001991', 't00001993', 't00001475', 't00001982']`
- Dispatch output tensor ids: `['t00001989', 't00002005']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00001992', 'consumer_event_op_index': 29, 'consumer_op_name': 'index.Tensor'}, {'tensor_id': 't00001995', 'consumer_event_op_index': 30, 'consumer_op_name': 'unsqueeze.default'}, {'tensor_id': 't00001994', 'consumer_event_op_index': 31, 'consumer_op_name': 'index.Tensor'}, {'tensor_id': 't00001997', 'consumer_event_op_index': 32, 'consumer_op_name': 'unsqueeze.default'}, {'tensor_id': 't00001996', 'consumer_event_op_index': 33, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00002001', 'consumer_event_op_index': 36, 'consumer_op_name': 'neg.default'}, {'tensor_id': 't00002002', 'consumer_event_op_index': 37, 'consumer_op_name': 'cat.default'}, {'tensor_id': 't00002000', 'consumer_event_op_index': 37, 'consumer_op_name': 'cat.default'}, {'tensor_id': 't00002003', 'consumer_event_op_index': 38, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00001998', 'consumer_event_op_index': 38, 'consumer_op_name': 'mul.Tensor'}, {'tensor_id': 't00001999', 'consumer_event_op_index': 39, 'consumer_op_name': 'add.Tensor'}, {'tensor_id': 't00002004', 'consumer_event_op_index': 39, 'consumer_op_name': 'add.Tensor'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/export_stage_onnx.py::RoPEStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/rope.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/export_stage_onnx.py`

## Code Explanation

Gathers rotary-position cos/sin values and applies rotate-half RoPE to query and key heads.

## Review Comments

- This page binds `03_rope.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#20 add.Tensor` inputs=`['t00001988']` outputs=`['t00001989']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001991']` outputs=`['t00001992']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001993']` outputs=`['t00001994']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001992', 't00001475']` outputs=`['t00001995']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001995']` outputs=`['t00001996']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001994', 't00001475']` outputs=`['t00001997']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001997']` outputs=`['t00001998']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001982', 't00001996']` outputs=`['t00001999']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001982']` outputs=`['t00002000']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001982']` outputs=`['t00002001']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002001']` outputs=`['t00002002']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002002', 't00002000']` outputs=`['t00002003']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002003', 't00001998']` outputs=`['t00002004']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001999', 't00002004']` outputs=`['t00002005']` -> shape=[1, 32, 58, 128], dtype=float16

## Export Wrapper Source

```python
class RoPEStage(nn.Module):
    def __init__(self, cfg: FlowConfig, rope_cache: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("cos_cached", rope_cache["cos_cached"])
        self.register_buffer("sin_cached", rope_cache["sin_cached"])

    def forward(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_cache = {
            "cos_cached": self.cos_cached,
            "sin_cached": self.sin_cached,
        }
        out = apply_rope(q_heads, k_heads, position_ids, rope_cache, self.cfg)
        return out["q_rope"], out["k_rope"]
```

## Primary Source: `rope.py`

```python
from __future__ import annotations

import torch

from config import FlowConfig


def rotate_half(x: torch.Tensor, cfg: FlowConfig) -> torch.Tensor:
    left = x[..., : cfg.half_dim]
    right = x[..., cfg.half_dim :]
    return torch.cat((-right, left), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    rope_cache: dict[str, torch.Tensor],
    cfg: FlowConfig,
) -> dict[str, torch.Tensor]:
    flattened_position_ids = position_ids.reshape(-1)
    cos = rope_cache["cos_cached"].index_select(0, flattened_position_ids)
    sin = rope_cache["sin_cached"].index_select(0, flattened_position_ids)
    cos_for_heads = cos.unsqueeze(0)
    sin_for_heads = sin.unsqueeze(0)
    q_rot = rotate_half(q, cfg)
    k_rot = rotate_half(k, cfg)
    q_rope = q * cos_for_heads + q_rot * sin_for_heads
    k_rope = k * cos_for_heads + k_rot * sin_for_heads
    return {
        "position_ids_flat": flattened_position_ids,
        "cos": cos,
        "sin": sin,
        "cos_for_heads": cos_for_heads,
        "sin_for_heads": sin_for_heads,
        "q_rot": q_rot,
        "k_rot": k_rot,
        "q_rope": q_rope,
        "k_rope": k_rope,
    }
```
