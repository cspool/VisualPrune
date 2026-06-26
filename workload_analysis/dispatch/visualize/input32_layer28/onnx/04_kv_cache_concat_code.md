# input32_layer28 `04_kv_cache_concat.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/onnx/04_kv_cache_concat.onnx`
- Stage: `kv_cache_concat`
- Stage title: K/V Cache Concatenation
- ONNX nodes: `2`
- ONNX initializers: `0`

### ONNX Inputs

- `k_current_rope`: `[4, 1, 8]`
- `v_current`: `[4, 1, 8]`
- `past_k`: `[4, 15, 8]`
- `past_v`: `[4, 15, 8]`

### ONNX Outputs

- `k_heads`: `[4, 16, 8]`
- `v_heads`: `[4, 16, 8]`

## Corresponding `torch_flow` Code

- Export wrapper: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/torch_flow/export_stage_onnx.py::KVCacheConcatStage`
- Primary implementation: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/torch_flow/kv_cache.py`
- Support files: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/torch_flow/config.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/torch_flow/init_data.py`, `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer28/torch_flow/export_stage_onnx.py`

## Code Explanation

Concatenates past decode K/V cache with the current token's K/V heads before rectangular attention.

## Review Comments

- This page binds `04_kv_cache_concat.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#47 cat.default` -> shape=[1, 32, 79, 128], dtype=float16
- `#48 cat.default` -> shape=[1, 32, 79, 128], dtype=float16

## Export Wrapper Source

```python
class KVCacheConcatStage(nn.Module):
    def forward(
        self,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = concat_kv_cache(k_current, v_current, past_k, past_v)
        return out["k_heads"], out["v_heads"]
```

## Primary Source: `kv_cache.py`

```python
from __future__ import annotations

import torch


def concat_kv_cache(
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    k_heads = torch.cat((past_k, k_current), dim=-2)
    v_heads = torch.cat((past_v, v_current), dim=-2)
    return {
        "k_heads": k_heads,
        "v_heads": v_heads,
    }
```
