# input1_layer14 `05_visipruner_similarity_check.onnx` Code Review

## ONNX Artifact

- ONNX file: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer14/onnx/05_visipruner_similarity_check.onnx`
- Stage: `visipruner_similarity_check`
- Stage title: VisiPrune Similarity Check
- ONNX nodes: `32`
- ONNX initializers: `0`

### ONNX Inputs

- `hidden_states`: `[16, 32]`

### ONNX Outputs

- `similarity`: `[15]`
- `any_close`: `[]`

### Dispatch Tensor ID Inputs/Outputs

- Dispatch input tensor ids: `['t00000974', 't00000057', 't00001012', 't00001022', 't00001021', 't00001024', 't00001027']`
- Dispatch output tensor ids: `['t00001025', 't00001028']`
- Dispatch tensor-id dependencies inside evidence rows: `[{'tensor_id': 't00000975', 'consumer_event_op_index': 22, 'consumer_op_name': 'is_nonzero.default'}, {'tensor_id': 't00001008', 'consumer_event_op_index': 59, 'consumer_op_name': 'is_nonzero.default'}, {'tensor_id': 't00001023', 'consumer_event_op_index': 77, 'consumer_op_name': 'cosine_similarity.default'}]`

## Corresponding `torch_flow` Code

- Export wrapper: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/export_stage_onnx.py::VisiPrunerSimilarityCheckStage`
- Primary implementation: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/visipruner_similarity.py`
- Support files: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/config.py`, `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/init_data.py`, `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/export_stage_onnx.py`

## Code Explanation

Computes cosine-similarity based VisiPrune probe/check signals when dispatch contains the similarity sub-process.

## Review Comments

- This page binds `05_visipruner_similarity_check.onnx` to the exact `torch_flow` source used to define or export that ONNX stage.
- The export wrapper defines the ONNX boundary: input names, output names, buffers/initializers, and the `forward()` method traced by `torch.onnx.export`.
- The primary implementation file contains the small-shape tensor computation being wrapped for visualization.
- Dispatch evidence below is the core trace evidence used to justify the stage in this layer reconstruction.

## Dispatch Evidence Notes

- `#21 gt.Scalar` inputs=`['t00000974']` outputs=`['t00000975']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000975']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001008']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001008']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001012']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001022', 't00001021']` outputs=`['t00001023']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001023', 't00001024']` outputs=`['t00001025']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001027']` outputs=`['t00001028']` -> shape=[], dtype=bool

## Export Wrapper Source

```python
class VisiPrunerSimilarityCheckStage(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = visipruner_similarity_check(hidden_states)
        return out["similarity"], out["any_close"].to(torch.float32)
```

## Primary Source: `visipruner_similarity.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def visipruner_similarity_check(hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
    if hidden_states.shape[0] < 2:
        reference = hidden_states
        candidate = hidden_states
    else:
        reference = hidden_states[:-1]
        candidate = hidden_states[1:]
    similarity = F.cosine_similarity(reference, candidate, dim=-1)
    threshold = torch.tensor(0.9, dtype=similarity.dtype, device=similarity.device)
    above_threshold = similarity > threshold
    any_close = torch.any(above_threshold)
    return {
        "reference": reference,
        "candidate": candidate,
        "similarity": similarity,
        "above_threshold": above_threshold,
        "any_close": any_close,
    }
```
