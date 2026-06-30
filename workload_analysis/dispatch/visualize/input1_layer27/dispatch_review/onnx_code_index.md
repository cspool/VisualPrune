# input1_layer27 ONNX To Torch Flow Code Index

Each ONNX file below is linked to the exact `torch_flow` code used to define or export that stage, with a per-ONNX explanation and review comments.

| ONNX | Stage | Code Review | Primary Code | Export Wrapper |
|---|---|---|---|---|
| `01_input_rmsnorm.onnx` | `input_rmsnorm` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/01_input_rmsnorm_code.md` | `rmsnorm.py` | `InputRMSNormStage` |
| `02_qkv_projection.onnx` | `qkv_projection` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/02_qkv_projection_code.md` | `qkv_projection.py` | `QKVProjectionStage` |
| `03_rope.onnx` | `rope` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/03_rope_code.md` | `rope.py` | `RoPEStage` |
| `04_attention.onnx` | `attention` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/04_attention_code.md` | `attention.py` | `AttentionStage` |
| `05_visipruner_similarity_check.onnx` | `visipruner_similarity_check` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/05_visipruner_similarity_check_code.md` | `visipruner_similarity.py` | `VisiPrunerSimilarityCheckStage` |
| `06_attention_output.onnx` | `attention_output` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/06_attention_output_code.md` | `attention_output.py` | `AttentionOutputStage` |
| `07_mlp.onnx` | `mlp` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/07_mlp_code.md` | `mlp.py`<br>`rmsnorm.py` | `MLPStage` |
| `08_full_flow.onnx` | `full_flow` | `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/onnx_code/08_full_flow_code.md` | `run_full_flow.py` | `FullFlowStage` |
