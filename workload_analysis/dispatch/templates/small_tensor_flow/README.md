# input1_layer0 Small Tensor Flow

This directory contains an executable small-tensor reconstruction of the
`input1_layer0` dispatch computation.

It uses the same toy dimensions as the Zetane ONNX:

```text
seq=16
hidden=32
heads=4
head_dim=8
ffn=64
visual_start=3
visual_end=13
tail_start=13
```

Run:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/templates/small_tensor_flow/run_full_flow.py
```

Optional tensor dump:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/templates/small_tensor_flow/run_full_flow.py \
  --save-npz /tmp/input1_layer0_small_flow_tensors.npz
```

Files are split by computation stage:

- `init_data.py`: deterministic toy inputs, weights, and RoPE cache.
- `rmsnorm.py`: RMSNorm primitives.
- `qkv_projection.py`: Q/K/V linear projection and head split.
- `rope.py`: RoPE gather, rotate-half, and apply.
- `attention.py`: scaled dot-product attention.
- `visual_adjust.py`: shallow/full-visual attention adjustment.
- `attention_output.py`: context matmul, output projection, residual.
- `mlp.py`: post-attention RMSNorm and gated MLP.
- `run_full_flow.py`: complete flow orchestration and tensor summaries.
- `export_stage_onnx.py`: exports one ONNX file per stage.

Export per-stage ONNX files:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/templates/small_tensor_flow/export_stage_onnx.py \
  --out-dir /workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer0/small_tensor_flow_onnx
```

Outputs are written to:

```text
/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer0/small_tensor_flow_onnx
```
