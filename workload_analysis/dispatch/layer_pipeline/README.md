# Dispatch Layer Pipeline

This is the refactored entry point for dispatch layer analysis.

It keeps dispatch scripts, generated torch flows, review artifacts, and ONNX outputs under one dispatch workspace:

```text
/workspace/VisiPrune/workload_analysis/dispatch/visualize/<event_id>/
```

Generated per-layer torch code is placed beside the corresponding layer visualization output:

```text
/workspace/VisiPrune/workload_analysis/dispatch/visualize/<event_id>/torch_flow/
```

## Goals

1. Analyze selected dispatch layers.
2. Split each layer into a dispatch review summary and a reconstructed torch compute flow.
3. Execute the layer with smaller tensors and export each sub-process to ONNX.

## Usage

List available events:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --layers input1_layer0 \
  --list-events
```

Analyze selected events:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --layers input1_layer0 input1_layer5
```

Plain layer numbers use `input1_` by default:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --layers 0,5,6
```

Adjust toy tensor size:

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --layers input1_layer0 \
  --small-seq 16 \
  --small-hidden 32 \
  --small-heads 4 \
  --small-ffn 64
```

## Per-Layer Output

Each layer directory contains:

```text
dispatch_review/dispatch_ops.csv
dispatch_review/summary.json
dispatch_review/tensor_compute_process.md
dispatch_review/process_manifest.json
dispatch_review/process_code_index.md
dispatch_review/onnx_code_index.md
dispatch_review/onnx_code_map.json
dispatch_review/onnx_code/*.md
dispatch_review/alignment_audit.json
dispatch_review/alignment_audit.md
onnx/*.onnx
onnx/manifest.json
onnx/*_code.md
layer_manifest.json
```

The generated process code lives in the layer directory:

```text
/workspace/VisiPrune/workload_analysis/dispatch/visualize/<event_id>/torch_flow/
```

`dispatch_review/` contains dispatch-derived review artifacts for the generated ONNX and process code. `onnx_code_index.md` maps every ONNX file to the exact `torch_flow` code used to export it, with per-ONNX code explanation, review comments, dispatch evidence notes, and embedded source snippets. The per-ONNX Markdown pages are also mirrored directly into `onnx/` so each visualization folder keeps the ONNX file and its explanation side by side.

The split small-tensor torch flow under `torch_flow/` includes:

- `rmsnorm.py`
- `qkv_projection.py`
- `rope.py`
- `attention.py`
- `visual_adjust.py`
- `attention_output.py`
- `mlp.py`
- `run_full_flow.py`
- `export_stage_onnx.py`
- `dispatch_reconstructed.py`
- `toy_tensor_compute.py`
- `layer_profile.json`
- `process_index.md`
