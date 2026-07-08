# VisiPruner Full Eager FX-SAME_INPUT Layer-wise Nsight Report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `visipruner-full`
- description: Native VisPrune full path: shallow + middle + deep.
- max_new_tokens: `32`
- use_flash_attn: `False`
- use_visipruner: `True`
- visipruner_decode_backend: `eager`
- resolved_visipruner_decode_backend: `eager`
- sync_timing: `off`

## FX-matched input contract

| field | FX trace value | performance run value | match |
|---|---|---|---|
| `config` | `visipruner-full` | `visipruner-full` | yes |
| `model_path` | `liuhaotian/llava-v1.5-7b` | `liuhaotian/llava-v1.5-7b` | yes |
| `image_path` | `/workspace/VisiPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg` | `/workspace/VisiPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg` | yes |
| `prompt` | `Describe the image briefly.` | `Describe the image briefly.` | yes |
| `conv_mode` | `llava_v1` | `llava_v1` | yes |
| `max_new_tokens` | `32` | `32` | yes |

## Data sources

- profile json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_fxsameinput_visipruner_full_eager_32tok.json`
- FX run metadata: `/workspace/VisiPrune/workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/run_metadata.json`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_fxsameinput_visipruner_full_eager_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_fxsameinput_visipruner_full_eager_32tok_layer_kernel_breakdown.csv`
- all input-layer performance table: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_fxsameinput_visipruner_full_eager_32tok_all_input_layer_performance.csv`
- attribution check status: `pass`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## Nsight-only timing note

This report intentionally omits clock/sync timing. Layer timing columns come from Nsight NVTX CPU ranges and CUPTI launch-owned kernel sums.

## Human-draft-style workload reading

forward 1

- layer 0: q_len=624, kv_len=624, workload=eager_visipruner_prefill_shallow_layer0_mass_fold; operator=eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs.
- layer 1-5: q_len=624, kv_len=624, workload=eager_visipruner_prefill_shallow_text_to_vision_mask; operator=eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs.
- layer 6: q_len=624, kv_len=624, workload=eager_visipruner_prefill_last_query_proxy_or_selection; operator=eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs.
- layer 7-18: q_len=624, kv_len=624, workload=eager_visipruner_prefill_last_query_proxy_or_selection; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.
- layer 19-27: q_len=58, kv_len=58, workload=eager_visipruner_middle_pruned_compact_prefill; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.
- layer 28-31: q_len=48, kv_len=48, workload=eager_visipruner_deep_removed_prefill; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.

forward 2

- layer 0-18: q_len=1, kv_len=625, workload=eager_visipruner_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 19-27: q_len=1, kv_len=59, workload=eager_visipruner_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=49, workload=eager_visipruner_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

forward 32

- layer 0-18: q_len=1, kv_len=655, workload=eager_visipruner_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 19-27: q_len=1, kv_len=89, workload=eager_visipruner_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=79, workload=eager_visipruner_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

## Complete input-layer coverage and data access

- covered input-layer rows: `1024` total = `32` prefill rows + `992` decode rows.
- decode coverage: forward `2` to `32`, occurrence `0` to `30`, 32 layers per decode forward.
- complete machine-readable input-layer table: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_fxsameinput_visipruner_full_eager_32tok_all_input_layer_performance.csv`
- performance source for every input-layer row: `layer_events.csv` provides forward/layer/q_len/kv_len/workload/operator metadata; `layer_kernel_breakdown.csv` provides the `component=total` NVTX CPU range and CUPTI launch-owned kernel sum matched by `(phase, layer_idx, occurrence)`.
- component-level data access: filter `layer_kernel_breakdown.csv` by the same `(phase, layer_idx, occurrence)` and choose `component=attn` or `component=mlp`; the compact table below shows only the total component.
- no row in this section is estimated. Missing per-row timing is rendered as `-` instead of being filled from a mean.

### Later input-layer pattern

- prefill is forward `1`, occurrence `0`; it contains layers `0-31` once.
- decode is forward `2-32`, occurrence `0-30`; each occurrence contains layers `0-31` once and uses `q_len=1`.
- for layers `0-18`, decode keeps the full KV cache and `kv_len = 625 + occurrence`.
- for layers `19-27`, decode uses the middle pruned KV cache and `kv_len = 59 + occurrence`.
- for layers `28-31`, decode uses the deep removed KV cache and `kv_len = 49 + occurrence`.
- workload type and operator path are stable within each of the three decode layer groups; later input-layer timings are obtained directly from their own NVTX/CUPTI rows, not by applying the first or last decode forward as a proxy.

### Complete compact input-layer table

| event | forward | phase | occurrence | layer | q_len | kv_len | workload type | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| 0 | 1 | prefill | 0 | 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | 3.054 | 2.077 | gemm_tensorcore (1.696 ms) |
| 1 | 1 | prefill | 0 | 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | 2.140 | 2.179 | gemm_tensorcore (1.828 ms) |
| 2 | 1 | prefill | 0 | 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | 2.196 | 2.048 | gemm_tensorcore (1.709 ms) |
| 3 | 1 | prefill | 0 | 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | 2.144 | 2.057 | gemm_tensorcore (1.701 ms) |
| 4 | 1 | prefill | 0 | 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | 2.258 | 2.053 | gemm_tensorcore (1.712 ms) |
| 5 | 1 | prefill | 0 | 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | 2.190 | 2.061 | gemm_tensorcore (1.707 ms) |
| 6 | 1 | prefill | 0 | 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.161 | 2.051 | gemm_tensorcore (1.712 ms) |
| 7 | 1 | prefill | 0 | 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.668 | 2.240 | gemm_tensorcore (1.707 ms) |
| 8 | 1 | prefill | 0 | 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.683 | 2.096 | gemm_tensorcore (1.702 ms) |
| 9 | 1 | prefill | 0 | 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.462 | 2.116 | gemm_tensorcore (1.709 ms) |
| 10 | 1 | prefill | 0 | 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.589 | 2.107 | gemm_tensorcore (1.709 ms) |
| 11 | 1 | prefill | 0 | 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.288 | 2.222 | gemm_tensorcore (1.749 ms) |
| 12 | 1 | prefill | 0 | 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.405 | 2.111 | gemm_tensorcore (1.715 ms) |
| 13 | 1 | prefill | 0 | 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.215 | 2.113 | gemm_tensorcore (1.707 ms) |
| 14 | 1 | prefill | 0 | 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.626 | 2.103 | gemm_tensorcore (1.709 ms) |
| 15 | 1 | prefill | 0 | 15 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.364 | 2.116 | gemm_tensorcore (1.708 ms) |
| 16 | 1 | prefill | 0 | 16 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.757 | 2.164 | gemm_tensorcore (1.707 ms) |
| 17 | 1 | prefill | 0 | 17 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.367 | 2.120 | gemm_tensorcore (1.711 ms) |
| 18 | 1 | prefill | 0 | 18 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | 2.886 | 2.127 | gemm_tensorcore (1.711 ms) |
| 19 | 1 | prefill | 0 | 19 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.458 | 0.625 | gemm_tensorcore (0.508 ms) |
| 20 | 1 | prefill | 0 | 20 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.413 | 0.619 | gemm_tensorcore (0.501 ms) |
| 21 | 1 | prefill | 0 | 21 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.258 | 0.729 | gemm_tensorcore (0.497 ms) |
| 22 | 1 | prefill | 0 | 22 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.314 | 0.619 | gemm_tensorcore (0.502 ms) |
| 23 | 1 | prefill | 0 | 23 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.167 | 0.619 | gemm_tensorcore (0.502 ms) |
| 24 | 1 | prefill | 0 | 24 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.340 | 0.617 | gemm_tensorcore (0.500 ms) |
| 25 | 1 | prefill | 0 | 25 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.234 | 0.619 | gemm_tensorcore (0.502 ms) |
| 26 | 1 | prefill | 0 | 26 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.484 | 0.626 | gemm_tensorcore (0.509 ms) |
| 27 | 1 | prefill | 0 | 27 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | 2.293 | 0.618 | gemm_tensorcore (0.501 ms) |
| 28 | 1 | prefill | 0 | 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | 1.811 | 0.607 | gemm_tensorcore (0.495 ms) |
| 29 | 1 | prefill | 0 | 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | 1.938 | 0.580 | gemm_tensorcore (0.497 ms) |
| 30 | 1 | prefill | 0 | 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | 1.797 | 0.580 | gemm_tensorcore (0.498 ms) |
| 31 | 1 | prefill | 0 | 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | 1.749 | 0.583 | gemm_tensorcore (0.500 ms) |
| 32 | 2 | decode | 0 | 0 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.600 | 0.538 | gemv_decode_cublas (0.452 ms) |
| 33 | 2 | decode | 0 | 1 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.537 | gemv_decode_cublas (0.452 ms) |
| 34 | 2 | decode | 0 | 2 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.505 | 0.551 | gemv_decode_cublas (0.464 ms) |
| 35 | 2 | decode | 0 | 3 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.508 | 0.554 | gemv_decode_cublas (0.467 ms) |
| 36 | 2 | decode | 0 | 4 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.424 | 0.615 | gemv_decode_cublas (0.459 ms) |
| 37 | 2 | decode | 0 | 5 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 2.111 | 0.546 | gemv_decode_cublas (0.460 ms) |
| 38 | 2 | decode | 0 | 6 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 2.129 | 0.545 | gemv_decode_cublas (0.459 ms) |
| 39 | 2 | decode | 0 | 7 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 2.111 | 0.550 | gemv_decode_cublas (0.462 ms) |
| 40 | 2 | decode | 0 | 8 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.658 | 0.551 | gemv_decode_cublas (0.464 ms) |
| 41 | 2 | decode | 0 | 9 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.721 | 0.550 | gemv_decode_cublas (0.463 ms) |
| 42 | 2 | decode | 0 | 10 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.562 | 0.549 | gemv_decode_cublas (0.462 ms) |
| 43 | 2 | decode | 0 | 11 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.827 | 0.549 | gemv_decode_cublas (0.462 ms) |
| 44 | 2 | decode | 0 | 12 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.718 | 0.677 | gemv_decode_cublas (0.503 ms) |
| 45 | 2 | decode | 0 | 13 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.758 | 0.543 | gemv_decode_cublas (0.457 ms) |
| 46 | 2 | decode | 0 | 14 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.585 | 0.547 | gemv_decode_cublas (0.461 ms) |
| 47 | 2 | decode | 0 | 15 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.775 | 0.544 | gemv_decode_cublas (0.458 ms) |
| 48 | 2 | decode | 0 | 16 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.571 | 0.548 | gemv_decode_cublas (0.461 ms) |
| 49 | 2 | decode | 0 | 17 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.955 | 0.550 | gemv_decode_cublas (0.462 ms) |
| 50 | 2 | decode | 0 | 18 | 1 | 625 | eager_visipruner_decode_full_kv_cache | 1.592 | 0.549 | gemv_decode_cublas (0.462 ms) |
| 51 | 2 | decode | 0 | 19 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.593 | 0.517 | gemv_decode_cublas (0.456 ms) |
| 52 | 2 | decode | 0 | 20 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.540 | 0.510 | gemv_decode_cublas (0.450 ms) |
| 53 | 2 | decode | 0 | 21 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.620 | 0.588 | gemv_decode_cublas (0.479 ms) |
| 54 | 2 | decode | 0 | 22 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.626 | 0.509 | gemv_decode_cublas (0.448 ms) |
| 55 | 2 | decode | 0 | 23 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.527 | 0.508 | gemv_decode_cublas (0.447 ms) |
| 56 | 2 | decode | 0 | 24 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.650 | 0.507 | gemv_decode_cublas (0.446 ms) |
| 57 | 2 | decode | 0 | 25 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.503 | 0.507 | gemv_decode_cublas (0.446 ms) |
| 58 | 2 | decode | 0 | 26 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.687 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 59 | 2 | decode | 0 | 27 | 1 | 59 | eager_visipruner_decode_middle_pruned_kv_cache | 1.543 | 0.508 | gemv_decode_cublas (0.448 ms) |
| 60 | 2 | decode | 0 | 28 | 1 | 49 | eager_visipruner_decode_deep_removed_kv_cache | 1.588 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 61 | 2 | decode | 0 | 29 | 1 | 49 | eager_visipruner_decode_deep_removed_kv_cache | 1.552 | 0.510 | gemv_decode_cublas (0.450 ms) |
| 62 | 2 | decode | 0 | 30 | 1 | 49 | eager_visipruner_decode_deep_removed_kv_cache | 1.753 | 0.595 | gemv_decode_cublas (0.534 ms) |
| 63 | 2 | decode | 0 | 31 | 1 | 49 | eager_visipruner_decode_deep_removed_kv_cache | 1.572 | 0.507 | gemv_decode_cublas (0.446 ms) |
| 64 | 3 | decode | 1 | 0 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.610 | 0.524 | gemv_decode_cublas (0.455 ms) |
| 65 | 3 | decode | 1 | 1 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.557 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 66 | 3 | decode | 1 | 2 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.556 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 67 | 3 | decode | 1 | 3 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.562 | 0.535 | gemv_decode_cublas (0.463 ms) |
| 68 | 3 | decode | 1 | 4 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.526 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 69 | 3 | decode | 1 | 5 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.500 | 0.544 | gemv_decode_cublas (0.470 ms) |
| 70 | 3 | decode | 1 | 6 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.536 | gemv_decode_cublas (0.464 ms) |
| 71 | 3 | decode | 1 | 7 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.523 | 0.634 | gemv_decode_cublas (0.501 ms) |
| 72 | 3 | decode | 1 | 8 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.481 | 0.527 | gemv_decode_cublas (0.458 ms) |
| 73 | 3 | decode | 1 | 9 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 74 | 3 | decode | 1 | 10 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 75 | 3 | decode | 1 | 11 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 76 | 3 | decode | 1 | 12 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.581 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 77 | 3 | decode | 1 | 13 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.544 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 78 | 3 | decode | 1 | 14 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.577 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 79 | 3 | decode | 1 | 15 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 80 | 3 | decode | 1 | 16 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 81 | 3 | decode | 1 | 17 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.684 | 0.643 | gemv_decode_cublas (0.573 ms) |
| 82 | 3 | decode | 1 | 18 | 1 | 626 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.530 | gemv_decode_cublas (0.460 ms) |
| 83 | 3 | decode | 1 | 19 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.537 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 84 | 3 | decode | 1 | 20 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.542 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 85 | 3 | decode | 1 | 21 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.534 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 86 | 3 | decode | 1 | 22 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.524 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 87 | 3 | decode | 1 | 23 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.499 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 88 | 3 | decode | 1 | 24 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.483 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 89 | 3 | decode | 1 | 25 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 90 | 3 | decode | 1 | 26 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.480 | 0.600 | gemv_decode_cublas (0.542 ms) |
| 91 | 3 | decode | 1 | 27 | 1 | 60 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 92 | 3 | decode | 1 | 28 | 1 | 50 | eager_visipruner_decode_deep_removed_kv_cache | 1.495 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 93 | 3 | decode | 1 | 29 | 1 | 50 | eager_visipruner_decode_deep_removed_kv_cache | 1.484 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 94 | 3 | decode | 1 | 30 | 1 | 50 | eager_visipruner_decode_deep_removed_kv_cache | 1.515 | 0.502 | gemv_decode_cublas (0.445 ms) |
| 95 | 3 | decode | 1 | 31 | 1 | 50 | eager_visipruner_decode_deep_removed_kv_cache | 1.517 | 0.504 | gemv_decode_cublas (0.447 ms) |
| 96 | 4 | decode | 2 | 0 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.616 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 97 | 4 | decode | 2 | 1 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 98 | 4 | decode | 2 | 2 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.551 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 99 | 4 | decode | 2 | 3 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.601 | 0.614 | gemv_decode_cublas (0.541 ms) |
| 100 | 4 | decode | 2 | 4 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.588 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 101 | 4 | decode | 2 | 5 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.585 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 102 | 4 | decode | 2 | 6 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 103 | 4 | decode | 2 | 7 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 104 | 4 | decode | 2 | 8 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 105 | 4 | decode | 2 | 9 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 106 | 4 | decode | 2 | 10 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.578 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 107 | 4 | decode | 2 | 11 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 108 | 4 | decode | 2 | 12 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.421 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 109 | 4 | decode | 2 | 13 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.428 | 0.607 | gemv_decode_cublas (0.458 ms) |
| 110 | 4 | decode | 2 | 14 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.415 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 111 | 4 | decode | 2 | 15 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.422 | 0.528 | gemv_decode_cublas (0.456 ms) |
| 112 | 4 | decode | 2 | 16 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.398 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 113 | 4 | decode | 2 | 17 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.414 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 114 | 4 | decode | 2 | 18 | 1 | 627 | eager_visipruner_decode_full_kv_cache | 1.496 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 115 | 4 | decode | 2 | 19 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.508 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 116 | 4 | decode | 2 | 20 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.498 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 117 | 4 | decode | 2 | 21 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.492 | 0.512 | gemv_decode_cublas (0.454 ms) |
| 118 | 4 | decode | 2 | 22 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.489 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 119 | 4 | decode | 2 | 23 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.696 | 0.621 | gemv_decode_cublas (0.468 ms) |
| 120 | 4 | decode | 2 | 24 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.559 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 121 | 4 | decode | 2 | 25 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 122 | 4 | decode | 2 | 26 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.523 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 123 | 4 | decode | 2 | 27 | 1 | 61 | eager_visipruner_decode_middle_pruned_kv_cache | 1.539 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 124 | 4 | decode | 2 | 28 | 1 | 51 | eager_visipruner_decode_deep_removed_kv_cache | 1.536 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 125 | 4 | decode | 2 | 29 | 1 | 51 | eager_visipruner_decode_deep_removed_kv_cache | 1.521 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 126 | 4 | decode | 2 | 30 | 1 | 51 | eager_visipruner_decode_deep_removed_kv_cache | 1.515 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 127 | 4 | decode | 2 | 31 | 1 | 51 | eager_visipruner_decode_deep_removed_kv_cache | 1.496 | 0.503 | gemv_decode_cublas (0.446 ms) |
| 128 | 5 | decode | 3 | 0 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.660 | 0.539 | gemv_decode_cublas (0.455 ms) |
| 129 | 5 | decode | 3 | 1 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 130 | 5 | decode | 3 | 2 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 131 | 5 | decode | 3 | 3 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 132 | 5 | decode | 3 | 4 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 133 | 5 | decode | 3 | 5 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 134 | 5 | decode | 3 | 6 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 135 | 5 | decode | 3 | 7 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 136 | 5 | decode | 3 | 8 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.509 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 137 | 5 | decode | 3 | 9 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.577 | 0.540 | gemv_decode_cublas (0.467 ms) |
| 138 | 5 | decode | 3 | 10 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.621 | 0.577 | gemv_decode_cublas (0.457 ms) |
| 139 | 5 | decode | 3 | 11 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.554 | 0.531 | gemv_decode_cublas (0.462 ms) |
| 140 | 5 | decode | 3 | 12 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 141 | 5 | decode | 3 | 13 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.513 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 142 | 5 | decode | 3 | 14 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 143 | 5 | decode | 3 | 15 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.494 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 144 | 5 | decode | 3 | 16 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.536 | gemv_decode_cublas (0.464 ms) |
| 145 | 5 | decode | 3 | 17 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.499 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 146 | 5 | decode | 3 | 18 | 1 | 628 | eager_visipruner_decode_full_kv_cache | 1.489 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 147 | 5 | decode | 3 | 19 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.482 | 0.581 | gemv_decode_cublas (0.522 ms) |
| 148 | 5 | decode | 3 | 20 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.439 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 149 | 5 | decode | 3 | 21 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.438 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 150 | 5 | decode | 3 | 22 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.492 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 151 | 5 | decode | 3 | 23 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.507 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 152 | 5 | decode | 3 | 24 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 153 | 5 | decode | 3 | 25 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.496 | 0.504 | gemv_decode_cublas (0.447 ms) |
| 154 | 5 | decode | 3 | 26 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.503 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 155 | 5 | decode | 3 | 27 | 1 | 62 | eager_visipruner_decode_middle_pruned_kv_cache | 1.483 | 0.512 | gemv_decode_cublas (0.454 ms) |
| 156 | 5 | decode | 3 | 28 | 1 | 52 | eager_visipruner_decode_deep_removed_kv_cache | 1.413 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 157 | 5 | decode | 3 | 29 | 1 | 52 | eager_visipruner_decode_deep_removed_kv_cache | 1.369 | 0.620 | gemv_decode_cublas (0.550 ms) |
| 158 | 5 | decode | 3 | 30 | 1 | 52 | eager_visipruner_decode_deep_removed_kv_cache | 1.580 | 0.502 | gemv_decode_cublas (0.444 ms) |
| 159 | 5 | decode | 3 | 31 | 1 | 52 | eager_visipruner_decode_deep_removed_kv_cache | 1.386 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 160 | 6 | decode | 4 | 0 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.459 | 0.522 | gemv_decode_cublas (0.452 ms) |
| 161 | 6 | decode | 4 | 1 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.479 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 162 | 6 | decode | 4 | 2 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.471 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 163 | 6 | decode | 4 | 3 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 164 | 6 | decode | 4 | 4 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 165 | 6 | decode | 4 | 5 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.504 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 166 | 6 | decode | 4 | 6 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.464 | 0.541 | gemv_decode_cublas (0.468 ms) |
| 167 | 6 | decode | 4 | 7 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.512 | 0.569 | gemv_decode_cublas (0.456 ms) |
| 168 | 6 | decode | 4 | 8 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.606 | 0.530 | gemv_decode_cublas (0.460 ms) |
| 169 | 6 | decode | 4 | 9 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.575 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 170 | 6 | decode | 4 | 10 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.560 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 171 | 6 | decode | 4 | 11 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.500 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 172 | 6 | decode | 4 | 12 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.572 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 173 | 6 | decode | 4 | 13 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 174 | 6 | decode | 4 | 14 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.607 | 0.541 | gemv_decode_cublas (0.468 ms) |
| 175 | 6 | decode | 4 | 15 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.560 | 0.582 | gemv_decode_cublas (0.509 ms) |
| 176 | 6 | decode | 4 | 16 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 177 | 6 | decode | 4 | 17 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 178 | 6 | decode | 4 | 18 | 1 | 629 | eager_visipruner_decode_full_kv_cache | 1.515 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 179 | 6 | decode | 4 | 19 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.516 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 180 | 6 | decode | 4 | 20 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.523 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 181 | 6 | decode | 4 | 21 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.471 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 182 | 6 | decode | 4 | 22 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.399 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 183 | 6 | decode | 4 | 23 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.437 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 184 | 6 | decode | 4 | 24 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.470 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 185 | 6 | decode | 4 | 25 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.512 | gemv_decode_cublas (0.454 ms) |
| 186 | 6 | decode | 4 | 26 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.536 | 0.590 | gemv_decode_cublas (0.449 ms) |
| 187 | 6 | decode | 4 | 27 | 1 | 63 | eager_visipruner_decode_middle_pruned_kv_cache | 1.465 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 188 | 6 | decode | 4 | 28 | 1 | 53 | eager_visipruner_decode_deep_removed_kv_cache | 1.475 | 0.504 | gemv_decode_cublas (0.447 ms) |
| 189 | 6 | decode | 4 | 29 | 1 | 53 | eager_visipruner_decode_deep_removed_kv_cache | 1.843 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 190 | 6 | decode | 4 | 30 | 1 | 53 | eager_visipruner_decode_deep_removed_kv_cache | 1.586 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 191 | 6 | decode | 4 | 31 | 1 | 53 | eager_visipruner_decode_deep_removed_kv_cache | 1.530 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 192 | 7 | decode | 5 | 0 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.632 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 193 | 7 | decode | 5 | 1 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.551 | 0.536 | gemv_decode_cublas (0.465 ms) |
| 194 | 7 | decode | 5 | 2 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.564 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 195 | 7 | decode | 5 | 3 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.652 | 0.650 | gemv_decode_cublas (0.580 ms) |
| 196 | 7 | decode | 5 | 4 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.541 | 0.530 | gemv_decode_cublas (0.460 ms) |
| 197 | 7 | decode | 5 | 5 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.526 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 198 | 7 | decode | 5 | 6 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 199 | 7 | decode | 5 | 7 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.531 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 200 | 7 | decode | 5 | 8 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.510 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 201 | 7 | decode | 5 | 9 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.477 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 202 | 7 | decode | 5 | 10 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.414 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 203 | 7 | decode | 5 | 11 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.447 | 0.541 | gemv_decode_cublas (0.467 ms) |
| 204 | 7 | decode | 5 | 12 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.485 | 0.610 | gemv_decode_cublas (0.517 ms) |
| 205 | 7 | decode | 5 | 13 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.462 | 0.528 | gemv_decode_cublas (0.458 ms) |
| 206 | 7 | decode | 5 | 14 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.389 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 207 | 7 | decode | 5 | 15 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.506 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 208 | 7 | decode | 5 | 16 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.500 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 209 | 7 | decode | 5 | 17 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 210 | 7 | decode | 5 | 18 | 1 | 630 | eager_visipruner_decode_full_kv_cache | 1.480 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 211 | 7 | decode | 5 | 19 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.605 | 0.514 | gemv_decode_cublas (0.456 ms) |
| 212 | 7 | decode | 5 | 20 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.552 | 0.513 | gemv_decode_cublas (0.455 ms) |
| 213 | 7 | decode | 5 | 21 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.533 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 214 | 7 | decode | 5 | 22 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.730 | 0.536 | gemv_decode_cublas (0.449 ms) |
| 215 | 7 | decode | 5 | 23 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 216 | 7 | decode | 5 | 24 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.529 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 217 | 7 | decode | 5 | 25 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.531 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 218 | 7 | decode | 5 | 26 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 219 | 7 | decode | 5 | 27 | 1 | 64 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 220 | 7 | decode | 5 | 28 | 1 | 54 | eager_visipruner_decode_deep_removed_kv_cache | 1.516 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 221 | 7 | decode | 5 | 29 | 1 | 54 | eager_visipruner_decode_deep_removed_kv_cache | 1.484 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 222 | 7 | decode | 5 | 30 | 1 | 54 | eager_visipruner_decode_deep_removed_kv_cache | 1.481 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 223 | 7 | decode | 5 | 31 | 1 | 54 | eager_visipruner_decode_deep_removed_kv_cache | 1.710 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 224 | 8 | decode | 6 | 0 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.586 | 0.523 | gemv_decode_cublas (0.453 ms) |
| 225 | 8 | decode | 6 | 1 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.573 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 226 | 8 | decode | 6 | 2 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 227 | 8 | decode | 6 | 3 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.485 | 0.531 | gemv_decode_cublas (0.457 ms) |
| 228 | 8 | decode | 6 | 4 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.510 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 229 | 8 | decode | 6 | 5 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 230 | 8 | decode | 6 | 6 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 231 | 8 | decode | 6 | 7 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.571 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 232 | 8 | decode | 6 | 8 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.599 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 233 | 8 | decode | 6 | 9 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.606 | 0.631 | gemv_decode_cublas (0.561 ms) |
| 234 | 8 | decode | 6 | 10 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.530 | gemv_decode_cublas (0.460 ms) |
| 235 | 8 | decode | 6 | 11 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.519 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 236 | 8 | decode | 6 | 12 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 237 | 8 | decode | 6 | 13 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.518 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 238 | 8 | decode | 6 | 14 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.410 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 239 | 8 | decode | 6 | 15 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 240 | 8 | decode | 6 | 16 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.471 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 241 | 8 | decode | 6 | 17 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 242 | 8 | decode | 6 | 18 | 1 | 631 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 243 | 8 | decode | 6 | 19 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.651 | 0.562 | gemv_decode_cublas (0.454 ms) |
| 244 | 8 | decode | 6 | 20 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.512 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 245 | 8 | decode | 6 | 21 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.560 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 246 | 8 | decode | 6 | 22 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.539 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 247 | 8 | decode | 6 | 23 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.525 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 248 | 8 | decode | 6 | 24 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 249 | 8 | decode | 6 | 25 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.491 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 250 | 8 | decode | 6 | 26 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.541 | 0.513 | gemv_decode_cublas (0.455 ms) |
| 251 | 8 | decode | 6 | 27 | 1 | 65 | eager_visipruner_decode_middle_pruned_kv_cache | 1.553 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 252 | 8 | decode | 6 | 28 | 1 | 55 | eager_visipruner_decode_deep_removed_kv_cache | 1.621 | 0.567 | gemv_decode_cublas (0.466 ms) |
| 253 | 8 | decode | 6 | 29 | 1 | 55 | eager_visipruner_decode_deep_removed_kv_cache | 1.533 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 254 | 8 | decode | 6 | 30 | 1 | 55 | eager_visipruner_decode_deep_removed_kv_cache | 1.520 | 0.502 | gemv_decode_cublas (0.444 ms) |
| 255 | 8 | decode | 6 | 31 | 1 | 55 | eager_visipruner_decode_deep_removed_kv_cache | 1.533 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 256 | 9 | decode | 7 | 0 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.558 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 257 | 9 | decode | 7 | 1 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.498 | 0.529 | gemv_decode_cublas (0.459 ms) |
| 258 | 9 | decode | 7 | 2 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.502 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 259 | 9 | decode | 7 | 3 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.482 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 260 | 9 | decode | 7 | 4 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 261 | 9 | decode | 7 | 5 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.618 | gemv_decode_cublas (0.531 ms) |
| 262 | 9 | decode | 7 | 6 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.470 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 263 | 9 | decode | 7 | 7 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.465 | 0.534 | gemv_decode_cublas (0.463 ms) |
| 264 | 9 | decode | 7 | 8 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.531 | gemv_decode_cublas (0.459 ms) |
| 265 | 9 | decode | 7 | 9 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.476 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 266 | 9 | decode | 7 | 10 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 267 | 9 | decode | 7 | 11 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.507 | 0.529 | gemv_decode_cublas (0.456 ms) |
| 268 | 9 | decode | 7 | 12 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 269 | 9 | decode | 7 | 13 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.558 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 270 | 9 | decode | 7 | 14 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.557 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 271 | 9 | decode | 7 | 15 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.691 | 0.647 | gemv_decode_cublas (0.568 ms) |
| 272 | 9 | decode | 7 | 16 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.574 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 273 | 9 | decode | 7 | 17 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 274 | 9 | decode | 7 | 18 | 1 | 632 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 275 | 9 | decode | 7 | 19 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.531 | 0.508 | gemv_decode_cublas (0.451 ms) |
| 276 | 9 | decode | 7 | 20 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.537 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 277 | 9 | decode | 7 | 21 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.536 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 278 | 9 | decode | 7 | 22 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.435 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 279 | 9 | decode | 7 | 23 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.419 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 280 | 9 | decode | 7 | 24 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.401 | 0.615 | gemv_decode_cublas (0.542 ms) |
| 281 | 9 | decode | 7 | 25 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.399 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 282 | 9 | decode | 7 | 26 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.409 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 283 | 9 | decode | 7 | 27 | 1 | 66 | eager_visipruner_decode_middle_pruned_kv_cache | 1.406 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 284 | 9 | decode | 7 | 28 | 1 | 56 | eager_visipruner_decode_deep_removed_kv_cache | 1.595 | 0.505 | gemv_decode_cublas (0.445 ms) |
| 285 | 9 | decode | 7 | 29 | 1 | 56 | eager_visipruner_decode_deep_removed_kv_cache | 1.520 | 0.505 | gemv_decode_cublas (0.444 ms) |
| 286 | 9 | decode | 7 | 30 | 1 | 56 | eager_visipruner_decode_deep_removed_kv_cache | 1.522 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 287 | 9 | decode | 7 | 31 | 1 | 56 | eager_visipruner_decode_deep_removed_kv_cache | 1.539 | 0.505 | gemv_decode_cublas (0.445 ms) |
| 288 | 10 | decode | 8 | 0 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.643 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 289 | 10 | decode | 8 | 1 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.591 | 0.622 | gemv_decode_cublas (0.527 ms) |
| 290 | 10 | decode | 8 | 2 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.524 | gemv_decode_cublas (0.454 ms) |
| 291 | 10 | decode | 8 | 3 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 292 | 10 | decode | 8 | 4 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 293 | 10 | decode | 8 | 5 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 294 | 10 | decode | 8 | 6 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 295 | 10 | decode | 8 | 7 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.472 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 296 | 10 | decode | 8 | 8 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.504 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 297 | 10 | decode | 8 | 9 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 298 | 10 | decode | 8 | 10 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 299 | 10 | decode | 8 | 11 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.600 | gemv_decode_cublas (0.455 ms) |
| 300 | 10 | decode | 8 | 12 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 301 | 10 | decode | 8 | 13 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.530 | gemv_decode_cublas (0.460 ms) |
| 302 | 10 | decode | 8 | 14 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.573 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 303 | 10 | decode | 8 | 15 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 304 | 10 | decode | 8 | 16 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.531 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 305 | 10 | decode | 8 | 17 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.553 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 306 | 10 | decode | 8 | 18 | 1 | 633 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 307 | 10 | decode | 8 | 19 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 2.949 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 308 | 10 | decode | 8 | 20 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.595 | 0.509 | gemv_decode_cublas (0.448 ms) |
| 309 | 10 | decode | 8 | 21 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.548 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 310 | 10 | decode | 8 | 22 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.548 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 311 | 10 | decode | 8 | 23 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.564 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 312 | 10 | decode | 8 | 24 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 313 | 10 | decode | 8 | 25 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.485 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 314 | 10 | decode | 8 | 26 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.526 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 315 | 10 | decode | 8 | 27 | 1 | 67 | eager_visipruner_decode_middle_pruned_kv_cache | 1.546 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 316 | 10 | decode | 8 | 28 | 1 | 57 | eager_visipruner_decode_deep_removed_kv_cache | 1.511 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 317 | 10 | decode | 8 | 29 | 1 | 57 | eager_visipruner_decode_deep_removed_kv_cache | 1.523 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 318 | 10 | decode | 8 | 30 | 1 | 57 | eager_visipruner_decode_deep_removed_kv_cache | 1.584 | 0.515 | gemv_decode_cublas (0.451 ms) |
| 319 | 10 | decode | 8 | 31 | 1 | 57 | eager_visipruner_decode_deep_removed_kv_cache | 1.504 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 320 | 11 | decode | 9 | 0 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.482 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 321 | 11 | decode | 9 | 1 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.541 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 322 | 11 | decode | 9 | 2 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 323 | 11 | decode | 9 | 3 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 324 | 11 | decode | 9 | 4 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 325 | 11 | decode | 9 | 5 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.586 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 326 | 11 | decode | 9 | 6 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.532 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 327 | 11 | decode | 9 | 7 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.600 | 0.626 | gemv_decode_cublas (0.556 ms) |
| 328 | 11 | decode | 9 | 8 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.571 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 329 | 11 | decode | 9 | 9 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 330 | 11 | decode | 9 | 10 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 331 | 11 | decode | 9 | 11 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 332 | 11 | decode | 9 | 12 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.523 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 333 | 11 | decode | 9 | 13 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 334 | 11 | decode | 9 | 14 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.438 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 335 | 11 | decode | 9 | 15 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.409 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 336 | 11 | decode | 9 | 16 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.413 | 0.640 | gemv_decode_cublas (0.567 ms) |
| 337 | 11 | decode | 9 | 17 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.424 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 338 | 11 | decode | 9 | 18 | 1 | 634 | eager_visipruner_decode_full_kv_cache | 1.402 | 0.531 | gemv_decode_cublas (0.461 ms) |
| 339 | 11 | decode | 9 | 19 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 340 | 11 | decode | 9 | 20 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.540 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 341 | 11 | decode | 9 | 21 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.526 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 342 | 11 | decode | 9 | 22 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.576 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 343 | 11 | decode | 9 | 23 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.546 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 344 | 11 | decode | 9 | 24 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.516 | 0.514 | gemv_decode_cublas (0.456 ms) |
| 345 | 11 | decode | 9 | 25 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.524 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 346 | 11 | decode | 9 | 26 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.533 | 0.570 | gemv_decode_cublas (0.512 ms) |
| 347 | 11 | decode | 9 | 27 | 1 | 68 | eager_visipruner_decode_middle_pruned_kv_cache | 1.481 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 348 | 11 | decode | 9 | 28 | 1 | 58 | eager_visipruner_decode_deep_removed_kv_cache | 1.482 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 349 | 11 | decode | 9 | 29 | 1 | 58 | eager_visipruner_decode_deep_removed_kv_cache | 1.496 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 350 | 11 | decode | 9 | 30 | 1 | 58 | eager_visipruner_decode_deep_removed_kv_cache | 1.750 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 351 | 11 | decode | 9 | 31 | 1 | 58 | eager_visipruner_decode_deep_removed_kv_cache | 1.502 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 352 | 12 | decode | 10 | 0 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.410 | 0.528 | gemv_decode_cublas (0.458 ms) |
| 353 | 12 | decode | 10 | 1 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.407 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 354 | 12 | decode | 10 | 2 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.415 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 355 | 12 | decode | 10 | 3 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.489 | 0.635 | gemv_decode_cublas (0.562 ms) |
| 356 | 12 | decode | 10 | 4 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.482 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 357 | 12 | decode | 10 | 5 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 358 | 12 | decode | 10 | 6 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.483 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 359 | 12 | decode | 10 | 7 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 360 | 12 | decode | 10 | 8 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 361 | 12 | decode | 10 | 9 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.502 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 362 | 12 | decode | 10 | 10 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 363 | 12 | decode | 10 | 11 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.562 | 0.540 | gemv_decode_cublas (0.467 ms) |
| 364 | 12 | decode | 10 | 12 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.572 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 365 | 12 | decode | 10 | 13 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.613 | 0.616 | gemv_decode_cublas (0.475 ms) |
| 366 | 12 | decode | 10 | 14 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.523 | gemv_decode_cublas (0.454 ms) |
| 367 | 12 | decode | 10 | 15 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 368 | 12 | decode | 10 | 16 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.544 | 0.531 | gemv_decode_cublas (0.458 ms) |
| 369 | 12 | decode | 10 | 17 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.574 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 370 | 12 | decode | 10 | 18 | 1 | 635 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 371 | 12 | decode | 10 | 19 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.509 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 372 | 12 | decode | 10 | 20 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.482 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 373 | 12 | decode | 10 | 21 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.499 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 374 | 12 | decode | 10 | 22 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.539 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 375 | 12 | decode | 10 | 23 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.644 | 0.604 | gemv_decode_cublas (0.473 ms) |
| 376 | 12 | decode | 10 | 24 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.479 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 377 | 12 | decode | 10 | 25 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.482 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 378 | 12 | decode | 10 | 26 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.560 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 379 | 12 | decode | 10 | 27 | 1 | 69 | eager_visipruner_decode_middle_pruned_kv_cache | 1.499 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 380 | 12 | decode | 10 | 28 | 1 | 59 | eager_visipruner_decode_deep_removed_kv_cache | 1.485 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 381 | 12 | decode | 10 | 29 | 1 | 59 | eager_visipruner_decode_deep_removed_kv_cache | 1.524 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 382 | 12 | decode | 10 | 30 | 1 | 59 | eager_visipruner_decode_deep_removed_kv_cache | 1.574 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 383 | 12 | decode | 10 | 31 | 1 | 59 | eager_visipruner_decode_deep_removed_kv_cache | 1.842 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 384 | 13 | decode | 11 | 0 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.646 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 385 | 13 | decode | 11 | 1 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.579 | 0.528 | gemv_decode_cublas (0.458 ms) |
| 386 | 13 | decode | 11 | 2 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.572 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 387 | 13 | decode | 11 | 3 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 388 | 13 | decode | 11 | 4 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 389 | 13 | decode | 11 | 5 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 390 | 13 | decode | 11 | 6 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.530 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 391 | 13 | decode | 11 | 7 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.540 | gemv_decode_cublas (0.467 ms) |
| 392 | 13 | decode | 11 | 8 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.507 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 393 | 13 | decode | 11 | 9 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.633 | gemv_decode_cublas (0.454 ms) |
| 394 | 13 | decode | 11 | 10 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.498 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 395 | 13 | decode | 11 | 11 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.501 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 396 | 13 | decode | 11 | 12 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.467 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 397 | 13 | decode | 11 | 13 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.506 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 398 | 13 | decode | 11 | 14 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.507 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 399 | 13 | decode | 11 | 15 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 400 | 13 | decode | 11 | 16 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 401 | 13 | decode | 11 | 17 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.592 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 402 | 13 | decode | 11 | 18 | 1 | 636 | eager_visipruner_decode_full_kv_cache | 1.563 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 403 | 13 | decode | 11 | 19 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.680 | 0.586 | gemv_decode_cublas (0.449 ms) |
| 404 | 13 | decode | 11 | 20 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 405 | 13 | decode | 11 | 21 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.547 | 0.504 | gemv_decode_cublas (0.445 ms) |
| 406 | 13 | decode | 11 | 22 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.545 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 407 | 13 | decode | 11 | 23 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.521 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 408 | 13 | decode | 11 | 24 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.481 | 0.507 | gemv_decode_cublas (0.450 ms) |
| 409 | 13 | decode | 11 | 25 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.481 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 410 | 13 | decode | 11 | 26 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.537 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 411 | 13 | decode | 11 | 27 | 1 | 70 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 412 | 13 | decode | 11 | 28 | 1 | 60 | eager_visipruner_decode_deep_removed_kv_cache | 1.534 | 0.631 | gemv_decode_cublas (0.455 ms) |
| 413 | 13 | decode | 11 | 29 | 1 | 60 | eager_visipruner_decode_deep_removed_kv_cache | 1.539 | 0.502 | gemv_decode_cublas (0.444 ms) |
| 414 | 13 | decode | 11 | 30 | 1 | 60 | eager_visipruner_decode_deep_removed_kv_cache | 1.562 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 415 | 13 | decode | 11 | 31 | 1 | 60 | eager_visipruner_decode_deep_removed_kv_cache | 1.549 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 416 | 14 | decode | 12 | 0 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.612 | 0.523 | gemv_decode_cublas (0.453 ms) |
| 417 | 14 | decode | 12 | 1 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.553 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 418 | 14 | decode | 12 | 2 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.566 | 0.529 | gemv_decode_cublas (0.456 ms) |
| 419 | 14 | decode | 12 | 3 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.585 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 420 | 14 | decode | 12 | 4 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.573 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 421 | 14 | decode | 12 | 5 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.600 | 0.609 | gemv_decode_cublas (0.453 ms) |
| 422 | 14 | decode | 12 | 6 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.609 | 0.524 | gemv_decode_cublas (0.454 ms) |
| 423 | 14 | decode | 12 | 7 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.531 | gemv_decode_cublas (0.459 ms) |
| 424 | 14 | decode | 12 | 8 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 425 | 14 | decode | 12 | 9 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.514 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 426 | 14 | decode | 12 | 10 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 427 | 14 | decode | 12 | 11 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 428 | 14 | decode | 12 | 12 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.476 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 429 | 14 | decode | 12 | 13 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.519 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 430 | 14 | decode | 12 | 14 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 431 | 14 | decode | 12 | 15 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.609 | 0.590 | gemv_decode_cublas (0.455 ms) |
| 432 | 14 | decode | 12 | 16 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.470 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 433 | 14 | decode | 12 | 17 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.508 | 0.528 | gemv_decode_cublas (0.456 ms) |
| 434 | 14 | decode | 12 | 18 | 1 | 637 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 435 | 14 | decode | 12 | 19 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 436 | 14 | decode | 12 | 20 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 437 | 14 | decode | 12 | 21 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.530 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 438 | 14 | decode | 12 | 22 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.558 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 439 | 14 | decode | 12 | 23 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.575 | 0.514 | gemv_decode_cublas (0.455 ms) |
| 440 | 14 | decode | 12 | 24 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.565 | 0.570 | gemv_decode_cublas (0.452 ms) |
| 441 | 14 | decode | 12 | 25 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.545 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 442 | 14 | decode | 12 | 26 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.563 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 443 | 14 | decode | 12 | 27 | 1 | 71 | eager_visipruner_decode_middle_pruned_kv_cache | 1.547 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 444 | 14 | decode | 12 | 28 | 1 | 61 | eager_visipruner_decode_deep_removed_kv_cache | 1.520 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 445 | 14 | decode | 12 | 29 | 1 | 61 | eager_visipruner_decode_deep_removed_kv_cache | 1.528 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 446 | 14 | decode | 12 | 30 | 1 | 61 | eager_visipruner_decode_deep_removed_kv_cache | 1.514 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 447 | 14 | decode | 12 | 31 | 1 | 61 | eager_visipruner_decode_deep_removed_kv_cache | 1.736 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 448 | 15 | decode | 13 | 0 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.463 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 449 | 15 | decode | 13 | 1 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.553 | gemv_decode_cublas (0.481 ms) |
| 450 | 15 | decode | 13 | 2 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.501 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 451 | 15 | decode | 13 | 3 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 452 | 15 | decode | 13 | 4 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.419 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 453 | 15 | decode | 13 | 5 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.521 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 454 | 15 | decode | 13 | 6 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.502 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 455 | 15 | decode | 13 | 7 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 456 | 15 | decode | 13 | 8 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.497 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 457 | 15 | decode | 13 | 9 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.497 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 458 | 15 | decode | 13 | 10 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.557 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 459 | 15 | decode | 13 | 11 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.639 | gemv_decode_cublas (0.457 ms) |
| 460 | 15 | decode | 13 | 12 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.495 | 0.529 | gemv_decode_cublas (0.459 ms) |
| 461 | 15 | decode | 13 | 13 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.531 | gemv_decode_cublas (0.459 ms) |
| 462 | 15 | decode | 13 | 14 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 463 | 15 | decode | 13 | 15 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.489 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 464 | 15 | decode | 13 | 16 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 465 | 15 | decode | 13 | 17 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.467 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 466 | 15 | decode | 13 | 18 | 1 | 638 | eager_visipruner_decode_full_kv_cache | 1.426 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 467 | 15 | decode | 13 | 19 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.425 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 468 | 15 | decode | 13 | 20 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.449 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 469 | 15 | decode | 13 | 21 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.438 | 0.540 | gemv_decode_cublas (0.474 ms) |
| 470 | 15 | decode | 13 | 22 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.540 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 471 | 15 | decode | 13 | 23 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.512 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 472 | 15 | decode | 13 | 24 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.559 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 473 | 15 | decode | 13 | 25 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.547 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 474 | 15 | decode | 13 | 26 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.527 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 475 | 15 | decode | 13 | 27 | 1 | 72 | eager_visipruner_decode_middle_pruned_kv_cache | 1.510 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 476 | 15 | decode | 13 | 28 | 1 | 62 | eager_visipruner_decode_deep_removed_kv_cache | 1.534 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 477 | 15 | decode | 13 | 29 | 1 | 62 | eager_visipruner_decode_deep_removed_kv_cache | 1.573 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 478 | 15 | decode | 13 | 30 | 1 | 62 | eager_visipruner_decode_deep_removed_kv_cache | 1.539 | 0.514 | gemv_decode_cublas (0.456 ms) |
| 479 | 15 | decode | 13 | 31 | 1 | 62 | eager_visipruner_decode_deep_removed_kv_cache | 1.461 | 0.550 | gemv_decode_cublas (0.446 ms) |
| 480 | 16 | decode | 14 | 0 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.477 | 0.524 | gemv_decode_cublas (0.453 ms) |
| 481 | 16 | decode | 14 | 1 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.499 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 482 | 16 | decode | 14 | 2 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 483 | 16 | decode | 14 | 3 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.489 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 484 | 16 | decode | 14 | 4 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.499 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 485 | 16 | decode | 14 | 5 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.487 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 486 | 16 | decode | 14 | 6 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 487 | 16 | decode | 14 | 7 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 488 | 16 | decode | 14 | 8 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.584 | 0.639 | gemv_decode_cublas (0.568 ms) |
| 489 | 16 | decode | 14 | 9 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.462 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 490 | 16 | decode | 14 | 10 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.416 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 491 | 16 | decode | 14 | 11 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.505 | 0.535 | gemv_decode_cublas (0.460 ms) |
| 492 | 16 | decode | 14 | 12 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 493 | 16 | decode | 14 | 13 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.484 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 494 | 16 | decode | 14 | 14 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.498 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 495 | 16 | decode | 14 | 15 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.516 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 496 | 16 | decode | 14 | 16 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 497 | 16 | decode | 14 | 17 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.633 | gemv_decode_cublas (0.560 ms) |
| 498 | 16 | decode | 14 | 18 | 1 | 639 | eager_visipruner_decode_full_kv_cache | 1.559 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 499 | 16 | decode | 14 | 19 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.556 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 500 | 16 | decode | 14 | 20 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.525 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 501 | 16 | decode | 14 | 21 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.523 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 502 | 16 | decode | 14 | 22 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.535 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 503 | 16 | decode | 14 | 23 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.534 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 504 | 16 | decode | 14 | 24 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.529 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 505 | 16 | decode | 14 | 25 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.511 | 0.513 | gemv_decode_cublas (0.455 ms) |
| 506 | 16 | decode | 14 | 26 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.479 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 507 | 16 | decode | 14 | 27 | 1 | 73 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.614 | gemv_decode_cublas (0.555 ms) |
| 508 | 16 | decode | 14 | 28 | 1 | 63 | eager_visipruner_decode_deep_removed_kv_cache | 1.534 | 0.504 | gemv_decode_cublas (0.445 ms) |
| 509 | 16 | decode | 14 | 29 | 1 | 63 | eager_visipruner_decode_deep_removed_kv_cache | 1.564 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 510 | 16 | decode | 14 | 30 | 1 | 63 | eager_visipruner_decode_deep_removed_kv_cache | 1.528 | 0.502 | gemv_decode_cublas (0.444 ms) |
| 511 | 16 | decode | 14 | 31 | 1 | 63 | eager_visipruner_decode_deep_removed_kv_cache | 1.524 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 512 | 17 | decode | 15 | 0 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.624 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 513 | 17 | decode | 15 | 1 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 514 | 17 | decode | 15 | 2 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.566 | 0.531 | gemv_decode_cublas (0.458 ms) |
| 515 | 17 | decode | 15 | 3 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.580 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 516 | 17 | decode | 15 | 4 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.589 | 0.677 | gemv_decode_cublas (0.458 ms) |
| 517 | 17 | decode | 15 | 5 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.536 | gemv_decode_cublas (0.465 ms) |
| 518 | 17 | decode | 15 | 6 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.563 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 519 | 17 | decode | 15 | 7 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 520 | 17 | decode | 15 | 8 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 521 | 17 | decode | 15 | 9 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.533 | gemv_decode_cublas (0.458 ms) |
| 522 | 17 | decode | 15 | 10 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.531 | 0.531 | gemv_decode_cublas (0.458 ms) |
| 523 | 17 | decode | 15 | 11 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 524 | 17 | decode | 15 | 12 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.541 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 525 | 17 | decode | 15 | 13 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 526 | 17 | decode | 15 | 14 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.648 | 0.652 | gemv_decode_cublas (0.582 ms) |
| 527 | 17 | decode | 15 | 15 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.535 | gemv_decode_cublas (0.463 ms) |
| 528 | 17 | decode | 15 | 16 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.595 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 529 | 17 | decode | 15 | 17 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.561 | 0.531 | gemv_decode_cublas (0.457 ms) |
| 530 | 17 | decode | 15 | 18 | 1 | 640 | eager_visipruner_decode_full_kv_cache | 1.576 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 531 | 17 | decode | 15 | 19 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.586 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 532 | 17 | decode | 15 | 20 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.543 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 533 | 17 | decode | 15 | 21 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.549 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 534 | 17 | decode | 15 | 22 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.540 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 535 | 17 | decode | 15 | 23 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.697 | 0.549 | gemv_decode_cublas (0.463 ms) |
| 536 | 17 | decode | 15 | 24 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.551 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 537 | 17 | decode | 15 | 25 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.568 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 538 | 17 | decode | 15 | 26 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.535 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 539 | 17 | decode | 15 | 27 | 1 | 74 | eager_visipruner_decode_middle_pruned_kv_cache | 1.557 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 540 | 17 | decode | 15 | 28 | 1 | 64 | eager_visipruner_decode_deep_removed_kv_cache | 1.533 | 0.504 | gemv_decode_cublas (0.447 ms) |
| 541 | 17 | decode | 15 | 29 | 1 | 64 | eager_visipruner_decode_deep_removed_kv_cache | 1.532 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 542 | 17 | decode | 15 | 30 | 1 | 64 | eager_visipruner_decode_deep_removed_kv_cache | 1.522 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 543 | 17 | decode | 15 | 31 | 1 | 64 | eager_visipruner_decode_deep_removed_kv_cache | 1.500 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 544 | 18 | decode | 16 | 0 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.722 | 0.656 | gemv_decode_cublas (0.506 ms) |
| 545 | 18 | decode | 16 | 1 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.520 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 546 | 18 | decode | 16 | 2 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.554 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 547 | 18 | decode | 16 | 3 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.512 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 548 | 18 | decode | 16 | 4 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.499 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 549 | 18 | decode | 16 | 5 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 550 | 18 | decode | 16 | 6 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.528 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 551 | 18 | decode | 16 | 7 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 552 | 18 | decode | 16 | 8 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.537 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 553 | 18 | decode | 16 | 9 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.508 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 554 | 18 | decode | 16 | 10 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.568 | 0.604 | gemv_decode_cublas (0.456 ms) |
| 555 | 18 | decode | 16 | 11 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.506 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 556 | 18 | decode | 16 | 12 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 557 | 18 | decode | 16 | 13 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 558 | 18 | decode | 16 | 14 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.467 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 559 | 18 | decode | 16 | 15 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.477 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 560 | 18 | decode | 16 | 16 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.459 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 561 | 18 | decode | 16 | 17 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.389 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 562 | 18 | decode | 16 | 18 | 1 | 641 | eager_visipruner_decode_full_kv_cache | 1.384 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 563 | 18 | decode | 16 | 19 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.396 | 0.515 | gemv_decode_cublas (0.456 ms) |
| 564 | 18 | decode | 16 | 20 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.556 | gemv_decode_cublas (0.449 ms) |
| 565 | 18 | decode | 16 | 21 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.408 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 566 | 18 | decode | 16 | 22 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.384 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 567 | 18 | decode | 16 | 23 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.522 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 568 | 18 | decode | 16 | 24 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.479 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 569 | 18 | decode | 16 | 25 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.499 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 570 | 18 | decode | 16 | 26 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.505 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 571 | 18 | decode | 16 | 27 | 1 | 75 | eager_visipruner_decode_middle_pruned_kv_cache | 1.595 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 572 | 18 | decode | 16 | 28 | 1 | 65 | eager_visipruner_decode_deep_removed_kv_cache | 1.475 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 573 | 18 | decode | 16 | 29 | 1 | 65 | eager_visipruner_decode_deep_removed_kv_cache | 1.488 | 0.632 | gemv_decode_cublas (0.555 ms) |
| 574 | 18 | decode | 16 | 30 | 1 | 65 | eager_visipruner_decode_deep_removed_kv_cache | 1.493 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 575 | 18 | decode | 16 | 31 | 1 | 65 | eager_visipruner_decode_deep_removed_kv_cache | 1.524 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 576 | 19 | decode | 17 | 0 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.593 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 577 | 19 | decode | 17 | 1 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.499 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 578 | 19 | decode | 17 | 2 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 579 | 19 | decode | 17 | 3 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 580 | 19 | decode | 17 | 4 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 581 | 19 | decode | 17 | 5 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.500 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 582 | 19 | decode | 17 | 6 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.485 | 0.639 | gemv_decode_cublas (0.566 ms) |
| 583 | 19 | decode | 17 | 7 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 584 | 19 | decode | 17 | 8 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.481 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 585 | 19 | decode | 17 | 9 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 586 | 19 | decode | 17 | 10 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.502 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 587 | 19 | decode | 17 | 11 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.473 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 588 | 19 | decode | 17 | 12 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.484 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 589 | 19 | decode | 17 | 13 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.514 | 0.539 | gemv_decode_cublas (0.466 ms) |
| 590 | 19 | decode | 17 | 14 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 591 | 19 | decode | 17 | 15 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.556 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 592 | 19 | decode | 17 | 16 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.572 | 0.623 | gemv_decode_cublas (0.550 ms) |
| 593 | 19 | decode | 17 | 17 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.526 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 594 | 19 | decode | 17 | 18 | 1 | 642 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 595 | 19 | decode | 17 | 19 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.529 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 596 | 19 | decode | 17 | 20 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.529 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 597 | 19 | decode | 17 | 21 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.529 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 598 | 19 | decode | 17 | 22 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.455 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 599 | 19 | decode | 17 | 23 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.482 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 600 | 19 | decode | 17 | 24 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.509 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 601 | 19 | decode | 17 | 25 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.515 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 602 | 19 | decode | 17 | 26 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.511 | 0.594 | gemv_decode_cublas (0.535 ms) |
| 603 | 19 | decode | 17 | 27 | 1 | 76 | eager_visipruner_decode_middle_pruned_kv_cache | 1.492 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 604 | 19 | decode | 17 | 28 | 1 | 66 | eager_visipruner_decode_deep_removed_kv_cache | 1.479 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 605 | 19 | decode | 17 | 29 | 1 | 66 | eager_visipruner_decode_deep_removed_kv_cache | 1.519 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 606 | 19 | decode | 17 | 30 | 1 | 66 | eager_visipruner_decode_deep_removed_kv_cache | 1.494 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 607 | 19 | decode | 17 | 31 | 1 | 66 | eager_visipruner_decode_deep_removed_kv_cache | 1.483 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 608 | 20 | decode | 18 | 0 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 609 | 20 | decode | 18 | 1 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.530 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 610 | 20 | decode | 18 | 2 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.562 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 611 | 20 | decode | 18 | 3 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.526 | 0.631 | gemv_decode_cublas (0.459 ms) |
| 612 | 20 | decode | 18 | 4 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.567 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 613 | 20 | decode | 18 | 5 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.589 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 614 | 20 | decode | 18 | 6 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.529 | gemv_decode_cublas (0.456 ms) |
| 615 | 20 | decode | 18 | 7 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 616 | 20 | decode | 18 | 8 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 617 | 20 | decode | 18 | 9 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 618 | 20 | decode | 18 | 10 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.537 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 619 | 20 | decode | 18 | 11 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.454 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 620 | 20 | decode | 18 | 12 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.407 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 621 | 20 | decode | 18 | 13 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.645 | 0.660 | gemv_decode_cublas (0.487 ms) |
| 622 | 20 | decode | 18 | 14 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.510 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 623 | 20 | decode | 18 | 15 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.518 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 624 | 20 | decode | 18 | 16 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.609 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 625 | 20 | decode | 18 | 17 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.560 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 626 | 20 | decode | 18 | 18 | 1 | 643 | eager_visipruner_decode_full_kv_cache | 1.579 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 627 | 20 | decode | 18 | 19 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.564 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 628 | 20 | decode | 18 | 20 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.542 | 0.516 | gemv_decode_cublas (0.457 ms) |
| 629 | 20 | decode | 18 | 21 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.550 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 630 | 20 | decode | 18 | 22 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.588 | gemv_decode_cublas (0.449 ms) |
| 631 | 20 | decode | 18 | 23 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 632 | 20 | decode | 18 | 24 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.545 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 633 | 20 | decode | 18 | 25 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.602 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 634 | 20 | decode | 18 | 26 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.553 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 635 | 20 | decode | 18 | 27 | 1 | 77 | eager_visipruner_decode_middle_pruned_kv_cache | 1.633 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 636 | 20 | decode | 18 | 28 | 1 | 67 | eager_visipruner_decode_deep_removed_kv_cache | 1.647 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 637 | 20 | decode | 18 | 29 | 1 | 67 | eager_visipruner_decode_deep_removed_kv_cache | 1.569 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 638 | 20 | decode | 18 | 30 | 1 | 67 | eager_visipruner_decode_deep_removed_kv_cache | 1.566 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 639 | 20 | decode | 18 | 31 | 1 | 67 | eager_visipruner_decode_deep_removed_kv_cache | 1.857 | 0.607 | gemv_decode_cublas (0.549 ms) |
| 640 | 21 | decode | 19 | 0 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.658 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 641 | 21 | decode | 19 | 1 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.580 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 642 | 21 | decode | 19 | 2 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 643 | 21 | decode | 19 | 3 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.506 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 644 | 21 | decode | 19 | 4 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.531 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 645 | 21 | decode | 19 | 5 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.530 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 646 | 21 | decode | 19 | 6 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.496 | 0.540 | gemv_decode_cublas (0.467 ms) |
| 647 | 21 | decode | 19 | 7 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 648 | 21 | decode | 19 | 8 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.582 | 0.598 | gemv_decode_cublas (0.525 ms) |
| 649 | 21 | decode | 19 | 9 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.559 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 650 | 21 | decode | 19 | 10 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 651 | 21 | decode | 19 | 11 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.529 | gemv_decode_cublas (0.456 ms) |
| 652 | 21 | decode | 19 | 12 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.551 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 653 | 21 | decode | 19 | 13 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 654 | 21 | decode | 19 | 14 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.556 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 655 | 21 | decode | 19 | 15 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.518 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 656 | 21 | decode | 19 | 16 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.513 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 657 | 21 | decode | 19 | 17 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 658 | 21 | decode | 19 | 18 | 1 | 644 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.582 | gemv_decode_cublas (0.454 ms) |
| 659 | 21 | decode | 19 | 19 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.560 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 660 | 21 | decode | 19 | 20 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.568 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 661 | 21 | decode | 19 | 21 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.593 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 662 | 21 | decode | 19 | 22 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.539 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 663 | 21 | decode | 19 | 23 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.543 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 664 | 21 | decode | 19 | 24 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.567 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 665 | 21 | decode | 19 | 25 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.517 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 666 | 21 | decode | 19 | 26 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.491 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 667 | 21 | decode | 19 | 27 | 1 | 78 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 668 | 21 | decode | 19 | 28 | 1 | 68 | eager_visipruner_decode_deep_removed_kv_cache | 1.525 | 0.558 | gemv_decode_cublas (0.444 ms) |
| 669 | 21 | decode | 19 | 29 | 1 | 68 | eager_visipruner_decode_deep_removed_kv_cache | 1.549 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 670 | 21 | decode | 19 | 30 | 1 | 68 | eager_visipruner_decode_deep_removed_kv_cache | 1.519 | 0.504 | gemv_decode_cublas (0.445 ms) |
| 671 | 21 | decode | 19 | 31 | 1 | 68 | eager_visipruner_decode_deep_removed_kv_cache | 1.550 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 672 | 22 | decode | 20 | 0 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.595 | 0.526 | gemv_decode_cublas (0.455 ms) |
| 673 | 22 | decode | 20 | 1 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.554 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 674 | 22 | decode | 20 | 2 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 675 | 22 | decode | 20 | 3 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.497 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 676 | 22 | decode | 20 | 4 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.785 | 0.585 | gemv_decode_cublas (0.504 ms) |
| 677 | 22 | decode | 20 | 5 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.494 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 678 | 22 | decode | 20 | 6 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.498 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 679 | 22 | decode | 20 | 7 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.549 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 680 | 22 | decode | 20 | 8 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.550 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 681 | 22 | decode | 20 | 9 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.518 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 682 | 22 | decode | 20 | 10 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.566 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 683 | 22 | decode | 20 | 11 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 684 | 22 | decode | 20 | 12 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 685 | 22 | decode | 20 | 13 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.563 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 686 | 22 | decode | 20 | 14 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.645 | 0.648 | gemv_decode_cublas (0.577 ms) |
| 687 | 22 | decode | 20 | 15 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.559 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 688 | 22 | decode | 20 | 16 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.529 | gemv_decode_cublas (0.456 ms) |
| 689 | 22 | decode | 20 | 17 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.520 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 690 | 22 | decode | 20 | 18 | 1 | 645 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 691 | 22 | decode | 20 | 19 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.501 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 692 | 22 | decode | 20 | 20 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.395 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 693 | 22 | decode | 20 | 21 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.407 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 694 | 22 | decode | 20 | 22 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.477 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 695 | 22 | decode | 20 | 23 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.546 | 0.621 | gemv_decode_cublas (0.562 ms) |
| 696 | 22 | decode | 20 | 24 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.530 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 697 | 22 | decode | 20 | 25 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.522 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 698 | 22 | decode | 20 | 26 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.527 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 699 | 22 | decode | 20 | 27 | 1 | 79 | eager_visipruner_decode_middle_pruned_kv_cache | 1.575 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 700 | 22 | decode | 20 | 28 | 1 | 69 | eager_visipruner_decode_deep_removed_kv_cache | 1.525 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 701 | 22 | decode | 20 | 29 | 1 | 69 | eager_visipruner_decode_deep_removed_kv_cache | 1.527 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 702 | 22 | decode | 20 | 30 | 1 | 69 | eager_visipruner_decode_deep_removed_kv_cache | 1.507 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 703 | 22 | decode | 20 | 31 | 1 | 69 | eager_visipruner_decode_deep_removed_kv_cache | 1.531 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 704 | 23 | decode | 21 | 0 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.592 | 0.636 | gemv_decode_cublas (0.565 ms) |
| 705 | 23 | decode | 21 | 1 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.526 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 706 | 23 | decode | 21 | 2 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.551 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 707 | 23 | decode | 21 | 3 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.544 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 708 | 23 | decode | 21 | 4 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.528 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 709 | 23 | decode | 21 | 5 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 710 | 23 | decode | 21 | 6 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.470 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 711 | 23 | decode | 21 | 7 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.452 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 712 | 23 | decode | 21 | 8 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 713 | 23 | decode | 21 | 9 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.510 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 714 | 23 | decode | 21 | 10 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.575 | gemv_decode_cublas (0.472 ms) |
| 715 | 23 | decode | 21 | 11 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 716 | 23 | decode | 21 | 12 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 717 | 23 | decode | 21 | 13 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.530 | gemv_decode_cublas (0.458 ms) |
| 718 | 23 | decode | 21 | 14 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 719 | 23 | decode | 21 | 15 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.514 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 720 | 23 | decode | 21 | 16 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.517 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 721 | 23 | decode | 21 | 17 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 722 | 23 | decode | 21 | 18 | 1 | 646 | eager_visipruner_decode_full_kv_cache | 1.533 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 723 | 23 | decode | 21 | 19 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.540 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 724 | 23 | decode | 21 | 20 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.580 | 0.583 | gemv_decode_cublas (0.524 ms) |
| 725 | 23 | decode | 21 | 21 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.570 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 726 | 23 | decode | 21 | 22 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.521 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 727 | 23 | decode | 21 | 23 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.521 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 728 | 23 | decode | 21 | 24 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.550 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 729 | 23 | decode | 21 | 25 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.523 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 730 | 23 | decode | 21 | 26 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 731 | 23 | decode | 21 | 27 | 1 | 80 | eager_visipruner_decode_middle_pruned_kv_cache | 1.463 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 732 | 23 | decode | 21 | 28 | 1 | 70 | eager_visipruner_decode_deep_removed_kv_cache | 1.445 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 733 | 23 | decode | 21 | 29 | 1 | 70 | eager_visipruner_decode_deep_removed_kv_cache | 1.470 | 0.619 | gemv_decode_cublas (0.561 ms) |
| 734 | 23 | decode | 21 | 30 | 1 | 70 | eager_visipruner_decode_deep_removed_kv_cache | 1.444 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 735 | 23 | decode | 21 | 31 | 1 | 70 | eager_visipruner_decode_deep_removed_kv_cache | 1.476 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 736 | 24 | decode | 22 | 0 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.608 | 0.524 | gemv_decode_cublas (0.453 ms) |
| 737 | 24 | decode | 22 | 1 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.553 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 738 | 24 | decode | 22 | 2 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.564 | 0.527 | gemv_decode_cublas (0.455 ms) |
| 739 | 24 | decode | 22 | 3 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.556 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 740 | 24 | decode | 22 | 4 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 741 | 24 | decode | 22 | 5 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.534 | gemv_decode_cublas (0.462 ms) |
| 742 | 24 | decode | 22 | 6 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.585 | 0.636 | gemv_decode_cublas (0.562 ms) |
| 743 | 24 | decode | 22 | 7 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.526 | gemv_decode_cublas (0.455 ms) |
| 744 | 24 | decode | 22 | 8 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.532 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 745 | 24 | decode | 22 | 9 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 746 | 24 | decode | 22 | 10 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 747 | 24 | decode | 22 | 11 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 748 | 24 | decode | 22 | 12 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.463 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 749 | 24 | decode | 22 | 13 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.405 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 750 | 24 | decode | 22 | 14 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.441 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 751 | 24 | decode | 22 | 15 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.529 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 752 | 24 | decode | 22 | 16 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.550 | 0.590 | gemv_decode_cublas (0.517 ms) |
| 753 | 24 | decode | 22 | 17 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.519 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 754 | 24 | decode | 22 | 18 | 1 | 647 | eager_visipruner_decode_full_kv_cache | 1.537 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 755 | 24 | decode | 22 | 19 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.546 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 756 | 24 | decode | 22 | 20 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.781 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 757 | 24 | decode | 22 | 21 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.568 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 758 | 24 | decode | 22 | 22 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.558 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 759 | 24 | decode | 22 | 23 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.543 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 760 | 24 | decode | 22 | 24 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.562 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 761 | 24 | decode | 22 | 25 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.579 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 762 | 24 | decode | 22 | 26 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.621 | 0.574 | gemv_decode_cublas (0.446 ms) |
| 763 | 24 | decode | 22 | 27 | 1 | 81 | eager_visipruner_decode_middle_pruned_kv_cache | 1.592 | 0.503 | gemv_decode_cublas (0.445 ms) |
| 764 | 24 | decode | 22 | 28 | 1 | 71 | eager_visipruner_decode_deep_removed_kv_cache | 1.526 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 765 | 24 | decode | 22 | 29 | 1 | 71 | eager_visipruner_decode_deep_removed_kv_cache | 1.485 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 766 | 24 | decode | 22 | 30 | 1 | 71 | eager_visipruner_decode_deep_removed_kv_cache | 1.545 | 0.504 | gemv_decode_cublas (0.446 ms) |
| 767 | 24 | decode | 22 | 31 | 1 | 71 | eager_visipruner_decode_deep_removed_kv_cache | 1.495 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 768 | 25 | decode | 23 | 0 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.573 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 769 | 25 | decode | 23 | 1 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.567 | 0.535 | gemv_decode_cublas (0.463 ms) |
| 770 | 25 | decode | 23 | 2 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.626 | gemv_decode_cublas (0.521 ms) |
| 771 | 25 | decode | 23 | 3 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.526 | gemv_decode_cublas (0.456 ms) |
| 772 | 25 | decode | 23 | 4 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.507 | 0.535 | gemv_decode_cublas (0.463 ms) |
| 773 | 25 | decode | 23 | 5 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.483 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 774 | 25 | decode | 23 | 6 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.530 | 0.530 | gemv_decode_cublas (0.456 ms) |
| 775 | 25 | decode | 23 | 7 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.452 | 0.534 | gemv_decode_cublas (0.459 ms) |
| 776 | 25 | decode | 23 | 8 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.424 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 777 | 25 | decode | 23 | 9 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.399 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 778 | 25 | decode | 23 | 10 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.521 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 779 | 25 | decode | 23 | 11 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.532 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 780 | 25 | decode | 23 | 12 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.618 | gemv_decode_cublas (0.544 ms) |
| 781 | 25 | decode | 23 | 13 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.570 | 0.529 | gemv_decode_cublas (0.459 ms) |
| 782 | 25 | decode | 23 | 14 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.535 | gemv_decode_cublas (0.463 ms) |
| 783 | 25 | decode | 23 | 15 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.536 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 784 | 25 | decode | 23 | 16 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.548 | 0.533 | gemv_decode_cublas (0.458 ms) |
| 785 | 25 | decode | 23 | 17 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.531 | gemv_decode_cublas (0.457 ms) |
| 786 | 25 | decode | 23 | 18 | 1 | 648 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.531 | gemv_decode_cublas (0.458 ms) |
| 787 | 25 | decode | 23 | 19 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.531 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 788 | 25 | decode | 23 | 20 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.533 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 789 | 25 | decode | 23 | 21 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.526 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 790 | 25 | decode | 23 | 22 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.578 | 0.555 | gemv_decode_cublas (0.482 ms) |
| 791 | 25 | decode | 23 | 23 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.520 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 792 | 25 | decode | 23 | 24 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.531 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 793 | 25 | decode | 23 | 25 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.583 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 794 | 25 | decode | 23 | 26 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.526 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 795 | 25 | decode | 23 | 27 | 1 | 82 | eager_visipruner_decode_middle_pruned_kv_cache | 1.497 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 796 | 25 | decode | 23 | 28 | 1 | 72 | eager_visipruner_decode_deep_removed_kv_cache | 1.544 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 797 | 25 | decode | 23 | 29 | 1 | 72 | eager_visipruner_decode_deep_removed_kv_cache | 1.577 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 798 | 25 | decode | 23 | 30 | 1 | 72 | eager_visipruner_decode_deep_removed_kv_cache | 1.564 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 799 | 25 | decode | 23 | 31 | 1 | 72 | eager_visipruner_decode_deep_removed_kv_cache | 1.547 | 0.617 | gemv_decode_cublas (0.559 ms) |
| 800 | 26 | decode | 24 | 0 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.627 | 0.524 | gemv_decode_cublas (0.453 ms) |
| 801 | 26 | decode | 24 | 1 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.567 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 802 | 26 | decode | 24 | 2 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.563 | 0.526 | gemv_decode_cublas (0.455 ms) |
| 803 | 26 | decode | 24 | 3 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.515 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 804 | 26 | decode | 24 | 4 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.504 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 805 | 26 | decode | 24 | 5 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.445 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 806 | 26 | decode | 24 | 6 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.515 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 807 | 26 | decode | 24 | 7 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.546 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 808 | 26 | decode | 24 | 8 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.555 | 0.640 | gemv_decode_cublas (0.566 ms) |
| 809 | 26 | decode | 24 | 9 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 810 | 26 | decode | 24 | 10 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 811 | 26 | decode | 24 | 11 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.559 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 812 | 26 | decode | 24 | 12 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.571 | 0.533 | gemv_decode_cublas (0.459 ms) |
| 813 | 26 | decode | 24 | 13 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.496 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 814 | 26 | decode | 24 | 14 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.480 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 815 | 26 | decode | 24 | 15 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.513 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 816 | 26 | decode | 24 | 16 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.543 | gemv_decode_cublas (0.469 ms) |
| 817 | 26 | decode | 24 | 17 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.576 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 818 | 26 | decode | 24 | 18 | 1 | 649 | eager_visipruner_decode_full_kv_cache | 1.572 | 0.643 | gemv_decode_cublas (0.508 ms) |
| 819 | 26 | decode | 24 | 19 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.543 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 820 | 26 | decode | 24 | 20 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.533 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 821 | 26 | decode | 24 | 21 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.554 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 822 | 26 | decode | 24 | 22 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.537 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 823 | 26 | decode | 24 | 23 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.489 | 0.508 | gemv_decode_cublas (0.450 ms) |
| 824 | 26 | decode | 24 | 24 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.459 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 825 | 26 | decode | 24 | 25 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.483 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 826 | 26 | decode | 24 | 26 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.552 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 827 | 26 | decode | 24 | 27 | 1 | 83 | eager_visipruner_decode_middle_pruned_kv_cache | 1.878 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 828 | 26 | decode | 24 | 28 | 1 | 73 | eager_visipruner_decode_deep_removed_kv_cache | 1.570 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 829 | 26 | decode | 24 | 29 | 1 | 73 | eager_visipruner_decode_deep_removed_kv_cache | 1.549 | 0.578 | gemv_decode_cublas (0.499 ms) |
| 830 | 26 | decode | 24 | 30 | 1 | 73 | eager_visipruner_decode_deep_removed_kv_cache | 1.592 | 0.504 | gemv_decode_cublas (0.445 ms) |
| 831 | 26 | decode | 24 | 31 | 1 | 73 | eager_visipruner_decode_deep_removed_kv_cache | 1.549 | 0.504 | gemv_decode_cublas (0.445 ms) |
| 832 | 27 | decode | 25 | 0 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.645 | 0.525 | gemv_decode_cublas (0.455 ms) |
| 833 | 27 | decode | 25 | 1 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.628 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 834 | 27 | decode | 25 | 2 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.600 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 835 | 27 | decode | 25 | 3 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.547 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 836 | 27 | decode | 25 | 4 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.583 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 837 | 27 | decode | 25 | 5 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.551 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 838 | 27 | decode | 25 | 6 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.537 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 839 | 27 | decode | 25 | 7 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.540 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 840 | 27 | decode | 25 | 8 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.535 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 841 | 27 | decode | 25 | 9 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 842 | 27 | decode | 25 | 10 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.480 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 843 | 27 | decode | 25 | 11 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.392 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 844 | 27 | decode | 25 | 12 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.407 | 0.543 | gemv_decode_cublas (0.469 ms) |
| 845 | 27 | decode | 25 | 13 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.404 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 846 | 27 | decode | 25 | 14 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.399 | 0.536 | gemv_decode_cublas (0.463 ms) |
| 847 | 27 | decode | 25 | 15 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.425 | 0.625 | gemv_decode_cublas (0.466 ms) |
| 848 | 27 | decode | 25 | 16 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.528 | gemv_decode_cublas (0.457 ms) |
| 849 | 27 | decode | 25 | 17 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.421 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 850 | 27 | decode | 25 | 18 | 1 | 650 | eager_visipruner_decode_full_kv_cache | 1.521 | 0.527 | gemv_decode_cublas (0.455 ms) |
| 851 | 27 | decode | 25 | 19 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.527 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 852 | 27 | decode | 25 | 20 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.551 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 853 | 27 | decode | 25 | 21 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.552 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 854 | 27 | decode | 25 | 22 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.536 | 0.553 | gemv_decode_cublas (0.494 ms) |
| 855 | 27 | decode | 25 | 23 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.553 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 856 | 27 | decode | 25 | 24 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.569 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 857 | 27 | decode | 25 | 25 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.547 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 858 | 27 | decode | 25 | 26 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.489 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 859 | 27 | decode | 25 | 27 | 1 | 84 | eager_visipruner_decode_middle_pruned_kv_cache | 1.482 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 860 | 27 | decode | 25 | 28 | 1 | 74 | eager_visipruner_decode_deep_removed_kv_cache | 1.539 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 861 | 27 | decode | 25 | 29 | 1 | 74 | eager_visipruner_decode_deep_removed_kv_cache | 1.500 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 862 | 27 | decode | 25 | 30 | 1 | 74 | eager_visipruner_decode_deep_removed_kv_cache | 1.430 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 863 | 27 | decode | 25 | 31 | 1 | 74 | eager_visipruner_decode_deep_removed_kv_cache | 1.508 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 864 | 28 | decode | 26 | 0 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.607 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 865 | 28 | decode | 26 | 1 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.558 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 866 | 28 | decode | 26 | 2 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.563 | 0.597 | gemv_decode_cublas (0.458 ms) |
| 867 | 28 | decode | 26 | 3 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.532 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 868 | 28 | decode | 26 | 4 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.552 | 0.531 | gemv_decode_cublas (0.459 ms) |
| 869 | 28 | decode | 26 | 5 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.510 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 870 | 28 | decode | 26 | 6 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.531 | gemv_decode_cublas (0.457 ms) |
| 871 | 28 | decode | 26 | 7 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 872 | 28 | decode | 26 | 8 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.549 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 873 | 28 | decode | 26 | 9 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.578 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 874 | 28 | decode | 26 | 10 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.509 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 875 | 28 | decode | 26 | 11 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.505 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 876 | 28 | decode | 26 | 12 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.493 | 0.621 | gemv_decode_cublas (0.500 ms) |
| 877 | 28 | decode | 26 | 13 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.534 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 878 | 28 | decode | 26 | 14 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.524 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 879 | 28 | decode | 26 | 15 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.542 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 880 | 28 | decode | 26 | 16 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.507 | 0.535 | gemv_decode_cublas (0.460 ms) |
| 881 | 28 | decode | 26 | 17 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.486 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 882 | 28 | decode | 26 | 18 | 1 | 651 | eager_visipruner_decode_full_kv_cache | 1.482 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 883 | 28 | decode | 26 | 19 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.517 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 884 | 28 | decode | 26 | 20 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.555 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 885 | 28 | decode | 26 | 21 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.501 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 886 | 28 | decode | 26 | 22 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.479 | 0.637 | gemv_decode_cublas (0.474 ms) |
| 887 | 28 | decode | 26 | 23 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.487 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 888 | 28 | decode | 26 | 24 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.600 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 889 | 28 | decode | 26 | 25 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.489 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 890 | 28 | decode | 26 | 26 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 891 | 28 | decode | 26 | 27 | 1 | 85 | eager_visipruner_decode_middle_pruned_kv_cache | 1.531 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 892 | 28 | decode | 26 | 28 | 1 | 75 | eager_visipruner_decode_deep_removed_kv_cache | 1.564 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 893 | 28 | decode | 26 | 29 | 1 | 75 | eager_visipruner_decode_deep_removed_kv_cache | 1.556 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 894 | 28 | decode | 26 | 30 | 1 | 75 | eager_visipruner_decode_deep_removed_kv_cache | 1.548 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 895 | 28 | decode | 26 | 31 | 1 | 75 | eager_visipruner_decode_deep_removed_kv_cache | 1.566 | 0.574 | gemv_decode_cublas (0.515 ms) |
| 896 | 29 | decode | 27 | 0 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.622 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 897 | 29 | decode | 27 | 1 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.561 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 898 | 29 | decode | 27 | 2 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.543 | 0.529 | gemv_decode_cublas (0.457 ms) |
| 899 | 29 | decode | 27 | 3 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.567 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 900 | 29 | decode | 27 | 4 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.544 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 901 | 29 | decode | 27 | 5 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.461 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 902 | 29 | decode | 27 | 6 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.494 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 903 | 29 | decode | 27 | 7 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.500 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 904 | 29 | decode | 27 | 8 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.597 | 0.533 | gemv_decode_cublas (0.462 ms) |
| 905 | 29 | decode | 27 | 9 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.529 | gemv_decode_cublas (0.458 ms) |
| 906 | 29 | decode | 27 | 10 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 907 | 29 | decode | 27 | 11 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 908 | 29 | decode | 27 | 12 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.478 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 909 | 29 | decode | 27 | 13 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 910 | 29 | decode | 27 | 14 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.506 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 911 | 29 | decode | 27 | 15 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.511 | 0.571 | gemv_decode_cublas (0.467 ms) |
| 912 | 29 | decode | 27 | 16 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.496 | 0.537 | gemv_decode_cublas (0.464 ms) |
| 913 | 29 | decode | 27 | 17 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.501 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 914 | 29 | decode | 27 | 18 | 1 | 652 | eager_visipruner_decode_full_kv_cache | 1.406 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 915 | 29 | decode | 27 | 19 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.407 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 916 | 29 | decode | 27 | 20 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.377 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 917 | 29 | decode | 27 | 21 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.348 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 918 | 29 | decode | 27 | 22 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.400 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 919 | 29 | decode | 27 | 23 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.469 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 920 | 29 | decode | 27 | 24 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.511 | gemv_decode_cublas (0.453 ms) |
| 921 | 29 | decode | 27 | 25 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.494 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 922 | 29 | decode | 27 | 26 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.519 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 923 | 29 | decode | 27 | 27 | 1 | 86 | eager_visipruner_decode_middle_pruned_kv_cache | 1.578 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 924 | 29 | decode | 27 | 28 | 1 | 76 | eager_visipruner_decode_deep_removed_kv_cache | 1.625 | 0.573 | gemv_decode_cublas (0.448 ms) |
| 925 | 29 | decode | 27 | 29 | 1 | 76 | eager_visipruner_decode_deep_removed_kv_cache | 1.530 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 926 | 29 | decode | 27 | 30 | 1 | 76 | eager_visipruner_decode_deep_removed_kv_cache | 1.566 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 927 | 29 | decode | 27 | 31 | 1 | 76 | eager_visipruner_decode_deep_removed_kv_cache | 1.540 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 928 | 30 | decode | 28 | 0 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.587 | 0.531 | gemv_decode_cublas (0.460 ms) |
| 929 | 30 | decode | 28 | 1 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.570 | 0.532 | gemv_decode_cublas (0.460 ms) |
| 930 | 30 | decode | 28 | 2 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.576 | 0.569 | gemv_decode_cublas (0.463 ms) |
| 931 | 30 | decode | 28 | 3 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.587 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 932 | 30 | decode | 28 | 4 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.528 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 933 | 30 | decode | 28 | 5 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.558 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 934 | 30 | decode | 28 | 6 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.527 | 0.533 | gemv_decode_cublas (0.460 ms) |
| 935 | 30 | decode | 28 | 7 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.497 | 0.534 | gemv_decode_cublas (0.459 ms) |
| 936 | 30 | decode | 28 | 8 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.481 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 937 | 30 | decode | 28 | 9 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.433 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 938 | 30 | decode | 28 | 10 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.411 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 939 | 30 | decode | 28 | 11 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.534 | gemv_decode_cublas (0.461 ms) |
| 940 | 30 | decode | 28 | 12 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.502 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 941 | 30 | decode | 28 | 13 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.515 | 0.652 | gemv_decode_cublas (0.564 ms) |
| 942 | 30 | decode | 28 | 14 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.528 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 943 | 30 | decode | 28 | 15 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.472 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 944 | 30 | decode | 28 | 16 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.484 | 0.530 | gemv_decode_cublas (0.457 ms) |
| 945 | 30 | decode | 28 | 17 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.454 | 0.535 | gemv_decode_cublas (0.460 ms) |
| 946 | 30 | decode | 28 | 18 | 1 | 653 | eager_visipruner_decode_full_kv_cache | 1.408 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 947 | 30 | decode | 28 | 19 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.387 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 948 | 30 | decode | 28 | 20 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.374 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 949 | 30 | decode | 28 | 21 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.511 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 950 | 30 | decode | 28 | 22 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.556 | 0.512 | gemv_decode_cublas (0.453 ms) |
| 951 | 30 | decode | 28 | 23 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.539 | 0.617 | gemv_decode_cublas (0.558 ms) |
| 952 | 30 | decode | 28 | 24 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.538 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 953 | 30 | decode | 28 | 25 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.547 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 954 | 30 | decode | 28 | 26 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.535 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 955 | 30 | decode | 28 | 27 | 1 | 87 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 956 | 30 | decode | 28 | 28 | 1 | 77 | eager_visipruner_decode_deep_removed_kv_cache | 1.551 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 957 | 30 | decode | 28 | 29 | 1 | 77 | eager_visipruner_decode_deep_removed_kv_cache | 1.425 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 958 | 30 | decode | 28 | 30 | 1 | 77 | eager_visipruner_decode_deep_removed_kv_cache | 1.459 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 959 | 30 | decode | 28 | 31 | 1 | 77 | eager_visipruner_decode_deep_removed_kv_cache | 1.399 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 960 | 31 | decode | 29 | 0 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.497 | 0.647 | gemv_decode_cublas (0.558 ms) |
| 961 | 31 | decode | 29 | 1 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.421 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 962 | 31 | decode | 29 | 2 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.406 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 963 | 31 | decode | 29 | 3 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.423 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 964 | 31 | decode | 29 | 4 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.492 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 965 | 31 | decode | 29 | 5 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.481 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 966 | 31 | decode | 29 | 6 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.447 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 967 | 31 | decode | 29 | 7 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.440 | 0.535 | gemv_decode_cublas (0.462 ms) |
| 968 | 31 | decode | 29 | 8 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.491 | 0.538 | gemv_decode_cublas (0.465 ms) |
| 969 | 31 | decode | 29 | 9 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.564 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 970 | 31 | decode | 29 | 10 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.579 | 0.583 | gemv_decode_cublas (0.469 ms) |
| 971 | 31 | decode | 29 | 11 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.518 | 0.527 | gemv_decode_cublas (0.457 ms) |
| 972 | 31 | decode | 29 | 12 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.539 | 0.532 | gemv_decode_cublas (0.461 ms) |
| 973 | 31 | decode | 29 | 13 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.545 | 0.531 | gemv_decode_cublas (0.458 ms) |
| 974 | 31 | decode | 29 | 14 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.501 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 975 | 31 | decode | 29 | 15 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.485 | 0.540 | gemv_decode_cublas (0.465 ms) |
| 976 | 31 | decode | 29 | 16 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.377 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 977 | 31 | decode | 29 | 17 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.369 | 0.532 | gemv_decode_cublas (0.458 ms) |
| 978 | 31 | decode | 29 | 18 | 1 | 654 | eager_visipruner_decode_full_kv_cache | 1.418 | 0.542 | gemv_decode_cublas (0.467 ms) |
| 979 | 31 | decode | 29 | 19 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.419 | 0.513 | gemv_decode_cublas (0.454 ms) |
| 980 | 31 | decode | 29 | 20 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.487 | 0.653 | gemv_decode_cublas (0.527 ms) |
| 981 | 31 | decode | 29 | 21 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.476 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 982 | 31 | decode | 29 | 22 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.493 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 983 | 31 | decode | 29 | 23 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.480 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 984 | 31 | decode | 29 | 24 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.452 | 0.506 | gemv_decode_cublas (0.448 ms) |
| 985 | 31 | decode | 29 | 25 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.457 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 986 | 31 | decode | 29 | 26 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.405 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 987 | 31 | decode | 29 | 27 | 1 | 88 | eager_visipruner_decode_middle_pruned_kv_cache | 1.677 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 988 | 31 | decode | 29 | 28 | 1 | 78 | eager_visipruner_decode_deep_removed_kv_cache | 1.513 | 0.510 | gemv_decode_cublas (0.452 ms) |
| 989 | 31 | decode | 29 | 29 | 1 | 78 | eager_visipruner_decode_deep_removed_kv_cache | 1.556 | 0.507 | gemv_decode_cublas (0.449 ms) |
| 990 | 31 | decode | 29 | 30 | 1 | 78 | eager_visipruner_decode_deep_removed_kv_cache | 1.569 | 0.629 | gemv_decode_cublas (0.445 ms) |
| 991 | 31 | decode | 29 | 31 | 1 | 78 | eager_visipruner_decode_deep_removed_kv_cache | 1.531 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 992 | 32 | decode | 30 | 0 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.578 | 0.523 | gemv_decode_cublas (0.453 ms) |
| 993 | 32 | decode | 30 | 1 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.520 | 0.527 | gemv_decode_cublas (0.455 ms) |
| 994 | 32 | decode | 30 | 2 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.498 | 0.528 | gemv_decode_cublas (0.456 ms) |
| 995 | 32 | decode | 30 | 3 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.379 | 0.534 | gemv_decode_cublas (0.460 ms) |
| 996 | 32 | decode | 30 | 4 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.419 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 997 | 32 | decode | 30 | 5 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.474 | 0.540 | gemv_decode_cublas (0.466 ms) |
| 998 | 32 | decode | 30 | 6 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.476 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 999 | 32 | decode | 30 | 7 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.597 | gemv_decode_cublas (0.523 ms) |
| 1000 | 32 | decode | 30 | 8 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.538 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 1001 | 32 | decode | 30 | 9 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.469 | 0.527 | gemv_decode_cublas (0.456 ms) |
| 1002 | 32 | decode | 30 | 10 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.532 | 0.526 | gemv_decode_cublas (0.452 ms) |
| 1003 | 32 | decode | 30 | 11 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.525 | 0.532 | gemv_decode_cublas (0.459 ms) |
| 1004 | 32 | decode | 30 | 12 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.503 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 1005 | 32 | decode | 30 | 13 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.490 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 1006 | 32 | decode | 30 | 14 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.522 | 0.536 | gemv_decode_cublas (0.462 ms) |
| 1007 | 32 | decode | 30 | 15 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.560 | 0.535 | gemv_decode_cublas (0.461 ms) |
| 1008 | 32 | decode | 30 | 16 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.527 | 0.537 | gemv_decode_cublas (0.463 ms) |
| 1009 | 32 | decode | 30 | 17 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.595 | 0.618 | gemv_decode_cublas (0.455 ms) |
| 1010 | 32 | decode | 30 | 18 | 1 | 655 | eager_visipruner_decode_full_kv_cache | 1.549 | 0.530 | gemv_decode_cublas (0.459 ms) |
| 1011 | 32 | decode | 30 | 19 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.566 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 1012 | 32 | decode | 30 | 20 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.533 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 1013 | 32 | decode | 30 | 21 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.528 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 1014 | 32 | decode | 30 | 22 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.535 | 0.507 | gemv_decode_cublas (0.448 ms) |
| 1015 | 32 | decode | 30 | 23 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.509 | 0.509 | gemv_decode_cublas (0.450 ms) |
| 1016 | 32 | decode | 30 | 24 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.522 | 0.509 | gemv_decode_cublas (0.451 ms) |
| 1017 | 32 | decode | 30 | 25 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.496 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 1018 | 32 | decode | 30 | 26 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.480 | 0.596 | gemv_decode_cublas (0.537 ms) |
| 1019 | 32 | decode | 30 | 27 | 1 | 89 | eager_visipruner_decode_middle_pruned_kv_cache | 1.467 | 0.505 | gemv_decode_cublas (0.445 ms) |
| 1020 | 32 | decode | 30 | 28 | 1 | 79 | eager_visipruner_decode_deep_removed_kv_cache | 1.489 | 0.505 | gemv_decode_cublas (0.446 ms) |
| 1021 | 32 | decode | 30 | 29 | 1 | 79 | eager_visipruner_decode_deep_removed_kv_cache | 1.480 | 0.505 | gemv_decode_cublas (0.447 ms) |
| 1022 | 32 | decode | 30 | 30 | 1 | 79 | eager_visipruner_decode_deep_removed_kv_cache | 1.469 | 0.506 | gemv_decode_cublas (0.447 ms) |
| 1023 | 32 | decode | 30 | 31 | 1 | 79 | eager_visipruner_decode_deep_removed_kv_cache | 1.497 | 0.505 | gemv_decode_cublas (0.446 ms) |

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | operator path | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---|
| 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.054 | 2.077 | gemm_tensorcore (1.696 ms) |
| 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.140 | 2.179 | gemm_tensorcore (1.828 ms) |
| 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.196 | 2.048 | gemm_tensorcore (1.709 ms) |
| 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.144 | 2.057 | gemm_tensorcore (1.701 ms) |
| 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.258 | 2.053 | gemm_tensorcore (1.712 ms) |
| 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.190 | 2.061 | gemm_tensorcore (1.707 ms) |
| 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs | 2.161 | 2.051 | gemm_tensorcore (1.712 ms) |
| 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.668 | 2.240 | gemm_tensorcore (1.707 ms) |
| 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.683 | 2.096 | gemm_tensorcore (1.702 ms) |
| 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.462 | 2.116 | gemm_tensorcore (1.709 ms) |
| 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.589 | 2.107 | gemm_tensorcore (1.709 ms) |
| 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.288 | 2.222 | gemm_tensorcore (1.749 ms) |
| 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.405 | 2.111 | gemm_tensorcore (1.715 ms) |
| 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.215 | 2.113 | gemm_tensorcore (1.707 ms) |
| 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.626 | 2.103 | gemm_tensorcore (1.709 ms) |
| 15 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.364 | 2.116 | gemm_tensorcore (1.708 ms) |
| 16 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.757 | 2.164 | gemm_tensorcore (1.707 ms) |
| 17 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.367 | 2.120 | gemm_tensorcore (1.711 ms) |
| 18 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.886 | 2.127 | gemm_tensorcore (1.711 ms) |
| 19 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.458 | 0.625 | gemm_tensorcore (0.508 ms) |
| 20 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.413 | 0.619 | gemm_tensorcore (0.501 ms) |
| 21 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.258 | 0.729 | gemm_tensorcore (0.497 ms) |
| 22 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.314 | 0.619 | gemm_tensorcore (0.502 ms) |
| 23 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.167 | 0.619 | gemm_tensorcore (0.502 ms) |
| 24 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.340 | 0.617 | gemm_tensorcore (0.500 ms) |
| 25 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.234 | 0.619 | gemm_tensorcore (0.502 ms) |
| 26 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.484 | 0.626 | gemm_tensorcore (0.509 ms) |
| 27 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.293 | 0.618 | gemm_tensorcore (0.501 ms) |
| 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.811 | 0.607 | gemm_tensorcore (0.495 ms) |
| 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.938 | 0.580 | gemm_tensorcore (0.497 ms) |
| 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.797 | 0.580 | gemm_tensorcore (0.498 ms) |
| 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.749 | 0.583 | gemm_tensorcore (0.500 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---|
| 0 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.585 | 0.539 | gemv_decode_cublas (0.464 ms) |
| 1 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.539 | 0.533 | gemv_decode_cublas (0.461 ms) |
| 2 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.532 | 0.536 | gemv_decode_cublas (0.460 ms) |
| 3 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.527 | 0.546 | gemv_decode_cublas (0.470 ms) |
| 4 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.528 | 0.543 | gemv_decode_cublas (0.462 ms) |
| 5 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.539 | 0.540 | gemv_decode_cublas (0.463 ms) |
| 6 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.534 | 0.541 | gemv_decode_cublas (0.467 ms) |
| 7 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.537 | 0.544 | gemv_decode_cublas (0.467 ms) |
| 8 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.533 | 0.543 | gemv_decode_cublas (0.470 ms) |
| 9 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.525 | 0.540 | gemv_decode_cublas (0.463 ms) |
| 10 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.523 | 0.540 | gemv_decode_cublas (0.460 ms) |
| 11 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.522 | 0.539 | gemv_decode_cublas (0.460 ms) |
| 12 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.516 | 0.546 | gemv_decode_cublas (0.468 ms) |
| 13 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.529 | 0.547 | gemv_decode_cublas (0.465 ms) |
| 14 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.516 | 0.541 | gemv_decode_cublas (0.468 ms) |
| 15 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.527 | 0.545 | gemv_decode_cublas (0.466 ms) |
| 16 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.514 | 0.543 | gemv_decode_cublas (0.469 ms) |
| 17 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.528 | 0.543 | gemv_decode_cublas (0.467 ms) |
| 18 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.508 | 0.539 | gemv_decode_cublas (0.462 ms) |
| 19 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.569 | 0.517 | gemv_decode_cublas (0.454 ms) |
| 20 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.520 | 0.518 | gemv_decode_cublas (0.455 ms) |
| 21 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.515 | 0.512 | gemv_decode_cublas (0.452 ms) |
| 22 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.522 | 0.518 | gemv_decode_cublas (0.452 ms) |
| 23 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.530 | 0.523 | gemv_decode_cublas (0.458 ms) |
| 24 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.523 | 0.513 | gemv_decode_cublas (0.452 ms) |
| 25 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.514 | 0.508 | gemv_decode_cublas (0.449 ms) |
| 26 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.522 | 0.522 | gemv_decode_cublas (0.459 ms) |
| 27 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.536 | 0.510 | gemv_decode_cublas (0.451 ms) |
| 28 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.529 | 0.516 | gemv_decode_cublas (0.448 ms) |
| 29 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.528 | 0.519 | gemv_decode_cublas (0.459 ms) |
| 30 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.542 | 0.512 | gemv_decode_cublas (0.450 ms) |
| 31 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.547 | 0.516 | gemv_decode_cublas (0.456 ms) |
