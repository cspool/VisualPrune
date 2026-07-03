# E2 visipruner-full eager 32-token layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `visipruner-full`
- description: Native VisPrune full path: shallow + middle + deep.
- max_new_tokens: `32`
- use_flash_attn: `False`
- use_visipruner: `True`
- visipruner_decode_backend: `eager`

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_eager_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_eager_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_eager_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_eager_32tok_layer_kernel_breakdown.csv`
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1544.539 |
| generate_total_ms | 1508.613 |
| prepare_multimodal_ms | 12.726 |
| vision_encode_project_ms | 11.223 |
| forward_prefill_ms | 84.792 |
| forward_decode_sum_ms | 1389.038 |
| value_aware_token_selection_ms | 5.945 |

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

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | nsys range ms | nsys kernel ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.317 | 1.689 | 1.199 | 2.931 | 1.116 | gemm_tensorcore (0.786 ms) |
| 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.871 | 1.312 | 1.196 | 2.173 | 1.020 | gemm_tensorcore (0.722 ms) |
| 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.783 | 1.239 | 1.199 | 2.282 | 1.063 | gemm_tensorcore (0.776 ms) |
| 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.689 | 1.159 | 1.198 | 1.984 | 1.063 | gemm_tensorcore (0.759 ms) |
| 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.745 | 1.219 | 1.195 | 2.037 | 1.051 | gemm_tensorcore (0.764 ms) |
| 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.750 | 1.196 | 1.206 | 1.929 | 1.056 | gemm_tensorcore (0.754 ms) |
| 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs | 3.003 | 1.184 | 1.327 | 1.974 | 1.051 | gemm_tensorcore (0.764 ms) |
| 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.064 | 1.506 | 1.200 | 2.482 | 1.233 | gemm_tensorcore (0.885 ms) |
| 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.091 | 1.529 | 1.202 | 2.458 | 1.109 | gemm_tensorcore (0.766 ms) |
| 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.996 | 1.437 | 1.205 | 2.197 | 1.118 | gemm_tensorcore (0.762 ms) |
| 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.985 | 1.466 | 1.196 | 2.269 | 1.101 | gemm_tensorcore (0.755 ms) |
| 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.969 | 1.413 | 1.211 | 2.121 | 1.107 | gemm_tensorcore (0.749 ms) |
| 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.394 | 1.852 | 1.199 | 2.309 | 1.089 | gemm_tensorcore (0.744 ms) |
| 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.998 | 1.442 | 1.200 | 2.127 | 1.108 | gemm_tensorcore (0.753 ms) |
| 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.119 | 1.549 | 1.213 | 2.327 | 1.126 | gemm_tensorcore (0.782 ms) |
| 15 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.028 | 1.461 | 1.204 | 2.138 | 1.110 | gemm_tensorcore (0.753 ms) |
| 16 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.056 | 1.507 | 1.204 | 2.320 | 1.092 | gemm_tensorcore (0.748 ms) |
| 17 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.122 | 1.580 | 1.203 | 2.160 | 1.199 | gemm_tensorcore (0.747 ms) |
| 18 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.289 | 1.730 | 1.206 | 2.742 | 1.127 | gemm_tensorcore (0.763 ms) |
| 19 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.022 | 1.260 | 0.397 | 2.293 | 0.473 | gemm_tensorcore (0.363 ms) |
| 20 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.960 | 1.075 | 0.391 | 2.228 | 0.459 | gemm_tensorcore (0.348 ms) |
| 21 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.797 | 1.059 | 0.387 | 2.084 | 0.445 | gemm_tensorcore (0.335 ms) |
| 22 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.937 | 1.055 | 0.394 | 2.582 | 0.459 | gemm_tensorcore (0.349 ms) |
| 23 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.795 | 1.055 | 0.387 | 1.979 | 0.443 | gemm_tensorcore (0.333 ms) |
| 24 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.974 | 1.119 | 0.390 | 2.034 | 0.438 | gemm_tensorcore (0.328 ms) |
| 25 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.756 | 1.030 | 0.384 | 1.913 | 0.455 | gemm_tensorcore (0.346 ms) |
| 26 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.887 | 1.034 | 0.389 | 2.089 | 0.440 | gemm_tensorcore (0.330 ms) |
| 27 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.711 | 1.010 | 0.382 | 1.967 | 0.446 | gemm_tensorcore (0.336 ms) |
| 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.442 | 0.729 | 0.382 | 1.555 | 0.403 | gemm_tensorcore (0.327 ms) |
| 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.594 | 0.732 | 0.387 | 1.738 | 0.405 | gemm_tensorcore (0.328 ms) |
| 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.448 | 0.720 | 0.383 | 1.601 | 0.408 | gemm_tensorcore (0.332 ms) |
| 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.438 | 0.723 | 0.382 | 1.686 | 0.508 | gemm_tensorcore (0.342 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | nsys range mean ms | nsys kernel mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.336 | 0.647 | 0.342 | 1.518 | 0.389 | gemv_decode_cublas (9.781 ms) |
| 1 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.324 | 0.628 | 0.344 | 1.464 | 0.383 | gemv_decode_cublas (9.678 ms) |
| 2 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.299 | 0.620 | 0.344 | 1.457 | 0.389 | gemv_decode_cublas (9.666 ms) |
| 3 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.311 | 0.624 | 0.349 | 1.455 | 0.386 | gemv_decode_cublas (9.664 ms) |
| 4 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.619 | 0.341 | 1.446 | 0.397 | gemv_decode_cublas (9.826 ms) |
| 5 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.297 | 0.614 | 0.345 | 1.449 | 0.390 | gemv_decode_cublas (9.802 ms) |
| 6 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.279 | 0.608 | 0.341 | 1.474 | 0.387 | gemv_decode_cublas (9.680 ms) |
| 7 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.302 | 0.627 | 0.341 | 1.449 | 0.395 | gemv_decode_cublas (9.838 ms) |
| 8 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.619 | 0.344 | 1.435 | 0.384 | gemv_decode_cublas (9.655 ms) |
| 9 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.618 | 0.342 | 1.447 | 0.386 | gemv_decode_cublas (9.569 ms) |
| 10 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.306 | 0.617 | 0.347 | 1.444 | 0.392 | gemv_decode_cublas (9.782 ms) |
| 11 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.304 | 0.623 | 0.346 | 1.448 | 0.392 | gemv_decode_cublas (9.758 ms) |
| 12 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.292 | 0.611 | 0.343 | 1.444 | 0.396 | gemv_decode_cublas (9.856 ms) |
| 13 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.302 | 0.624 | 0.344 | 1.432 | 0.388 | gemv_decode_cublas (9.586 ms) |
| 14 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.293 | 0.618 | 0.341 | 1.429 | 0.391 | gemv_decode_cublas (9.605 ms) |
| 15 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.300 | 0.615 | 0.349 | 1.428 | 0.388 | gemv_decode_cublas (9.619 ms) |
| 16 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.612 | 0.346 | 1.426 | 0.384 | gemv_decode_cublas (9.657 ms) |
| 17 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.289 | 0.614 | 0.341 | 1.461 | 0.392 | gemv_decode_cublas (9.525 ms) |
| 18 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.292 | 0.617 | 0.341 | 1.436 | 0.383 | gemv_decode_cublas (9.636 ms) |
| 19 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.295 | 0.620 | 0.340 | 1.438 | 0.364 | gemv_decode_cublas (9.477 ms) |
| 20 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.287 | 0.608 | 0.344 | 1.438 | 0.363 | gemv_decode_cublas (9.437 ms) |
| 21 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.608 | 0.340 | 1.439 | 0.362 | gemv_decode_cublas (9.406 ms) |
| 22 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.297 | 0.611 | 0.345 | 1.454 | 0.375 | gemv_decode_cublas (9.641 ms) |
| 23 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.288 | 0.613 | 0.340 | 1.442 | 0.364 | gemv_decode_cublas (9.429 ms) |
| 24 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.613 | 0.342 | 1.453 | 0.371 | gemv_decode_cublas (9.679 ms) |
| 25 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.289 | 0.615 | 0.342 | 1.448 | 0.376 | gemv_decode_cublas (9.716 ms) |
| 26 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.286 | 0.615 | 0.340 | 1.436 | 0.360 | gemv_decode_cublas (9.340 ms) |
| 27 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.616 | 0.340 | 1.449 | 0.368 | gemv_decode_cublas (9.531 ms) |
| 28 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.290 | 0.621 | 0.336 | 1.434 | 0.363 | gemv_decode_cublas (9.448 ms) |
| 29 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.275 | 0.608 | 0.338 | 1.431 | 0.360 | gemv_decode_cublas (9.371 ms) |
| 30 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.300 | 0.633 | 0.337 | 1.460 | 0.364 | gemv_decode_cublas (9.479 ms) |
| 31 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.318 | 0.641 | 0.339 | 1.482 | 0.361 | gemv_decode_cublas (9.401 ms) |
