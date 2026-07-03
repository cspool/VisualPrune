# E2 visipruner-full-fa2 32-token layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `visipruner-full-fa2`
- description: VisPrune full path with optimized backend auto-selection; currently Triton VP-FA prefill when available.
- max_new_tokens: `32`
- use_flash_attn: `True`
- use_visipruner: `True`
- visipruner_decode_backend: `auto`

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_fa2_32tok_layer_kernel_breakdown.csv`
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1755.986 |
| generate_total_ms | 1705.834 |
| prepare_multimodal_ms | 18.968 |
| vision_encode_project_ms | 16.726 |
| forward_prefill_ms | 113.715 |
| forward_decode_sum_ms | 1545.759 |
| value_aware_token_selection_ms | 10.077 |

## Human-draft-style workload reading

forward 1

- layer 0: q_len=624, kv_len=624, workload=eager_visipruner_prefill_shallow_layer0_mass_fold; operator=eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs.
- layer 1-5: q_len=624, kv_len=624, workload=eager_visipruner_prefill_shallow_text_to_vision_mask; operator=eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs.
- layer 6: q_len=624, kv_len=624, workload=eager_visipruner_prefill_last_query_proxy_or_selection; operator=eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs.
- layer 7-14: q_len=624, kv_len=624, workload=eager_visipruner_prefill_last_query_proxy_or_selection; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.
- layer 15-27: q_len=56, kv_len=56, workload=eager_visipruner_middle_pruned_compact_prefill; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.
- layer 28-31: q_len=48, kv_len=48, workload=eager_visipruner_deep_removed_prefill; operator=eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs.

forward 2

- layer 0-14: q_len=1, kv_len=625, workload=eager_visipruner_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 15-27: q_len=1, kv_len=57, workload=eager_visipruner_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=49, workload=eager_visipruner_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

forward 32

- layer 0-14: q_len=1, kv_len=655, workload=eager_visipruner_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 15-27: q_len=1, kv_len=87, workload=eager_visipruner_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=79, workload=eager_visipruner_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | nsys range ms | nsys kernel ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.315 | 1.426 | 1.276 | 2.788 | 1.524 | gemm_tensorcore (1.178 ms) |
| 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.163 | 1.319 | 1.272 | 2.388 | 1.319 | gemm_tensorcore (0.987 ms) |
| 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.052 | 1.280 | 1.270 | 2.325 | 1.288 | gemm_tensorcore (0.971 ms) |
| 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.074 | 1.286 | 1.270 | 2.334 | 1.291 | gemm_tensorcore (0.972 ms) |
| 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.229 | 1.443 | 1.272 | 2.468 | 1.301 | gemm_tensorcore (0.970 ms) |
| 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.054 | 1.271 | 1.268 | 2.350 | 1.295 | gemm_tensorcore (0.972 ms) |
| 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs | 3.039 | 1.261 | 1.271 | 2.321 | 1.288 | gemm_tensorcore (0.971 ms) |
| 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.992 | 2.195 | 1.279 | 3.436 | 1.383 | gemm_tensorcore (0.972 ms) |
| 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.903 | 2.069 | 1.297 | 3.459 | 1.469 | gemm_tensorcore (0.982 ms) |
| 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 4.007 | 2.173 | 1.276 | 3.261 | 1.402 | gemm_tensorcore (0.979 ms) |
| 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.978 | 2.133 | 1.274 | 3.225 | 1.400 | gemm_tensorcore (0.977 ms) |
| 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.875 | 2.068 | 1.277 | 3.206 | 1.380 | gemm_tensorcore (0.973 ms) |
| 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.914 | 2.097 | 1.280 | 3.311 | 1.481 | gemm_tensorcore (0.976 ms) |
| 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 4.128 | 2.315 | 1.277 | 3.169 | 1.376 | gemm_tensorcore (0.968 ms) |
| 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 4.314 | 2.461 | 1.286 | 3.493 | 1.413 | gemm_tensorcore (0.975 ms) |
| 15 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.203 | 2.155 | 0.464 | 3.243 | 0.555 | gemm_tensorcore (0.417 ms) |
| 16 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.865 | 1.867 | 0.434 | 2.942 | 0.527 | gemm_tensorcore (0.390 ms) |
| 17 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.236 | 2.224 | 0.427 | 3.051 | 0.529 | gemm_tensorcore (0.392 ms) |
| 18 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.254 | 1.919 | 0.501 | 3.027 | 0.538 | gemm_tensorcore (0.400 ms) |
| 19 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.985 | 1.949 | 0.434 | 2.831 | 0.516 | gemm_tensorcore (0.381 ms) |
| 20 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.247 | 1.938 | 0.447 | 3.011 | 0.526 | gemm_tensorcore (0.391 ms) |
| 21 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.906 | 1.910 | 0.433 | 3.807 | 0.599 | gemm_tensorcore (0.395 ms) |
| 22 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.416 | 2.043 | 0.448 | 3.076 | 0.529 | gemm_tensorcore (0.391 ms) |
| 23 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.027 | 2.010 | 0.435 | 2.913 | 0.520 | gemm_tensorcore (0.385 ms) |
| 24 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.300 | 1.948 | 0.446 | 3.024 | 0.522 | gemm_tensorcore (0.387 ms) |
| 25 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.965 | 1.942 | 0.432 | 2.884 | 0.543 | gemm_tensorcore (0.405 ms) |
| 26 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.284 | 1.965 | 0.446 | 3.085 | 0.589 | gemm_tensorcore (0.383 ms) |
| 27 | 56 | 56 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.550 | 2.503 | 0.436 | 2.977 | 0.519 | gemm_tensorcore (0.384 ms) |
| 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.515 | 1.439 | 0.436 | 2.384 | 0.489 | gemm_tensorcore (0.389 ms) |
| 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.748 | 1.420 | 0.440 | 2.561 | 0.493 | gemm_tensorcore (0.391 ms) |
| 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.448 | 1.419 | 0.432 | 2.372 | 0.482 | gemm_tensorcore (0.384 ms) |
| 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.558 | 1.420 | 0.441 | 2.377 | 0.496 | gemm_tensorcore (0.395 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | nsys range mean ms | nsys kernel mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.500 | 0.740 | 0.350 | 2.168 | 0.451 | gemv_decode_cublas (11.568 ms) |
| 1 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.450 | 0.711 | 0.348 | 2.099 | 0.442 | gemv_decode_cublas (11.322 ms) |
| 2 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.440 | 0.701 | 0.349 | 2.092 | 0.451 | gemv_decode_cublas (11.342 ms) |
| 3 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.453 | 0.705 | 0.355 | 2.110 | 0.471 | gemv_decode_cublas (12.057 ms) |
| 4 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.455 | 0.717 | 0.354 | 2.152 | 0.459 | gemv_decode_cublas (11.821 ms) |
| 5 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.436 | 0.705 | 0.350 | 2.089 | 0.473 | gemv_decode_cublas (11.669 ms) |
| 6 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.456 | 0.716 | 0.352 | 2.091 | 0.454 | gemv_decode_cublas (11.500 ms) |
| 7 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.440 | 0.707 | 0.349 | 2.087 | 0.474 | gemv_decode_cublas (11.608 ms) |
| 8 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.462 | 0.708 | 0.355 | 2.093 | 0.462 | gemv_decode_cublas (11.662 ms) |
| 9 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.438 | 0.706 | 0.348 | 2.089 | 0.458 | gemv_decode_cublas (11.631 ms) |
| 10 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.447 | 0.706 | 0.352 | 2.091 | 0.465 | gemv_decode_cublas (11.405 ms) |
| 11 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.431 | 0.692 | 0.351 | 2.092 | 0.458 | gemv_decode_cublas (11.673 ms) |
| 12 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.463 | 0.716 | 0.352 | 2.100 | 0.456 | gemv_decode_cublas (11.502 ms) |
| 13 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.431 | 0.696 | 0.349 | 2.078 | 0.461 | gemv_decode_cublas (11.554 ms) |
| 14 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.483 | 0.734 | 0.360 | 2.112 | 0.453 | gemv_decode_cublas (11.406 ms) |
| 15 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.709 | 0.345 | 2.079 | 0.419 | gemv_decode_cublas (11.010 ms) |
| 16 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.433 | 0.695 | 0.347 | 2.074 | 0.437 | gemv_decode_cublas (11.219 ms) |
| 17 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.430 | 0.689 | 0.350 | 2.067 | 0.432 | gemv_decode_cublas (11.337 ms) |
| 18 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.429 | 0.694 | 0.347 | 2.073 | 0.419 | gemv_decode_cublas (11.055 ms) |
| 19 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.420 | 0.686 | 0.352 | 2.074 | 0.443 | gemv_decode_cublas (11.495 ms) |
| 20 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.437 | 0.704 | 0.344 | 2.076 | 0.430 | gemv_decode_cublas (11.181 ms) |
| 21 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.425 | 0.686 | 0.348 | 2.052 | 0.431 | gemv_decode_cublas (11.188 ms) |
| 22 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.420 | 0.692 | 0.345 | 2.068 | 0.427 | gemv_decode_cublas (11.165 ms) |
| 23 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.410 | 0.681 | 0.346 | 2.058 | 0.440 | gemv_decode_cublas (11.400 ms) |
| 24 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.424 | 0.693 | 0.345 | 2.046 | 0.424 | gemv_decode_cublas (11.028 ms) |
| 25 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.418 | 0.682 | 0.348 | 2.053 | 0.425 | gemv_decode_cublas (11.119 ms) |
| 26 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.705 | 0.347 | 2.076 | 0.422 | gemv_decode_cublas (11.117 ms) |
| 27 | 57 -> 87 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.418 | 0.687 | 0.347 | 2.056 | 0.422 | gemv_decode_cublas (10.941 ms) |
| 28 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.434 | 0.691 | 0.351 | 2.061 | 0.441 | gemv_decode_cublas (11.320 ms) |
| 29 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.426 | 0.687 | 0.352 | 2.061 | 0.438 | gemv_decode_cublas (11.447 ms) |
| 30 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.443 | 0.712 | 0.345 | 2.096 | 0.418 | gemv_decode_cublas (11.033 ms) |
| 31 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.722 | 0.342 | 2.128 | 0.426 | gemv_decode_cublas (11.243 ms) |
