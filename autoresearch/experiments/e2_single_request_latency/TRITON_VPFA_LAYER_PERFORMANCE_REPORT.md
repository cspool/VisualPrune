# Triton VP-FA Single-Request Layer Performance Report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.csv`
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1859.271 |
| generate_total_ms | 1813.852 |
| prepare_multimodal_ms | 15.998 |
| vision_encode_project_ms | 13.973 |
| forward_prefill_ms | 112.241 |
| forward_decode_sum_ms | 1658.759 |
| value_aware_token_selection_ms | 8.801 |

## Human-draft-style workload reading

forward 1

- layer 0: full prefill, Triton VP-FA attention applies the layer-0 shallow visual mass fold in-kernel.
- layer 1-5: full prefill, Triton VP-FA attention applies shallow text-to-vision masking in-kernel.
- layer 7-18: full prefill keeps the expanded sequence and computes the last-query pruning proxy where selection is needed.
- layer 18: observed q_len=624, kv_len=624; this is the middle-selection decision point when selected visual tokens are emitted.
- layer 19-27: compact prefill after middle pruning, observed layer19 q_len=58, kv_len=58.
- layer 28-31: deep-removed prefill, observed layer28 q_len=48, kv_len=48.

forward 2

- layer 18: q_len=1, kv_len=625, workload=decode_full_kv_cache.
- layer 19: q_len=1, kv_len=59, workload=decode_middle_pruned_kv_cache.
- layer 28: q_len=1, kv_len=49, workload=decode_deep_removed_kv_cache.

forward 32

- layer 18: q_len=1, kv_len=655, workload=decode_full_kv_cache.
- layer 19: q_len=1, kv_len=89, workload=decode_middle_pruned_kv_cache.
- layer 28: q_len=1, kv_len=79, workload=decode_deep_removed_kv_cache.

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | clock total ms | attn ms | mlp ms | nsys range ms | nsys kernel ms | dominant kernel family |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | full_prefill_shallow_layer0_mass_fold | 3.355 | 1.571 | 1.226 | 2.992 | 1.517 | gemm_tensorcore (1.191 ms) |
| 1 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.885 | 1.211 | 1.214 | 2.436 | 1.213 | gemm_tensorcore (0.916 ms) |
| 2 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.855 | 1.195 | 1.215 | 2.463 | 1.308 | gemm_tensorcore (0.994 ms) |
| 3 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.850 | 1.183 | 1.221 | 2.404 | 1.269 | gemm_tensorcore (0.956 ms) |
| 4 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 3.436 | 1.580 | 1.296 | 2.394 | 1.216 | gemm_tensorcore (0.916 ms) |
| 5 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 3.211 | 1.204 | 1.290 | 2.503 | 1.269 | gemm_tensorcore (0.955 ms) |
| 6 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 2.894 | 1.203 | 1.219 | 2.405 | 1.277 | gemm_tensorcore (0.966 ms) |
| 7 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.679 | 1.996 | 1.222 | 3.481 | 1.728 | gemm_tensorcore (1.294 ms) |
| 8 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.734 | 1.896 | 1.227 | 3.000 | 1.233 | gemm_tensorcore (0.852 ms) |
| 9 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.672 | 2.632 | 1.284 | 2.971 | 1.231 | gemm_tensorcore (0.852 ms) |
| 10 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.597 | 1.892 | 1.227 | 2.900 | 1.319 | gemm_tensorcore (0.922 ms) |
| 11 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.663 | 1.944 | 1.236 | 3.095 | 1.293 | gemm_tensorcore (0.912 ms) |
| 12 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.661 | 1.946 | 1.229 | 2.778 | 1.293 | gemm_tensorcore (0.909 ms) |
| 13 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.999 | 2.175 | 1.225 | 2.835 | 1.223 | gemm_tensorcore (0.840 ms) |
| 14 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.780 | 2.062 | 1.229 | 2.831 | 1.287 | gemm_tensorcore (0.907 ms) |
| 15 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.773 | 2.011 | 1.233 | 3.044 | 1.297 | gemm_tensorcore (0.917 ms) |
| 16 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.732 | 1.970 | 1.233 | 2.759 | 1.295 | gemm_tensorcore (0.914 ms) |
| 17 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.111 | 2.007 | 1.377 | 2.945 | 1.273 | gemm_tensorcore (0.890 ms) |
| 18 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.481 | 2.680 | 1.237 | 2.983 | 1.175 | gemm_tensorcore (0.776 ms) |
| 19 | 58 | 58 | middle_pruned_compact_prefill | 3.020 | 2.040 | 0.441 | 2.679 | 0.486 | gemm_tensorcore (0.357 ms) |
| 20 | 58 | 58 | middle_pruned_compact_prefill | 2.857 | 1.699 | 0.434 | 2.515 | 0.472 | gemm_tensorcore (0.344 ms) |
| 21 | 58 | 58 | middle_pruned_compact_prefill | 4.081 | 2.919 | 0.448 | 3.254 | 0.486 | gemm_tensorcore (0.356 ms) |
| 22 | 58 | 58 | middle_pruned_compact_prefill | 3.060 | 1.783 | 0.502 | 2.780 | 0.477 | gemm_tensorcore (0.348 ms) |
| 23 | 58 | 58 | middle_pruned_compact_prefill | 2.615 | 1.685 | 0.423 | 2.491 | 0.470 | gemm_tensorcore (0.342 ms) |
| 24 | 58 | 58 | middle_pruned_compact_prefill | 2.879 | 1.691 | 0.432 | 2.653 | 0.474 | gemm_tensorcore (0.344 ms) |
| 25 | 58 | 58 | middle_pruned_compact_prefill | 2.680 | 1.730 | 0.422 | 2.372 | 0.469 | gemm_tensorcore (0.340 ms) |
| 26 | 58 | 58 | middle_pruned_compact_prefill | 2.980 | 1.777 | 0.434 | 2.413 | 0.434 | gemm_tensorcore (0.305 ms) |
| 27 | 58 | 58 | middle_pruned_compact_prefill | 3.300 | 2.327 | 0.421 | 2.219 | 0.459 | gemm_tensorcore (0.331 ms) |
| 28 | 48 | 48 | deep_removed_prefill | 2.277 | 1.319 | 0.426 | 1.775 | 0.427 | gemm_tensorcore (0.335 ms) |
| 29 | 48 | 48 | deep_removed_prefill | 2.425 | 1.237 | 0.426 | 2.037 | 0.444 | gemm_tensorcore (0.349 ms) |
| 30 | 48 | 48 | deep_removed_prefill | 2.161 | 1.234 | 0.419 | 1.852 | 0.415 | gemm_tensorcore (0.320 ms) |
| 31 | 48 | 48 | deep_removed_prefill | 2.154 | 1.219 | 0.434 | 1.822 | 0.430 | gemm_tensorcore (0.337 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | clock total mean ms | attn mean ms | mlp mean ms | nsys range mean ms | nsys kernel mean ms | dominant kernel family |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | decode_full_kv_cache | 1.615 | 0.811 | 0.354 | 1.494 | 0.371 | gemv_decode_cublas (9.329 ms) |
| 1 | 625 -> 655 | decode_full_kv_cache | 1.564 | 0.775 | 0.356 | 1.438 | 0.366 | gemv_decode_cublas (9.194 ms) |
| 2 | 625 -> 655 | decode_full_kv_cache | 1.539 | 0.761 | 0.360 | 1.431 | 0.360 | gemv_decode_cublas (8.970 ms) |
| 3 | 625 -> 655 | decode_full_kv_cache | 1.551 | 0.773 | 0.350 | 1.451 | 0.372 | gemv_decode_cublas (9.320 ms) |
| 4 | 625 -> 655 | decode_full_kv_cache | 1.554 | 0.781 | 0.351 | 1.452 | 0.368 | gemv_decode_cublas (9.213 ms) |
| 5 | 625 -> 655 | decode_full_kv_cache | 1.552 | 0.773 | 0.362 | 1.415 | 0.367 | gemv_decode_cublas (9.197 ms) |
| 6 | 625 -> 655 | decode_full_kv_cache | 1.555 | 0.773 | 0.354 | 1.448 | 0.349 | gemv_decode_cublas (8.642 ms) |
| 7 | 625 -> 655 | decode_full_kv_cache | 1.526 | 0.760 | 0.349 | 1.423 | 0.370 | gemv_decode_cublas (9.263 ms) |
| 8 | 625 -> 655 | decode_full_kv_cache | 1.568 | 0.768 | 0.363 | 1.414 | 0.371 | gemv_decode_cublas (9.318 ms) |
| 9 | 625 -> 655 | decode_full_kv_cache | 1.593 | 0.774 | 0.365 | 1.433 | 0.361 | gemv_decode_cublas (8.984 ms) |
| 10 | 625 -> 655 | decode_full_kv_cache | 1.523 | 0.745 | 0.360 | 1.450 | 0.375 | gemv_decode_cublas (9.418 ms) |
| 11 | 625 -> 655 | decode_full_kv_cache | 1.521 | 0.748 | 0.363 | 1.424 | 0.365 | gemv_decode_cublas (9.122 ms) |
| 12 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.753 | 0.353 | 1.466 | 0.375 | gemv_decode_cublas (9.414 ms) |
| 13 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.742 | 0.366 | 1.423 | 0.369 | gemv_decode_cublas (9.239 ms) |
| 14 | 625 -> 655 | decode_full_kv_cache | 1.550 | 0.752 | 0.363 | 1.413 | 0.366 | gemv_decode_cublas (9.141 ms) |
| 15 | 625 -> 655 | decode_full_kv_cache | 1.505 | 0.729 | 0.353 | 1.414 | 0.358 | gemv_decode_cublas (8.907 ms) |
| 16 | 625 -> 655 | decode_full_kv_cache | 1.548 | 0.765 | 0.360 | 1.423 | 0.363 | gemv_decode_cublas (9.067 ms) |
| 17 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.745 | 0.353 | 1.428 | 0.355 | gemv_decode_cublas (8.798 ms) |
| 18 | 625 -> 655 | decode_full_kv_cache | 1.547 | 0.773 | 0.351 | 1.456 | 0.367 | gemv_decode_cublas (9.188 ms) |
| 19 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.549 | 0.763 | 0.353 | 1.405 | 0.342 | gemv_decode_cublas (8.806 ms) |
| 20 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.531 | 0.747 | 0.353 | 1.428 | 0.346 | gemv_decode_cublas (8.929 ms) |
| 21 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.505 | 0.741 | 0.345 | 1.411 | 0.351 | gemv_decode_cublas (9.063 ms) |
| 22 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.561 | 0.775 | 0.350 | 1.440 | 0.343 | gemv_decode_cublas (8.850 ms) |
| 23 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.517 | 0.740 | 0.350 | 1.395 | 0.348 | gemv_decode_cublas (8.991 ms) |
| 24 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.554 | 0.767 | 0.346 | 1.403 | 0.348 | gemv_decode_cublas (8.996 ms) |
| 25 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.515 | 0.746 | 0.349 | 1.426 | 0.347 | gemv_decode_cublas (8.952 ms) |
| 26 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.518 | 0.743 | 0.364 | 1.429 | 0.345 | gemv_decode_cublas (8.886 ms) |
| 27 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.520 | 0.754 | 0.346 | 1.422 | 0.354 | gemv_decode_cublas (9.153 ms) |
| 28 | 49 -> 79 | decode_deep_removed_kv_cache | 1.526 | 0.742 | 0.360 | 1.418 | 0.348 | gemv_decode_cublas (8.999 ms) |
| 29 | 49 -> 79 | decode_deep_removed_kv_cache | 1.514 | 0.743 | 0.357 | 1.448 | 0.347 | gemv_decode_cublas (8.963 ms) |
| 30 | 49 -> 79 | decode_deep_removed_kv_cache | 1.550 | 0.777 | 0.355 | 1.441 | 0.347 | gemv_decode_cublas (8.977 ms) |
| 31 | 49 -> 79 | decode_deep_removed_kv_cache | 1.602 | 0.809 | 0.351 | 1.454 | 0.334 | gemv_decode_cublas (8.555 ms) |
