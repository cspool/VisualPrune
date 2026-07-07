# Triton VP-FA Single-Request Layer Performance Report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.csv`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
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

| layer | q_len | kv_len | workload type | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | full_prefill_shallow_layer0_mass_fold | 3.355 | 1.571 | 1.226 | 2.992 | 2.005 | gemm_tensorcore (1.643 ms) |
| 1 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.885 | 1.211 | 1.214 | 2.436 | 1.993 | gemm_tensorcore (1.644 ms) |
| 2 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.855 | 1.195 | 1.215 | 2.463 | 1.994 | gemm_tensorcore (1.644 ms) |
| 3 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 2.850 | 1.183 | 1.221 | 2.404 | 1.993 | gemm_tensorcore (1.643 ms) |
| 4 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 3.436 | 1.580 | 1.296 | 2.394 | 1.995 | gemm_tensorcore (1.642 ms) |
| 5 | 624 | 624 | full_prefill_shallow_text_to_vision_mask | 3.211 | 1.204 | 1.290 | 2.503 | 1.995 | gemm_tensorcore (1.645 ms) |
| 6 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 2.894 | 1.203 | 1.219 | 2.405 | 1.993 | gemm_tensorcore (1.644 ms) |
| 7 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.679 | 1.996 | 1.222 | 3.481 | 2.088 | gemm_tensorcore (1.646 ms) |
| 8 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.734 | 1.896 | 1.227 | 3.000 | 2.079 | gemm_tensorcore (1.645 ms) |
| 9 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.672 | 2.632 | 1.284 | 2.971 | 2.074 | gemm_tensorcore (1.643 ms) |
| 10 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.597 | 1.892 | 1.227 | 2.900 | 2.082 | gemm_tensorcore (1.648 ms) |
| 11 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.663 | 1.944 | 1.236 | 3.095 | 2.079 | gemm_tensorcore (1.646 ms) |
| 12 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.661 | 1.946 | 1.229 | 2.778 | 2.084 | gemm_tensorcore (1.648 ms) |
| 13 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.999 | 2.175 | 1.225 | 2.835 | 2.083 | gemm_tensorcore (1.647 ms) |
| 14 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.780 | 2.062 | 1.229 | 2.831 | 2.084 | gemm_tensorcore (1.652 ms) |
| 15 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.773 | 2.011 | 1.233 | 3.044 | 2.085 | gemm_tensorcore (1.653 ms) |
| 16 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 3.732 | 1.970 | 1.233 | 2.759 | 2.082 | gemm_tensorcore (1.649 ms) |
| 17 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.111 | 2.007 | 1.377 | 2.945 | 2.080 | gemm_tensorcore (1.646 ms) |
| 18 | 624 | 624 | full_prefill_last_query_proxy_or_selection | 4.481 | 2.680 | 1.237 | 2.983 | 2.099 | gemm_tensorcore (1.648 ms) |
| 19 | 58 | 58 | middle_pruned_compact_prefill | 3.020 | 2.040 | 0.441 | 2.679 | 0.628 | gemm_tensorcore (0.492 ms) |
| 20 | 58 | 58 | middle_pruned_compact_prefill | 2.857 | 1.699 | 0.434 | 2.515 | 0.632 | gemm_tensorcore (0.496 ms) |
| 21 | 58 | 58 | middle_pruned_compact_prefill | 4.081 | 2.919 | 0.448 | 3.254 | 0.636 | gemm_tensorcore (0.499 ms) |
| 22 | 58 | 58 | middle_pruned_compact_prefill | 3.060 | 1.783 | 0.502 | 2.780 | 0.629 | gemm_tensorcore (0.492 ms) |
| 23 | 58 | 58 | middle_pruned_compact_prefill | 2.615 | 1.685 | 0.423 | 2.491 | 0.631 | gemm_tensorcore (0.495 ms) |
| 24 | 58 | 58 | middle_pruned_compact_prefill | 2.879 | 1.691 | 0.432 | 2.653 | 0.633 | gemm_tensorcore (0.495 ms) |
| 25 | 58 | 58 | middle_pruned_compact_prefill | 2.680 | 1.730 | 0.422 | 2.372 | 0.631 | gemm_tensorcore (0.495 ms) |
| 26 | 58 | 58 | middle_pruned_compact_prefill | 2.980 | 1.777 | 0.434 | 2.413 | 0.635 | gemm_tensorcore (0.499 ms) |
| 27 | 58 | 58 | middle_pruned_compact_prefill | 3.300 | 2.327 | 0.421 | 2.219 | 0.631 | gemm_tensorcore (0.496 ms) |
| 28 | 48 | 48 | deep_removed_prefill | 2.277 | 1.319 | 0.426 | 1.775 | 0.589 | gemm_tensorcore (0.491 ms) |
| 29 | 48 | 48 | deep_removed_prefill | 2.425 | 1.237 | 0.426 | 2.037 | 0.592 | gemm_tensorcore (0.490 ms) |
| 30 | 48 | 48 | deep_removed_prefill | 2.161 | 1.234 | 0.419 | 1.852 | 0.592 | gemm_tensorcore (0.491 ms) |
| 31 | 48 | 48 | deep_removed_prefill | 2.154 | 1.219 | 0.434 | 1.822 | 0.590 | gemm_tensorcore (0.491 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | decode_full_kv_cache | 1.615 | 0.811 | 0.354 | 1.494 | 0.523 | gemv_decode_cublas (0.451 ms) |
| 1 | 625 -> 655 | decode_full_kv_cache | 1.564 | 0.775 | 0.356 | 1.438 | 0.524 | gemv_decode_cublas (0.452 ms) |
| 2 | 625 -> 655 | decode_full_kv_cache | 1.539 | 0.761 | 0.360 | 1.431 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 3 | 625 -> 655 | decode_full_kv_cache | 1.551 | 0.773 | 0.350 | 1.451 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 4 | 625 -> 655 | decode_full_kv_cache | 1.554 | 0.781 | 0.351 | 1.452 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 5 | 625 -> 655 | decode_full_kv_cache | 1.552 | 0.773 | 0.362 | 1.415 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 6 | 625 -> 655 | decode_full_kv_cache | 1.555 | 0.773 | 0.354 | 1.448 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 7 | 625 -> 655 | decode_full_kv_cache | 1.526 | 0.760 | 0.349 | 1.423 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 8 | 625 -> 655 | decode_full_kv_cache | 1.568 | 0.768 | 0.363 | 1.414 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 9 | 625 -> 655 | decode_full_kv_cache | 1.593 | 0.774 | 0.365 | 1.433 | 0.529 | gemv_decode_cublas (0.455 ms) |
| 10 | 625 -> 655 | decode_full_kv_cache | 1.523 | 0.745 | 0.360 | 1.450 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 11 | 625 -> 655 | decode_full_kv_cache | 1.521 | 0.748 | 0.363 | 1.424 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 12 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.753 | 0.353 | 1.466 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 13 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.742 | 0.366 | 1.423 | 0.527 | gemv_decode_cublas (0.455 ms) |
| 14 | 625 -> 655 | decode_full_kv_cache | 1.550 | 0.752 | 0.363 | 1.413 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 15 | 625 -> 655 | decode_full_kv_cache | 1.505 | 0.729 | 0.353 | 1.414 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 16 | 625 -> 655 | decode_full_kv_cache | 1.548 | 0.765 | 0.360 | 1.423 | 0.528 | gemv_decode_cublas (0.454 ms) |
| 17 | 625 -> 655 | decode_full_kv_cache | 1.533 | 0.745 | 0.353 | 1.428 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 18 | 625 -> 655 | decode_full_kv_cache | 1.547 | 0.773 | 0.351 | 1.456 | 0.528 | gemv_decode_cublas (0.455 ms) |
| 19 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.549 | 0.763 | 0.353 | 1.405 | 0.509 | gemv_decode_cublas (0.448 ms) |
| 20 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.531 | 0.747 | 0.353 | 1.428 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 21 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.505 | 0.741 | 0.345 | 1.411 | 0.508 | gemv_decode_cublas (0.447 ms) |
| 22 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.561 | 0.775 | 0.350 | 1.440 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 23 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.517 | 0.740 | 0.350 | 1.395 | 0.507 | gemv_decode_cublas (0.446 ms) |
| 24 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.554 | 0.767 | 0.346 | 1.403 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 25 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.515 | 0.746 | 0.349 | 1.426 | 0.506 | gemv_decode_cublas (0.446 ms) |
| 26 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.518 | 0.743 | 0.364 | 1.429 | 0.506 | gemv_decode_cublas (0.446 ms) |
| 27 | 59 -> 89 | decode_middle_pruned_kv_cache | 1.520 | 0.754 | 0.346 | 1.422 | 0.507 | gemv_decode_cublas (0.447 ms) |
| 28 | 49 -> 79 | decode_deep_removed_kv_cache | 1.526 | 0.742 | 0.360 | 1.418 | 0.506 | gemv_decode_cublas (0.446 ms) |
| 29 | 49 -> 79 | decode_deep_removed_kv_cache | 1.514 | 0.743 | 0.357 | 1.448 | 0.506 | gemv_decode_cublas (0.446 ms) |
| 30 | 49 -> 79 | decode_deep_removed_kv_cache | 1.550 | 0.777 | 0.355 | 1.441 | 0.506 | gemv_decode_cublas (0.446 ms) |
| 31 | 49 -> 79 | decode_deep_removed_kv_cache | 1.602 | 0.809 | 0.351 | 1.454 | 0.506 | gemv_decode_cublas (0.446 ms) |
