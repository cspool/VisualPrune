# Same-input dense-FA2 layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `dense-fa2`
- description: Dense LLaVA reference with FlashAttention2; no VisPrune pruning.
- max_new_tokens: `32`
- use_flash_attn: `True`
- use_visipruner: `False`
- visipruner_decode_backend: `off`

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/dense_fa2/clock_sameinput_dense_fa2_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/dense_fa2/clock_sameinput_dense_fa2_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/dense_fa2/clock_sameinput_dense_fa2_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/dense_fa2/nsys_sameinput_dense_fa2_32tok_layer_kernel_breakdown.csv`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1552.875 |
| generate_total_ms | 1509.667 |
| prepare_multimodal_ms | 15.938 |
| vision_encode_project_ms | 14.073 |
| forward_prefill_ms | 85.092 |
| forward_decode_sum_ms | 1385.150 |
| value_aware_token_selection_ms | 0.000 |

## Human-draft-style workload reading

forward 1

- layer 0-31: q_len=624, kv_len=624, workload=dense_fa2_full_prefill; operator=FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs.

forward 2

- layer 0-31: q_len=1, kv_len=625, workload=dense_fa2_decode_full_kv_cache; operator=FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV.

forward 32

- layer 0-31: q_len=1, kv_len=655, workload=dense_fa2_decode_full_kv_cache; operator=FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV.

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.692 | 0.890 | 1.259 | 2.548 | 1.892 | gemm_tensorcore (1.642 ms) |
| 1 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.488 | 0.771 | 1.253 | 1.988 | 1.890 | gemm_tensorcore (1.636 ms) |
| 2 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.498 | 0.784 | 1.259 | 1.922 | 1.896 | gemm_tensorcore (1.644 ms) |
| 3 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.452 | 0.766 | 1.251 | 1.897 | 1.903 | gemm_tensorcore (1.652 ms) |
| 4 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.692 | 0.764 | 1.368 | 1.874 | 1.898 | gemm_tensorcore (1.648 ms) |
| 5 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.470 | 0.765 | 1.252 | 1.933 | 2.016 | gemm_tensorcore (1.644 ms) |
| 6 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.464 | 0.764 | 1.258 | 1.897 | 1.903 | gemm_tensorcore (1.650 ms) |
| 7 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.487 | 0.789 | 1.254 | 1.992 | 1.893 | gemm_tensorcore (1.643 ms) |
| 8 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.464 | 0.773 | 1.257 | 1.944 | 1.898 | gemm_tensorcore (1.647 ms) |
| 9 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.445 | 0.764 | 1.252 | 1.935 | 1.895 | gemm_tensorcore (1.644 ms) |
| 10 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.729 | 0.764 | 1.385 | 1.929 | 1.895 | gemm_tensorcore (1.645 ms) |
| 11 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.458 | 0.766 | 1.252 | 1.903 | 2.039 | gemm_tensorcore (1.789 ms) |
| 12 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.460 | 0.766 | 1.256 | 1.896 | 1.904 | gemm_tensorcore (1.654 ms) |
| 13 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.481 | 0.769 | 1.255 | 1.970 | 1.903 | gemm_tensorcore (1.652 ms) |
| 14 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.465 | 0.766 | 1.255 | 1.948 | 1.900 | gemm_tensorcore (1.649 ms) |
| 15 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.492 | 0.773 | 1.256 | 1.957 | 1.897 | gemm_tensorcore (1.645 ms) |
| 16 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.464 | 0.768 | 1.258 | 1.980 | 1.907 | gemm_tensorcore (1.658 ms) |
| 17 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.491 | 0.822 | 1.216 | 1.935 | 2.049 | gemm_tensorcore (1.800 ms) |
| 18 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.407 | 0.750 | 1.215 | 1.962 | 1.898 | gemm_tensorcore (1.647 ms) |
| 19 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.392 | 0.736 | 1.213 | 2.026 | 1.896 | gemm_tensorcore (1.645 ms) |
| 20 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.433 | 0.760 | 1.226 | 2.084 | 1.905 | gemm_tensorcore (1.653 ms) |
| 21 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.444 | 0.758 | 1.222 | 1.984 | 1.905 | gemm_tensorcore (1.652 ms) |
| 22 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.430 | 0.741 | 1.220 | 2.049 | 1.902 | gemm_tensorcore (1.651 ms) |
| 23 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.662 | 0.739 | 1.364 | 1.987 | 1.963 | gemm_tensorcore (1.714 ms) |
| 24 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.419 | 0.742 | 1.221 | 2.068 | 1.902 | gemm_tensorcore (1.653 ms) |
| 25 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.444 | 0.757 | 1.220 | 2.032 | 1.897 | gemm_tensorcore (1.646 ms) |
| 26 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.439 | 0.759 | 1.221 | 2.117 | 1.903 | gemm_tensorcore (1.654 ms) |
| 27 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.448 | 0.745 | 1.224 | 2.072 | 1.903 | gemm_tensorcore (1.652 ms) |
| 28 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.463 | 0.764 | 1.225 | 2.080 | 1.897 | gemm_tensorcore (1.646 ms) |
| 29 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.488 | 0.761 | 1.225 | 2.057 | 1.998 | gemm_tensorcore (1.662 ms) |
| 30 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.608 | 0.919 | 1.225 | 2.160 | 1.908 | gemm_tensorcore (1.658 ms) |
| 31 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.448 | 0.757 | 1.224 | 2.093 | 1.902 | gemm_tensorcore (1.652 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.358 | 0.589 | 0.359 | 1.895 | 0.543 | gemv_decode_cublas (0.462 ms) |
| 1 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.308 | 0.563 | 0.361 | 1.830 | 0.554 | gemv_decode_cublas (0.471 ms) |
| 2 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.293 | 0.553 | 0.354 | 1.813 | 0.540 | gemv_decode_cublas (0.451 ms) |
| 3 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.274 | 0.545 | 0.350 | 1.825 | 0.549 | gemv_decode_cublas (0.452 ms) |
| 4 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.305 | 0.560 | 0.355 | 1.853 | 0.555 | gemv_decode_cublas (0.462 ms) |
| 5 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.298 | 0.549 | 0.359 | 1.820 | 0.546 | gemv_decode_cublas (0.457 ms) |
| 6 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.553 | 0.359 | 1.784 | 0.545 | gemv_decode_cublas (0.463 ms) |
| 7 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.543 | 0.356 | 1.810 | 0.535 | gemv_decode_cublas (0.453 ms) |
| 8 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.270 | 0.541 | 0.349 | 1.839 | 0.543 | gemv_decode_cublas (0.451 ms) |
| 9 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.297 | 0.553 | 0.358 | 1.851 | 0.544 | gemv_decode_cublas (0.455 ms) |
| 10 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.286 | 0.549 | 0.355 | 1.829 | 0.548 | gemv_decode_cublas (0.455 ms) |
| 11 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.289 | 0.551 | 0.358 | 1.800 | 0.557 | gemv_decode_cublas (0.457 ms) |
| 12 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.547 | 0.358 | 1.784 | 0.548 | gemv_decode_cublas (0.461 ms) |
| 13 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.274 | 0.550 | 0.349 | 1.784 | 0.554 | gemv_decode_cublas (0.464 ms) |
| 14 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.263 | 0.541 | 0.348 | 1.797 | 0.545 | gemv_decode_cublas (0.464 ms) |
| 15 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.555 | 0.353 | 1.800 | 0.535 | gemv_decode_cublas (0.454 ms) |
| 16 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.262 | 0.541 | 0.347 | 1.830 | 0.555 | gemv_decode_cublas (0.464 ms) |
| 17 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.285 | 0.550 | 0.352 | 1.802 | 0.541 | gemv_decode_cublas (0.452 ms) |
| 18 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.548 | 0.349 | 1.787 | 0.536 | gemv_decode_cublas (0.455 ms) |
| 19 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.559 | 0.348 | 1.798 | 0.541 | gemv_decode_cublas (0.454 ms) |
| 20 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.280 | 0.544 | 0.360 | 1.822 | 0.561 | gemv_decode_cublas (0.460 ms) |
| 21 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.550 | 0.353 | 1.796 | 0.551 | gemv_decode_cublas (0.456 ms) |
| 22 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.261 | 0.537 | 0.349 | 1.763 | 0.534 | gemv_decode_cublas (0.450 ms) |
| 23 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.286 | 0.552 | 0.357 | 1.840 | 0.545 | gemv_decode_cublas (0.460 ms) |
| 24 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.267 | 0.544 | 0.348 | 1.789 | 0.544 | gemv_decode_cublas (0.458 ms) |
| 25 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.278 | 0.552 | 0.350 | 1.773 | 0.536 | gemv_decode_cublas (0.450 ms) |
| 26 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.295 | 0.559 | 0.349 | 1.780 | 0.548 | gemv_decode_cublas (0.460 ms) |
| 27 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.271 | 0.549 | 0.349 | 1.783 | 0.536 | gemv_decode_cublas (0.450 ms) |
| 28 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.261 | 0.544 | 0.348 | 1.789 | 0.541 | gemv_decode_cublas (0.452 ms) |
| 29 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.552 | 0.353 | 1.824 | 0.566 | gemv_decode_cublas (0.477 ms) |
| 30 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.270 | 0.539 | 0.350 | 1.799 | 0.538 | gemv_decode_cublas (0.457 ms) |
| 31 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.293 | 0.557 | 0.356 | 1.822 | 0.545 | gemv_decode_cublas (0.464 ms) |
