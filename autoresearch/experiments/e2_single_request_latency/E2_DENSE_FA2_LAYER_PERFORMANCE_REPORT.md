# E2 dense-FA2 layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `dense-fa2`
- description: Dense LLaVA reference with FlashAttention2; no VisPrune pruning.
- max_new_tokens: `32`
- use_flash_attn: `True`
- use_visipruner: `False`
- visipruner_decode_backend: `off`

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_dense_fa2_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_dense_fa2_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_e2_dense_fa2_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_e2_dense_fa2_32tok_layer_kernel_breakdown.csv`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1352.631 |
| generate_total_ms | 1318.549 |
| prepare_multimodal_ms | 12.419 |
| vision_encode_project_ms | 11.114 |
| forward_prefill_ms | 78.016 |
| forward_decode_sum_ms | 1212.669 |
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
| 0 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.605 | 0.721 | 1.476 | 2.269 | 2.011 | gemm_tensorcore (1.703 ms) |
| 1 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.310 | 0.711 | 1.213 | 1.805 | 1.893 | gemm_tensorcore (1.639 ms) |
| 2 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.275 | 0.711 | 1.204 | 1.716 | 1.899 | gemm_tensorcore (1.648 ms) |
| 3 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.252 | 0.708 | 1.194 | 1.664 | 1.899 | gemm_tensorcore (1.650 ms) |
| 4 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.273 | 0.712 | 1.203 | 1.642 | 1.896 | gemm_tensorcore (1.646 ms) |
| 5 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.713 | 1.199 | 1.589 | 1.904 | gemm_tensorcore (1.654 ms) |
| 6 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.252 | 0.706 | 1.196 | 1.587 | 1.899 | gemm_tensorcore (1.648 ms) |
| 7 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.548 | 0.707 | 1.473 | 1.644 | 2.015 | gemm_tensorcore (1.766 ms) |
| 8 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.269 | 0.713 | 1.207 | 1.554 | 1.903 | gemm_tensorcore (1.653 ms) |
| 9 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.286 | 0.713 | 1.206 | 1.519 | 1.900 | gemm_tensorcore (1.649 ms) |
| 10 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.711 | 1.200 | 1.520 | 1.898 | gemm_tensorcore (1.648 ms) |
| 11 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.249 | 0.709 | 1.197 | 1.438 | 1.905 | gemm_tensorcore (1.655 ms) |
| 12 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.260 | 0.711 | 1.203 | 1.441 | 1.909 | gemm_tensorcore (1.659 ms) |
| 13 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.258 | 0.713 | 1.199 | 1.645 | 1.898 | gemm_tensorcore (1.650 ms) |
| 14 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.462 | 0.711 | 1.268 | 1.500 | 2.028 | gemm_tensorcore (1.702 ms) |
| 15 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.251 | 0.707 | 1.204 | 1.513 | 1.903 | gemm_tensorcore (1.652 ms) |
| 16 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.715 | 1.205 | 1.454 | 1.904 | gemm_tensorcore (1.655 ms) |
| 17 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.266 | 0.714 | 1.201 | 1.417 | 1.904 | gemm_tensorcore (1.654 ms) |
| 18 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.258 | 0.711 | 1.210 | 1.447 | 1.909 | gemm_tensorcore (1.660 ms) |
| 19 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.231 | 0.706 | 1.209 | 1.446 | 1.905 | gemm_tensorcore (1.655 ms) |
| 20 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.707 | 1.217 | 1.445 | 1.895 | gemm_tensorcore (1.645 ms) |
| 21 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.510 | 0.711 | 1.469 | 1.542 | 2.014 | gemm_tensorcore (1.731 ms) |
| 22 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.259 | 0.713 | 1.201 | 1.429 | 1.913 | gemm_tensorcore (1.664 ms) |
| 23 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.713 | 1.201 | 1.514 | 1.902 | gemm_tensorcore (1.652 ms) |
| 24 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.264 | 0.710 | 1.201 | 1.583 | 1.904 | gemm_tensorcore (1.655 ms) |
| 25 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.273 | 0.714 | 1.205 | 1.526 | 1.902 | gemm_tensorcore (1.653 ms) |
| 26 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.272 | 0.713 | 1.213 | 1.556 | 1.910 | gemm_tensorcore (1.660 ms) |
| 27 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.277 | 0.718 | 1.205 | 1.581 | 1.905 | gemm_tensorcore (1.655 ms) |
| 28 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.516 | 0.716 | 1.320 | 1.570 | 2.027 | gemm_tensorcore (1.780 ms) |
| 29 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.261 | 0.715 | 1.197 | 1.534 | 1.915 | gemm_tensorcore (1.665 ms) |
| 30 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.270 | 0.713 | 1.211 | 1.564 | 1.913 | gemm_tensorcore (1.664 ms) |
| 31 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.713 | 1.198 | 1.513 | 1.906 | gemm_tensorcore (1.657 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.184 | 0.497 | 0.343 | 1.321 | 0.532 | gemv_decode_cublas (0.454 ms) |
| 1 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.146 | 0.474 | 0.343 | 1.270 | 0.531 | gemv_decode_cublas (0.451 ms) |
| 2 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.141 | 0.473 | 0.342 | 1.269 | 0.532 | gemv_decode_cublas (0.449 ms) |
| 3 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.143 | 0.469 | 0.342 | 1.281 | 0.542 | gemv_decode_cublas (0.462 ms) |
| 4 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.124 | 0.465 | 0.339 | 1.264 | 0.532 | gemv_decode_cublas (0.453 ms) |
| 5 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.122 | 0.461 | 0.339 | 1.260 | 0.532 | gemv_decode_cublas (0.449 ms) |
| 6 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.128 | 0.465 | 0.346 | 1.268 | 0.533 | gemv_decode_cublas (0.450 ms) |
| 7 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.130 | 0.458 | 0.346 | 1.262 | 0.537 | gemv_decode_cublas (0.451 ms) |
| 8 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.115 | 0.461 | 0.339 | 1.265 | 0.542 | gemv_decode_cublas (0.456 ms) |
| 9 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.466 | 0.339 | 1.280 | 0.534 | gemv_decode_cublas (0.450 ms) |
| 10 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.115 | 0.456 | 0.341 | 1.258 | 0.535 | gemv_decode_cublas (0.452 ms) |
| 11 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.130 | 0.461 | 0.347 | 1.278 | 0.538 | gemv_decode_cublas (0.453 ms) |
| 12 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.459 | 0.341 | 1.260 | 0.531 | gemv_decode_cublas (0.449 ms) |
| 13 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.338 | 1.258 | 0.535 | gemv_decode_cublas (0.456 ms) |
| 14 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.116 | 0.459 | 0.342 | 1.262 | 0.540 | gemv_decode_cublas (0.455 ms) |
| 15 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.134 | 0.476 | 0.339 | 1.264 | 0.528 | gemv_decode_cublas (0.449 ms) |
| 16 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.139 | 0.467 | 0.346 | 1.293 | 0.532 | gemv_decode_cublas (0.452 ms) |
| 17 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.143 | 0.470 | 0.345 | 1.256 | 0.538 | gemv_decode_cublas (0.459 ms) |
| 18 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.339 | 1.249 | 0.536 | gemv_decode_cublas (0.451 ms) |
| 19 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.124 | 0.469 | 0.338 | 1.261 | 0.538 | gemv_decode_cublas (0.459 ms) |
| 20 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.127 | 0.469 | 0.338 | 1.265 | 0.538 | gemv_decode_cublas (0.452 ms) |
| 21 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.137 | 0.472 | 0.341 | 1.262 | 0.533 | gemv_decode_cublas (0.453 ms) |
| 22 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.339 | 1.250 | 0.532 | gemv_decode_cublas (0.452 ms) |
| 23 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.129 | 0.470 | 0.339 | 1.256 | 0.535 | gemv_decode_cublas (0.456 ms) |
| 24 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.133 | 0.466 | 0.342 | 1.259 | 0.534 | gemv_decode_cublas (0.452 ms) |
| 25 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.132 | 0.470 | 0.342 | 1.261 | 0.530 | gemv_decode_cublas (0.451 ms) |
| 26 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.460 | 0.341 | 1.280 | 0.535 | gemv_decode_cublas (0.452 ms) |
| 27 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.118 | 0.466 | 0.338 | 1.270 | 0.533 | gemv_decode_cublas (0.452 ms) |
| 28 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.117 | 0.459 | 0.339 | 1.255 | 0.534 | gemv_decode_cublas (0.454 ms) |
| 29 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.126 | 0.465 | 0.341 | 1.274 | 0.538 | gemv_decode_cublas (0.456 ms) |
| 30 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.125 | 0.461 | 0.342 | 1.271 | 0.542 | gemv_decode_cublas (0.457 ms) |
| 31 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.129 | 0.466 | 0.342 | 1.277 | 0.533 | gemv_decode_cublas (0.453 ms) |
