# E2 dense-fa2 32-token layer performance report

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

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | nsys range ms | nsys kernel ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.605 | 0.721 | 1.476 | 2.269 | 1.457 | gemm_tensorcore (1.244 ms) |
| 1 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.310 | 0.711 | 1.213 | 1.805 | 1.155 | gemm_tensorcore (0.937 ms) |
| 2 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.275 | 0.711 | 1.204 | 1.716 | 1.130 | gemm_tensorcore (0.917 ms) |
| 3 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.252 | 0.708 | 1.194 | 1.664 | 1.117 | gemm_tensorcore (0.919 ms) |
| 4 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.273 | 0.712 | 1.203 | 1.642 | 1.085 | gemm_tensorcore (0.888 ms) |
| 5 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.713 | 1.199 | 1.589 | 1.099 | gemm_tensorcore (0.902 ms) |
| 6 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.252 | 0.706 | 1.196 | 1.587 | 1.104 | gemm_tensorcore (0.905 ms) |
| 7 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.548 | 0.707 | 1.473 | 1.644 | 1.060 | gemm_tensorcore (0.863 ms) |
| 8 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.269 | 0.713 | 1.207 | 1.554 | 1.098 | gemm_tensorcore (0.901 ms) |
| 9 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.286 | 0.713 | 1.206 | 1.519 | 1.082 | gemm_tensorcore (0.883 ms) |
| 10 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.711 | 1.200 | 1.520 | 1.077 | gemm_tensorcore (0.879 ms) |
| 11 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.249 | 0.709 | 1.197 | 1.438 | 1.070 | gemm_tensorcore (0.871 ms) |
| 12 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.260 | 0.711 | 1.203 | 1.441 | 1.061 | gemm_tensorcore (0.863 ms) |
| 13 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.258 | 0.713 | 1.199 | 1.645 | 1.080 | gemm_tensorcore (0.884 ms) |
| 14 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.462 | 0.711 | 1.268 | 1.500 | 1.081 | gemm_tensorcore (0.807 ms) |
| 15 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.251 | 0.707 | 1.204 | 1.513 | 1.058 | gemm_tensorcore (0.861 ms) |
| 16 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.715 | 1.205 | 1.454 | 1.070 | gemm_tensorcore (0.873 ms) |
| 17 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.266 | 0.714 | 1.201 | 1.417 | 1.069 | gemm_tensorcore (0.872 ms) |
| 18 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.258 | 0.711 | 1.210 | 1.447 | 1.074 | gemm_tensorcore (0.878 ms) |
| 19 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.231 | 0.706 | 1.209 | 1.446 | 1.073 | gemm_tensorcore (0.875 ms) |
| 20 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.257 | 0.707 | 1.217 | 1.445 | 1.107 | gemm_tensorcore (0.910 ms) |
| 21 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.510 | 0.711 | 1.469 | 1.542 | 1.225 | gemm_tensorcore (0.995 ms) |
| 22 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.259 | 0.713 | 1.201 | 1.429 | 1.085 | gemm_tensorcore (0.887 ms) |
| 23 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.713 | 1.201 | 1.514 | 1.085 | gemm_tensorcore (0.887 ms) |
| 24 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.264 | 0.710 | 1.201 | 1.583 | 1.088 | gemm_tensorcore (0.892 ms) |
| 25 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.273 | 0.714 | 1.205 | 1.526 | 1.078 | gemm_tensorcore (0.881 ms) |
| 26 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.272 | 0.713 | 1.213 | 1.556 | 1.108 | gemm_tensorcore (0.910 ms) |
| 27 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.277 | 0.718 | 1.205 | 1.581 | 1.076 | gemm_tensorcore (0.878 ms) |
| 28 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.516 | 0.716 | 1.320 | 1.570 | 1.149 | gemm_tensorcore (0.954 ms) |
| 29 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.261 | 0.715 | 1.197 | 1.534 | 1.069 | gemm_tensorcore (0.871 ms) |
| 30 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.270 | 0.713 | 1.211 | 1.564 | 1.110 | gemm_tensorcore (0.913 ms) |
| 31 | 624 | 624 | dense_fa2_full_prefill | FlashAttention2 dense full attention; o_proj GEMM; MLP GEMMs | 2.254 | 0.713 | 1.198 | 1.513 | 1.070 | gemm_tensorcore (0.873 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | nsys range mean ms | nsys kernel mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.184 | 0.497 | 0.343 | 1.321 | 0.388 | gemv_decode_cublas (9.674 ms) |
| 1 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.146 | 0.474 | 0.343 | 1.270 | 0.380 | gemv_decode_cublas (9.355 ms) |
| 2 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.141 | 0.473 | 0.342 | 1.269 | 0.384 | gemv_decode_cublas (9.411 ms) |
| 3 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.143 | 0.469 | 0.342 | 1.281 | 0.387 | gemv_decode_cublas (9.560 ms) |
| 4 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.124 | 0.465 | 0.339 | 1.264 | 0.380 | gemv_decode_cublas (9.406 ms) |
| 5 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.122 | 0.461 | 0.339 | 1.260 | 0.383 | gemv_decode_cublas (9.382 ms) |
| 6 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.128 | 0.465 | 0.346 | 1.268 | 0.384 | gemv_decode_cublas (9.384 ms) |
| 7 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.130 | 0.458 | 0.346 | 1.262 | 0.385 | gemv_decode_cublas (9.365 ms) |
| 8 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.115 | 0.461 | 0.339 | 1.265 | 0.386 | gemv_decode_cublas (9.369 ms) |
| 9 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.466 | 0.339 | 1.280 | 0.386 | gemv_decode_cublas (9.407 ms) |
| 10 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.115 | 0.456 | 0.341 | 1.258 | 0.379 | gemv_decode_cublas (9.272 ms) |
| 11 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.130 | 0.461 | 0.347 | 1.278 | 0.390 | gemv_decode_cublas (9.510 ms) |
| 12 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.459 | 0.341 | 1.260 | 0.377 | gemv_decode_cublas (9.220 ms) |
| 13 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.338 | 1.258 | 0.384 | gemv_decode_cublas (9.513 ms) |
| 14 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.116 | 0.459 | 0.342 | 1.262 | 0.384 | gemv_decode_cublas (9.305 ms) |
| 15 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.134 | 0.476 | 0.339 | 1.264 | 0.377 | gemv_decode_cublas (9.326 ms) |
| 16 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.139 | 0.467 | 0.346 | 1.293 | 0.377 | gemv_decode_cublas (9.271 ms) |
| 17 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.143 | 0.470 | 0.345 | 1.256 | 0.384 | gemv_decode_cublas (9.528 ms) |
| 18 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.339 | 1.249 | 0.385 | gemv_decode_cublas (9.332 ms) |
| 19 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.124 | 0.469 | 0.338 | 1.261 | 0.381 | gemv_decode_cublas (9.457 ms) |
| 20 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.127 | 0.469 | 0.338 | 1.265 | 0.385 | gemv_decode_cublas (9.330 ms) |
| 21 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.137 | 0.472 | 0.341 | 1.262 | 0.382 | gemv_decode_cublas (9.422 ms) |
| 22 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.120 | 0.463 | 0.339 | 1.250 | 0.380 | gemv_decode_cublas (9.364 ms) |
| 23 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.129 | 0.470 | 0.339 | 1.256 | 0.380 | gemv_decode_cublas (9.395 ms) |
| 24 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.133 | 0.466 | 0.342 | 1.259 | 0.382 | gemv_decode_cublas (9.368 ms) |
| 25 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.132 | 0.470 | 0.342 | 1.261 | 0.379 | gemv_decode_cublas (9.381 ms) |
| 26 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.121 | 0.460 | 0.341 | 1.280 | 0.391 | gemv_decode_cublas (9.601 ms) |
| 27 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.118 | 0.466 | 0.338 | 1.270 | 0.374 | gemv_decode_cublas (9.130 ms) |
| 28 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.117 | 0.459 | 0.339 | 1.255 | 0.381 | gemv_decode_cublas (9.402 ms) |
| 29 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.126 | 0.465 | 0.341 | 1.274 | 0.382 | gemv_decode_cublas (9.347 ms) |
| 30 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.125 | 0.461 | 0.342 | 1.271 | 0.390 | gemv_decode_cublas (9.528 ms) |
| 31 | 625 -> 655 | dense_fa2_decode_full_kv_cache | FlashAttention2 q_len=1 attention over dense KV cache; o_proj GEMV; MLP GEMV | 1.129 | 0.466 | 0.342 | 1.277 | 0.385 | gemv_decode_cublas (9.512 ms) |
