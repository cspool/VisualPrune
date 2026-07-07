# E2 eager VisiPruner full layer performance report

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
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
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

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.317 | 1.689 | 1.199 | 2.931 | 2.077 | gemm_tensorcore (1.696 ms) |
| 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.871 | 1.312 | 1.196 | 2.173 | 2.076 | gemm_tensorcore (1.727 ms) |
| 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.783 | 1.239 | 1.199 | 2.282 | 2.040 | gemm_tensorcore (1.701 ms) |
| 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.689 | 1.159 | 1.198 | 1.984 | 2.058 | gemm_tensorcore (1.701 ms) |
| 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.745 | 1.219 | 1.195 | 2.037 | 2.043 | gemm_tensorcore (1.703 ms) |
| 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.750 | 1.196 | 1.206 | 1.929 | 2.062 | gemm_tensorcore (1.708 ms) |
| 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs | 3.003 | 1.184 | 1.327 | 1.974 | 2.054 | gemm_tensorcore (1.716 ms) |
| 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.064 | 1.506 | 1.200 | 2.482 | 2.231 | gemm_tensorcore (1.830 ms) |
| 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.091 | 1.529 | 1.202 | 2.458 | 2.104 | gemm_tensorcore (1.709 ms) |
| 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.996 | 1.437 | 1.205 | 2.197 | 2.115 | gemm_tensorcore (1.706 ms) |
| 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.985 | 1.466 | 1.196 | 2.269 | 2.156 | gemm_tensorcore (1.757 ms) |
| 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.969 | 1.413 | 1.211 | 2.121 | 2.106 | gemm_tensorcore (1.696 ms) |
| 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.394 | 1.852 | 1.199 | 2.309 | 2.109 | gemm_tensorcore (1.713 ms) |
| 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.998 | 1.442 | 1.200 | 2.127 | 2.115 | gemm_tensorcore (1.708 ms) |
| 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.119 | 1.549 | 1.213 | 2.327 | 2.099 | gemm_tensorcore (1.703 ms) |
| 15 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.028 | 1.461 | 1.204 | 2.138 | 2.110 | gemm_tensorcore (1.701 ms) |
| 16 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.056 | 1.507 | 1.204 | 2.320 | 2.093 | gemm_tensorcore (1.697 ms) |
| 17 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.122 | 1.580 | 1.203 | 2.160 | 2.215 | gemm_tensorcore (1.710 ms) |
| 18 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.289 | 1.730 | 1.206 | 2.742 | 2.120 | gemm_tensorcore (1.705 ms) |
| 19 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.022 | 1.260 | 0.397 | 2.293 | 0.627 | gemm_tensorcore (0.510 ms) |
| 20 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.960 | 1.075 | 0.391 | 2.228 | 0.620 | gemm_tensorcore (0.502 ms) |
| 21 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.797 | 1.059 | 0.387 | 2.084 | 0.619 | gemm_tensorcore (0.502 ms) |
| 22 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.937 | 1.055 | 0.394 | 2.582 | 0.617 | gemm_tensorcore (0.499 ms) |
| 23 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.795 | 1.055 | 0.387 | 1.979 | 0.619 | gemm_tensorcore (0.502 ms) |
| 24 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.974 | 1.119 | 0.390 | 2.034 | 0.618 | gemm_tensorcore (0.501 ms) |
| 25 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.756 | 1.030 | 0.384 | 1.913 | 0.618 | gemm_tensorcore (0.501 ms) |
| 26 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.887 | 1.034 | 0.389 | 2.089 | 0.618 | gemm_tensorcore (0.501 ms) |
| 27 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.711 | 1.010 | 0.382 | 1.967 | 0.619 | gemm_tensorcore (0.502 ms) |
| 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.442 | 0.729 | 0.382 | 1.555 | 0.584 | gemm_tensorcore (0.501 ms) |
| 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.594 | 0.732 | 0.387 | 1.738 | 0.583 | gemm_tensorcore (0.499 ms) |
| 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.448 | 0.720 | 0.383 | 1.601 | 0.581 | gemm_tensorcore (0.498 ms) |
| 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.438 | 0.723 | 0.382 | 1.686 | 0.667 | gemm_tensorcore (0.494 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.336 | 0.647 | 0.342 | 1.518 | 0.541 | gemv_decode_cublas (0.465 ms) |
| 1 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.324 | 0.628 | 0.344 | 1.464 | 0.540 | gemv_decode_cublas (0.466 ms) |
| 2 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.299 | 0.620 | 0.344 | 1.457 | 0.544 | gemv_decode_cublas (0.464 ms) |
| 3 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.311 | 0.624 | 0.349 | 1.455 | 0.540 | gemv_decode_cublas (0.463 ms) |
| 4 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.619 | 0.341 | 1.446 | 0.551 | gemv_decode_cublas (0.469 ms) |
| 5 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.297 | 0.614 | 0.345 | 1.449 | 0.541 | gemv_decode_cublas (0.465 ms) |
| 6 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.279 | 0.608 | 0.341 | 1.474 | 0.543 | gemv_decode_cublas (0.465 ms) |
| 7 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.302 | 0.627 | 0.341 | 1.449 | 0.549 | gemv_decode_cublas (0.469 ms) |
| 8 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.619 | 0.344 | 1.435 | 0.537 | gemv_decode_cublas (0.462 ms) |
| 9 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.618 | 0.342 | 1.447 | 0.545 | gemv_decode_cublas (0.465 ms) |
| 10 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.306 | 0.617 | 0.347 | 1.444 | 0.544 | gemv_decode_cublas (0.465 ms) |
| 11 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.304 | 0.623 | 0.346 | 1.448 | 0.544 | gemv_decode_cublas (0.464 ms) |
| 12 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.292 | 0.611 | 0.343 | 1.444 | 0.548 | gemv_decode_cublas (0.467 ms) |
| 13 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.302 | 0.624 | 0.344 | 1.432 | 0.542 | gemv_decode_cublas (0.461 ms) |
| 14 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.293 | 0.618 | 0.341 | 1.429 | 0.551 | gemv_decode_cublas (0.468 ms) |
| 15 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.300 | 0.615 | 0.349 | 1.428 | 0.541 | gemv_decode_cublas (0.461 ms) |
| 16 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.296 | 0.612 | 0.346 | 1.426 | 0.540 | gemv_decode_cublas (0.464 ms) |
| 17 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.289 | 0.614 | 0.341 | 1.461 | 0.552 | gemv_decode_cublas (0.465 ms) |
| 18 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.292 | 0.617 | 0.341 | 1.436 | 0.536 | gemv_decode_cublas (0.461 ms) |
| 19 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.295 | 0.620 | 0.340 | 1.438 | 0.518 | gemv_decode_cublas (0.458 ms) |
| 20 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.287 | 0.608 | 0.344 | 1.438 | 0.516 | gemv_decode_cublas (0.456 ms) |
| 21 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.281 | 0.608 | 0.340 | 1.439 | 0.515 | gemv_decode_cublas (0.454 ms) |
| 22 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.297 | 0.611 | 0.345 | 1.454 | 0.524 | gemv_decode_cublas (0.457 ms) |
| 23 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.288 | 0.613 | 0.340 | 1.442 | 0.514 | gemv_decode_cublas (0.452 ms) |
| 24 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.613 | 0.342 | 1.453 | 0.518 | gemv_decode_cublas (0.457 ms) |
| 25 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.289 | 0.615 | 0.342 | 1.448 | 0.524 | gemv_decode_cublas (0.460 ms) |
| 26 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.286 | 0.615 | 0.340 | 1.436 | 0.513 | gemv_decode_cublas (0.452 ms) |
| 27 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.294 | 0.616 | 0.340 | 1.449 | 0.520 | gemv_decode_cublas (0.457 ms) |
| 28 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.290 | 0.621 | 0.336 | 1.434 | 0.513 | gemv_decode_cublas (0.452 ms) |
| 29 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.275 | 0.608 | 0.338 | 1.431 | 0.515 | gemv_decode_cublas (0.455 ms) |
| 30 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.300 | 0.633 | 0.337 | 1.460 | 0.511 | gemv_decode_cublas (0.452 ms) |
| 31 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.318 | 0.641 | 0.339 | 1.482 | 0.511 | gemv_decode_cublas (0.451 ms) |
