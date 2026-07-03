# Same-input eager VisiPruner full layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `visipruner-full`
- description: Native VisPrune full path: shallow + middle + deep.
- max_new_tokens: `32`
- use_flash_attn: `False`
- use_visipruner: `True`
- visipruner_decode_backend: `eager`

## Data sources

- clock json: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_eager_32tok.json`
- clock ranges: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_eager_32tok_ranges.csv`
- layer events: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_eager_32tok_layer_events.csv`
- Nsight layer kernels: `/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_sameinput_visipruner_full_eager_32tok_layer_kernel_breakdown.csv`
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1596.428 |
| generate_total_ms | 1559.544 |
| prepare_multimodal_ms | 15.134 |
| vision_encode_project_ms | 13.527 |
| forward_prefill_ms | 87.377 |
| forward_decode_sum_ms | 1433.784 |
| value_aware_token_selection_ms | 6.138 |

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
| 0 | 624 | 624 | eager_visipruner_prefill_shallow_layer0_mass_fold | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 3.404 | 1.735 | 1.204 | 2.793 | 1.117 | gemm_tensorcore (0.788 ms) |
| 1 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.795 | 1.221 | 1.194 | 1.987 | 1.063 | gemm_tensorcore (0.765 ms) |
| 2 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.890 | 1.326 | 1.197 | 2.291 | 1.212 | gemm_tensorcore (0.769 ms) |
| 3 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.764 | 1.206 | 1.196 | 1.996 | 1.094 | gemm_tensorcore (0.790 ms) |
| 4 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.838 | 1.274 | 1.196 | 2.090 | 1.051 | gemm_tensorcore (0.764 ms) |
| 5 | 624 | 624 | eager_visipruner_prefill_shallow_text_to_vision_mask | eager QK^T/softmax/AV attention; shallow VisiPruner post-softmax edits; o_proj GEMM; MLP GEMMs | 2.954 | 1.382 | 1.201 | 2.022 | 1.080 | gemm_tensorcore (0.778 ms) |
| 6 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; o_proj GEMM; MLP GEMMs | 2.798 | 1.225 | 1.206 | 2.009 | 1.056 | gemm_tensorcore (0.769 ms) |
| 7 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.150 | 1.561 | 1.209 | 2.426 | 1.091 | gemm_tensorcore (0.734 ms) |
| 8 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.198 | 1.624 | 1.200 | 2.479 | 1.122 | gemm_tensorcore (0.779 ms) |
| 9 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.065 | 1.488 | 1.215 | 2.268 | 1.128 | gemm_tensorcore (0.770 ms) |
| 10 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.341 | 1.581 | 1.236 | 2.456 | 1.128 | gemm_tensorcore (0.783 ms) |
| 11 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.039 | 1.467 | 1.198 | 2.355 | 1.133 | gemm_tensorcore (0.776 ms) |
| 12 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.125 | 1.549 | 1.215 | 3.104 | 1.129 | gemm_tensorcore (0.759 ms) |
| 13 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.051 | 1.471 | 1.210 | 2.296 | 1.124 | gemm_tensorcore (0.770 ms) |
| 14 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.109 | 1.548 | 1.208 | 2.532 | 1.129 | gemm_tensorcore (0.786 ms) |
| 15 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.262 | 1.462 | 1.324 | 2.293 | 1.125 | gemm_tensorcore (0.770 ms) |
| 16 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.152 | 1.574 | 1.203 | 2.456 | 1.129 | gemm_tensorcore (0.771 ms) |
| 17 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.073 | 1.475 | 1.236 | 2.602 | 1.228 | gemm_tensorcore (0.873 ms) |
| 18 | 624 | 624 | eager_visipruner_prefill_last_query_proxy_or_selection | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 3.461 | 1.859 | 1.218 | 2.778 | 1.128 | gemm_tensorcore (0.764 ms) |
| 19 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.063 | 1.292 | 0.391 | 2.374 | 0.477 | gemm_tensorcore (0.367 ms) |
| 20 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 2.005 | 1.104 | 0.404 | 2.229 | 0.463 | gemm_tensorcore (0.352 ms) |
| 21 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.944 | 1.190 | 0.388 | 2.117 | 0.456 | gemm_tensorcore (0.347 ms) |
| 22 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.967 | 1.074 | 0.394 | 2.595 | 0.468 | gemm_tensorcore (0.357 ms) |
| 23 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.839 | 1.095 | 0.389 | 2.141 | 0.469 | gemm_tensorcore (0.359 ms) |
| 24 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.931 | 1.050 | 0.394 | 2.316 | 0.465 | gemm_tensorcore (0.355 ms) |
| 25 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.783 | 1.054 | 0.385 | 2.150 | 0.464 | gemm_tensorcore (0.355 ms) |
| 26 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.931 | 1.046 | 0.393 | 2.157 | 0.466 | gemm_tensorcore (0.356 ms) |
| 27 | 58 | 58 | eager_visipruner_middle_pruned_compact_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.817 | 1.085 | 0.387 | 2.011 | 0.455 | gemm_tensorcore (0.346 ms) |
| 28 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.509 | 0.769 | 0.386 | 1.670 | 0.416 | gemm_tensorcore (0.340 ms) |
| 29 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.659 | 0.773 | 0.390 | 1.946 | 0.454 | gemm_tensorcore (0.353 ms) |
| 30 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.600 | 0.761 | 0.384 | 1.708 | 0.430 | gemm_tensorcore (0.354 ms) |
| 31 | 48 | 48 | eager_visipruner_deep_removed_prefill | eager QK^T/softmax/AV attention; last-query pruning proxy/score computation; o_proj GEMM; MLP GEMMs | 1.483 | 0.738 | 0.386 | 1.726 | 0.438 | gemm_tensorcore (0.362 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | nsys range mean ms | nsys kernel mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.380 | 0.667 | 0.347 | 1.523 | 0.391 | gemv_decode_cublas (9.920 ms) |
| 1 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.341 | 0.642 | 0.342 | 1.480 | 0.388 | gemv_decode_cublas (9.668 ms) |
| 2 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.339 | 0.646 | 0.342 | 1.500 | 0.386 | gemv_decode_cublas (9.737 ms) |
| 3 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.339 | 0.639 | 0.347 | 1.465 | 0.396 | gemv_decode_cublas (9.796 ms) |
| 4 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.330 | 0.640 | 0.343 | 1.449 | 0.391 | gemv_decode_cublas (9.792 ms) |
| 5 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.336 | 0.640 | 0.343 | 1.443 | 0.386 | gemv_decode_cublas (9.768 ms) |
| 6 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.328 | 0.638 | 0.343 | 1.437 | 0.391 | gemv_decode_cublas (9.655 ms) |
| 7 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.326 | 0.635 | 0.343 | 1.444 | 0.394 | gemv_decode_cublas (9.815 ms) |
| 8 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.342 | 0.644 | 0.348 | 1.434 | 0.381 | gemv_decode_cublas (9.531 ms) |
| 9 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.342 | 0.652 | 0.343 | 1.450 | 0.386 | gemv_decode_cublas (9.702 ms) |
| 10 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.324 | 0.634 | 0.345 | 1.437 | 0.390 | gemv_decode_cublas (9.711 ms) |
| 11 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.356 | 0.640 | 0.354 | 1.458 | 0.391 | gemv_decode_cublas (9.700 ms) |
| 12 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.356 | 0.634 | 0.367 | 1.441 | 0.380 | gemv_decode_cublas (9.492 ms) |
| 13 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.351 | 0.642 | 0.350 | 1.463 | 0.395 | gemv_decode_cublas (9.867 ms) |
| 14 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.340 | 0.641 | 0.346 | 1.441 | 0.382 | gemv_decode_cublas (9.639 ms) |
| 15 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.334 | 0.635 | 0.345 | 1.451 | 0.390 | gemv_decode_cublas (9.826 ms) |
| 16 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.341 | 0.634 | 0.351 | 1.445 | 0.387 | gemv_decode_cublas (9.719 ms) |
| 17 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.343 | 0.644 | 0.349 | 1.457 | 0.389 | gemv_decode_cublas (9.810 ms) |
| 18 | 625 -> 655 | eager_visipruner_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.330 | 0.632 | 0.346 | 1.440 | 0.380 | gemv_decode_cublas (9.503 ms) |
| 19 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.332 | 0.647 | 0.340 | 1.448 | 0.363 | gemv_decode_cublas (9.481 ms) |
| 20 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.328 | 0.641 | 0.339 | 1.449 | 0.366 | gemv_decode_cublas (9.539 ms) |
| 21 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.325 | 0.638 | 0.340 | 1.445 | 0.364 | gemv_decode_cublas (9.396 ms) |
| 22 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.330 | 0.646 | 0.339 | 1.450 | 0.365 | gemv_decode_cublas (9.549 ms) |
| 23 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.326 | 0.641 | 0.339 | 1.444 | 0.364 | gemv_decode_cublas (9.369 ms) |
| 24 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.317 | 0.636 | 0.338 | 1.454 | 0.366 | gemv_decode_cublas (9.601 ms) |
| 25 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.319 | 0.634 | 0.339 | 1.447 | 0.369 | gemv_decode_cublas (9.527 ms) |
| 26 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.333 | 0.645 | 0.342 | 1.459 | 0.365 | gemv_decode_cublas (9.504 ms) |
| 27 | 59 -> 89 | eager_visipruner_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.336 | 0.644 | 0.343 | 1.468 | 0.360 | gemv_decode_cublas (9.234 ms) |
| 28 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.323 | 0.636 | 0.340 | 1.450 | 0.363 | gemv_decode_cublas (9.496 ms) |
| 29 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.323 | 0.632 | 0.342 | 1.453 | 0.366 | gemv_decode_cublas (9.522 ms) |
| 30 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.355 | 0.652 | 0.345 | 1.492 | 0.371 | gemv_decode_cublas (9.628 ms) |
| 31 | 49 -> 79 | eager_visipruner_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.359 | 0.664 | 0.343 | 1.482 | 0.362 | gemv_decode_cublas (9.384 ms) |
