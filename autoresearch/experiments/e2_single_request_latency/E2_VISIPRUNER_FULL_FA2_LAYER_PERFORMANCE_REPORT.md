# E2 VisiPruner VP-FA layer performance report

This report follows the workload-reading style of `workload_analysis/human_draft.md`; that source file is read-only and is not modified.

## Run metadata

- config: `visipruner-full-vp-fa (legacy output config: visipruner-full-fa2)`
- description: VisPrune full path with optimized backend auto-selection; currently Triton VP-FA prefill when available.
- max_new_tokens: `32`
- use_flash_attn: `True`
- use_visipruner: `True`
- visipruner_decode_backend: `vp-fa (legacy output backend request: auto)`
- resolved_visipruner_decode_backend: `vp_fa`

## Data sources

- clock json: `autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok.json`
- clock ranges: `autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok_ranges.csv`
- layer events: `autoresearch/experiments/e2_single_request_latency/output/clock_e2_visipruner_full_fa2_32tok_layer_events.csv`
- Nsight layer kernels: `autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_fa2_32tok_layer_kernel_breakdown.csv`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
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

- layer 0: q_len=624, kv_len=624, workload=triton_vpfa_prefill_shallow_layer0_mass_fold; operator=Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs.
- layer 1-5: q_len=624, kv_len=624, workload=triton_vpfa_prefill_shallow_text_to_vision_mask; operator=Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs.
- layer 6: q_len=624, kv_len=624, workload=triton_vpfa_prefill_last_query_proxy_or_selection; operator=Triton causal VP-FA prefill attention; o_proj GEMM; MLP GEMMs.
- layer 7-14: q_len=624, kv_len=624, workload=triton_vpfa_prefill_last_query_proxy_or_selection; operator=Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs.
- layer 15-27: q_len=56, kv_len=56, workload=triton_vpfa_middle_pruned_compact_prefill; operator=Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs.
- layer 28-31: q_len=48, kv_len=48, workload=triton_vpfa_deep_removed_prefill; operator=Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs.

forward 2

- layer 0-14: q_len=1, kv_len=625, workload=triton_vpfa_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 15-27: q_len=1, kv_len=57, workload=triton_vpfa_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=49, workload=triton_vpfa_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

forward 32

- layer 0-14: q_len=1, kv_len=655, workload=triton_vpfa_decode_full_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 15-27: q_len=1, kv_len=87, workload=triton_vpfa_decode_middle_pruned_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.
- layer 28-31: q_len=1, kv_len=79, workload=triton_vpfa_decode_deep_removed_kv_cache; operator=eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV.

## Forward 1: Prefill layer workload and latency

| layer | q_len | kv_len | workload type | operator path | clock total ms | attn ms | mlp ms | NVTX CPU range ms | CUPTI launch-owned kernel sum ms | dominant kernel family |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | 624 | 624 | triton_vpfa_prefill_shallow_layer0_mass_fold | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.315 | 1.426 | 1.276 | 2.788 | 2.113 | gemm_tensorcore (1.730 ms) |
| 1 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.163 | 1.319 | 1.272 | 2.388 | 2.097 | gemm_tensorcore (1.729 ms) |
| 2 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.052 | 1.280 | 1.270 | 2.325 | 2.102 | gemm_tensorcore (1.732 ms) |
| 3 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.074 | 1.286 | 1.270 | 2.334 | 2.204 | gemm_tensorcore (1.804 ms) |
| 4 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.229 | 1.443 | 1.272 | 2.468 | 2.100 | gemm_tensorcore (1.732 ms) |
| 5 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.054 | 1.271 | 1.268 | 2.350 | 2.099 | gemm_tensorcore (1.731 ms) |
| 6 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; o_proj GEMM; MLP GEMMs | 3.039 | 1.261 | 1.271 | 2.321 | 2.098 | gemm_tensorcore (1.732 ms) |
| 7 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.992 | 2.195 | 1.279 | 3.436 | 2.192 | gemm_tensorcore (1.733 ms) |
| 8 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.903 | 2.069 | 1.297 | 3.459 | 2.258 | gemm_tensorcore (1.735 ms) |
| 9 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.007 | 2.173 | 1.276 | 3.261 | 2.193 | gemm_tensorcore (1.732 ms) |
| 10 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.978 | 2.133 | 1.274 | 3.225 | 2.195 | gemm_tensorcore (1.735 ms) |
| 11 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.875 | 2.068 | 1.277 | 3.206 | 2.195 | gemm_tensorcore (1.736 ms) |
| 12 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.914 | 2.097 | 1.280 | 3.311 | 2.294 | gemm_tensorcore (1.739 ms) |
| 13 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.128 | 2.315 | 1.277 | 3.169 | 2.196 | gemm_tensorcore (1.736 ms) |
| 14 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.314 | 2.461 | 1.286 | 3.493 | 2.213 | gemm_tensorcore (1.736 ms) |
| 15 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.203 | 2.155 | 0.464 | 3.243 | 0.649 | gemm_tensorcore (0.507 ms) |
| 16 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.865 | 1.867 | 0.434 | 2.942 | 0.642 | gemm_tensorcore (0.500 ms) |
| 17 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.236 | 2.224 | 0.427 | 3.051 | 0.636 | gemm_tensorcore (0.494 ms) |
| 18 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.254 | 1.919 | 0.501 | 3.027 | 0.642 | gemm_tensorcore (0.499 ms) |
| 19 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.985 | 1.949 | 0.434 | 2.831 | 0.638 | gemm_tensorcore (0.496 ms) |
| 20 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.247 | 1.938 | 0.447 | 3.011 | 0.648 | gemm_tensorcore (0.505 ms) |
| 21 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.906 | 1.910 | 0.433 | 3.807 | 0.707 | gemm_tensorcore (0.498 ms) |
| 22 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.416 | 2.043 | 0.448 | 3.076 | 0.639 | gemm_tensorcore (0.496 ms) |
| 23 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.027 | 2.010 | 0.435 | 2.913 | 0.640 | gemm_tensorcore (0.498 ms) |
| 24 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.300 | 1.948 | 0.446 | 3.024 | 0.640 | gemm_tensorcore (0.497 ms) |
| 25 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.965 | 1.942 | 0.432 | 2.884 | 0.642 | gemm_tensorcore (0.500 ms) |
| 26 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.284 | 1.965 | 0.446 | 3.085 | 0.713 | gemm_tensorcore (0.499 ms) |
| 27 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.550 | 2.503 | 0.436 | 2.977 | 0.638 | gemm_tensorcore (0.496 ms) |
| 28 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.515 | 1.439 | 0.436 | 2.384 | 0.601 | gemm_tensorcore (0.495 ms) |
| 29 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.748 | 1.420 | 0.440 | 2.561 | 0.601 | gemm_tensorcore (0.495 ms) |
| 30 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.448 | 1.419 | 0.432 | 2.372 | 0.602 | gemm_tensorcore (0.497 ms) |
| 31 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.558 | 1.420 | 0.441 | 2.377 | 0.600 | gemm_tensorcore (0.495 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.500 | 0.740 | 0.350 | 2.168 | 0.554 | gemv_decode_cublas (0.475 ms) |
| 1 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.450 | 0.711 | 0.348 | 2.099 | 0.540 | gemv_decode_cublas (0.462 ms) |
| 2 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.440 | 0.701 | 0.349 | 2.092 | 0.551 | gemv_decode_cublas (0.464 ms) |
| 3 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.453 | 0.705 | 0.355 | 2.110 | 0.571 | gemv_decode_cublas (0.487 ms) |
| 4 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.455 | 0.717 | 0.354 | 2.152 | 0.559 | gemv_decode_cublas (0.480 ms) |
| 5 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.436 | 0.705 | 0.350 | 2.089 | 0.567 | gemv_decode_cublas (0.469 ms) |
| 6 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.456 | 0.716 | 0.352 | 2.091 | 0.555 | gemv_decode_cublas (0.469 ms) |
| 7 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.440 | 0.707 | 0.349 | 2.087 | 0.580 | gemv_decode_cublas (0.479 ms) |
| 8 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.462 | 0.708 | 0.355 | 2.093 | 0.561 | gemv_decode_cublas (0.474 ms) |
| 9 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.438 | 0.706 | 0.348 | 2.089 | 0.555 | gemv_decode_cublas (0.471 ms) |
| 10 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.447 | 0.706 | 0.352 | 2.091 | 0.563 | gemv_decode_cublas (0.464 ms) |
| 11 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.431 | 0.692 | 0.351 | 2.092 | 0.564 | gemv_decode_cublas (0.481 ms) |
| 12 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.463 | 0.716 | 0.352 | 2.100 | 0.554 | gemv_decode_cublas (0.467 ms) |
| 13 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.431 | 0.696 | 0.349 | 2.078 | 0.565 | gemv_decode_cublas (0.475 ms) |
| 14 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.483 | 0.734 | 0.360 | 2.112 | 0.569 | gemv_decode_cublas (0.481 ms) |
| 15 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.709 | 0.345 | 2.079 | 0.526 | gemv_decode_cublas (0.460 ms) |
| 16 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.433 | 0.695 | 0.347 | 2.074 | 0.541 | gemv_decode_cublas (0.464 ms) |
| 17 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.430 | 0.689 | 0.350 | 2.067 | 0.529 | gemv_decode_cublas (0.461 ms) |
| 18 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.429 | 0.694 | 0.347 | 2.073 | 0.520 | gemv_decode_cublas (0.456 ms) |
| 19 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.420 | 0.686 | 0.352 | 2.074 | 0.543 | gemv_decode_cublas (0.470 ms) |
| 20 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.437 | 0.704 | 0.344 | 2.076 | 0.528 | gemv_decode_cublas (0.458 ms) |
| 21 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.425 | 0.686 | 0.348 | 2.052 | 0.527 | gemv_decode_cublas (0.455 ms) |
| 22 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.420 | 0.692 | 0.345 | 2.068 | 0.522 | gemv_decode_cublas (0.455 ms) |
| 23 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.410 | 0.681 | 0.346 | 2.058 | 0.549 | gemv_decode_cublas (0.475 ms) |
| 24 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.424 | 0.693 | 0.345 | 2.046 | 0.531 | gemv_decode_cublas (0.461 ms) |
| 25 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.418 | 0.682 | 0.348 | 2.053 | 0.517 | gemv_decode_cublas (0.449 ms) |
| 26 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.705 | 0.347 | 2.076 | 0.522 | gemv_decode_cublas (0.457 ms) |
| 27 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.418 | 0.687 | 0.347 | 2.056 | 0.525 | gemv_decode_cublas (0.454 ms) |
| 28 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.434 | 0.691 | 0.351 | 2.061 | 0.541 | gemv_decode_cublas (0.463 ms) |
| 29 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.426 | 0.687 | 0.352 | 2.061 | 0.543 | gemv_decode_cublas (0.473 ms) |
| 30 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.443 | 0.712 | 0.345 | 2.096 | 0.512 | gemv_decode_cublas (0.448 ms) |
| 31 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.722 | 0.342 | 2.128 | 0.528 | gemv_decode_cublas (0.463 ms) |
