# Same-input VisiPruner VP-FA layer performance report

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

- clock json: `autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_fa2_32tok.json`
- clock ranges: `autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_fa2_32tok_ranges.csv`
- layer events: `autoresearch/experiments/e2_single_request_latency/output/clock_sameinput_visipruner_full_fa2_32tok_layer_events.csv`
- Nsight layer kernels: `autoresearch/experiments/e2_single_request_latency/output/nsys_sameinput_visipruner_full_fa2_32tok_layer_kernel_breakdown.csv`
- Nsight/CUPTI kernel attribution: CUDA Runtime API `correlationId` -> CUPTI GPU kernel `correlationId`; the runtime API call start must fall inside the NVTX CPU range. This is CUPTI launch-owned kernel attribution, not kernel-vs-range execution overlap.
- human draft reference: `/workspace/VisiPrune/workload_analysis/human_draft.md`

## End-to-end clock summary

| metric | ms |
|---|---:|
| request_total_ms | 1762.533 |
| generate_total_ms | 1713.769 |
| prepare_multimodal_ms | 18.798 |
| vision_encode_project_ms | 16.572 |
| forward_prefill_ms | 113.206 |
| forward_decode_sum_ms | 1554.962 |
| value_aware_token_selection_ms | 9.545 |

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
| 0 | 624 | 624 | triton_vpfa_prefill_shallow_layer0_mass_fold | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.409 | 1.486 | 1.287 | 3.056 | 2.117 | gemm_tensorcore (1.733 ms) |
| 1 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.135 | 1.319 | 1.276 | 2.548 | 2.097 | gemm_tensorcore (1.728 ms) |
| 2 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.146 | 1.309 | 1.282 | 2.470 | 2.106 | gemm_tensorcore (1.736 ms) |
| 3 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.092 | 1.280 | 1.279 | 3.412 | 2.254 | gemm_tensorcore (1.883 ms) |
| 4 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.343 | 1.469 | 1.280 | 2.611 | 2.105 | gemm_tensorcore (1.736 ms) |
| 5 | 624 | 624 | triton_vpfa_prefill_shallow_text_to_vision_mask | Triton causal VP-FA prefill attention; shallow VisiPruner post-softmax edits in-kernel; o_proj GEMM; MLP GEMMs | 3.073 | 1.292 | 1.273 | 2.589 | 2.098 | gemm_tensorcore (1.730 ms) |
| 6 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; o_proj GEMM; MLP GEMMs | 3.189 | 1.315 | 1.297 | 2.573 | 2.098 | gemm_tensorcore (1.731 ms) |
| 7 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.163 | 2.336 | 1.285 | 3.668 | 2.194 | gemm_tensorcore (1.733 ms) |
| 8 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.295 | 2.157 | 1.469 | 3.708 | 2.338 | gemm_tensorcore (1.737 ms) |
| 9 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.923 | 2.109 | 1.281 | 3.596 | 2.190 | gemm_tensorcore (1.731 ms) |
| 10 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.894 | 2.086 | 1.281 | 3.614 | 2.197 | gemm_tensorcore (1.737 ms) |
| 11 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.916 | 2.090 | 1.287 | 3.644 | 2.396 | gemm_tensorcore (1.936 ms) |
| 12 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.195 | 2.125 | 1.525 | 3.693 | 2.198 | gemm_tensorcore (1.739 ms) |
| 13 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.010 | 2.151 | 1.288 | 3.755 | 2.193 | gemm_tensorcore (1.734 ms) |
| 14 | 624 | 624 | triton_vpfa_prefill_last_query_proxy_or_selection | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.296 | 2.427 | 1.293 | 4.076 | 2.216 | gemm_tensorcore (1.738 ms) |
| 15 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.276 | 2.239 | 0.451 | 4.762 | 0.651 | gemm_tensorcore (0.508 ms) |
| 16 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.884 | 1.862 | 0.438 | 3.454 | 0.639 | gemm_tensorcore (0.497 ms) |
| 17 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.298 | 2.334 | 0.441 | 3.700 | 0.639 | gemm_tensorcore (0.496 ms) |
| 18 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.044 | 1.779 | 0.444 | 3.651 | 0.641 | gemm_tensorcore (0.498 ms) |
| 19 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.778 | 1.810 | 0.430 | 3.513 | 0.717 | gemm_tensorcore (0.496 ms) |
| 20 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.021 | 1.787 | 0.444 | 3.691 | 0.643 | gemm_tensorcore (0.500 ms) |
| 21 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 4.640 | 3.316 | 0.667 | 4.814 | 0.641 | gemm_tensorcore (0.499 ms) |
| 22 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.121 | 1.855 | 0.447 | 3.767 | 0.642 | gemm_tensorcore (0.499 ms) |
| 23 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.832 | 1.799 | 0.434 | 3.470 | 0.719 | gemm_tensorcore (0.497 ms) |
| 24 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 3.022 | 1.764 | 0.440 | 3.679 | 0.639 | gemm_tensorcore (0.496 ms) |
| 25 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.778 | 1.818 | 0.431 | 3.442 | 0.644 | gemm_tensorcore (0.502 ms) |
| 26 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.991 | 1.767 | 0.444 | 3.702 | 0.644 | gemm_tensorcore (0.502 ms) |
| 27 | 56 | 56 | triton_vpfa_middle_pruned_compact_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.882 | 1.897 | 0.433 | 3.413 | 0.844 | gemm_tensorcore (0.700 ms) |
| 28 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.306 | 1.332 | 0.429 | 2.850 | 0.600 | gemm_tensorcore (0.495 ms) |
| 29 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.534 | 1.284 | 0.439 | 3.153 | 0.601 | gemm_tensorcore (0.495 ms) |
| 30 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.293 | 1.303 | 0.427 | 2.837 | 0.601 | gemm_tensorcore (0.496 ms) |
| 31 | 48 | 48 | triton_vpfa_deep_removed_prefill | Triton causal VP-FA prefill attention; Triton last-query pruning proxy where needed; o_proj GEMM; MLP GEMMs | 2.313 | 1.290 | 0.430 | 2.786 | 0.609 | gemm_tensorcore (0.503 ms) |

## Decode forwards: per-layer repeated-token workload and latency

| layer | decode kv_len first -> last | workload type | operator path | clock total mean ms | attn mean ms | mlp mean ms | NVTX CPU range mean ms | CUPTI launch-owned kernel sum mean ms | dominant kernel family |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.494 | 0.748 | 0.349 | 2.361 | 0.570 | gemv_decode_cublas (0.480 ms) |
| 1 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.431 | 0.704 | 0.347 | 2.300 | 0.553 | gemv_decode_cublas (0.462 ms) |
| 2 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.438 | 0.701 | 0.353 | 2.229 | 0.552 | gemv_decode_cublas (0.463 ms) |
| 3 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.437 | 0.708 | 0.349 | 2.232 | 0.551 | gemv_decode_cublas (0.464 ms) |
| 4 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.438 | 0.706 | 0.353 | 2.360 | 0.562 | gemv_decode_cublas (0.474 ms) |
| 5 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.579 | 0.691 | 0.513 | 2.095 | 0.557 | gemv_decode_cublas (0.472 ms) |
| 6 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.449 | 0.714 | 0.351 | 2.198 | 0.577 | gemv_decode_cublas (0.482 ms) |
| 7 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.423 | 0.693 | 0.350 | 2.189 | 0.558 | gemv_decode_cublas (0.480 ms) |
| 8 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.457 | 0.716 | 0.355 | 2.269 | 0.551 | gemv_decode_cublas (0.471 ms) |
| 9 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.443 | 0.687 | 0.364 | 2.250 | 0.558 | gemv_decode_cublas (0.474 ms) |
| 10 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.463 | 0.717 | 0.355 | 2.133 | 0.553 | gemv_decode_cublas (0.473 ms) |
| 11 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.426 | 0.685 | 0.357 | 2.121 | 0.555 | gemv_decode_cublas (0.468 ms) |
| 12 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.439 | 0.702 | 0.351 | 2.227 | 0.560 | gemv_decode_cublas (0.470 ms) |
| 13 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.404 | 0.685 | 0.348 | 2.164 | 0.589 | gemv_decode_cublas (0.490 ms) |
| 14 | 625 -> 655 | triton_vpfa_decode_full_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.508 | 0.716 | 0.420 | 2.238 | 0.555 | gemv_decode_cublas (0.476 ms) |
| 15 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.447 | 0.706 | 0.354 | 2.124 | 0.529 | gemv_decode_cublas (0.464 ms) |
| 16 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.435 | 0.704 | 0.346 | 2.059 | 0.524 | gemv_decode_cublas (0.453 ms) |
| 17 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.443 | 0.703 | 0.352 | 2.107 | 0.536 | gemv_decode_cublas (0.468 ms) |
| 18 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.498 | 0.745 | 0.353 | 2.184 | 0.539 | gemv_decode_cublas (0.467 ms) |
| 19 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.472 | 0.724 | 0.350 | 2.235 | 0.522 | gemv_decode_cublas (0.458 ms) |
| 20 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.499 | 0.751 | 0.352 | 2.247 | 0.552 | gemv_decode_cublas (0.471 ms) |
| 21 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.417 | 0.688 | 0.351 | 2.089 | 0.521 | gemv_decode_cublas (0.457 ms) |
| 22 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.415 | 0.697 | 0.343 | 2.079 | 0.533 | gemv_decode_cublas (0.463 ms) |
| 23 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.407 | 0.683 | 0.344 | 2.088 | 0.513 | gemv_decode_cublas (0.449 ms) |
| 24 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.432 | 0.696 | 0.348 | 2.095 | 0.538 | gemv_decode_cublas (0.460 ms) |
| 25 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.405 | 0.687 | 0.344 | 2.151 | 0.526 | gemv_decode_cublas (0.459 ms) |
| 26 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.419 | 0.700 | 0.343 | 2.137 | 0.540 | gemv_decode_cublas (0.467 ms) |
| 27 | 57 -> 87 | triton_vpfa_decode_middle_pruned_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.424 | 0.707 | 0.344 | 2.117 | 0.533 | gemv_decode_cublas (0.455 ms) |
| 28 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.421 | 0.700 | 0.346 | 2.214 | 0.527 | gemv_decode_cublas (0.454 ms) |
| 29 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.410 | 0.683 | 0.347 | 2.252 | 0.538 | gemv_decode_cublas (0.464 ms) |
| 30 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.453 | 0.730 | 0.344 | 2.250 | 0.515 | gemv_decode_cublas (0.448 ms) |
| 31 | 49 -> 79 | triton_vpfa_decode_deep_removed_kv_cache | eager q_len=1 QK^T/softmax/AV over KV cache; o_proj GEMV; MLP GEMV | 1.463 | 0.737 | 0.350 | 2.356 | 0.520 | gemv_decode_cublas (0.456 ms) |
