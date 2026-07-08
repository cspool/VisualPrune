# VisiPruner Full Eager SAME_INPUT Process-wise Performance Report

## Sources

- Nsight sqlite: `autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager/nsys_sameinput_visipruner_full_eager_32tok.sqlite`
- Handoff: `autoresearch/experiments/e2_single_request_latency/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md`
- Fragment CSV: `autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv`
- Aggregated process CSV: `autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv`

## Handoff Summary

- Variant scope: `visipruner-full-eager`
- Handoff inventory rows: 11
- Attribution method: launch-owned CUDA runtime `correlationId` to CUPTI kernels.
- FX process data is used for semantic validation and process naming, not as a timing source.

## Process-level NVTX Coverage

- Process NVTX fragment ranges: 371
- Aggregated process rows: 301
- Parent layer ranges with process coverage: 35
- Validation status counts: {'partially_validated': 81, 'validated': 290}

## Process Launch-owned Kernel Breakdown

| CUPTI kernel ms | NVTX CPU ms | process_cupti_pct_in_parent | process_nvtx_pct_in_parent | phase | forward_id | layer | process_id | process_title | fragment_id | matched_kernel_families | runtime API calls | kernel instances | validation status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.268 | 0.255 | 57.15% | 9.94% | prefill | 1 | 11 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.165 | 0.229 | 54.88% | 7.34% | prefill | 1 | 16 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.165 | 0.275 | 55.41% | 9.95% | prefill | 1 | 8 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.165 | 0.297 | 56.82% | 13.01% | prefill | 1 | 6 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.163 | 0.260 | 55.27% | 8.99% | prefill | 1 | 12 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.161 | 0.245 | 55.10% | 9.93% | prefill | 1 | 13 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.160 | 0.281 | 56.49% | 12.26% | prefill | 1 | 5 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.160 | 0.281 | 55.26% | 10.14% | prefill | 1 | 10 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.159 | 0.260 | 54.87% | 10.10% | prefill | 1 | 9 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.159 | 0.245 | 55.21% | 9.47% | prefill | 1 | 14 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.157 | 0.280 | 52.14% | 9.77% | prefill | 1 | 7 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.156 | 0.320 | 55.85% | 10.47% | prefill | 1 | 0 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.156 | 0.247 | 54.22% | 10.20% | prefill | 1 | 15 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.155 | 0.232 | 54.69% | 9.92% | prefill | 1 | 17 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 1.155 | 0.228 | 54.54% | 8.19% | prefill | 1 | 18 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 13 | 6 | validated |
| 0.404 | 0.192 | 19.11% | 7.48% | prefill | 1 | 9 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.403 | 0.153 | 19.08% | 6.54% | prefill | 1 | 17 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.402 | 0.197 | 19.62% | 8.63% | prefill | 1 | 6 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.402 | 0.151 | 19.00% | 5.43% | prefill | 1 | 18 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.402 | 0.184 | 19.11% | 6.35% | prefill | 1 | 12 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.402 | 0.181 | 18.11% | 6.33% | prefill | 1 | 7 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.401 | 0.156 | 18.83% | 6.45% | prefill | 1 | 15 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.401 | 0.172 | 18.89% | 5.51% | prefill | 1 | 16 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.401 | 0.182 | 19.11% | 6.58% | prefill | 1 | 10 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.401 | 0.165 | 19.11% | 6.38% | prefill | 1 | 14 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.401 | 0.196 | 19.06% | 7.10% | prefill | 1 | 8 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.400 | 0.202 | 19.48% | 8.81% | prefill | 1 | 5 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.400 | 0.170 | 18.01% | 6.65% | prefill | 1 | 11 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.399 | 0.217 | 18.95% | 8.79% | prefill | 1 | 13 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.398 | 0.225 | 19.23% | 7.37% | prefill | 1 | 0 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemm_tensorcore | 9 | 3 | validated |
| 0.336 | 0.288 | 54.32% | 11.77% | prefill | 1 | 25 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.336 | 0.298 | 53.41% | 11.62% | prefill | 1 | 19 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.336 | 0.279 | 54.19% | 12.18% | prefill | 1 | 23 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.336 | 0.268 | 46.81% | 11.53% | prefill | 1 | 21 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.336 | 0.289 | 54.10% | 11.84% | prefill | 1 | 20 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.335 | 0.290 | 54.17% | 11.29% | prefill | 1 | 26 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.335 | 0.258 | 54.17% | 10.93% | prefill | 1 | 22 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.335 | 0.285 | 54.24% | 11.86% | prefill | 1 | 24 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.334 | 0.284 | 47.55% | 11.52% | prefill | 1 | 27 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.333 | 0.284 | 57.15% | 14.49% | prefill | 1 | 28 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | 15 | 7 | partially_validated |
| 0.302 | 0.225 | 57.89% | 13.76% | decode | 32 | 31 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.301 | 0.216 | 55.62% | 13.24% | decode | 32 | 18 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.300 | 0.227 | 54.63% | 13.74% | decode | 2 | 18 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.299 | 0.214 | 57.44% | 13.18% | decode | 32 | 19 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.298 | 0.222 | 57.94% | 13.34% | decode | 2 | 19 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.296 | 0.230 | 57.50% | 14.08% | decode | 32 | 28 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.295 | 0.229 | 57.28% | 14.50% | decode | 32 | 27 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.294 | 0.226 | 57.95% | 13.57% | decode | 2 | 28 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.294 | 0.209 | 58.10% | 13.22% | decode | 2 | 27 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.293 | 0.231 | 48.12% | 13.40% | decode | 2 | 31 | mlp | MLP and final residual | aggregated(part01, part02) | elementwise_norm_activation, gemv_decode_cublas | 12 | 6 | validated |
| 0.272 | 0.172 | 12.25% | 6.00% | prefill | 1 | 7 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.216 | 0.137 | 35.51% | 7.99% | decode | 2 | 31 | qkv_projection | Q/K/V projection and head reshape | aggregated(whole) | gemv_decode_cublas | 6 | 3 | validated |
| 0.191 | 0.896 | 9.25% | 29.31% | prefill | 1 | 0 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan, softmax | 34 | 9 | partially_validated |
| 0.189 | 0.171 | 8.86% | 7.08% | prefill | 1 | 15 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.172 | 0.872 | 8.12% | 27.93% | prefill | 1 | 16 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.167 | 0.322 | 8.11% | 14.05% | prefill | 1 | 5 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 20 | 7 | validated |
| 0.163 | 0.175 | 7.34% | 6.83% | prefill | 1 | 11 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.162 | 0.146 | 7.68% | 6.24% | prefill | 1 | 17 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.162 | 0.168 | 7.67% | 6.54% | prefill | 1 | 9 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.160 | 0.158 | 7.59% | 6.39% | prefill | 1 | 13 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 10 | 5 | validated |
| 0.151 | 0.298 | 7.18% | 10.28% | prefill | 1 | 12 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.150 | 0.314 | 7.16% | 11.32% | prefill | 1 | 10 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.150 | 0.350 | 7.10% | 12.57% | prefill | 1 | 18 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.149 | 0.314 | 7.26% | 13.72% | prefill | 1 | 6 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.149 | 0.298 | 7.09% | 11.53% | prefill | 1 | 14 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.149 | 0.303 | 7.07% | 10.98% | prefill | 1 | 8 | attention_scores | QK scores, mask, softmax | aggregated(whole) | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, softmax | 12 | 5 | validated |
| 0.146 | 0.104 | 7.12% | 4.56% | prefill | 1 | 6 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.146 | 0.111 | 6.94% | 3.85% | prefill | 1 | 12 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.146 | 0.095 | 6.91% | 4.06% | prefill | 1 | 17 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.146 | 0.099 | 6.93% | 3.83% | prefill | 1 | 14 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.145 | 0.099 | 6.83% | 3.16% | prefill | 1 | 16 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.145 | 0.099 | 6.88% | 4.02% | prefill | 1 | 13 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.145 | 0.097 | 6.85% | 3.50% | prefill | 1 | 18 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.145 | 0.115 | 6.85% | 4.48% | prefill | 1 | 9 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.107 | 6.77% | 4.41% | prefill | 1 | 15 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.117 | 6.86% | 4.22% | prefill | 1 | 10 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.109 | 6.85% | 3.95% | prefill | 1 | 8 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.113 | 6.48% | 4.41% | prefill | 1 | 11 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.122 | 6.94% | 3.99% | prefill | 1 | 0 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |
| 0.144 | 0.127 | 6.48% | 4.44% | prefill | 1 | 7 | output_projection | Attention output projection and residual | aggregated(part01, part02) | elementwise_norm_activation, gemm_tensorcore | 5 | 2 | validated |


## FX Process Validation

- Unexpected kernel-family rows: 0
- No-kernel rows: 0
- Partially validated rows: 81
- unresolved rows: 0

| CUPTI kernel ms | NVTX CPU ms | phase | forward_id | layer | process_id | fragment_id | expected_kernel_families | matched_kernel_families | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.039 | 0.256 | prefill | 1 | 0 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.191 | 0.896 | prefill | 1 | 0 | attention_scores | whole | gemm_tensorcore, gemv_decode_cublas, softmax, copy_gather_cat, elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan, softmax | Decode cache K/V concatenation appears in this FX stage; value cache work is included here because it is launched by the same cache update block. |
| 0.039 | 0.196 | prefill | 1 | 0 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.204 | prefill | 1 | 5 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.039 | 0.184 | prefill | 1 | 5 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.207 | prefill | 1 | 6 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.041 | 0.187 | prefill | 1 | 6 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.206 | prefill | 1 | 7 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.180 | prefill | 1 | 7 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.184 | prefill | 1 | 8 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.041 | 0.199 | prefill | 1 | 8 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.196 | prefill | 1 | 9 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.196 | prefill | 1 | 9 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.201 | prefill | 1 | 10 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.041 | 0.183 | prefill | 1 | 10 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.201 | prefill | 1 | 11 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.175 | prefill | 1 | 11 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.187 | prefill | 1 | 12 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.406 | prefill | 1 | 12 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.188 | prefill | 1 | 13 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.182 | prefill | 1 | 13 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.183 | prefill | 1 | 14 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.042 | 0.165 | prefill | 1 | 14 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.190 | prefill | 1 | 15 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.165 | prefill | 1 | 15 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.182 | prefill | 1 | 16 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.041 | 0.185 | prefill | 1 | 16 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.039 | 0.168 | prefill | 1 | 17 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.040 | 0.172 | prefill | 1 | 17 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.038 | 0.182 | prefill | 1 | 18 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.042 | 0.177 | prefill | 1 | 18 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.017 | 0.197 | prefill | 1 | 19 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.017 | 0.169 | prefill | 1 | 19 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.334 | 0.273 | prefill | 1 | 19 | mlp | part01 | gemm_tensorcore, gemv_decode_cublas, elementwise_norm_activation | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | First fragment of MLP FX process. |
| 0.018 | 0.179 | prefill | 1 | 20 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.018 | 0.309 | prefill | 1 | 20 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.333 | 0.256 | prefill | 1 | 20 | mlp | part01 | gemm_tensorcore, gemv_decode_cublas, elementwise_norm_activation | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | First fragment of MLP FX process. |
| 0.018 | 0.185 | prefill | 1 | 21 | input_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch elementwise/norm kernels or fused norm kernels depending runtime. |
| 0.017 | 0.165 | prefill | 1 | 21 | post_attention_rmsnorm | whole | elementwise_norm_activation | copy_gather_cat, elementwise_norm_activation, selection_reduce_scan | CPU range should launch post-attention norm kernels or fused norm kernels. |
| 0.334 | 0.245 | prefill | 1 | 21 | mlp | part01 | gemm_tensorcore, gemv_decode_cublas, elementwise_norm_activation | elementwise_norm_activation, gemm_tensorcore, selection_reduce_scan | First fragment of MLP FX process. |


## Residual and Unresolved Work

Residual is parent layer launch-owned time minus the sum of instrumented process fragments in the same traced parent scope. Negative values can appear when asynchronous launch attribution or nested fragments double-count related launch work; inspect the fragment CSV before interpreting them as savings.

| parent_layer_range | forward_id | layer | parent_CUPTI kernel ms | process_CUPTI kernel ms | residual_CUPTI kernel ms | parent_NVTX CPU ms | process_NVTX CPU ms | residual_NVTX CPU ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visprune.layer21.prefill | 1 | 21 | 0.717 | 0.711 | 0.006 | 2.322 | 1.983 | 0.339 |
| visprune.layer28.prefill | 1 | 28 | 0.583 | 0.577 | 0.006 | 1.961 | 1.627 | 0.334 |
| visprune.layer20.prefill | 1 | 20 | 0.620 | 0.614 | 0.006 | 2.442 | 2.101 | 0.341 |
| visprune.layer27.prefill | 1 | 27 | 0.702 | 0.696 | 0.006 | 2.469 | 2.159 | 0.310 |
| visprune.layer22.prefill | 1 | 22 | 0.618 | 0.612 | 0.006 | 2.359 | 2.038 | 0.321 |
| visprune.layer26.prefill | 1 | 26 | 0.619 | 0.613 | 0.006 | 2.569 | 2.213 | 0.357 |
| visprune.layer23.prefill | 1 | 23 | 0.620 | 0.613 | 0.006 | 2.293 | 1.961 | 0.333 |
| visprune.layer25.prefill | 1 | 25 | 0.619 | 0.613 | 0.006 | 2.451 | 2.106 | 0.345 |
| visprune.layer19.prefill | 1 | 19 | 0.629 | 0.623 | 0.006 | 2.562 | 2.204 | 0.357 |
| visprune.layer24.prefill | 1 | 24 | 0.617 | 0.611 | 0.006 | 2.406 | 2.075 | 0.331 |
| visprune.layer07.prefill | 1 | 7 | 2.219 | 2.215 | 0.004 | 2.862 | 2.431 | 0.431 |
| visprune.layer18.prefill | 1 | 18 | 2.117 | 2.114 | 0.004 | 2.786 | 2.524 | 0.262 |
| visprune.layer13.prefill | 1 | 13 | 2.107 | 2.104 | 0.003 | 2.473 | 2.186 | 0.286 |
| visprune.layer14.prefill | 1 | 14 | 2.098 | 2.095 | 0.003 | 2.582 | 2.313 | 0.270 |
| visprune.layer17.prefill | 1 | 17 | 2.111 | 2.108 | 0.003 | 2.339 | 2.069 | 0.270 |
| visprune.layer08.prefill | 1 | 8 | 2.103 | 2.100 | 0.003 | 2.763 | 2.462 | 0.301 |
| visprune.layer10.prefill | 1 | 10 | 2.098 | 2.095 | 0.003 | 2.773 | 2.462 | 0.311 |
| visprune.layer16.prefill | 1 | 16 | 2.123 | 2.120 | 0.003 | 3.121 | 2.840 | 0.281 |
| visprune.layer09.prefill | 1 | 9 | 2.112 | 2.109 | 0.003 | 2.573 | 2.278 | 0.295 |
| visprune.layer11.prefill | 1 | 11 | 2.219 | 2.216 | 0.003 | 2.561 | 2.279 | 0.281 |
| visprune.layer12.prefill | 1 | 12 | 2.104 | 2.101 | 0.003 | 2.894 | 2.599 | 0.295 |
| visprune.layer15.prefill | 1 | 15 | 2.132 | 2.129 | 0.003 | 2.420 | 2.142 | 0.278 |
| visprune.layer19.decode | 32 | 19 | 0.520 | 0.519 | 0.001 | 1.627 | 1.444 | 0.183 |
| visprune.layer18.decode | 32 | 18 | 0.542 | 0.541 | 0.001 | 1.632 | 1.437 | 0.195 |
| visprune.layer27.decode | 32 | 27 | 0.516 | 0.515 | 0.001 | 1.577 | 1.383 | 0.194 |
| visprune.layer28.decode | 32 | 28 | 0.515 | 0.513 | 0.001 | 1.636 | 1.448 | 0.188 |
| visprune.layer31.decode | 32 | 31 | 0.521 | 0.520 | 0.001 | 1.638 | 1.450 | 0.188 |
| visprune.layer06.prefill | 1 | 6 | 2.050 | 2.049 | 0.001 | 2.285 | 2.052 | 0.233 |
| visprune.layer31.decode | 2 | 31 | 0.609 | 0.608 | 0.001 | 1.721 | 1.528 | 0.192 |
| visprune.layer18.decode | 2 | 18 | 0.550 | 0.548 | 0.001 | 1.649 | 1.450 | 0.199 |
| visprune.layer19.decode | 2 | 19 | 0.515 | 0.514 | 0.001 | 1.664 | 1.461 | 0.204 |
| visprune.layer27.decode | 2 | 27 | 0.506 | 0.505 | 0.001 | 1.582 | 1.401 | 0.181 |
| visprune.layer28.decode | 2 | 28 | 0.508 | 0.507 | 0.001 | 1.664 | 1.476 | 0.188 |
| visprune.layer05.prefill | 1 | 5 | 2.054 | 2.053 | 0.001 | 2.291 | 2.041 | 0.250 |
| visprune.layer00.prefill | 1 | 0 | 2.070 | 2.069 | 0.001 | 3.059 | 2.771 | 0.288 |


## Interpretation Notes

- `CUPTI kernel ms` is the sum of full GPU kernel durations launched by CUDA runtime calls that started inside the process NVTX CPU range.
- `NVTX CPU ms` is the CPU-side duration of the process range and can include Python/framework/runtime launch overhead.
- Percentages are computed against the containing layer range in the same metric scope, not against whole-request latency.
- `nvtx_gpu_proj_sum` can be used as a diagnostic for overlap, but this report uses launch ownership as the primary attribution relation.
- Cross-function processes such as `output_projection` and `mlp` are aggregated by `aggregation_key`; fragment rows remain in the CSV.
