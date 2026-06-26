# E2 Results: Single-Request VisPrune Latency

Date: 2026-06-20

## Scope

This run profiles one native VisPrune/LLaVA request with two methods:

- clock timing: synchronized wall-clock breakdown from Python ranges
- Nsight Systems: NVTX/CUDA timeline and kernel breakdown

No throughput metric is reported.

## Request

```text
model: liuhaotian/llava-v1.5-7b
image: /workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg
prompt: Describe the image briefly.
prompt tokens before image expansion: 49
prefill sequence length after image expansion: 624
max_new_tokens: 32
decode forward calls observed: 31
output_token_count: 33
```

VisPrune selected 10 visual tokens at layer 18 in this request. Deep-exit
checks were observed on layers 19-27.

## Clock Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_visprune_single_request.sh
```

Output:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_visprune_full_32tok.json
```

Synchronized latency:

```text
stage                         ms        pct_request
request_total                 1456.85   100.00
non_generate                  53.37     3.66
generate_total                1403.48   96.34
prepare_multimodal            17.68     1.21
vision_encode_project         15.71     1.08
forward_prefill               98.52     6.76
forward_decode_sum            1265.12   86.84
generate_other                22.16     1.52
value_aware_token_selection   8.97      0.62
```

Key observation: single-request latency is dominated by autoregressive decode,
not image preprocessing, prefill, or VisPrune token selection. Decode forward
accounts for 1265.12 ms of the 1456.85 ms request.

## Nsight Systems Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_nsys_visprune_single_request.sh
```

Outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok_stats_cuda_gpu_kern_sum.csv
```

NVTX host timeline:

```text
range                                 instances  ms
:visprune.request                     1          1758.22
:visprune.generate_total              1          1720.61
:visprune.forward_decode              31         1579.80
:visprune.forward_prefill             1          94.45
:visprune.image_preprocess_cpu        1          27.09
:visprune.prepare_multimodal          32         19.42
:visprune.vision_encode_project       1          17.73
:visprune.value_aware_token_selection 21         9.00
:visprune.load_image                  1          6.91
:visprune.build_prompt_tokenize       1          1.01
:visprune.input_h2d                   1          0.80
```

NVTX GPU projection:

```text
range                                 gpu_ms   gpu_ops
:visprune.request                     1721.38  60170
:visprune.generate_total              1720.31  60165
:visprune.forward_decode              1583.78  56048
:visprune.forward_prefill             94.64    2594
:visprune.prepare_multimodal          18.63    680
:visprune.vision_encode_project       17.44    635
:visprune.value_aware_token_selection 7.71     407
:visprune.input_h2d                   0.58     4
```

CUDA kernel summary:

```text
total kernel duration: 596.87 ms
top kernel family 1: gemvx, 304.64 ms, 3007 calls, 51.0%
top kernel family 2: gemvx, 157.22 ms, 4163 calls, 26.3%
top GEMM: ampere_fp16_s1688gemm_fp16_64x128..., 25.30 ms, 114 calls, 4.2%
cat/copy kernels: 12.51 ms, 1920 calls, 2.1%
elementwise multiply: 8.39 ms, 4182 calls, 1.4%
```

Nsight absolute wall time is higher than the clock run because profiling adds
overhead. The ordering is consistent: decode dominates, prefill is second, and
VisPrune value-aware token selection is small.

## Conclusion

For this single request, the latency bottleneck is the many small decode-step
forward passes. VisPrune's value-aware selection is visible in both clock and
Nsight, but it is not the main latency contributor. The largest kernel-time
contributors are GEMV-style decode kernels, which is consistent with
batch-size-1 autoregressive generation.
