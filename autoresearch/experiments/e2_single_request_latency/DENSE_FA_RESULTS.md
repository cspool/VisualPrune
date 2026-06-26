# Dense Eager Single-Request Results

Date: 2026-06-20

## Scope

This is the no-pruning dense baseline for native LLaVA/VisPrune single-request
inference. No throughput metric is reported.

In this document, `dense-fa` is the historical dense full-attention eager native
LLaVA path:

```text
use_visipruner: false
use_flash_attn: false
pruning_config: null
```

This is not a FlashAttention2 run. FlashAttention2 is now exposed to
`/workspace/VisPrune/venv_profiling` through targeted symlinks from
`/home/descfly/.host-home/.local/lib/python3.12/site-packages`; the actual FA2
baseline is recorded separately in `DENSE_FA2_RESULTS.md`.

## Environment

```text
python: /workspace/VisPrune/venv_profiling/bin/python
torch: 2.12.0+cu132
cuda: 13.2
transformers: 4.37.2
nsys: 2026.3.1
gpu: NVIDIA GeForce RTX 4090
CUDA_VISIBLE_DEVICES: 1
HF_HOME: /workspace/VisPrune/models
HUGGINGFACE_HUB_CACHE: /workspace/VisPrune/models/hub
HF_HUB_OFFLINE: 1
TRANSFORMERS_OFFLINE: 1
flash_attn: 2.8.3.post1 available through venv symlinks for separate FA2 runs
```

The conda `cu132` environment has CUDA/PyTorch but is missing the LLaVA package
dependencies. The profiling environment is therefore `venv_profiling`.

## Request

```text
model: liuhaotian/llava-v1.5-7b
image: /workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg
prompt: Describe the image briefly.
prompt tokens before image expansion: 49
prefill sequence length after image expansion: 624
max_new_tokens: 32
warmup: 1 request, outside the measured request
decode forward calls observed: 31
output_token_count: 33
```

The JSON result confirms no pruning path was used:

```text
use_visipruner=false
use_flash_attn=false
pruning_config=null
value_aware_token_selection_ms=0.0
selected_visual_token_counts=[]
deep_exit_checks=[]
```

## Clock Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_dense_fa_single_request.sh
```

Output:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa_32tok.json
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa_32tok_ranges.csv
```

Synchronized latency:

```text
stage                         ms        pct_request
request_total                 1131.38   100.00
non_generate                  34.93     3.09
generate_total                1096.46   96.91
prepare_multimodal            13.01     1.15
vision_encode_project         11.64     1.03
forward_prefill               74.78     6.61
forward_decode_sum            993.28    87.79
generate_other                15.39     1.36
value_aware_token_selection   0.00      0.00
```

## Nsight Systems Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_nsys_dense_fa_single_request.sh
```

Outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_cuda_gpu_kern_sum.csv
```

NVTX host timeline:

```text
range                           instances  ms
:visprune.request               1          1606.29
:visprune.generate_total        1          1562.70
:visprune.forward_decode        31         1430.26
:visprune.forward_prefill       1          83.85
:visprune.image_preprocess_cpu  1          30.28
:visprune.prepare_multimodal    32         23.72
:visprune.vision_encode_project 1          21.62
:visprune.load_image            1          8.93
:visprune.build_prompt_tokenize 1          1.78
:visprune.input_h2d             1          1.16
```

NVTX GPU projection:

```text
range                           gpu_ms   gpu_ops
:visprune.request               1563.76  52193
:visprune.generate_total        1562.36  52188
:visprune.forward_decode        1434.50  49104
:visprune.forward_prefill       85.52    1721
:visprune.prepare_multimodal    22.94    680
:visprune.vision_encode_project 21.36    635
:visprune.input_h2d             0.87     4
```

CUDA kernel summary:

```text
total kernel duration: 625.11 ms
top kernel family 1: gemvx, 305.55 ms, 3007 calls, 48.9%
top kernel family 2: gemvx, 157.90 ms, 3968 calls, 25.3%
top GEMM: ampere_fp16_s1688gemm_fp16_64x128..., 42.93 ms, 192 calls, 6.9%
cat/copy kernels: 17.94 ms, 1920 calls, 2.9%
128x64 GEMM: 11.79 ms, 32 calls, 1.9%
```

## Conclusion

The dense no-pruning single request is decode dominated. In the synchronized
clock run, decode forward accounts for 993.28 ms of the 1131.38 ms request
latency. Nsight Systems shows the same ordering, with decode forward dominating
both NVTX host time and GPU projected time. The value-aware VisPrune selection
range is absent, confirming this run does not use pruning.
