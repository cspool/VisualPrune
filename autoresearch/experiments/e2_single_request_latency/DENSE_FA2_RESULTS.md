# Dense FlashAttention2 Single-Request Results

Date: 2026-06-20

## Scope

This is the no-pruning dense baseline for native LLaVA/VisPrune single-request
inference with FlashAttention2 enabled. No throughput metric is reported.

```text
config: dense-fa2
use_visipruner: false
use_flash_attn: true
pruning_config: null
```

## Correct FA Configuration

FlashAttention is installed under the host-home user site, not under the normal
Python user site:

```text
/home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn
/home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
```

`/home/descfly/.host-home/.local/bin` is in `PATH`, but `PATH` only affects
executable lookup. It does not make Python packages importable. The profiling
venv has therefore been configured with symlinks to only the FlashAttention
files:

```text
/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn
  -> /home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn
/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn-2.8.3.post1.dist-info
  -> /home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn-2.8.3.post1.dist-info
/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
  -> /home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
```

This keeps `PYTHONPATH` unset and avoids shadowing unrelated packages from the
host-home site.

Verified import:

```text
python: /workspace/VisPrune/venv_profiling/bin/python
torch: 2.12.0+cu132
cuda: 13.2
flash_attn: 2.8.3.post1
flash_attn path: /workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn/__init__.py
PYTHONPATH: unset
```

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
use_flash_attn=true
pruning_config=null
value_aware_token_selection_ms=0.0
selected_visual_token_counts=[]
deep_exit_checks=[]
```

## Clock Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_dense_fa2_single_request.sh
```

Output:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa2_32tok.json
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa2_32tok_ranges.csv
```

Synchronized latency:

```text
stage                         ms        pct_request
request_total                 1140.97   100.00
non_generate                  40.35     3.54
generate_total                1100.62   96.46
prepare_multimodal            13.06     1.14
vision_encode_project         11.64     1.02
forward_prefill               68.95     6.04
forward_decode_sum            1003.61   87.96
generate_other                15.00     1.31
value_aware_token_selection   0.00      0.00
```

Compared with the eager dense clock run, FA2 reduces prefill from 74.78 ms to
68.95 ms, while request total remains effectively unchanged for this batch-1
decode-heavy request.

## Nsight Systems Breakdown

Command:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_nsys_dense_fa2_single_request.sh
```

Outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_cuda_gpu_kern_sum.csv
```

NVTX host timeline:

```text
range                           instances  ms
:visprune.request               1          1546.85
:visprune.generate_total        1          1509.57
:visprune.forward_decode        31         1388.11
:visprune.forward_prefill       1          75.58
:visprune.image_preprocess_cpu  1          25.31
:visprune.prepare_multimodal    32         21.17
:visprune.vision_encode_project 1          19.26
:visprune.load_image            1          9.08
:visprune.build_prompt_tokenize 1          0.94
:visprune.input_h2d             1          0.78
```

NVTX GPU projection:

```text
range                           gpu_ms   gpu_ops
:visprune.request               1510.31  47960
:visprune.generate_total        1509.26  47955
:visprune.forward_decode        1392.75  45074
:visprune.forward_prefill       76.95    1518
:visprune.prepare_multimodal    20.47    680
:visprune.vision_encode_project 19.04    635
:visprune.input_h2d             0.53     4
```

CUDA kernel summary confirms FA2 kernels were used:

```text
flash_fwd_splitkv_kernel          11.48 ms   992 calls
flash_fwd_splitkv_combine_kernel  2.99 ms    992 calls
flash_fwd_kernel                  1.35 ms    32 calls
```

The dominant GPU time is still GEMV-style decode work:

```text
top GEMV family 1: 303.83 ms, 3007 calls, 49.5%
top GEMV family 2: 157.00 ms, 3968 calls, 25.6%
```

## Interpretation

The FA2 configuration is correctly active and visible in both the JSON
environment record and Nsight kernel names. For this specific single-request,
batch-size-1 workload, FA2 is not expected to materially reduce total latency:
decode dominates and most GPU time remains in GEMV-style projection kernels.
FA2 mainly improves the attention portion, which is a small fraction of this
request.
