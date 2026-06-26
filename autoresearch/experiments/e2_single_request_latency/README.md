# E2 Single-Request LLaVA/VisPrune Latency

This experiment profiles native LLaVA/VisPrune single-request inference. It
does not measure throughput.

Start from `EXPERIMENT_IMPLEMENTATION_AND_RESULTS_REPORT.md` for the complete
implementation and results report, including environment, workload trace, code
trace, results, and visualization artifacts.

`EXPERIMENT_VALIDATION_AUDIT.md` records the implementation/profiling audit:
what the current experiment can support, what it cannot support, and which
caveats apply to the dense-FA2 vs native VisiPrune comparison.

## Environment

Use the local profiling environment:

```bash
/workspace/VisPrune/venv_profiling/bin/python
```

Verified stack:

```text
torch 2.12.0+cu132
cuda 13.2
transformers 4.37.2
nsys 2026.3.1
GPU NVIDIA GeForce RTX 4090
flash_attn 2.8.3.post1 linked into venv_profiling
```

The conda `cu132` environment has CUDA/PyTorch but is missing the LLaVA
dependencies, so this experiment uses `venv_profiling`.

FlashAttention is present under the host-home user site:

```text
/home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn
/home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
```

`PATH` contains `/home/descfly/.host-home/.local/bin`, but that does not make
Python packages importable. The clean configuration is to link only the FA files
into the profiling venv, avoiding full host `site-packages` shadowing:

```bash
SITE=/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages
HOST_SITE=/home/descfly/.host-home/.local/lib/python3.12/site-packages
ln -sfn "${HOST_SITE}/flash_attn" "${SITE}/flash_attn"
ln -sfn "${HOST_SITE}/flash_attn-2.8.3.post1.dist-info" "${SITE}/flash_attn-2.8.3.post1.dist-info"
ln -sfn "${HOST_SITE}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so" "${SITE}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
```

The FA2 scripts also accept an explicit `FLASH_ATTN_SITE_PACKAGES=...`
`PYTHONPATH` fallback, but it is not used by default because adding the whole
host site also exposes unrelated packages.

The run scripts set:

```text
HF_HOME=/workspace/VisPrune/models
HUGGINGFACE_HUB_CACHE=/workspace/VisPrune/models/hub
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Override the cache with `VISPRUNE_HF_HOME` or `VISPRUNE_HUB_CACHE` if needed.

## Dense Eager No-Pruning Run

`dense-fa` is kept as the historical dense full-attention eager baseline. It
does not enable VisPrune pruning and does not use FlashAttention2.

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_dense_fa_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/run_nsys_dense_fa_single_request.sh
```

Dense outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa_32tok.json
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa_32tok_stats_cuda_gpu_kern_sum.csv
```

See `DENSE_FA_RESULTS.md` for the recorded dense eager no-pruning results.

## Dense FlashAttention2 No-Pruning Run

`dense-fa2` enables `use_flash_attn=true`, which passes
`attn_implementation="flash_attention_2"` through the LLaVA loader. The JSON
output records the imported `flash_attn` version and path.

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_dense_fa2_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/run_nsys_dense_fa2_single_request.sh
```

FA2 outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_dense_fa2_32tok.json
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_dense_fa2_32tok_stats_cuda_gpu_kern_sum.csv
```

See `DENSE_FA2_RESULTS.md` for the recorded FA2 no-pruning results.

## GEMV Runtime Analysis

`GEMV_ACCELERATION_ANALYSIS.md` summarizes why the current single-request
VisiPrune path does not beat dense-FA2 and evaluates runtime designs for making
the GEMV-heavy decode path faster through batching or fusion.

Supporting profiling artifacts:

```text
autoresearch/experiments/e2_single_request_latency/code/benchmark_gemv_concurrency.py
autoresearch/experiments/e2_single_request_latency/code/analyze_decode_iterations.py
autoresearch/experiments/e2_single_request_latency/output/gemv_concurrency_bench.json
autoresearch/experiments/e2_single_request_latency/output/gemv_concurrency_bench.csv
autoresearch/experiments/e2_single_request_latency/output/kernel_family_summary.json
autoresearch/experiments/e2_single_request_latency/output/decode_iteration_kernel_breakdown.json
autoresearch/experiments/e2_single_request_latency/output/decode_iteration_kernel_breakdown.csv
```

## Visualization

`LATENCY_VISUALIZATION_TOOLS.md` summarizes the official/open-source
visualization options and the generated artifacts.

Generated local views:

```text
autoresearch/experiments/e2_single_request_latency/output/visualizations/single_request_latency_breakdown.png
autoresearch/experiments/e2_single_request_latency/output/visualizations/single_request_latency_dashboard.html
autoresearch/experiments/e2_single_request_latency/output/visualizations/single_request_latency_trace_chrome.json
```

Regenerate:

```bash
cd /workspace/VisPrune
/workspace/VisPrune/venv_profiling/bin/python \
  autoresearch/experiments/e2_single_request_latency/code/visualize_latency_breakdown.py
```

## Pruning Reference Workload

```text
config          visipruner-full
model           liuhaotian/llava-v1.5-7b
image           /workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg
prompt          Describe the image briefly.
max_new_tokens  32
warmup          1 request, outside the measured request
gpu             CUDA_VISIBLE_DEVICES=1
```

VisPrune config:

```json
{"mode":["shallow","middle","deep"],"shallow_mid_layer":6,"layer_threshold":0.995,"tokens_threshold":0.2}
```

## Pruning Reference Clock Run

Clock profiling uses explicit CUDA synchronization around each measured range.
This is intended for latency breakdown, not for throughput.

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_visprune_single_request.sh
```

Outputs:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_visprune_full_32tok.json
autoresearch/experiments/e2_single_request_latency/output/clock_visprune_full_32tok_ranges.csv
```

## Pruning Reference Nsight Systems Run

Nsight Systems profiling uses NVTX ranges and CUDA profiler API capture. The
script disables explicit synchronization timing to reduce profiler distortion.

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
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok_stats_nvtx_kern_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok_stats_cuda_gpu_kern_sum.csv
```

## Interpretation

Use the clock result for synchronized end-to-end latency. Use Nsight Systems to
inspect the CUDA/NVTX timeline and kernel distribution. Nsight adds overhead, so
its absolute wall time is expected to be higher than the clock run.
