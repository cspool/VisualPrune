# Triton VP-FA Layer Performance Status

Date: 2026-07-02 UTC

## Objective

Run single-request end-to-end Triton VP-FA inference performance tests:

- clock timing;
- Nsight Systems CUDA/NVTX profiling;
- per-layer workload/operator/latency report in the style of
  `workload_analysis/human_draft.md`.

The human draft file is a read-only reference and must not be modified.

## Current State

Completed on 2026-07-02 UTC after GPU/NVML became available again.

```text
nvidia-smi -> NVIDIA-SMI 595.71.05, CUDA 13.2, 2x NVIDIA GeForce RTX 4090
torch -> 2.12.0+cu132
torch.cuda.is_available() -> True
torch.cuda.device_count() -> 2
torch.cuda.get_device_name(0) -> NVIDIA GeForce RTX 4090
nsys --version -> NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0
```

The clock, Nsight Systems, layer-kernel attribution, and Markdown report
pipeline has now run successfully for `visipruner-full-vp-fa` with
`max_new_tokens=32`.

## Commands Run

```bash
autoresearch/experiments/e2_single_request_latency/code/run_clock_triton_vpfa_layer_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/run_nsys_triton_vpfa_layer_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/generate_triton_vpfa_layer_report.sh
```

## Prepared Pipeline

Clock run:

```bash
autoresearch/experiments/e2_single_request_latency/code/run_clock_triton_vpfa_layer_single_request.sh
```

Nsight run:

```bash
autoresearch/experiments/e2_single_request_latency/code/run_nsys_triton_vpfa_layer_single_request.sh
```

Report generation:

```bash
autoresearch/experiments/e2_single_request_latency/code/generate_triton_vpfa_layer_report.sh
```

Expected final report:

```text
autoresearch/experiments/e2_single_request_latency/TRITON_VPFA_LAYER_PERFORMANCE_REPORT.md
```

## Expected Artifacts

Clock:

```text
output/clock_triton_vpfa_layer_32tok.json
output/clock_triton_vpfa_layer_32tok_ranges.csv
output/clock_triton_vpfa_layer_32tok_layer_events.csv
```

Nsight:

```text
output/nsys_triton_vpfa_layer_32tok.nsys-rep
output/nsys_triton_vpfa_layer_32tok.sqlite
output/nsys_triton_vpfa_layer_32tok_stats_cuda_gpu_kern_sum.csv
output/nsys_triton_vpfa_layer_32tok_stats_nvtx_sum.csv
output/nsys_triton_vpfa_layer_32tok_layer_events.csv
output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.json
output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.csv
```

## Validation Already Done

- `profile_visprune_single_request.py` compiles after adding
  `--layer-profile`.
- `analyze_layer_nsys.py` compiles and can parse an existing Nsight SQLite
  without crashing.
- `generate_triton_vpfa_layer_report.py` compiles and can emit a Markdown report
  template from available inputs.
- `git diff --check` passes for the new profiling/report scripts.
- Clock run produced 1,024 layer events and populated timing ranges.
- Nsight run produced 3,072 layer NVTX ranges and 54,598 captured CUDA kernels.
- `generate_triton_vpfa_layer_report.sh` produced a populated Markdown report
  with measured clock and Nsight values.

## Key Measured Results

Clock run:

```text
request_total_ms = 1859.271
generate_total_ms = 1813.852
forward_prefill_ms = 112.241
forward_decode_sum_ms = 1658.759
value_aware_token_selection_ms = 8.801
```

Nsight/layer attribution:

```text
nsys layer ranges = 3072
CUDA kernels captured = 54598
prefill attention kernel total = 16.010 ms
prefill attention Triton VP-FA kernels = 2.277 ms
decode attention kernel total = 179.318 ms
decode attention GEMV/cuBLAS kernels = 137.028 ms
```

Workload reading from the generated report:

```text
prefill layers 0-18: q_len=624, kv_len=624
prefill layers 19-27: q_len=58, kv_len=58
prefill layers 28-31: q_len=48, kv_len=48
decode layers 0-18: kv_len 625 -> 655
decode layers 19-27: kv_len 59 -> 89
decode layers 28-31: kv_len 49 -> 79
```

The run confirms that the Triton VP-FA prefill path is measurable and that this
single-request configuration is still dominated by repeated decode work, mainly
small GEMV/cuBLAS kernels.

## Completion Criteria

Completion audit:

1. GPU/NVML is available: passed.
2. The clock script produces the three expected clock artifacts: passed.
3. The Nsight script produces `.nsys-rep`, `.sqlite`, stats CSVs, and
   per-layer kernel breakdown files: passed.
4. `TRITON_VPFA_LAYER_PERFORMANCE_REPORT.md` contains measured clock and Nsight data,
   not placeholders: passed.
5. `workload_analysis/human_draft.md` remains unmodified: passed; `git diff --
   workload_analysis/human_draft.md` is empty.

## Final Artifacts

Clock:

```text
autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok.json
autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_ranges.csv
autoresearch/experiments/e2_single_request_latency/output/clock_triton_vpfa_layer_32tok_layer_events.csv
```

Nsight:

```text
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok.nsys-rep
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok.sqlite
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_stats_cuda_gpu_kern_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_stats_nvtx_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_stats_nvtx_gpu_proj_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_stats_nvtx_kern_sum.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_layer_events.csv
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.json
autoresearch/experiments/e2_single_request_latency/output/nsys_triton_vpfa_layer_32tok_layer_kernel_breakdown.csv
```

Report:

```text
autoresearch/experiments/e2_single_request_latency/TRITON_VPFA_LAYER_PERFORMANCE_REPORT.md
```

## Next Direction

Use `autoresearch/experiments/e3_single_request_kernel_acceleration/` to attack
the measured decode bottleneck. Prefill cleanup alone does not make the
single-request path fast because 31 decode forwards dominate the end-to-end
latency.
