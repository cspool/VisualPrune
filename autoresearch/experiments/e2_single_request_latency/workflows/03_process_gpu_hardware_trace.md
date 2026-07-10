# 03 Process GPU Hardware Trace

目标：在 workflow 02 的 strict process timing/timeline 基础上，补充代表 layer/process 的 GPU 硬件瓶颈证据。这个 workflow 用于解释优化空间，不替代 process latency attribution。

## Skill Reference

```text
$visipruner-process-performance-breakdown
```

该 skill 只作为参考：它提供 process/kernel 选择、launch-owned timing
边界，以及 “不要用硬件诊断替代 process latency” 的证据约束。

本 workflow 的目标需要自行完成：根据 workflow 02 的 process-wise trace
选择代表 process/kernel，创建必要的 NCU plan/report 脚本，执行 Nsight
Compute，并产出硬件瓶颈报告。不要因为没有专用 NCU skill 就停止。

## Required GPU

```text
GPU=1
```

GPU 1 当前负载更轻，性能干扰更小。NCU replay 开销大，更需要固定低干扰 GPU。

## Expected Output Directory

```text
output/visipruner_full_eager_process_wise_ncu/
```

## Required Inputs

```text
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok.sqlite
output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv
output/visipruner_full_eager_process_wise/process_kernel_launch_order.csv
output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
```

## Possible Tools or Scripts

Existing tools:

```text
ncu
code/profile_visprune_single_request.py
code/run_nsys_layer_profile_single_request.sh
code/generate_process_performance_breakdown.py
```

Optional scripts to create if needed:

```text
code/generate_process_ncu_plan.py
code/generate_process_ncu_report.py
```

## Nsight Systems vs NCU Boundary

Nsight Systems / CUPTI owns timing and launch order:

```text
process NVTX range
runtime correlationId
CUPTI kernel start/end
stream
kernel family
```

NCU owns hardware diagnostics:

```text
SM utilization
Tensor Core utilization
DRAM/L2 bandwidth
occupancy
warp stall reasons
register/shared memory pressure
```

Do not use NCU timing as process latency. Use workflow 02 for latency and launch order.

## NCU Target Selection

Use workflow 02 to select a small representative set:

```text
high-CUPTI processes
high-NVTX CPU processes
kernel-family transitions
attention_scores / attention_output
qkv_projection / output_projection
mlp
visual_process when present
```

Prefer layer/process pairs with clear process NVTX and stable kernel names. Avoid profiling all 1024 layers with NCU.

## Command Template

Use NCU with the same model input and GPU 1. Verify local `ncu --help` syntax before running; NVTX include syntax can vary by NCU version.

```bash
cd /workspace/VisiPrune

mkdir -p /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu

CUDA_VISIBLE_DEVICES=1 \
ncu \
  --target-processes all \
  --set full \
  --nvtx \
  --nvtx-include "visprune.fx_process.layer00.prefill.*.qkv_projection.*" \
  --export /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu/ncu_layer00_prefill_qkv_projection \
  --force-overwrite \
  /workspace/VisiPrune/venv_profiling/bin/python \
  /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py \
    --config visipruner-full \
    --image-path /workspace/VisiPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg \
    --prompt "Describe the image briefly." \
    --output-dir /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu \
    --max-new-tokens 32 \
    --warmup-iters 1 \
    --gpu 1 \
    --sync-timing off \
    --nvtx on \
    --layer-profile \
    --fx-process-profile on \
    --tag ncu_sameinput_visipruner_full_eager_32tok_layer00_qkv
```

Export CSV metrics after NCU collection:

```bash
ncu --import /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu/ncu_layer00_prefill_qkv_projection.ncu-rep \
  --csv \
  > /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu/ncu_layer00_prefill_qkv_projection_metrics.csv
```

## Expected Files

```text
ncu_<layer>_<phase>_<process>.ncu-rep
ncu_<layer>_<phase>_<process>_metrics.csv
ncu_process_selection_plan.csv
SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_NCU_REPORT.md
```

## Report Requirements

The NCU report should list, per profiled representative process:

```text
layer / phase / forward_id
process_id / process_title
matched kernels from workflow 02
NCU kernel filter or NVTX include rule
SM utilization
Tensor Core utilization
DRAM throughput
L2 throughput
occupancy
dominant stall reasons
hardware bottleneck interpretation
whether timing evidence still comes from workflow 02
```

## Checks

```bash
test -d /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu
rg -n 'SM|Tensor|DRAM|L2|occupancy|stall|workflow 02' \
  /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise_ncu/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_NCU_REPORT.md
```

## Constraints

- NCU is expensive and may replay kernels; profile only representative process/kernel targets.
- NCU results explain hardware bottlenecks, not end-to-end latency.
- Keep NCU outputs separate from `output/visipruner_full_eager_process_wise/` strict timing outputs.
- If NCU changes execution enough to alter kernel choices or pruning behavior, mark the row as diagnostic-only.
