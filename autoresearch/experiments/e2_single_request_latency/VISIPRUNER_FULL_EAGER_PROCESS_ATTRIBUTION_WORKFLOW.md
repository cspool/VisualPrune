# VisiPruner Full Eager Process Attribution Workflow Index

本文是 `e2_single_request_latency` 下 VisiPruner full eager process attribution 的索引。具体执行被拆成 4 个 workflow：

```text
workflows/01_layer_wise_end_to_end_trace.md
workflows/02_representative_fx_process_wise_trace.md
workflows/03_process_gpu_hardware_trace.md
workflows/04_full_layer_fx_process_wise_estimate.md
```

## Common Rules

- 所有 workflow 默认使用 `GPU=1`。当前机器上 GPU 1 负载更轻，性能干扰更小；除非重新审查 GPU 负载，否则不要改 GPU。
- 所有性能数据必须来自同一输入、权重、backend、生成参数、warmup 和 cache policy。
- 时间归因使用 launch ownership：`NVTX CPU range -> CUDA runtime correlationId -> CUPTI kernel`。
- `NVTX CPU ms`、`CUPTI kernel ms`、clock/wall time、NCU 硬件计数器是不同证据，不能混合为同一个 latency。
- `output_bk/` 是历史参考备份，已知有缺漏；不能直接当作完整证据继续补写报告。

## Expected Output Directories

```text
output/visipruner_full_eager_layer_wise/
output/visipruner_full_eager_process_wise/
output/visipruner_full_eager_process_wise_ncu/
output/visipruner_full_eager_full_layer_process_attribution/
```

## Workflow Order

1. `01_layer_wise_end_to_end_trace.md`
   - skill: `$visipruner-same-input-layer-wise-workflow`
   - purpose: 生成全量 input-layer layer-wise Nsight/CUPTI 分母。

2. `02_representative_fx_process_wise_trace.md`
   - skill: `$visipruner-fx-process-nvtx-instrumentation` + `$visipruner-process-performance-breakdown`
   - purpose: 生成代表 layer 的 strict FX process-wise trace、process GPU launch order、process timing。

3. `03_process_gpu_hardware_trace.md`
   - skill reference: `$visipruner-process-performance-breakdown` for process/kernel selection and timing boundaries; NCU 计划、执行和报告由该 workflow 自行完成。
   - purpose: 对代表 layer/process 补充 GPU 硬件瓶颈证据，例如 SM/Tensor Core/DRAM/L2/occupancy/stall。

4. `04_full_layer_fx_process_wise_estimate.md`
   - skill: `$visipruner-segmented-process-attribution`
   - purpose: 用代表 process template 估计全量 input-layer process-wise attribution，并保持每层 metric 守恒。

## Evidence Boundary

Nsight Systems / CUPTI 是 strict timing/timeline 证据：

```text
process NVTX range
CUDA runtime launch
CUPTI kernel start/end
correlationId
stream
kernel family
```

Nsight Compute / NCU 是硬件瓶颈诊断证据：

```text
SM utilization
Tensor Core utilization
DRAM/L2 bandwidth
occupancy
warp stall reasons
register/shared memory pressure
```

不要用 NCU timing 替代 Nsight Systems/CUPTI 的 process latency。NCU 解释为什么慢；Nsight Systems/CUPTI 负责 process 花了多少 launch-owned GPU kernel 时间，以及按什么顺序执行。
