# Single-Request Experiment Validation Audit

Date: 2026-06-22

## Verdict

当前代码和产物可以支撑一个明确范围内的结论:

- 支撑: 当前实现路径下，`dense-fa2` 单请求端到端延迟优于
  `visipruner-full`。VisiPrune 虽然降低了部分 prefill/GEMM kernel 时间，
  但没有把理论 token/attention 计算量下降转化为该单请求的端到端加速。
- 不支撑: 直接宣称 "同一 attention 后端下 VisiPrune 算法一定慢/快" 或
  "VisiPrune+FA2 的性能结论"。当前对比是 dense FlashAttention2 baseline
  对 native VisiPrune eager/custom attention 路径，不是后端相同的算法对比。

## Checked Artifacts

主要检查对象:

```text
code/profile_visprune_single_request.py
code/run_clock_dense_fa2_single_request.sh
code/run_clock_visprune_single_request.sh
code/run_nsys_dense_fa2_single_request.sh
code/run_nsys_visprune_single_request.sh
code/analyze_decode_iterations.py
code/visualize_latency_breakdown.py
repo/llava/model/builder.py
repo/llava/model/language_model/custom_modeling_llama.py
output/clock_dense_fa2_32tok.json
output/clock_visprune_full_32tok.json
output/nsys_dense_fa2_32tok.json
output/nsys_visprune_full_32tok.json
output/kernel_family_summary.json
output/decode_iteration_kernel_breakdown.json
```

## Requirement Matrix

| requirement | evidence | status |
|---|---|---|
| 单请求端到端推理 | `profile_visprune_single_request.py` 覆盖 prompt tokenize、image load/preprocess、H2D、`model.generate()`、prefill/decode forward | 支撑 |
| warmup 不混入被测请求 | warmup 使用 `warmup.*` NVTX/range 前缀，被测请求使用 `visprune.request`；Nsight capture 由 CUDA profiler API 包住被测请求 | 支撑 |
| clock 延迟可用于端到端比较 | clock scripts 使用 `--sync-timing on`，range 前后 `torch.cuda.synchronize()` | 支撑 |
| Nsight 可用于 kernel 归因 | nsys scripts 使用 `--trace=cuda,nvtx,cublas,osrt` 与 `--capture-range=cudaProfilerApi`，产出 `.nsys-rep/.sqlite/stats` | 支撑 |
| dense-FA2 后端确认 | `dense-fa2` 配置 `use_flash_attn=True`，loader 设置 `attn_implementation="flash_attention_2"`；Nsight 有 flash kernels | 支撑 |
| VisiPrune 路径确认 | `visipruner-full` 配置 `pruning_config`，tracker 记录 layer 18 选择 10 个视觉 token、layer 25/27 deep exit | 支撑 |
| 同后端算法对比 | 当前 `visipruner-full` 配置 `use_flash_attn=False`，dense baseline 是 FA2 | 不支撑 |
| 统计稳定性 | 当前每个配置是一条代表性单请求结果，没有重复运行分布 | 部分支撑 |

## Result Consistency

Clock 结果:

| method | request ms | generate ms | prefill ms | decode sum ms |
|---|---:|---:|---:|---:|
| dense-fa2 | 1140.97 | 1100.62 | 68.95 | 1003.61 |
| visipruner-full | 1456.85 | 1403.48 | 98.52 | 1265.12 |

`visipruner-full` 比 `dense-fa2` 慢 `315.88 ms`，即 `+27.7%`。

Nsight 结果:

| method | request NVTX ms | decode NVTX ms | CUDA kernel total ms |
|---|---:|---:|---:|
| dense-fa2 | 1546.85 | 1388.11 | 613.92 |
| visipruner-full | 1758.21 | 1579.53 | 596.87 |

Nsight 中 VisiPrune 的 CUDA kernel 总时长比 dense-fa2 少 `17.05 ms`
(`-2.8%`)，但 request/decode NVTX wall time 更高。这个现象与 clock 结果一致:
保存的 kernel 时间太小，且没有落在主导端到端延迟的 decode GEMV 链上。

## Why Theory Does Not Become Speedup Here

从 tracker 推导的 prefill sequence schedule:

| layer range | sequence length |
|---|---:|
| 0-18 | 624 |
| 19-27 | 58 |
| 28-31 | 48 |

这对应约 `37.0%` 的 linear token-layer row reduction 和 `40.3%` 的
quadratic attention-pair reduction。但当前请求的 measured bottleneck 是 31 次
autoregressive decode。每次 decode 只有一个 query token，仍要经过 32 层
batch-1 projection/GEMV。

Nsight kernel family summary:

| family | dense-fa2 ms | visipruner-full ms |
|---|---:|---:|
| GEMV decode cuBLAS | 460.84 | 467.34 |
| Tensor Core GEMM | 59.55 | 44.41 |
| FlashAttention | 15.82 | 0.00 |
| elementwise/norm/activation | 39.22 | 46.31 |
| copy/gather/cat | 32.48 | 27.91 |

可以看到 VisiPrune 确实降低了 Tensor Core GEMM 时间，但主导项
GEMV decode cuBLAS 没有下降。逐 decode iteration 归因也一致:

| metric | mean ms |
|---|---:|
| forward_decode NVTX range | 50.96 |
| CUDA kernel total inside range | 17.23 |
| GEMV kernels inside range | 14.86 |

GEMV 约占每次 decode kernel 时间的 `86.2%`。

## Implementation Caveats

1. 当前结论是 "dense-FA2 vs native VisiPrune eager/custom path" 的性能结论。
   如果实验设计目标是算法公平性，需要补充 `dense-eager vs visipruner-full`
   或 `dense-fa2 vs VisiPrune-FA2`。
2. 现有结果是单样本单请求，不包含 repeated runs、median、p10/p90 或置信区间。
   可以作为 case study 和瓶颈定位，不能单独作为统计稳定性结论。
3. `clock_visprune_full_32tok.json` 和 `nsys_visprune_full_32tok.json` 是较早
   schema 生成的产物，`use_visipruner/use_flash_attn` 字段为 `None`。实际
   pruning config、tracker、run script default 和 loader code 可以确认其路径；
   当前脚本已会写出这两个布尔字段。
4. `output_token_count=33` 与 `max_new_tokens=32` 不应作为精确 decode 步数解释。
   当前可靠的生成步数依据是 hook 记录的 `visprune.forward_decode` count = 31。
5. Kernel family 分类依赖 Nsight kernel name substring，例如 `gemvx::kernel`
   和 `flash::flash`。这足够支撑本机本版本 trace 的归因，但不是跨 CUDA/cuBLAS
   版本稳定 ABI。
6. Nsight wall time 带 profiler overhead，应该用于 timeline 和 kernel 归因；
   端到端延迟比较优先看 synchronized clock run。

## Recommendations

若要把实验升级为可支持更强结论的设计，建议补齐:

1. 同后端矩阵:
   `dense-eager` vs `visipruner-full`，以及 `dense-fa2` vs 一个明确的
   VisiPrune-FA2 配置。
2. 重复采样:
   每个配置至少 10 次，报告 median、mean、p10/p90，并固定 GPU clocks 或记录
   clocks/temperature。
3. 多并发服务场景:
   在 concurrency 4/8/16/32 下测 decode batching 或 serving runtime。当前
   microbenchmark 显示 row batching 才是改善 GEMV 利用率的关键。
4. 更细 NVTX:
   给 q/k/v/o projection、MLP gate/up/down 增加 per-layer/per-projection range，
   用来验证具体 decode GEMV 来源。

## Bottom Line

当前实验实现、采样工具和单请求端到端推理代码足以支撑现有报告的核心结论:
VisiPrune 在这个单请求 exact greedy decode workload 上没有达成理论 prefill
计算量下降对应的端到端性能提升。原因不是 token selection 本身耗时过高，而是
端到端延迟被 decode 阶段稳定重复的 batch-1 GEMV 主导。
