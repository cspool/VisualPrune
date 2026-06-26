# Filtered Dispatch Profile: VisiPrune Full 32 Tokens

本目录是一次 **VisiPrune-centric filtered TorchDispatch profile** 的输出。
它不是全量 profiler：运行时只在 manifest 选中的 35 个 `(forward_id,
layer_id)` 事件进入 `TorchDispatchMode`，没有把整个 `model.generate()` 包在
dispatch/profiler 里。

## 运行入口

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py \
  --gpu 1 \
  --tag filtered_dispatch_visipruner_full_32tok
```

过滤规则来自：

`/workspace/VisiPrune/workload_analysis/DISPATCH_FILTER_RULES.md`

源 algorithmic trace：

`/workspace/VisiPrune/workload_analysis/algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/algorithmic_trace.json`

## 结果摘要

- source layer events: `1024`
- dispatch target events: `35`
- observed forwards: `32`
- observed layer events: `1024`
- captured ATen dispatch ops: `3163`
- output text matches source trace: `true`

这里的 `observed_layer_events=1024` 只是普通 Python wrapper 对 layer 调用的
观测校验，不是 1024 层全量 dispatch profile。真正进入 dispatch 的只有
`dispatch_manifest.csv` 中的 35 个事件。

## 文件说明

- `dispatch_manifest.csv`
  过滤后的采集清单。每行是一个允许进入 dispatch 的 `(input_id, layer_id)`
  事件，包含 `priority`、`visipruner_role`、`q_len/kv_len`、保留原因和重复组。

- `dispatch_ops.csv`
  真正捕获到的 ATen dispatch trace。每行是目标 layer 内的一次 ATen op 调用，
  包含 `op_name`、`op_schema`、输入/输出 tensor 的 shape、dtype、device。

- `dispatch_op_summary.csv`
  对 `dispatch_ops.csv` 按 `event_id + op_schema` 聚合后的计数表。

- `observed_layer_events.csv`
  运行时观测到的全部 layer forward 事件，用于校验 forward/layer 编号是否与
  源 trace 对齐。这个文件不是 dispatch op trace。

- `run_metadata.json`
  本次运行的配置、输出路径、目标事件数、捕获 op 数和输出文本一致性检查。

## 如何解读

优先从 `dispatch_manifest.csv` 看哪些 VisiPrune 事件被采集：

- P0: middle selection / deep exit 决策层
- P1: token schedule 生效边界层
- P2: shallow mask 代表层
- P3: decode 阶段剪枝效果代表层

然后用 `dispatch_ops.csv` 对应的 `event_id` 查看这些层内部真实发生的 ATen
算子和输入 shape。例如：

- `input1_layer18`: full visual 状态下实际选择 10 个 visual tokens 的层。
- `input1_layer19`: `624 -> 58` 后第一个真实执行层。
- `input1_layer28`: `58 -> 48` 后第一个真实执行层。
- `input2_layer19` / `input32_layer19`: decode 首尾 step 中 middle-pruned
  cache regime 的代表。

注意：`dispatch_ops.csv` 是 PyTorch dispatch 级别的 op trace，不是 CUDA kernel
trace，也不是硬件 latency profile。它用于回答“目标 VisiPrune layer 内真实
调用了哪些 ATen 算子、输入输出 shape 是什么”。
