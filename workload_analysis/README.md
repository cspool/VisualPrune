# VisiPrune Workload Analysis

## `torch.profiler` 与当前 `workload_analysis` 的区别

`torch.profiler` 和当前 `workload_analysis` 都可以观察一次真实运行，但它们的目标不同。

- `torch.profiler` 面向性能分析。
- `workload_analysis` 面向算法执行理解、动态 workload 建模和 layer process 证据重建。

因此，两者记录的“op flow”不能按同一种语义解释。

## `torch.profiler`: 性能事件 profiler

`torch.profiler` 的主要目标是回答：

```text
一次真实运行中，哪些 PyTorch op / CUDA kernel / CPU activity 消耗了时间和显存？
```

它关注的数据通常包括：

- PyTorch op 名称
- CPU time / CUDA time
- CUDA kernel timeline
- op 调用次数
- memory allocation
- tensor shape
- Python stack
- Chrome trace

启用示例：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model.generate(...)
```

`torch.profiler` 的 shape、时间和显存数据对性能分析重要，但它不天然理解
VisiPrune 的算法语义。例如，它不会自动提供：

- `forward_id`
- prefill / decode phase
- 每层 `q_len / kv_len / past_len`
- VisiPrune token selection 事件
- deep exit 事件
- layer 在 pruning schedule 中的角色
- token 数变化边界，例如 `624 -> 58 -> 48`

`torch.profiler` 也不是严格的数据依赖图工具。它可以显示真实运行中发生过哪些
op，并提供时间线和统计表，但它的核心目标不是把这些 op 重建成一个可审计的
layer tensor process。

`with_flops=True` 只能对部分算子，例如 matmul / conv，给出有限 FLOPs 估计。
它不会自动推导 VisiPrune 这种动态 token pruning 算法的理论复杂度。

因此，`torch.profiler` 适合做：

- 新算法早期的通用热点初筛
- CPU/CUDA 时间线观察
- shape / memory / stack 辅助定位
- 判断哪些区域值得进一步做 nsys/ncu 或 dispatch 取证

但它不适合作为当前项目的算法 trace 或 layer process 重建主证据。

## `workload_analysis`: 算法行为与证据重建

当前 `workload_analysis` 分为两个核心层次：

1. `algorithmic_trace`
2. filtered dispatch profile

这两层都基于真实 `generate()` 或真实 eager forward 运行，但它们不以执行时间为核心。

### `algorithmic_trace`

`algorithmic_trace` 的目标是回答：

```text
VisiPrune 在一次真实 generate() 中如何改变 token schedule 和理论 workload？
```

它通过 wrapper / hook 记录上层动态执行 flow，包括：

- `forward_id`
- prefill / decode phase
- 每层 `q_len`
- 每层 `kv_len`
- 每层 `past_len`
- hidden state shape
- VisiPrune selection 事件
- deep exit 事件
- 每层动态 token schedule
- 理论 FLOPs

重要输出包括：

```text
algorithmic_trace.json
layer_trace.csv
selection_trace.csv
operator_flops.csv
```

这里的重点不是 wall-clock latency，也不是 CUDA kernel timeline，而是：

```text
算法在真实请求中走了哪条动态路径，以及这条路径对应什么理论 workload。
```

因此，`algorithmic_trace` 是 VisiPrune 动态 schedule 的权威来源。

### Filtered Dispatch Profile

filtered dispatch profile 的目标是回答：

```text
在选中的 layer / forward 事件内，真实 eager 执行时发生了哪些 ATen op？
这些 op 的 tensor shape、数据依赖、alias、inplace 行为是什么？
```

它使用：

```text
torch.utils._python_dispatch.TorchDispatchMode
__torch_dispatch__
```

这不是编译时 trace，也不是 `torch.compile` / FX / export IR。它是运行时 eager
执行过程中观察到的 ATen dispatch op 流。

它记录的数据包括：

- selected `event_id`
- `forward_id`
- `layer_id`
- prefill / decode phase
- `q_len / kv_len / past_len`
- op schema
- input tensor ids
- output tensor ids
- tensor shape / dtype / device
- alias / storage / inplace mutation 信息
- sampled module stack
- op count summary

重要输出包括：

```text
dispatch_manifest.csv
dispatch_ops.csv
dispatch_op_summary.csv
observed_layer_events.csv
run_metadata.json
```

其中：

- `dispatch_manifest.csv` 说明为什么选这些 layer / forward 事件。
- `dispatch_ops.csv` 是选中 layer 内真实 ATen dispatch op 的主要证据。
- `observed_layer_events.csv` 用于校验全局 layer 事件编号，不代表全量 dispatch profile。

filtered dispatch profile 的重点不是执行时间，而是：

```text
用真实运行时 op、tensor id、shape、alias 和 inplace 证据，反推出 layer 的实际 tensor process。
```

这些数据后续可以被 `dispatch-layer-reconstruct-onnx` 等流程消费，用来生成更可读的
process 表达、small-shape Torch flow 或 ONNX stage。

## 关键区别

| 维度 | `torch.profiler` | `workload_analysis` |
| --- | --- | --- |
| 主要目标 | 性能分析 | 算法理解与执行证据重建 |
| 运行方式 | profile 真实运行的 op/kernel/activity | wrapper 记录算法 schedule，dispatch mode 记录选中 layer ATen op |
| 时间数据 | 核心数据 | 非核心数据 |
| shape 数据 | 性能辅助信息 | process 重建证据 |
| FLOPs | 有限 op 级估计 | 基于算法 schedule 的理论 FLOPs |
| layer 语义 | 不天然提供 | 显式记录 `forward_id/layer_id/phase/q_len/kv_len` |
| VisiPrune selection | 不天然提供 | 显式记录 |
| 数据依赖 | 不是主要目标 | 通过 tensor ids / input-output ids / alias 信息重建 |
| 归因目标 | 找热点 | 解释 selected layer 实际执行过程 |
| 适合作为性能结论吗 | 可辅助，但本项目正式性能仍用 nsys/ncu | 不适合，主要不是性能工具 |

## 推荐使用方式

当分析一个还没有明确关注过程的新算法时，可以先用 `torch.profiler` 做通用初筛：

1. 跑一次真实 `generate()`。
2. 查看 op / CUDA kernel / memory / shape 热点。
3. 判断哪些模块或阶段值得重点分析。
4. 不把 `torch.profiler` 的结果直接作为 VisiPrune 算法语义证据。

然后进入 `workload_analysis` 主线：

1. 用 `algorithmic_trace` 记录真实动态 schedule。
2. 从 `selection_trace.csv + layer_trace.csv` 中选择重要 layer / forward 事件。
3. 用 filtered dispatch profile 捕获这些事件内的真实 ATen op 与 tensor 依赖。
4. 如需进一步重建 layer process，再进入 dispatch reconstruction / ONNX / small-shape flow。

## 结论

更准确的表述是：

```text
torch.profiler 是性能 profiler：
它记录真实运行中的 op/kernel 时间线，重点是时间、次数、shape、memory 和热点。

workload_analysis 是算法执行与证据重建工具：
algorithmic_trace 记录上层动态 schedule；
filtered dispatch profile 记录选中 layer 的运行时 ATen op、shape、tensor ids、
alias 和 inplace 关系；
这些数据用于理解 VisiPrune 实际执行过程，而不是做 wall-clock 性能结论。
```

需要特别注意：

```text
filtered dispatch profile 不是编译时 trace。
它是运行时 eager dispatch trace。
```
