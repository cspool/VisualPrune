# Autoresearch Notes

## Nsight Systems Range 与 Kernel Time 的含义

当前 E2 单请求实验使用 Nsight Systems 记录真实推理 timeline，但实现方式不是
“按 Python scope 精确归因每个 layer 的全部 GPU 工作”，而是：

1. 用 `nsys profile` 记录 CUDA / NVTX / cuBLAS / OS runtime 事件。
2. Python 侧给 `request`、`forward`、`layer`、`attn`、`mlp` 等代码段打 NVTX range。
3. 后处理脚本读取 nsys 导出的 SQLite。
4. 用 NVTX range 时间窗和 CUDA kernel 时间窗的重叠来粗略归因 kernel 时间。

因此，当前 `nsys range`、`nsys kernel` 字段必须按下面的边界理解。

### 当前 nsys 如何运行

入口脚本：

```bash
autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
```

核心命令：

```bash
nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --stats=true \
  ...
  profile_visprune_single_request.py \
    --sync-timing off \
    --nvtx on \
    --cuda-profiler-api \
    --layer-profile
```

含义：

- `cuda`: 记录 CUDA runtime/API 和 kernel activity。
- `nvtx`: 记录 Python 侧打出的 NVTX range。
- `cublas`: 记录 cuBLAS 调用。
- `osrt`: 记录部分 OS runtime 活动。
- `cudaProfilerApi`: 只采集 `torch.cuda.profiler.start()` 到
  `torch.cuda.profiler.stop()` 之间的 measured request，避开模型加载和 warmup。
- `--sync-timing off`: 不在每个 range 前后插入 CUDA 同步，尽量保留真实异步执行形态。

### Python/CPU 侧记录了什么

`profile_visprune_single_request.py` 中的 `LatencyRecorder.range()` 会在代码段前后调用：

```python
torch.cuda.nvtx.range_push(name)
...
torch.cuda.nvtx.range_pop()
```

`--layer-profile` 会 monkey patch 这些函数或模块：

- `model.forward`
- `layer.forward`
- `layer.self_attn.forward`
- `layer.mlp.forward`
- `value_aware_token_selection`

因此 nsys timeline 中会看到类似：

```text
visprune.request
visprune.forward_prefill
visprune.layer00.prefill
visprune.layer00.prefill.attn
visprune.layer00.prefill.mlp
```

这些 range 表示 CPU/Python 执行到对应代码段的时间窗。由于 CUDA 调用通常是异步的，
CPU 调用 PyTorch op 后只是 enqueue CUDA/cuBLAS/Triton/FlashAttention 工作，然后继续执行。
GPU 可能稍后才真正执行这些 kernel。

所以当前 `nsys range` 的严格含义是：

```text
CPU 进入这个 Python/module scope 到 CPU 离开这个 scope 的 NVTX 时间跨度。
```

它不是：

```text
GPU 完整执行这个 layer/attn/mlp 的耗时。
```

### GPU/CUDA 侧记录了什么

Nsight Systems 通过 CUPTI activity 记录 CUDA kernel 的 GPU timeline：

```text
kernel.start
kernel.end
kernel.name
```

单个 kernel 的 duration 通常是可信的。问题不在 kernel 本身，而在“这个 kernel
应该归到哪个 layer/range”。

不开同步时，常见执行关系是：

```text
CPU: enter layer00.mlp range
CPU: enqueue MLP GEMM
CPU: exit layer00.mlp range

GPU:                  MLP GEMM actually runs later
```

此时 MLP GEMM 的 kernel duration 是准确的，但它不一定和 `layer00.mlp` 的
NVTX 时间窗重叠。

### 后处理如何计算 nsys kernel time

后处理脚本：

```bash
autoresearch/experiments/e2_single_request_latency/code/analyze_layer_nsys.py
```

它读取：

- `NVTX_EVENTS`: 每个 range 的 `start/end/text`
- `CUPTI_ACTIVITY_KIND_KERNEL`: 每个 kernel 的 `start/end/name`

然后对每个 range 做时间窗重叠归因：

```text
kernel_total_ms(range)
  = sum(overlap(kernel.start/end, range.start/end))
```

也就是说，当前 `kernel_total_ms` 表示：

```text
这个 NVTX 时间窗内重叠到的 GPU kernel 执行时间总和。
```

它不等于：

```text
这个 layer/attn/mlp launch 出来的所有 kernel 总时间。
```

### 为什么会出现反直觉结果

例如 same-input layer0 prefill 中曾观察到：

```text
dense-fa2 layer0 total kernel > eager-vis layer0 total kernel
```

直觉上这不合理，因为 layer0 输入相同，FlashAttention2 是优化 kernel，不应该比
eager `QK^T / softmax / AV` 的实际 GPU 工作更慢。

这个反直觉结果来自归因方式，而不是说明 FA2 更慢：

- dense-FA2 的 `layer00.prefill` NVTX 时间窗可能刚好覆盖了更多实际执行中的 GEMM/FA kernel。
- eager-vis 的一些 kernel 可能在 CPU range 结束后才执行，没有被 overlap 算进该 range。
- component 级别更容易暴露这个问题，例如同形状 layer0 MLP 不应出现极大的 kernel
  时间差。如果出现，优先怀疑 range overlap 归因，而不是认为 MLP 实际计算量变化了。

因此：

- 单个 CUDA kernel duration: 可信。
- 按 kernel name / kernel family 跨整段 measured request 聚合: 较可信。
- 当前按 NVTX CPU 时间窗 overlap 得到的 per-layer/per-component `kernel_total_ms`: 只能作粗粒度参考。
- 不能用当前 `layer total kernel_total_ms` 证明 “FA2 layer kernel 比 eager layer kernel 更慢”。

### 当前 nsys range 是否合理

当前 `nsys range` 是合理的，但含义有限：

```text
它是低侵入的 CPU/NVTX timeline 标记，保留真实异步执行形态。
```

它适合用来：

- 标记 request / forward / layer / attn / mlp 在 CPU 侧发生的位置。
- 查看真实异步执行下的 timeline 结构。
- 粗略定位热点范围。
- 与正式端到端 nsys/clock 实验保持较低侵入性。

它不适合直接解释为：

```text
这个 layer 在 GPU 上完整执行了多久。
```

也不适合直接用：

```text
range_ms - kernel_total_ms
```

解释为严格的 kernel launch / 同步开销。因为 `kernel_total_ms` 不是该 range
发起的所有 kernel，而只是与该 CPU/NVTX 时间窗重叠的 kernel。

### 多轮相同输入的作用

相同输入多轮 nsys 有价值，可以降低随机波动：

- GPU clock / boost 状态波动
- OS / driver 调度波动
- allocator/cache 状态差异
- kernel launch 间隙的小幅波动
- Nsight 采集扰动

但多轮只能降低随机噪声，不能修正系统性归因偏差。如果归因逻辑是
“时间窗重叠即属于该 range”，多轮运行会让这个偏差更稳定，而不是让它变成严格正确。

### 如何做更严格的诊断

如果目标是验证 “相同输入 layer0 下 FA2 attention core 应该快于 eager
`QK^T / softmax / AV`”，不要直接使用当前 `layer total kernel_total_ms`。

更合理的诊断方式：

1. 继续使用无同步 nsys/clock 作为正式端到端性能证据。
2. 单独跑同步诊断版 profile，只用于归因，不用于端到端 latency 结论。
3. 在 attention 内部拆更细 NVTX：
   - q/k/v projection
   - RoPE
   - FA2 core 或 eager QK
   - softmax
   - AV
   - o_proj
4. 诊断版 range 应先同步再 push，结束前同步再 pop：

```python
torch.cuda.synchronize()
torch.cuda.nvtx.range_push(name)
...
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
```

5. 或者绕开 layer range 归因，按 kernel name / kernel family 在 measured request
   内聚合：
   - dense FA2: FlashAttention core kernel
   - eager attention: QK GEMM + softmax + AV GEMM
6. 如需更严格的 layer 归因，应使用 CUDA launch correlation / Nsight GPU projection，
   而不是仅用 NVTX CPU 时间窗 overlap。

### 推荐结论口径

正式报告中应区分：

- **端到端性能**: 使用无同步 clock/nsys，保留真实执行形态。
- **GPU kernel 实际耗时**: 使用 Nsight kernel duration 或按 kernel name 聚合。
- **layer/component 归因**: 当前 overlap 方法只作粗粒度参考；精确归因需要同步诊断或更严格的 correlation 方法。

因此，当前项目中的 nsys layer report 应避免把 `kernel_total_ms` 解释成
“该 layer 发起的全部 GPU kernel 时间”。更准确的说法是：

```text
kernel_total_ms 是与该 NVTX CPU 时间窗重叠的 CUDA kernel 时间总和。
```
