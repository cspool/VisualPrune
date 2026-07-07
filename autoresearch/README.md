# Autoresearch Notes

## Nsight Systems Range 与 Kernel Time 的含义

当前 E2 单请求实验使用 Nsight Systems 记录真实推理 timeline。现在的 layer/kernel
归因不再使用 “CUDA kernel 时间窗和 NVTX range 时间窗 overlap”，而是：

1. 用 `nsys profile` 记录 CUDA / NVTX / cuBLAS / OS runtime 事件。
2. Python 侧给 `request`、`forward`、`layer`、`attn`、`mlp` 等代码段打 NVTX range。
3. 后处理脚本读取 nsys 导出的 SQLite。
4. 先找 `start` 落在 NVTX range 内的 CUDA Runtime API 调用。
5. 再用 `CUPTI_ACTIVITY_KIND_RUNTIME.correlationId` 匹配
   `CUPTI_ACTIVITY_KIND_KERNEL.correlationId`，把这些 kernel 归到该 range。

因此，当前报告里的 `NVTX CPU range` 和 `CUPTI launch-owned kernel` 字段必须按下面的边界理解。

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

所以当前 `NVTX CPU range` 的严格含义是：

```text
CPU 进入这个 Python/module scope 到 CPU 离开这个 scope 的 NVTX 时间跨度。
```

它不是：

```text
GPU 完整执行这个 layer/attn/mlp 的耗时。
```

### GPU/CUDA 侧记录了什么

Nsight Systems 通过 CUPTI activity 记录 CUDA Runtime API 和 CUDA kernel：

```text
runtime.start
runtime.end
runtime.name
runtime.correlationId

kernel.start
kernel.end
kernel.name
kernel.correlationId
```

单个 kernel 的 duration 通常是可信的。归因问题不在 kernel duration 本身，而在
“这个 kernel 应该归到哪个 layer/range”。

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
- `CUPTI_ACTIVITY_KIND_RUNTIME`: 每个 CUDA Runtime API 调用的
  `start/end/name/correlationId`
- `CUPTI_ACTIVITY_KIND_KERNEL`: 每个 kernel 的 `start/end/name/correlationId`

然后对每个 range 做 launch ownership 归因：

```text
owned_runtime_api(range)
  = runtime where range.start <= runtime.start < range.end

owned_cupti_kernel(range)
  = CUPTI kernel where kernel.correlationId == owned_runtime_api.correlationId

kernel_total_ms(range)
  = sum(kernel.end - kernel.start for owned_cupti_kernel(range))
```

也就是说，当前报告里的 `CUPTI launch-owned kernel sum ms` / `kernel_total_ms` 表示：

```text
由这个 NVTX CPU range 内的 CUDA Runtime API 调用发起的 CUPTI GPU kernel 完整 duration 总和。
```

它不等于：

```text
这个 layer 在 GPU 上占用 wall-clock 的时间。
```

原因有三点：

- kernel 可以在 CPU range 结束后才开始或结束；launch ownership 仍会把它完整计入该 range。
- 多个 CUPTI kernel 如果并发执行，`kernel_total_ms` 是 duration 求和，不是 GPU 时间轴 span。
- `layer`、`attn`、`mlp` 是嵌套 range；同一个 kernel 可以同时属于父 range 和子 range，
  这些表项不能相加。

### 为什么仍然要区分端到端 latency 和 breakdown

端到端 latency 应优先看 clock JSON 里的外层 request/generate/forward 计时，而不是把
per-layer `kernel_total_ms` 相加。当前 `profile_visprune_single_request.py` 中，
`visprune.request` 包住完整 request，并且 `run_request()` 在返回前执行一次
`torch.cuda.synchronize()`；因此 `request_total_ms` 更接近用户可感知的单请求端到端时间。

`generate_total_ms`、`forward_prefill_ms`、`forward_decode_sum_ms` 等 breakdown 则是
CPU/NVTX scope 或 Python 计时分段。它们用于解释时间花在什么阶段，但不开
`--sync-timing on` 时，不应解释为每个阶段的完整 GPU completion latency。

### 当前 NVTX CPU range 是否合理

当前 `NVTX CPU range` 是合理的，但含义有限：

```text
它是低侵入的 CPU/NVTX timeline 标记，保留真实异步执行形态。
```

它适合用来：

- 标记 request / forward / layer / attn / mlp 在 CPU 侧发生的位置。
- 查看真实异步执行下的 timeline 结构。
- 按 CUDA Runtime correlationId 找到 range 内 launch 出来的 GPU kernel。
- 与正式端到端 nsys/clock 实验保持较低侵入性。

它不适合直接解释为：

```text
这个 layer 在 GPU 上完整执行了多久。
```

也不适合直接用：

```text
range_ms - kernel_total_ms
```

解释为严格的 kernel launch / 同步 / CPU-only overhead。因为 `range_ms` 是 CPU scope，
`kernel_total_ms` 是 CUPTI launch-owned kernel duration 求和，两者不是同一个物理量。

### 多轮相同输入的作用

相同输入多轮 nsys 有价值，可以降低随机波动：

- GPU clock / boost 状态波动
- OS / driver 调度波动
- allocator/cache 状态差异
- kernel launch 间隙的小幅波动
- Nsight 采集扰动

多轮不能消除所有系统性偏差。例如如果一个框架后台线程在同一时间窗提交 CUDA
工作，而分析脚本只按全局 timestamp 判断 range containment，就仍然可能需要线程或
stack 约束来进一步收窄归因。不过相比 kernel-vs-range overlap，当前 correlationId
方法已经避免了“只统计落在 CPU range 内的 kernel 片段”这个主要偏差。

### 如何做更严格的诊断

如果目标是验证 “相同输入 layer0 下 FA2 attention core 应该快于 eager
`QK^T / softmax / AV`”，不要只看 `layer total kernel_total_ms`。

更合理的诊断方式：

1. 继续使用无同步 nsys/clock 作为正式端到端性能证据。
2. layer/component breakdown 使用 Runtime correlationId -> kernel correlationId。
3. 在 attention 内部拆更细 NVTX：
   - q/k/v projection
   - RoPE
   - FA2 core 或 eager QK
   - softmax
   - AV
   - o_proj
4. 必要时单独跑同步诊断版 profile，只用于归因，不用于端到端 latency 结论：

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
6. 如果要分析多个 kernel 的并发关系，使用 Nsight Systems 的 GPU timeline/span、
   stream 和 kernel start/end；Nsight Compute 更适合单个 kernel 的 counters 和
   source-level 诊断，不适合直接复原原始多 kernel 并发 timeline。

### 推荐结论口径

正式报告中应区分：

- **端到端性能**: 使用 measured request 的 clock/nsys，保留真实执行形态；优先看
  `request_total_ms`。
- **GPU kernel 实际耗时**: 使用 Nsight kernel duration 或按 kernel name/family 聚合。
- **layer/component 归因**: 使用 Runtime correlationId -> kernel correlationId 的
  launch ownership；不要再用 kernel-vs-NVTX overlap。

因此，当前项目中的 nsys layer report 应把 `kernel_total_ms` 解释成：

```text
kernel_total_ms 是该 NVTX CPU range 内 CUDA Runtime API 调用发起的 CUPTI GPU kernel 完整 duration 总和。
```

## E2 单层推理过程 UML 时序图

下面用 `e2_single_request_latency` 中的一次真实 Nsight 记录建模一个 layer 的
推理过程。选择的对象是：

```text
run tag        nsys_e2_visipruner_full_eager_32tok
config         visipruner-full
backend        eager VisiPruner, use_flash_attn=False
layer event    event_id=0, visprune.layer00.prefill
shape          q_len=624, kv_len=624
workload       eager_visipruner_prefill_shallow_layer0_mass_fold
operator path  eager QK^T/softmax/AV attention; shallow post-softmax edits;
               o_proj GEMM; MLP GEMMs
```

证据来自：

- `autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_eager_32tok.json`
- `autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_eager_32tok_layer_events.csv`
- `autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_eager_32tok_layer_kernel_breakdown.csv`
- `autoresearch/experiments/e2_single_request_latency/output/nsys_e2_visipruner_full_eager_32tok.sqlite`
- `autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py`

该 layer 的关键观测值：

| 观测项 | 值 |
|---|---:|
| `visprune.layer00.prefill` NVTX CPU range | 2.930888 ms |
| `visprune.layer00.prefill.attn` NVTX CPU range | 2.079071 ms |
| `visprune.layer00.prefill.mlp` NVTX CPU range | 0.249649 ms |
| NVTX CPU range 内 CUDA Runtime API 调用数 | 53 |
| NVTX CPU range 内 CUPTI launch-owned kernel 数 | 53 |
| CUPTI launch-owned `kernel_total_ms` | 2.077156 ms |
| dominant kernel family | `gemm_tensorcore` |
| `gemm_tensorcore_ms` | 1.695774 ms |
| `elementwise_norm_activation_ms` | 0.237252 ms |
| `copy_gather_cat_ms` | 0.077825 ms |
| `softmax_ms` | 0.050721 ms |
| `selection_reduce_scan_ms` | 0.015584 ms |
| attention component CUPTI launch-owned `kernel_total_ms` | 0.827311 ms |
| MLP component CUPTI launch-owned `kernel_total_ms` | 1.153428 ms |

### 采样与归因机制

这个图说明当前 nsys 采样方式如何把 CPU range、CUDA Runtime、GPU kernel 和
OSRT 事件放到同一条 timeline 上。注意这里的归因规则是 “runtime API start
落在 NVTX range 内，并且 runtime/kernel correlationId 相同”，不是 kernel
执行时间窗和 NVTX range 的 overlap。

<pre>
Diagram: E2_NSIGHT_SAMPLING_AND_LAUNCH_OWNERSHIP
Time/order axis: left -> right.  The diagram is not PlantUML syntax.
Formula:
  owned_runtime(range) = runtime.start inside NVTX_RANGE
  owned_cupti_kernel(range) = kernel.correlationId == runtime.correlationId
  kernel_total_ms           = sum full duration(owned_cupti_kernel)

participant / event source          observed request order
                                     0                                                                  export
                                     ▲                                                                    ▲
CPU_PROCESS / Python+PyTorch    ──▶  |==================== PYTORCH_LAYER_FORWARD_LOOP ====================|
                                     meaning: 真实 Python 进程；执行 eager LLaVA/VisiPrune，打 NVTX range，并提交 CUDA 工作。
                                     action: push NVTX; run eager ATen ops; call CUDA runtime; request-level sync
                                                    │ runtime calls                     │ OS calls
                                                    ▼                                   ▼
NVTX_EVENTS                       ──▶        |==================== NVTX_LAYER_RANGE ====================|
                                             meaning: Nsight NVTX 事件表；记录 range_push/range_pop 的 start/end/text。
                                             action: start/end timestamps define where runtime API starts are accepted.
                                             ▲                                                   ▲
                                             │ range_push                                        │ range_pop

CUPTI_RUNTIME                     ──▶             ┌──CALL c1──┐ ┌──CALL c2──┐       ┌──CALL c3──┐
                                                  meaning: CUPTI Runtime 表；记录 CPU 发出的 launch/copy/sync/alloc API 调用。
                                                  action: if CALL.start is inside NVTX range, keep CALL.correlationId.
                                                  │launch    │ │memcpy    │  ...  │launch    │
                                                  └──────────┘ └──────────┘       └──────────┘
                                                     │            │                │
                                                     │corr=c1     │corr=c2         │corr=c3
                                                     ▼            ▼                ▼
CUDA_DRIVER_STREAM_QUEUE          ──▶               enqueue on stream 7 ─────────────────────────▶
                                                  meaning: CUDA driver/stream 排队视角；提交的工作在 stream 7 上等待 GPU 调度。
                                                               │
                                                               ▼
CUPTI_KERNELS                     ──▶                  ┌─KERNEL c1─┐ ┌─KERNEL c2─┐ ┌─KERNEL c3─┐
                                                       meaning: CUPTI kernel activity；记录真实 GPU kernel 的 start/end/name/correlationId。
                                                       action: full kernel duration is counted when correlationId matches an owned runtime call.
                                                       │ GEMM      │ │ softmax   │ │ tail GEMM │
                                                       └───────────┘ └───────────┘ └───────────┘
                                                                                     ▲
                                                                                     │ tail can finish after NVTX range

OSRT_API                          ──▶                         ┌─IOCTL/POLL/THREAD_EVENT─┐
                                                              meaning: OS runtime 采样；展示 driver ioctl、poll 和线程事件。
                                                              └─────────────────────────┘

NSIGHT_CUPTI_COLLECTOR            ──▶  records NVTX_EVENTS + CUPTI_RUNTIME + CUPTI_KERNEL + OSRT_API independently
                                       meaning: nsys 采集器；采集 cuda,nvtx,cublas,osrt 并导出 SQLite 证据表。
                                                                  │
                                                                  ▼
POSTPROCESS_ANALYZER              ──▶  |==================== SQLITE_RANGE_JOIN_AND_AGGREGATE ====================|
                                       meaning: analyze_layer_nsys.py；用 Runtime correlationId -> Kernel correlationId 计算 launch ownership。
                                       action: read SQLite; keep runtime.start inside range; aggregate matched full kernel durations by family.
                                                                  │
                                                                  ▼
OUTPUT_FILES                       ──▶  layer_events.csv + layer_kernel_breakdown.csv/json
                                       meaning: 后处理输出；本节观测值和 CUPTI launch-owned summary 的来源。

Representative elements:
RUNTIME_CALL = cudaLaunchKernel / cudaMemcpyAsync / cudaStreamSynchronize
KERNEL       = gemm_tensorcore / softmax / copy_gather_cat / selection_reduce_scan
OSRT_EVENT   = ioctl / poll / pthread_cond_signal
</pre>

### layer00.prefill 推理过程

这个图把上面的采样机制落到一个具体 layer。时间均以
`visprune.layer00.prefill` 的 NVTX start 为 `+0.000 ms`。

<pre>
Diagram: E2_LAYER00_PREFILL_SEQUENCE
Run: nsys_e2_visipruner_full_eager_32tok
Time axis: milliseconds from NVTX start of visprune.layer00.prefill.  Horizontal scale is compressed.

time(ms)              0.000        0.300        0.600        0.900        1.200        1.500        1.800        2.100        2.400        2.700        2.930  3.106
                      ▲            ▲            ▲            ▲            ▲            ▲            ▲            ▲            ▲            ▲            ▲      ▲

CPU_LAYER_RANGE       0.000000..2.930888  ──▶  |============================ LAYER00_PREFILL_CPU_RANGE ============================|
                                                meaning: layer.forward 的 CPU/NVTX scope；不是该 layer 的完整 GPU 执行耗时。
                                                note: q_len=624, kv_len=624, eager_visipruner_prefill_shallow_layer0_mass_fold

CPU_ATTN_RANGE        0.309738..2.388809  ──▶       |======================= ATTN_CPU_RANGE =======================|
                                                    meaning: layer.self_attn.forward 的 CPU/NVTX scope；GPU kernel 可能滞后执行。
                                                    note: eager QK^T, softmax, AV, shallow post-softmax edit, o_proj enqueue

CPU_MLP_RANGE         2.649489..2.899138  ──▶                                                                 |====== MLP_CPU_RANGE ======|
                                                                                                             meaning: layer.mlp.forward 的 CPU/NVTX scope。
                                                                                                             note: MLP GEMM enqueue; full GPU GEMM interval can extend past the scope.

RUNTIME_LAUNCHES      selected launch intervals ──▶      [73181] [73200] [73219]        [73452]    [73495]  [73563][73580][73599] [73623]     [73676]       [73805]
                                                         meaning: CUPTI Runtime API 调用；CPU 请求 CUDA launch/copy/sync，不代表 kernel 已完成。
                                                         0.404   0.476   0.522          1.455      1.811    1.989  2.043  2.095   2.173       2.355         2.721
                                                         │       │       │              │          │        │      │      │       │           │             │
                                                         ▼       ▼       ▼              ▼          ▼        ▼      ▼      ▼       ▼           ▼             ▼
GPU_STREAM7_KERNELS   kernel intervals      ──▶  |EARLY_NORM_COPY_REDUCE 0.048263..0.278859|
                                                 meaning: stream 7 上的真实 GPU kernel 区间；单个 duration 是 kernel 级证据。
                                                 |QKV_GEMM_CLUSTER       0.413837..0.816084|
                                                 |ROPE_ELEMWISE_COPY     0.932662..1.720708|
                                                 |SOFTMAX                1.822054..1.872775|
                                                 |SHALLOW_EDIT           1.873415..2.206669|
                                                 |O_PROJ_GEMM            2.363696..2.497490|
                                                 |POST_ATTN_NORM         2.498066..2.620532|
                                                 |MLP_GEMM_WITH_TAIL     2.730966..3.106333| ──▶ tail after layer end
LAUNCH_OWNERSHIP_RULE counted kernels       ──▶  runtime.start in 0.000000..2.930888 and matched kernel.correlationId are counted in full
                                                meaning: 后处理使用的归因规则；tail kernel 也完整计入，因为它由 range 内 runtime call 发起。

OSRT_API              OS events             ──▶                                             |IOCTL_BURST 1.277173..1.788821, ioctl x9|
                                             meaning: Nsight OS runtime 事件；解释 driver/system 活动，不是 VisiPrune 算法步骤。
                                             |POLL_BACKGROUND -3.327193..96.799379, crosses layer time window|

NSIGHT_CUPTI          collected records     ──▶  NVTX range + 53 owned runtime API calls + 53 owned GPU kernels
                                                meaning: 选中 layer 时间窗内的 nsys/CUPTI 采集结果。

POSTPROCESS_OUTPUT    CUPTI launch-owned summary ──▶  range_ms=2.930888; kernel_total_ms=2.077156; dominant_family=gemm_tensorcore
                                                meaning: analyze_layer_nsys.py 生成的 layer_events 与 layer_kernel_breakdown 摘要。


Kernel-family CUPTI launch-owned contribution from LAYER00_PREFILL_CPU_RANGE
Contribution axis: 0.000000 ms                                                                  2.077156 ms
                   ▲                                                                                ▲
filled bar width is proportional to kernel_total_ms=2.077156 ms
gemm_tensorcore             1.695774 ms 81.6% ──▶ |=================================================           |
elementwise_norm_activation 0.237252 ms 11.4% ──▶ |=======                                                     |
copy_gather_cat             0.077825 ms  3.7% ──▶ |==                                                          |
softmax                     0.050721 ms  2.4% ──▶ |=                                                           |
selection_reduce_scan       0.015584 ms  0.8% ──▶ |                                                            |

Representative correlation map:
73181  runtime 0.404209..0.415187  -> GPU GEMM 0.413837..0.547376
73495  runtime 1.811016..1.823188  -> GPU SOFTMAX 1.822054..1.872775
73676  runtime 2.355081..2.363461  -> GPU O_PROJ_GEMM 2.363696..2.497490
73805  runtime 2.720738..2.730981  -> GPU MLP_GEMM 2.730966..3.106333, counted in full by correlationId ownership
</pre>

### 读图边界

这个单层图可以说明：

- CPU 进入/退出 layer、attn、mlp Python/module scope 的时间窗；
- CUDA Runtime/Driver 如何在这些 CPU scope 内 enqueue kernel、memcpy、sync 和 allocator 工作；
- GPU kernel 的真实执行窗口可能滞后于 CPU launch；
- OSRT 中可以看到 driver `ioctl`、后台 `poll` 和少量线程事件；
- 当前 `kernel_total_ms` 如何通过 Runtime correlationId -> Kernel correlationId 计算 launch ownership。

这个单层图不能说明：

- `layer00.prefill` 的 CUPTI launch-owned kernel duration 之和就是 GPU wall-clock layer latency；
- `range_ms - kernel_total_ms` 是严格的 launch overhead 或 CPU-only overhead；
- 嵌套的 layer、attn、mlp breakdown 可以直接相加；
- 没有线程/stack 约束时，时间窗内所有 runtime call 都一定来自同一条 Python 调用栈。
