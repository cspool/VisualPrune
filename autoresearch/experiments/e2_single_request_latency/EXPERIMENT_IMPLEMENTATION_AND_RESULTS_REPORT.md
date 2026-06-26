# 单请求端到端推理实验实现与结果报告

日期: 2026-06-22

## 1. 实验目标

本实验基于当前 VisPrune/LLaVA 代码和已完成 profiling 结果，拆解单请求
端到端推理延迟，并给出可复现实验实现、负载 trace、性能采样 trace、结果
和可视化入口。

实验比较两条路径:

| 名称 | 配置 | 后端路径 |
|---|---|---|
| `dense-fa2` | no pruning, `use_flash_attn=true` | LLaVA dense baseline + FlashAttention2 |
| `visipruner-full` | shallow + middle + deep pruning | native VisiPruner attention, `use_flash_attn=false` |

注意: 代码中存在 `FA2VisiPrunerLlamaAttention`，但本次已记录的
`visipruner-full` 结果来自 native VisiPruner eager attention 路径。

## 2. 环境

### 2.1 软件与框架

| 项 | 值 |
|---|---|
| Python | `/workspace/VisPrune/venv_profiling/bin/python` |
| Framework | PyTorch / Transformers / local LLaVA-VisPrune repo |
| PyTorch | `2.12.0+cu132` |
| CUDA | `13.2` |
| Transformers | `4.37.2` |
| FlashAttention | `2.8.3.post1` |
| Nsight Systems | `2026.3.1` |
| Visualization | Matplotlib `3.11.0`, Plotly `6.8.0`, Perfetto/Chrome Trace JSON |
| GPU | NVIDIA GeForce RTX 4090 |

FlashAttention 位于 host-home user site，并通过 targeted symlink 暴露给
`venv_profiling`:

```text
/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn
  -> /home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn
/workspace/VisPrune/venv_profiling/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
  -> /home/descfly/.host-home/.local/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
```

### 2.2 缓存与离线设置

运行脚本统一设置 Hugging Face 本地缓存和离线模式:

```bash
export HF_HOME="${VISPRUNE_HF_HOME:-${ROOT_DIR}/models}"
export HUGGINGFACE_HUB_CACHE="${VISPRUNE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
```

来源:

- `code/run_clock_dense_fa2_single_request.sh`
- `code/run_clock_visprune_single_request.sh`
- `code/run_nsys_visprune_single_request.sh`

## 3. 负载 Trace

负载完全来自当前实验脚本 `profile_visprune_single_request.py`。

### 3.1 默认请求

源码摘录:

```python
DEFAULT_GPU = 1
DEFAULT_MODEL_PATH = "liuhaotian/llava-v1.5-7b"
DEFAULT_IMAGE_PATH = (
    "/workspace/VisPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg"
)
DEFAULT_PROMPT = "Describe the image briefly."
DEFAULT_OUTPUT_DIR = (
    "/workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output"
)
```

实际记录:

| 项 | 值 |
|---|---|
| model | `liuhaotian/llava-v1.5-7b` |
| image | `autoresearch/data/benchmark_images/002901d9d194c4fb.jpg` |
| image size | `1024 x 768` |
| prompt | `Describe the image briefly.` |
| prompt token count before image expansion | `49` |
| prefill sequence length after image expansion | `624` |
| max new tokens | `32` |
| observed output tokens | `33` |
| observed decode forward calls | `31` |
| warmup | `1` request outside measured request |

### 3.2 实验配置

源码摘录:

```python
VISIPRUNER_CONFIGS = {
    "dense-fa2": {
        "use_flash_attn": True,
        "use_visipruner": False,
        "pruning_config": None,
        "description": "Dense LLaVA reference with FlashAttention2; no VisPrune pruning.",
    },
    "visipruner-full": {
        "use_flash_attn": False,
        "use_visipruner": True,
        "pruning_config": {
            "mode": ["shallow", "middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "Native VisPrune full path: shallow + middle + deep.",
    },
}
```

### 3.3 负载执行入口

Clock run 使用同步 wall-clock + NVTX:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_clock_dense_fa2_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/run_clock_visprune_single_request.sh
```

Nsight run 使用 CUDA/NVTX/cuBLAS/osrt trace:

```bash
cd /workspace/VisPrune
autoresearch/experiments/e2_single_request_latency/code/run_nsys_dense_fa2_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/run_nsys_visprune_single_request.sh
```

`run_nsys_visprune_single_request.sh` 的关键采样参数:

```bash
"${NSYS_BIN}" profile \
  --trace=cuda,nvtx,cublas,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --stats=true \
  --output "${OUTPUT_DIR}/${TAG}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
    --config "${CONFIG:-visipruner-full}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --warmup-iters "${WARMUP_ITERS:-1}" \
    --gpu "${GPU:-1}" \
    --sync-timing off \
    --nvtx on \
    --cuda-profiler-api \
    --tag "${TAG}"
```

### 3.4 请求执行路径

请求由 `run_request()` 构造并调用 `model.generate()`:

```python
with recorder.range(f"{prefix}.build_prompt_tokenize"):
    prompt_text = build_prompt(prompt, conv_mode)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

with recorder.range(f"{prefix}.load_image"):
    image = Image.open(image_path).convert("RGB")
    image_size = image.size

with recorder.range(f"{prefix}.image_preprocess_cpu"):
    image_tensor = process_images([image], image_processor, model.config)

with recorder.range(f"{prefix}.input_h2d"):
    input_ids = input_ids.to(device="cuda", non_blocking=True)
    attention_mask = attention_mask.to(device="cuda", non_blocking=True)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

with recorder.range(f"{prefix}.generate_total"):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pruning_config=pruning_config,
            use_cache=True,
        )
```

## 4. 后端核心代码 Trace

### 4.1 Dense FA2 后端

`dense-fa2` 通过 `load_pretrained_model(..., use_flash_attn=True)` 进入
FlashAttention2 attention implementation:

```python
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base,
    model_name,
    device_map="cuda:0",
    use_flash_attn=config["use_flash_attn"],
    use_visipruner=config["use_visipruner"],
)
```

### 4.2 VisiPrune token selection 与 compaction

VisiPrune 后端在 attention 中执行 value-aware token selection，并在后续层
根据 selected visual tokens 压缩 hidden states。

FA2-capable VisiPruner attention 的代码结构如下，展示 QKV、pruning proxy、
FA2 attention 和 token selection 的关系:

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

need_pruning_proxy = (
    self.num_images > 0
    and q_len > 1
    and self.layer_idx > shallow_mid
    and ("middle" in pruning_mode or "deep" in pruning_mode)
)

if need_pruning_proxy:
    key_states_for_pruning = repeat_kv(key_states, self.num_key_value_groups)
    q_last = query_states[:, :, -1:, :]
    pruning_weights = torch.matmul(
        q_last, key_states_for_pruning.transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    pruning_weights = nn.functional.softmax(
        pruning_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)

attn_output = flash_attn_func(
    query_states_fa2,
    key_states_fa2,
    value_states_fa2,
    dropout_p=self.attention_dropout if self.training else 0.0,
    softmax_scale=1.0 / math.sqrt(self.head_dim),
    causal=is_causal,
    window_size=(-1, -1),
)

if need_pruning_proxy and pruning_weights is not None:
    important_vis_tokens = self.value_aware_token_selection(
        value_states_full, attn_output_for_selection, pw_full)
```

当前 `visipruner-full` 实测路径未启用 FA2，但 token compaction 逻辑相同:

```python
if important_vision_tokens != None:
    important_vision_hidden_states = hidden_states[:,important_vision_tokens,:]
    hidden_states = torch.cat(
        (hidden_states[:,:35,:],
         important_vision_hidden_states,
         hidden_states[:,self.last_image_token_index:,:]),
        dim=1
    )
    position_ids = torch.cat(
        (position_ids[:,:35],
         position_ids[:,important_vision_tokens],
         position_ids[:,self.last_image_token_index:]),
        dim=1
    )
    pruned_vis_len = hidden_states.shape[1]
    attention_mask = complete_attn_mask[:,:,:pruned_vis_len,:pruned_vis_len]
    self.last_image_token_index = 35 + important_vision_tokens.shape[0]
```

## 5. 性能采样代码 Trace

### 5.1 Clock + NVTX range recorder

`LatencyRecorder` 同时支持:

- clock run: range 前后 `torch.cuda.synchronize()`，得到同步 wall-clock;
- Nsight run: `torch.cuda.nvtx.range_push/pop()`，让 Nsight Systems 捕获 stage。

```python
class LatencyRecorder:
    def __init__(self, sync_cuda: bool, enable_nvtx: bool):
        self.sync_cuda = sync_cuda
        self.enable_nvtx = enable_nvtx and torch.cuda.is_available()
        self.ranges = {}

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def range(self, name: str):
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(name)
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.ranges.setdefault(name, RangeStats()).add(elapsed_ms)
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()
```

### 5.2 模型 monkey patch 采样点

脚本不改动模型数学计算，只包裹三个高层函数和 token selection:

```python
def wrapped_encode_images(images):
    with recorder.range(f"{prefix}.vision_encode_project"):
        return original_encode_images(images)

def wrapped_prepare(*args, **kwargs):
    with recorder.range(f"{prefix}.prepare_multimodal"):
        return original_prepare(*args, **kwargs)

def wrapped_forward(*args, **kwargs):
    if seq_len > 1:
        range_suffix = "forward_prefill"
        tracker["prefill_seq_lens"].append(seq_len)
    else:
        range_suffix = "forward_decode"
        tracker["decode_seq_lens"].append(seq_len)
    with recorder.range(f"{prefix}.{range_suffix}"):
        return original_forward(*args, **kwargs)

def wrapped_select(*args, **kwargs):
    with recorder.range(f"{prefix}.value_aware_token_selection"):
        result = orig(*args, **kwargs)
    if torch.is_tensor(result):
        tracker["value_select_tensor_layers"].append(idx)
        tracker["selected_visual_token_counts"].append(int(result.numel()))
    elif isinstance(result, bool):
        tracker["deep_exit_checks"].append({"layer": idx, "exit": bool(result)})
    return result
```

### 5.3 输出格式

每次运行输出:

- JSON: 完整环境、请求、tracker、range、derived breakdown;
- CSV: range 表格，包含 `total_ms`, `mean_ms`, `pct_request`,
  `pct_generate`;
- Nsight `.nsys-rep` 和 `.sqlite`;
- Nsight stats CSV: `nvtx_sum`, `nvtx_gpu_proj_sum`, `nvtx_kern_sum`,
  `cuda_gpu_kern_sum`。

输出逻辑:

```python
payload = {
    "config": args.config,
    "use_visipruner": config["use_visipruner"],
    "use_flash_attn": config["use_flash_attn"],
    "pruning_config": config["pruning_config"],
    "request": request,
    "tracker": tracker,
    "ranges": ranges,
    "derived_ms": derived_breakdown(ranges),
}
json.dump(payload, f, indent=2)
```

## 6. 结果

### 6.1 Clock end-to-end latency

| method | request ms | generate ms | prefill ms | decode sum ms | selection ms |
|---|---:|---:|---:|---:|---:|
| `dense-fa2` | 1140.97 | 1100.62 | 68.95 | 1003.61 | 0.00 |
| `visipruner-full` | 1456.85 | 1403.48 | 98.52 | 1265.12 | 8.97 |

结论:

- 单请求端到端延迟由 decode 主导。
- 当前 `visipruner-full` 比 `dense-fa2` 慢约 `27.7%`。
- VisiPrune token selection 本身只有 `8.97 ms`，不是最大瓶颈；更大的问题是
  decode 阶段仍然是 batch-1 小 GEMV 链。

### 6.2 Nsight timeline and kernel results

| method | request NVTX ms | decode NVTX ms | CUDA kernel total ms |
|---|---:|---:|---:|
| `dense-fa2` | 1546.85 | 1388.11 | 613.92 |
| `visipruner-full` | 1758.21 | 1579.53 | 596.87 |

Kernel family breakdown:

| family | dense-fa2 ms | visipruner-full ms |
|---|---:|---:|
| GEMV decode cuBLAS | 460.84 | 467.34 |
| Tensor Core GEMM | 59.55 | 44.41 |
| FlashAttention | 15.82 | 0.00 |
| elementwise/norm/activation | 39.22 | 46.31 |
| copy/gather/cat | 32.48 | 27.91 |
| selection/reduce/scan | 5.44 | 5.80 |
| softmax | 0.32 | 3.34 |
| other | 0.24 | 1.76 |

结论:

- VisiPrune 降低了 Tensor Core GEMM 时间，符合 prefill token 减少预期。
- 但主导项 GEMV decode cuBLAS 没有下降，反而从 `460.84 ms` 到
  `467.34 ms`。
- 因此 CUDA kernel 总时长只小幅下降，而端到端 NVTX/clock 延迟没有获益。

### 6.3 VisiPrune tracker

| tracker item | value |
|---|---|
| prefill seq lens | `[624]` |
| decode calls | `31` |
| selected layer | `18` |
| selected visual tokens | `10` |
| deep exit checks | layers `19-27` |
| deep exits true | layers `25`, `27` |

推导的 prefill sequence schedule:

| layer range | sequence length |
|---|---:|
| `0-18` | `624` |
| `19-27` | `58` |
| `28-31` | `48` |

### 6.4 Decode iteration breakdown

逐 decode iteration 归因来自 `decode_iteration_kernel_breakdown.json`，按 CUDA
kernel 与 `visprune.forward_decode` NVTX range 的重叠时间统计:

| metric | mean ms | min ms | max ms | stdev ms |
|---|---:|---:|---:|---:|
| forward_decode NVTX range | 50.96 | 49.44 | 56.94 | 1.40 |
| CUDA kernel total inside range | 17.23 | 17.10 | 17.62 | 0.10 |
| GEMV kernels | 14.86 | 14.69 | 15.04 | 0.09 |

每个 decode iteration 的 GEMV 占 kernel 时间约 `86.2%`。这说明不是某个
异常 token 拖慢，而是每个生成 token 都重复稳定的小 GEMV 计算链。

## 7. 可视化

可视化生成脚本:

```bash
cd /workspace/VisPrune
/workspace/VisPrune/venv_profiling/bin/python \
  autoresearch/experiments/e2_single_request_latency/code/visualize_latency_breakdown.py
```

### 7.1 本地报告图

```text
output/visualizations/single_request_latency_breakdown.png
```

包含三块:

1. clock request latency breakdown;
2. Nsight CUDA kernel family duration;
3. VisiPrune decode iteration breakdown.

### 7.2 交互式 HTML

```text
output/visualizations/single_request_latency_dashboard.html
```

这是 self-contained Plotly HTML，可直接用浏览器打开。

### 7.3 Perfetto / Chrome Trace

```text
output/visualizations/single_request_latency_trace_chrome.json
```

该 trace 包含:

- `dense-fa2` process lane;
- `visipruner-full` process lane;
- NVTX range track;
- CUDA kernel family track;
- 原始 kernel name 写入 event args。

可用工具:

- Perfetto UI: https://ui.perfetto.dev/
- Chrome trace viewer: `chrome://tracing`
- Nsight Systems GUI: `/opt/nvidia/nsight-systems/2026.3.1/bin/nsys-ui`

### 7.4 可视化代码 trace

`visualize_latency_breakdown.py` 的数据入口:

```python
RUNS = {
    "dense-fa2": {
        "clock": OUTPUT_DIR / "clock_dense_fa2_32tok.json",
        "nsys": OUTPUT_DIR / "nsys_dense_fa2_32tok.json",
        "sqlite": OUTPUT_DIR / "nsys_dense_fa2_32tok.sqlite",
        "kernel_key": "dense_fa2",
    },
    "visipruner-full": {
        "clock": OUTPUT_DIR / "clock_visprune_full_32tok.json",
        "nsys": OUTPUT_DIR / "nsys_visprune_full_32tok.json",
        "sqlite": OUTPUT_DIR / "nsys_visprune_full_32tok.sqlite",
        "kernel_key": "visipruner_full",
    },
}
```

生成图表时使用的 stage 和 kernel family:

```python
STAGE_COMPONENTS = [
    ("non_generate_ms", "non-generate"),
    ("vision_encode_project_ms", "vision encode/project"),
    ("prepare_without_vision_ms", "prepare multimodal"),
    ("forward_prefill_ms", "prefill forward"),
    ("forward_decode_sum_ms", "decode forward"),
    ("generate_other_ms", "generate other"),
]

KERNEL_FAMILIES = [
    ("gemv_decode_cublas", "GEMV decode cuBLAS"),
    ("gemm_tensorcore", "Tensor Core GEMM"),
    ("flash_attention", "FlashAttention"),
    ("elementwise_norm_activation", "elementwise/norm/activation"),
    ("copy_gather_cat", "copy/gather/cat"),
    ("selection_reduce_scan", "selection/reduce/scan"),
    ("softmax", "softmax"),
    ("other", "other"),
]
```

Trace JSON 写出逻辑:

```python
events.append({
    "name": row["text"],
    "cat": "NVTX",
    "ph": "X",
    "pid": pid,
    "tid": 1,
    "ts": (row["start"] - request_start) / 1000.0,
    "dur": (row["end"] - row["start"]) / 1000.0,
})

events.append({
    "name": fam,
    "cat": "CUDA kernel",
    "ph": "X",
    "pid": pid,
    "tid": 2,
    "ts": (row["start"] - request_start) / 1000.0,
    "dur": (row["end"] - row["start"]) / 1000.0,
    "args": {"kernel": row["name"][:240]},
})
```

## 8. 产物索引

核心实现:

```text
code/profile_visprune_single_request.py
code/run_clock_dense_fa2_single_request.sh
code/run_clock_visprune_single_request.sh
code/run_nsys_dense_fa2_single_request.sh
code/run_nsys_visprune_single_request.sh
code/analyze_decode_iterations.py
code/benchmark_gemv_concurrency.py
code/visualize_latency_breakdown.py
```

结果:

```text
output/clock_dense_fa2_32tok.json
output/clock_visprune_full_32tok.json
output/nsys_dense_fa2_32tok.json
output/nsys_visprune_full_32tok.json
output/kernel_family_summary.json
output/decode_iteration_kernel_breakdown.json
```

Nsight:

```text
output/nsys_dense_fa2_32tok.nsys-rep
output/nsys_visprune_full_32tok.nsys-rep
output/nsys_dense_fa2_32tok.sqlite
output/nsys_visprune_full_32tok.sqlite
```

可视化:

```text
output/visualizations/single_request_latency_breakdown.png
output/visualizations/single_request_latency_dashboard.html
output/visualizations/single_request_latency_trace_chrome.json
output/visualizations/latency_stage_breakdown.csv
output/visualizations/kernel_family_breakdown.csv
output/visualizations/decode_iteration_lines.csv
```

相关说明文档:

```text
DENSE_FA2_RESULTS.md
RESULTS.md
GEMV_ACCELERATION_ANALYSIS.md
LATENCY_VISUALIZATION_TOOLS.md
```

## 9. 总结

本实验基于现有 LLaVA/VisPrune PyTorch 代码、FlashAttention2 dense baseline
和 Nsight Systems CUDA/NVTX trace，完成了单请求端到端延迟拆解。

核心结论:

- `dense-fa2` 是当前单请求 baseline，request clock 为 `1140.97 ms`。
- `visipruner-full` request clock 为 `1456.85 ms`，未实现端到端加速。
- VisiPrune 的 prefill token pruning 确实减少了一部分 Tensor Core GEMM
  时间，但总体瓶颈是 decode 阶段稳定重复的小 GEMV。
- 可视化已经覆盖静态图、Plotly HTML 和 Perfetto/Chrome trace JSON，可用于
  报告、浏览器查看和 Nsight 深入分析。
