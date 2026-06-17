# E1 Baseline Gap — 实验代码 Spec

## 概述

为 E1 实验编写 `bench_e1_baseline.py`，支持 4 个 model config 的端到端推理，通过 nsys 和 ncu 两层 profiling 采集 GPU 性能数据。

**严格禁止**：Python wall-time 计时（`time.time()`, `CUDATimer`, `torch.cuda.Event`）、CUPTI / `torch.profiler`。所有性能数据必须来自 nsys 或 ncu。

## 文件位置

```
autoresearch/experiments/e1_baseline/code/bench_e1_baseline.py
```

## CLI 接口

```
python bench_e1_baseline.py \
    --model-path <str>           # 默认: liuhaotian/llava-v1.5-7b
    --model-base <str>           # 可选, LoRA base
    --image-path <str>           # 单张图片路径 (必需)
    --prompt <str>               # 默认: "Describe the image briefly."
    --config <str>               # dense-fa2 | dense-eager | visipruner-full | visipruner-shallow-only
    --mode <str>                 # nsys-profile | ncu-profile
    --max-new-tokens <int>       # 默认: nsys=128, ncu=32
    --cache-dir <str>            # HF cache, 默认: ~/.cache/huggingface
    --output-dir <str>           # 输出目录, 默认: profiling/results/e1_baseline
```

**两种 mode 的行为差异**：

| 行为 | nsys-profile | ncu-profile |
|------|-------------|-------------|
| 推理次数 | 1 次 (nsys 采样全量) | 1 次 (ncu replay 热点 kernel) |
| NVTX markers | ✅ 启用 (标记 prefill/decode) | ❌ 不需要 |
| max_new_tokens | 128 (足够的 decode 步数以观察分布) | 32 (限制 replay 量) |
| warmup | 无 (避免干扰时间线) | 无 (ncu 自带 skip) |
| stdout 输出 | 最少 — 只打印 config + mode + 完成状态 | 同左 |

## Config 定义

```python
CONFIGS = {
    "dense-fa2": {
        "attn_implementation": "flash_attention_2",
        "pruning_config": None,
        "description": "Dense + Flash Attention 2 (baseline)",
    },
    "dense-eager": {
        "attn_implementation": "eager",
        "pruning_config": None,
        "description": "Dense + Eager attention (no pruning)",
    },
    "visipruner-full": {
        "attn_implementation": "eager",
        "pruning_config": {
            "mode": ["middle", "deep"],
            "shallow_mid_layer": 6,
            "layer_threshold": 0.995,
            "tokens_threshold": 0.2,
        },
        "description": "VisiPruner full pipeline (shallow+middle+deep)",
    },
    "visipruner-shallow-only": {
        "attn_implementation": "eager",
        "pruning_config": {
            "mode": ["shallow"],
            "shallow_mid_layer": 6,
        },
        "description": "VisiPruner shallow-only",
    },
}
```

## 模块依赖

```python
import argparse, os, sys, json
import torch
from PIL import Image

# LLaVA imports (from repo/)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
```

**禁止依赖**：`profiling/utils.py`（CUDATimer, BenchResult 等）、`profiling/hardware.py`（HardwareProfiler, CUPTI 等）、`profiling/cupti_collector.py`。

## 核心执行流

### 1. 参数解析 + 环境设置

```python
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['HF_HOME'] = args.cache_dir

    # 写 system_info.json（GPU 名称、PyTorch/CUDA 版本等）
    # 仅基本信息，不做任何性能测量
    sys_info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_mem / 1024**3,
    }
    with open(os.path.join(args.output_dir, "system_info.json"), "w") as f:
        json.dump(sys_info, f, indent=2)
```

### 2. 图片加载

```python
def load_single_image(image_path, image_processor, model_config):
    """Load and process a single image. Returns (image_tensor, image_size)."""
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    image_tensor = process_images([image], image_processor, model_config)
    # process_images returns a tensor or list; ensure it's on correct device
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device='cuda', dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device='cuda', dtype=torch.float16)
    return image_tensor, image_size
```

### 3. 模型加载

```python
def load_model_for_config(model_path, model_base, config, cache_dir):
    """Load LLaVA model with the specified attention implementation."""
    cfg = CONFIGS[config]
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="auto",
        attn_implementation=cfg["attn_implementation"],
        cache_dir=cache_dir,
    )
    model.eval()

    # Apply pruning config if specified
    if cfg["pruning_config"] is not None:
        model.set_pruning_config(cfg["pruning_config"])

    return tokenizer, model, image_processor
```

**注意**：`load_pretrained_model` 的 `attn_implementation` 通过 `**kwargs` 传递给 `from_pretrained()`。对于 SDPA（非 E1 核心 config 但可能用于对照），需直接传 `attn_implementation="sdpa"`。当前 `builder.py:45-48` 的逻辑是 `use_flash_attn=True → FA2 else → Eager`，需确认 `attn_implementation` 是否已被 builder 支持为独立参数。如果 builder 不支持，需在 config dict 中使用 `use_flash_attn=True`（FA2）或 `use_flash_attn=False`（Eager）。

### 4. 推理执行（带 NVTX markers — nsys 模式）

```python
def run_inference_for_profiling(model, tokenizer, image_processor,
                                 image_tensor, image_size,
                                 prompt, max_new_tokens,
                                 pruning_config, use_nvtx=False):
    """
    Run end-to-end inference for profiling.

    When use_nvtx=True: wraps prefill and decode phases with NVTX ranges
    for nsys timeline visualization.

    Returns: None (all performance data captured by external profiler)
    """
    from llava.mm_utils import tokenizer_image_token

    # Build input
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    # NVTX markers for nsys
    if use_nvtx:
        torch.cuda.nvtx.range_push("e1_inference_total")

    # Run generate — all profiling data comes from nsys/ncu externally
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,                  # greedy for reproducibility
            pruning_config=pruning_config,    # only for visipruner configs
            use_cache=True,
        )

    if use_nvtx:
        torch.cuda.nvtx.range_pop()

    # Decode output (just to verify generation worked)
    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    return output_text
```

**关于 NVTX 细粒度标记**：

E1 的 nsys 模式不需要 per-phase NVTX（那是 E4 的需求）。E1 只需要粗粒度的 `e1_inference_total` range，用于在 nsys 时间线中定位推理区间。kernel 级别的时间分布全部来自 `nsys stats --report cuda_gpu_kern_sum`。

如果需要在 nsys timeline 中区分 prefill 和 decode（便于后续分析），可增加两个 NVTX range：

```python
# 在 model.generate() 内部无法插入 NVTX（HuggingFace 内部代码）
# 替代方案：使用 torch.cuda.cudart().cudaProfilerStart/Stop()
# 或者通过 monkey-patch LlamaModel.forward 添加 NVTX（范围太大，不建议 E1 做）
#
# E1 决策：只标记 e1_inference_total，prefill/decode 分解留给 E4
```

### 5. nsys-profile 模式入口

```python
def run_nsys_profile(args):
    """
    Step 1 of E1 protocol: nsys sampling to locate hot kernels.

    This script is designed to be launched BY nsys, not to invoke nsys itself.
    The user runs:
        nsys profile --trace cuda,nvtx,cublas --stats=true \
            --output e1_nsys_<config> \
            python bench_e1_baseline.py --config <config> --mode nsys-profile

    This function is the target of that nsys invocation.
    """
    print(f"[E1 nsys-profile] config={args.config}", flush=True)
    cfg = CONFIGS[args.config]

    # Load model
    tokenizer, model, image_processor = load_model_for_config(
        args.model_path, args.model_base, args.config, args.cache_dir
    )

    # Load image
    image_tensor, image_size = load_single_image(
        args.image_path, image_processor, model.config
    )

    # Warmup: one short generation to initialize CUDA context, allocate KV cache
    # This prevents first-run overhead from dominating the nsys trace
    print(f"[E1 nsys-profile] Warming up...", flush=True)
    run_inference_for_profiling(
        model, tokenizer, image_processor, image_tensor, image_size,
        "Describe briefly.", max_new_tokens=8,
        pruning_config=cfg["pruning_config"], use_nvtx=False,
    )
    torch.cuda.synchronize()

    # Profiled run
    print(f"[E1 nsys-profile] Profiled run (max_new_tokens={args.max_new_tokens})...", flush=True)
    output = run_inference_for_profiling(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=args.max_new_tokens,
        pruning_config=cfg["pruning_config"], use_nvtx=True,
    )
    torch.cuda.synchronize()

    print(f"[E1 nsys-profile] Done. Output: {output[:80]}...", flush=True)
    print(f"[E1 nsys-profile] nsys output file will be written by the nsys wrapper.", flush=True)

    # Cleanup
    del model, tokenizer, image_processor
    torch.cuda.empty_cache()
```

**关键设计决策**：
- warmup 不计入 nsys trace（`use_nvtx=False`），避免包含 CUDA context 初始化、JIT 编译等一次性开销
- 但 warmup 必须在 nsys 进程内完成（不能在 nsys 外部），否则 nsys 仍会捕获 warmup 的 kernel
- 实际上，更简洁的方式是**不做 warmup**，让 nsys 捕获完整的第一轮推理。在 post-process 时通过时间线剪掉前面的 context init kernel。这样避免了 warmup 引入的时序复杂性

**修正方案（更简洁）**：

```python
def run_nsys_profile(args):
    """No warmup — let nsys capture everything, filter in post-process."""
    print(f"[E1 nsys-profile] config={args.config}", flush=True)
    cfg = CONFIGS[args.config]

    tokenizer, model, image_processor = load_model_for_config(...)
    image_tensor, image_size = load_single_image(...)

    # Single profiled run — nsys captures everything
    torch.cuda.nvtx.range_push("e1_inference")
    output = run_inference_for_profiling(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=args.max_new_tokens,
        pruning_config=cfg["pruning_config"], use_nvtx=False,
    )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    print(f"[E1 nsys-profile] Done.", flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()
```

### 6. ncu-profile 模式入口

```python
def run_ncu_profile(args):
    """
    Step 2 of E1 protocol: ncu deep analysis on specific hot kernels.

    Launched BY ncu:
        ncu --set full \
            --kernel-name regex:"<kernel_pattern_from_nsys>" \
            --launch-count 50 --launch-skip 5 \
            --import-source yes \
            --output e1_ncu_<config>_<kernel> \
            python bench_e1_baseline.py --config <config> --mode ncu-profile

    This function is the target of that ncu invocation.
    It runs a minimal inference to trigger the target kernels.
    """
    print(f"[E1 ncu-profile] config={args.config}", flush=True)
    cfg = CONFIGS[args.config]

    tokenizer, model, image_processor = load_model_for_config(...)
    image_tensor, image_size = load_single_image(...)

    # ncu will replay only kernels matching --kernel-name
    # Keep max_new_tokens small to limit the number of kernel launches
    print(f"[E1 ncu-profile] Profiled run (max_new_tokens={args.max_new_tokens})...", flush=True)
    output = run_inference_for_profiling(
        model, tokenizer, image_processor, image_tensor, image_size,
        args.prompt, max_new_tokens=args.max_new_tokens,
        pruning_config=cfg["pruning_config"], use_nvtx=False,
    )
    torch.cuda.synchronize()

    print(f"[E1 ncu-profile] Done.", flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()
```

**ncu 调用的封装脚本（Shell）**：

因为 ncu 的 kernel pattern 需要在 nsys 结果出来后才知道，提供一个辅助脚本生成 ncu 命令：

```bash
# scripts/gen_e1_ncu_commands.sh
# 用法: bash gen_e1_ncu_commands.sh <nsys_kernel_csv_path> <config_name>
# 从 nsys top-10 kernel CSV 中提取 kernel 名，生成 ncu 命令
```

这是 Step 1 完成后才需要的，暂不细化。

### 7. 入口调度

```python
def main():
    args = parse_args()

    # Write system info (always)
    os.makedirs(args.output_dir, exist_ok=True)
    sys_info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "config": args.config,
        "mode": args.mode,
    }
    with open(os.path.join(args.output_dir, "system_info.json"), "w") as f:
        json.dump(sys_info, f, indent=2)

    if args.mode == "nsys-profile":
        run_nsys_profile(args)
    elif args.mode == "ncu-profile":
        run_ncu_profile(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
```

## nsys 后处理流程（不在 bench 脚本内）

bench 脚本只负责**运行**推理。nsys 数据采集由 nsys wrapper 完成，后处理用 `nsys stats` 命令：

```bash
# E1 Step 1 完整流程（人工或外层脚本执行）:

# 1. 采集
for config in dense-fa2 dense-eager visipruner-full visipruner-shallow-only; do
    nsys profile \
        --trace cuda,nvtx,cublas \
        --stats=true \
        --force-overwrite=true \
        --output e1_nsys_${config} \
        python bench_e1_baseline.py \
            --config ${config} \
            --mode nsys-profile \
            --image-path /path/to/test_image.jpg \
            --max-new-tokens 128
done

# 2. 导出 kernel 统计
for config in dense-fa2 dense-eager visipruner-full visipruner-shallow-only; do
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output e1_nsys_${config}_kernels.csv \
        e1_nsys_${config}.nsys-rep
done

# 3. 分析——提取 top-10 kernel、T_total、kernel 类别
python analysis/e1_analyze_nsys.py  # 不在本 spec 范围内
```

## 错误处理

1. **模型加载失败**：打印明确错误信息 → `sys.exit(1)`
2. **图片不存在**：打印路径 + `sys.exit(1)`
3. **未知 config**：打印合法 config 列表 + `sys.exit(1)`
4. **CUDA OOM**：让 Python 自然 crash（traceback 保留完整信息）
5. **nsys/ncu 未安装**：不在 bench 脚本内检查 —— 由外层 wrapper 处理

## 与已有代码的关系

| 组件 | 状态 | 处理方式 |
|------|------|---------|
| `profiling/utils.py` | 依赖 wall-time / CUPTI | **不引用** — 重写最小化版本 |
| `profiling/hardware.py` | 依赖 CUPTI | **不引用** |
| `profiling/cupti_collector.py` | CUPTI | **不引用** |
| `llava/model/builder.py` | 模型加载 | **直接使用** `load_pretrained_model` |
| `llava/mm_utils.py` | 图像处理 | **直接使用** `process_images`, `tokenizer_image_token` |
| `llava/model/language_model/llava_llama.py` | `generate()` 入口 | **直接调用** `model.generate()` |
| `llava/model/language_model/custom_modeling_llama.py` | Pruning 逻辑 | 通过 `pruning_config` 参数触发，**不修改源码** |

## 复杂度约束

- 总行数 ≤ 250 行（不含 shebang / imports / docstring）
- 依赖项 ≤ 6 个 import 模块
- 无类定义（纯函数式）
- 无中间结果缓存（每次运行即抛）
- stdout 输出 ≤ 5 行（避免污染 nsys/ncu 的输出解析）
