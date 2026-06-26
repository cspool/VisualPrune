# VisiPrune Workload Analysis

`workload_analysis` 是一个独立的负载分析工作区。这里的目标不是测量
Nsight/CUDA kernel 级运行时结果，而是生成 **algorithmic / theoretical
workload trace**：用真实 forward 暴露 VisiPrune 的动态 token schedule，再用
公式统计理论 FLOPs；同时用开源工具生成 dense LLM 的参考分析。

本文现在按实际功能组织目录，不再使用旧的抽象方案目录名。

## 一级目录

- `env/`
  运行环境 glue code。当前只有 `run_with_analysis_env.sh`，它复用已有
  `/workspace/VisiPrune/venv_profiling/bin/python`，设置 `HF_HOME`、
  offline HF 变量、`PYTHONPATH` 和本目录 pip cache。这里不重新安装解释器
  或包管理器。

- `external/`
  下载的开源工具源码，只放在本目录下。当前包括：
  `llm-analysis`、`llm-viewer`、`calculate-flops.pytorch`。

- `logs/`
  环境准备日志，例如轻量包安装日志。用于确认没有重复安装 Torch/CUDA
  这类通用运行环境。

- `algorithmic_trace/`
  端到端 algorithmic workload trace 的工具和产物。`tools/` 放 fresh-forward trace 与 trace 对比脚本；
  `runners/` 放一键入口；`traces/` 和 `comparisons/` 放对应输出；
  `verification/` 放 trace wrapper 等价性验证。

- `open_tool_dense_baseline/`
  Open-tool dense baseline 的工具和产物。`tools/` 放对照脚本；
  `dense_baseline/` 放 `llm-analysis`、`LLM-Viewer` 和比较报告。

- `dispatch/`
  Dispatch 相关工具和产物。`profiles/` 放 filtered TorchDispatch profile；
  `visualize/`、`layer_pipeline/`、`templates/` 放后续 layer reconstruction/ONNX 工作。

- `vendor/`
  少量缺失 Python 包的本地安装目录，使用 `pip --target ... --no-deps`。
  它只补 `tabulate/fvcore/calflops/fire/termcolor` 等轻量分析包，不放重复
  Torch/CUDA 环境。

## Algorithmic Trace

`algorithmic_trace/` 是 VisiPrune 的权威 workload trace 来源。

它做两件事：

1. 获取 VisiPrune 的动态 token schedule。
2. 按 LLaVA/LLaMA/CLIP/projector 结构用公式计算理论 FLOPs。

输出文件格式：

- `algorithmic_trace.json`: 完整 JSON，包括请求、模型维度、forward/layer/
  selection 事件和 FLOP summary。
- `layer_trace.csv`: 每个 forward、每层的 `q_len/kv_len/past_len`。
- `selection_trace.csv`: `value_aware_token_selection` 事件，包括 middle 选择
  和 deep exit。
- `operator_flops.csv`: 展开的理论 operator FLOPs。

## Filtered Dispatch Profile

Filtered dispatch profile 用于深入查看 VisiPrune 强相关 layer 内真实发生的
ATen operator 和输入/输出 shape。它不是全量 profiler：脚本先根据
`selection_trace.csv + layer_trace.csv` 生成 manifest，然后只在选中的
`(forward_id, layer_id)` 进入 `TorchDispatchMode`。

过滤规则：

`DISPATCH_FILTER_RULES.md`

脚本：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py \
  --gpu 1 \
  --tag filtered_dispatch_visipruner_full_32tok
```

当前输出：

`dispatch/profiles/filtered_dispatch_visipruner_full_32tok/`

当前运行从 1024 个 layer events 中选择 35 个 dispatch target events，捕获
3163 条 ATen dispatch op。`observed_layer_events.csv` 记录全部 layer 调用只作
编号校验，不代表全量 dispatch profile。

Algorithmic Trace 排除：

- model loading
- image file I/O
- CPU preprocessing
- host-device transfer
- CUDA kernel launch overhead
- runtime/compiler fusion
- memory traffic
- communication

## Fresh Forward

Fresh forward 是 Algorithmic Trace 的首选路径：真实加载 LLaVA/VisiPrune 模型，实际跑
一次 `generate()`，通过 hook 记录动态 schedule，然后离线用公式计数。

脚本：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/algorithmic_trace/tools/visipruner_algorithmic_trace.py \
  --config visipruner-full \
  --max-new-tokens 32 \
  --gpu 1 \
  --tag fresh_forward_visipruner_full_32tok
```

当前 fresh VisiPrune 输出：

`algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/`

当前 fresh dense-eager 输出：

`algorithmic_trace/traces/fresh_forward_dense_eager_32tok/`

完整 fresh-forward 一键入口：

```bash
GPU=1 TOKENS=32 /workspace/VisiPrune/workload_analysis/algorithmic_trace/runners/run_full_forward.sh
```

这个入口会依次生成：

- fresh VisiPrune trace
- 基于 fresh VisiPrune trace 的 open-tool dense baseline
- fresh dense-eager trace
- VisiPrune vs dense-eager 理论 FLOPs 对比

Wrapper 等价性验证：

```bash
CUDA_VISIBLE_DEVICES=1 /workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/algorithmic_trace/verification/tools/verify_wrapper_equivalence.py \
  --config visipruner-full \
  --max-new-tokens 32
```

这个验证会先运行未包裹的 `model.generate()`，再打 wrapper 运行同一请求，
比较两次 `output_ids` 是否完全一致。

## Reconstruct

当前工作区没有可运行的 e2 reconstruction 脚本；algorithmic trace 的权威入口是
`algorithmic_trace/tools/visipruner_algorithmic_trace.py` 的 fresh forward。保留
`algorithmic_trace/runners/run_all.sh` 作为旧入口占位，但它会在缺少 reconstruction
脚本时显式报错。

## Open-Tool Dense Baseline

`open_tool_dense_baseline/` 不是 VisiPrune trace 的来源。它用于把 algorithmic trace 的输入规模转换成
dense language-backbone 配置，然后用开源工具生成参考分析。

脚本：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/open_tool_dense_baseline/tools/open_tool_compare.py \
  --trace /workspace/VisiPrune/workload_analysis/algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/algorithmic_trace.json
```

输出目录：

`open_tool_dense_baseline/dense_baseline/`

主要输出：

- `dense_equivalent_config.json`: 从 algorithmic trace 派生出的 dense LLaMA 配置。
- `llm_analysis_dense_summary.json`: `llm-analysis` 的 dense baseline 输出。
- `llm_viewer_dense_summary.json`: `LLM-Viewer` 的 dense operator/roofline 输出。
- `dense_tool_comparison.csv`: algorithmic trace 和 dense open-tool 结果的简表。
- `open_tool_fit_report.json`: 工具适配边界说明。

## llm-analysis

本地源码：

`external/llm-analysis/`

用途：

- 理论分析 dense Transformer/LLM 的 inference latency/memory。
- 当前在本工作区中用于 dense LLaMA language-backbone baseline。

边界：

- 不原生表达 VisiPrune 的逐层动态视觉 token pruning。
- 不包含 LLaVA 的 CLIP vision tower 和 multimodal projector，除非手工扩展。
- 因此它是 open-tool dense baseline 的参考工具，不是 algorithmic trace 的权威计数器。

## LLM-Viewer

本地源码：

`external/llm-viewer/`

用途：

- 给 dense LLaMA backbone 生成 operator/roofline 风格分析。
- 输出 prefill/decode 的 OPs、memory access、bound、inference time 等。

边界：

- 默认假设 dense transformer 层和全局 sequence length。
- 不能直接表示 layer 18 后 `624 -> 58 -> 48` 这类 VisiPrune 动态长度变化。
- 因此它用于理解 dense baseline 的结构，不替代 VisiPrune trace。

## calculate-flops.pytorch / calflops

本地源码：

`external/calculate-flops.pytorch/`

用途：

- 可作为后续 module-level FLOP sanity check 工具。
- 当前完整分析主要依赖自定义公式和 `llm-analysis`/`LLM-Viewer` 对照。

边界：

- 通用 FLOP counter 通常不理解 VisiPrune 的数据相关 token 删除语义。
- 如果用于 VisiPrune，需要额外定制 hook 或输入 shape schedule。

## open_tool_compare.py

脚本：

`open_tool_dense_baseline/tools/open_tool_compare.py`

作用：

1. 读取 Algorithmic Trace 的 `algorithmic_trace.json`。
2. 提取 dense-equivalent language-backbone 配置。
3. 调用 `llm-analysis` 和 `LLM-Viewer`。
4. 写出 Open-Tool Dense Baseline 报告。

这个脚本中的 `open_tool` 指的是“开源/官方分析工具对照层”，不是
VisiPrune 本身的实现。

## Fresh Result Summary

当前 fresh-forward 32-token 结果：

- VisiPrune trace: `algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/`
- Dense trace: `algorithmic_trace/traces/fresh_forward_dense_eager_32tok/`
- VisiPrune vs dense 对比:
  `algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok.json`

关键 schedule：

- prompt tokens before image expansion: `49`
- visual tokens: `576`
- prefill length: `624`
- middle selection layer: `18`
- selected visual tokens: `10`
- layers `0-18`: `624`
- layers `19-27`: `58`
- layers `28-31`: `48`
- decode forwards: `31`

理论 FLOPs：

- dense-eager full VLM actual model path: `9.275895808000e12`
- VisiPrune full VLM actual model path: `6.044742909952e12`
- saved FLOPs: `3.231152898048e12`
- saved percentage: `34.83%`

更详细的结果见：

`ANALYSIS_SUMMARY.md`
