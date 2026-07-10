# VisiPrune 负载视角性能分析工作流草稿

本文梳理从“真实运行中的算法负载”出发，逐步得到 process-wise 性能归因的工作流。它覆盖当前仓库中的这些路径：

```text
workload_analysis/torch_profile
workload_analysis/algorithmic_trace
workload_analysis/dispatch
workload_analysis/fx
autoresearch/experiments/e2_single_request_latency
```

核心原则：

- 先用真实运行确定算法负载和动态 schedule，再做 selected-layer 的 op/process 取证。
- `torch.profiler`、`algorithmic_trace`、`dispatch`、`fx`、`nsys/CUPTI` 的证据层级不同，不能混用为同一种事实。
- 负载视角的 process 来自 dispatch/FX 的 op 证据和人工/规则重建；时间数据来自 NVTX/CUDA Runtime/CUPTI。
- 严格 process 性能归因必须依赖 process-level 或 fragment-level NVTX range；没有覆盖的 input-layer 只能做代表层模板估计，并显式标注 `template_scaled`。

## 总览

```text
0. torch_profile 通用初筛
   -> 1. algorithmic_trace 记录真实算法 schedule
   -> 2. layer 拆分并选择代表 input-layer
      -> 3A. dispatch 方法线: runtime ATen op / tensor-id 取证与 ONNX 小图重建
      -> 3B. fx 方法线: fixed-input FX DAG 与 process 重建/可视化
   -> 4. 将 FX process 归因到真实 Torch 代码并插入 process NVTX
   -> 5. SAME_INPUT layer-wise nsys/CUPTI 端到端 trace
   -> 6. 代表 input-layer strict process-wise breakdown
   -> 7. 用代表 input-layer 估计全量 input-layer process-wise breakdown
```

## 0. `torch_profile`: 通用 profiler 初筛

目的：

- 在不设计完整算法 trace 前，先观察一次真实 `generate()` 的 CPU/CUDA/PyTorch event flow。
- 检查 `record_shapes=True`、`with_stack=True`、`with_modules=True` 在当前 eager VisiPrune/LLaVA 路径下是否提供有效 stack/module 信息。
- 辅助决定后续需要显式 patch/wrap 的 request、forward、layer、attention、MLP、selection 边界。

方法：

- 使用 `torch.profiler.profile(...)` 包住真实请求。
- 同时插入 `record_function` scopes，避免完全依赖 profiler 自动 stack/module 元数据。
- 只把结果作为早期热点、shape、timeline、metadata 质量初筛，不作为算法 schedule 或 tensor dependency 主证据。

参考代码：

```text
workload_analysis/torch_profile/tools/torch_profiler_generate_trace.py
workload_analysis/torch_profile/runners/run_visipruner_full_profile.sh
workload_analysis/env/run_with_analysis_env.sh
```

复现命令：

```bash
GPU=1 TOKENS=1 \
/workspace/VisiPrune/workload_analysis/torch_profile/runners/run_visipruner_full_profile.sh
```

当前输出：

```text
workload_analysis/torch_profile/traces/visipruner_full_1tok_stack_modules/
  metadata.json
  chrome_trace.json
  profiler_events.csv
  profiler_key_averages.csv
  record_function_scopes.csv
  forward_events.csv
  layer_events.csv
  selection_events.csv
  module_stack_summary.csv
  process_view.md
```

使用的 skill：

```text
trace-patch-target-discovery
```

人工 review 重点：

- `metadata.json` 中 profiler 选项、输入、模型、GPU 是否符合本次分析目标。
- `module_stack_summary.csv` 和 profiler event stack 是否为空；当前 eager 路径下 stack/module metadata 可能为空。
- `record_function_scopes.csv` 是否足够指示后续需要 patch 的语义边界。

## 1. `algorithmic_trace`: 算法 layer-wise wrap trace

目的：

- 记录真实 `generate()` 中 VisiPrune 的动态 schedule。
- 明确每个 `forward_id/layer_idx` 的 `phase/q_len/kv_len/past_len`。
- 记录 middle token selection、deep exit、重要 visual token 数量变化。
- 为后续代表 input-layer 选择提供权威边界。

方法：

- 在真实模型运行中 wrap `model.forward`、decoder layer forward 和 VisiPrune selection 相关函数。
- 只记录小规模算法状态和 join keys，不复制大 tensor。
- 基于真实 layer trace 计算理论 FLOPs，而不是用 profiler 的有限 FLOPs 估计替代。

参考代码：

```text
workload_analysis/algorithmic_trace/tools/visipruner_algorithmic_trace.py
workload_analysis/algorithmic_trace/tools/compare_algorithmic_traces.py
workload_analysis/algorithmic_trace/runners/run_full_forward.sh
workload_analysis/algorithmic_trace/runners/run_all.sh
workload_analysis/env/run_with_analysis_env.sh
```

复现命令：

```bash
GPU=1 TOKENS=32 \
/workspace/VisiPrune/workload_analysis/algorithmic_trace/runners/run_full_forward.sh
```

当前输出：

```text
workload_analysis/algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/
  README.md
  algorithmic_trace.json
  layer_trace.csv
  selection_trace.csv
  operator_flops.csv

workload_analysis/algorithmic_trace/traces/fresh_forward_dense_eager_32tok/
  algorithmic_trace.json
  layer_trace.csv
  selection_trace.csv
  operator_flops.csv

workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok.json
workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok_ops.csv
workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok_phase.csv
```

当前 VisiPrune full trace 的关键事实：

```text
layer_trace.csv: 1024 rows
selection_trace.csv: 21 rows
forward events: 1 prefill + 31 decode
```

使用的 skill：

```text
trace-patch-target-discovery
visipruner-trace-dispatch-profile
```

人工 review 重点：

- `layer_trace.csv` 是否覆盖 32 层 x 32 次 forward。
- `selection_trace.csv` 是否显示 VisiPrune middle/deep/shallow 相关事件。
- token schedule 边界是否清楚，例如 prefill 中 `624 -> 58 -> 48`。
- dense trace 只能做 baseline，不可用来推断 VisiPrune-specific 层选择。

## 2. layer 拆分和代表 input-layer 选择

目的：

- 从全量 1024 个 input-layer 中选出少量代表事件，减少 dispatch/FX/process trace 数据量。
- 每个代表层都必须有明确负载角色：selection 边界、token 数变化边界、shallow/full-visual 代表、decode cache regime 代表等。

方法：

- 以 `algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/layer_trace.csv` 和 `selection_trace.csv` 为权威输入。
- 选层依据来自 `q_len/kv_len/past_len/phase`、selection/deep-exit 事件、workload 角色和重复组。
- 当前 dispatch 与 FX 使用同一批 35 个代表事件，保证两条方法线可对照。

参考代码和规则：

```text
workload_analysis/DISPATCH_FILTER_RULES.md
workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py
workload_analysis/fx/fx_dynamic_trace.py
```

当前代表层输出：

```text
workload_analysis/dispatch/profiles/filtered_dispatch_visipruner_full_32tok/dispatch_manifest.csv
workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/fx_layer_trace_manifest.csv
```

当前代表层规模：

```text
dispatch_manifest.csv: 35 target events
fx_layer_trace_manifest.csv: 35 target events
```

使用的 skill：

```text
visipruner-trace-dispatch-profile
visipruner-fx-trace-workflow
```

人工 review 重点：

- 每个代表事件是否能回到 `layer_trace.csv + selection_trace.csv` 的具体证据。
- 是否覆盖 prefill shallow、selection、middle pruned、deep removed、decode full-kv、decode pruned-kv、decode deep-removed 等角色。
- 代表层不是最终全量结论；它们只是后续 process 模板、op 证据和插桩定位的来源。

## 3A. `dispatch`: runtime ATen op / tensor-id 取证与 ONNX 重建

目的：

- 对选中的代表 input-layer，记录真实 eager 运行中的 ATen dispatch op 流。
- 得到 tensor id、input/output ids、shape、dtype、alias、inplace mutation、module stack 等 runtime 证据。
- 从 dispatch 证据重建 layer tensor process，并生成 small-shape Torch flow 与 ONNX stage。

方法：

- 使用 `TorchDispatchMode.__torch_dispatch__` 只包住 `dispatch_manifest.csv` 中的 35 个事件。
- 不做全量 dispatch profile；`observed_layer_events.csv` 只是全局 layer 编号校验。
- 对每个代表 layer 的 `dispatch_ops.csv` 做 op coverage、tensor dataflow、module split、process manifest 和 ONNX 导出。

参考代码：

```text
workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py
workload_analysis/dispatch/tools/split_dispatch_ops_by_event.py
workload_analysis/dispatch/layer_pipeline/run.py
workload_analysis/dispatch/layer_pipeline/analyze.py
workload_analysis/dispatch/layer_pipeline/flow_codegen.py
workload_analysis/dispatch/layer_pipeline/review_reconstruction.py
workload_analysis/dispatch/layer_pipeline/audit_layer_reconstruction.py
```

生成 filtered dispatch profile：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py \
  --gpu 1 \
  --tag filtered_dispatch_visipruner_full_32tok
```

重建选中 layer：

```bash
python /workspace/VisiPrune/workload_analysis/dispatch/layer_pipeline/run.py \
  --source-csv /workspace/VisiPrune/workload_analysis/dispatch/profiles/filtered_dispatch_visipruner_full_32tok/dispatch_ops.csv \
  --out-dir /workspace/VisiPrune/workload_analysis/dispatch/visualize \
  --layers input1_layer0 input1_layer5
```

当前输出：

```text
workload_analysis/dispatch/profiles/filtered_dispatch_visipruner_full_32tok/
  README.md
  dispatch_manifest.csv
  dispatch_ops.csv
  dispatch_op_summary.csv
  observed_layer_events.csv
  run_metadata.json

workload_analysis/dispatch/visualize/<event_id>/
  dispatch_review/
  torch_flow/
  onnx/
  layer_manifest.json
```

当前 filtered dispatch 的关键事实：

```text
source layer events: 1024
dispatch target events: 35
captured ATen dispatch ops: 3163
```

使用的 skill：

```text
visipruner-trace-dispatch-profile
dispatch-layer-reconstruct-onnx
```

人工 review 重点：

- `dispatch_manifest.csv` 的 35 个事件是否和算法 trace 选层理由一致。
- 每个 `dispatch/visualize/<event_id>/dispatch_review/dispatch_op_coverage.*` 是否覆盖该层所有 `event_op_index`。
- `tensor_dataflow.*` 的 producer-consumer edge 是否可由 `input_tensor_ids/output_tensor_ids` 重算。
- ONNX stage 是否只包含 dispatch 证据支持的过程；不要用固定模板覆盖不同层。

## 3B. `fx`: 代表 layer 的 FX trace、process 重建与可视化

目的：

- 对同一批代表 input-layer，得到 fixed-input `make_fx(...)` ATen DAG。
- 从 FX DAG 重建 readable process，例如 `input_rmsnorm`、`qkv_projection`、`rope`、`attention_scores`、`visual_process`、`mlp`。
- 用 FX process 的 op family、shape、node dependency 帮助理解负载过程，并为 NVTX 插桩定位提供语义模板。

方法：

- 真实 eager `generate()` 先运行，wrapper 只采样目标 layer 输入。
- 请求结束后离线 replay 采样输入并运行 `make_fx(...)`。
- 对 FX graph nodes 做 process reconstruction，生成 `fx_process_reconstruction.*`。
- process 可视化必须基于已生成的 FX reconstruction 手工解释，不把 process label 当成 PyTorch 官方模块语义。

参考代码：

```text
workload_analysis/fx/fx_dynamic_trace.py
workload_analysis/fx/fx_layer_process_reconstruct.py
```

生成 FX trace 示例：

```bash
python /workspace/VisiPrune/workload_analysis/fx/fx_dynamic_trace.py \
  --model-layer-trace \
  --layers input1_layer0,input1_layer5 \
  --gpu 1 \
  --tag fx_selected_layers
```

当前输出：

```text
workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/
  run_metadata.json
  fx_layer_events.csv
  fx_layer_trace_manifest.csv
  fx_process_reconstruction_manifest.csv
  fx_process_reconstruction_manifest.json
  <event_id>/
    fx_graph.py
    fx_graph.txt
    fx_graph_module.pt
    fx_graph_module/
    fx_nodes.json
    fx_process_nodes.csv
    fx_process_reconstruction.json
    fx_process_reconstruction.md
    fx_trace_metadata.json
```

当前 FX 关键事实：

```text
observed_layer_event_count: 1024
fx_sample_count: 35
fx_trace_count: 35
fx_trace_error_count: 0
```

使用的 skill：

```text
visipruner-fx-trace-workflow
visipruner-fx-process-visualization
```

人工 review 重点：

- `run_metadata.json` 的 input contract 是否和后续性能实验一致。
- `fx_layer_trace_manifest.csv` 是否 35 个代表层全部 `fx_traced`。
- `fx_process_reconstruction.md/json` 中每个 process 的 nodes、op families、shape 是否能解释对应负载。
- FX 是 fixed-input DAG，不是 runtime module ownership；如果要严格性能归因，必须继续做 Torch 代码插桩和 nsys/CUPTI。

## 4. FX process 归因到 Torch 代码并插入 NVTX

目的：

- 把 FX process 语义映射回真实 Torch 执行位置，建立严格性能采样所需的 process-level 或 fragment-level NVTX range。
- 解决跨 module / 跨函数 process：不能用一个宽泛 layer range 硬拆；需要可聚合 fragment。

方法：

- 读取 `fx_process_reconstruction.*`，提取 process id、title、nodes、op families、expected kernel families。
- 找到真实 launch-owning code path：layer forward、attention wrapper、token selection routine、backend wrapper、helper function。
- 在最小稳定代码区域插入 `visprune.fx_process.*` NVTX range。
- 输出 handoff 文档，交给 process performance breakdown 使用。

参考代码：

```text
autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py
autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/analyze_layer_nsys.py
repo/llava/model/language_model/custom_modeling_llama.py
```

当前输出：

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
```

使用的 skill：

```text
visipruner-fx-process-nvtx-instrumentation
```

人工 review 重点：

- Handoff 是否列出每个 process/fragment 的 `nvtx_range_name`、`aggregation_key`、code path、expected kernel families。
- 是否记录 same-input / same-weights / same-backend reproducibility contract。
- 是否存在 unresolved 或 ambiguous process；不能在性能报告中隐藏。
- 插桩只应引入 NVTX，不应改变算法语义或增加同步。

## 5. SAME_INPUT layer-wise 端到端 nsys/CUPTI trace

目的：

- 在与 FX trace 相同输入、权重、backend、生成参数下，得到全量 1024 个 input-layer 的 layer-wise 性能数据。
- 为后续全量 process attribution 提供每个 target input-layer 的守恒分母。

方法：

- 使用 layer/component NVTX ranges 包住 request、forward、layer、attn、MLP。
- 用 Nsight Systems 采集 NVTX、CUDA Runtime、CUPTI kernel。
- 通过 CUDA Runtime `correlationId` 将 NVTX CPU range 内发起的 runtime calls 连接到 CUPTI kernels。
- 报告 `NVTX CPU range ms` 和 `CUPTI launch-owned kernel sum ms`，不把二者相减，也不把 CUPTI kernel sum 当端到端 latency。

参考代码：

```text
autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py
autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
autoresearch/experiments/e2_single_request_latency/code/analyze_layer_nsys.py
autoresearch/experiments/e2_single_request_latency/code/generate_layer_performance_report.py
```

当前 VisiPruner full eager 输出：

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_layer_wise/
  SAME_INPUT_VISIPRUNER_FULL_EAGER_LAYER_PERFORMANCE_REPORT.md
  nsys_fxsameinput_visipruner_full_eager_32tok.json
  nsys_fxsameinput_visipruner_full_eager_32tok.nsys-rep
  nsys_fxsameinput_visipruner_full_eager_32tok.sqlite
  nsys_fxsameinput_visipruner_full_eager_32tok_layer_events.csv
  nsys_fxsameinput_visipruner_full_eager_32tok_layer_kernel_breakdown.csv
  nsys_fxsameinput_visipruner_full_eager_32tok_layer_kernel_breakdown.json
  nsys_fxsameinput_visipruner_full_eager_32tok_all_input_layer_performance.csv
  nsys_fxsameinput_visipruner_full_eager_32tok_stats_*.csv
```

当前关键事实：

```text
all input-layer rows: 1024 = 32 prefill + 992 decode
attribution_method: runtime_correlation_id
```

使用的 skill：

```text
visipruner-same-input-layer-wise-workflow
```

人工 review 重点：

- 报告中的 FX-matched input contract 是否全部 `yes`。
- `layer_events.csv` 是否和 FX/algorithmic layer sequence 对齐。
- `layer_kernel_breakdown.csv` 是否包含 `component=total/attn/mlp`，且 `attribution_method=runtime_correlation_id`。
- 后续 process 估计只能使用每个 input-layer 自己的 measured layer-wise latency 做归一化目标。

## 6. 代表 input-layer strict process-wise breakdown

目的：

- 对已经插入 process-level 或 fragment-level NVTX 的代表 input-layer，得到严格的 process-wise 性能数据。
- 为全量 process attribution 提供 observed process weights、process set、expected/matched kernel families。

方法：

- 消费 process-level NVTX trace。
- 对每个 `visprune.fx_process.*` range，选取 CPU range 内发起的 CUDA Runtime calls。
- 通过 `correlationId` 连接 CUPTI kernel，统计 launch-owned kernel duration 和 kernel family。
- 对 fragment-level range 按 `aggregation_key` 聚合成 process。

参考代码：

```text
autoresearch/experiments/e2_single_request_latency/code/generate_process_performance_breakdown.py
autoresearch/experiments/e2_single_request_latency/code/analyze_layer_nsys.py
autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
```

当前输出：

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/
  SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md
  SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
  nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv
  nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json
  same_input_visipruner_full_eager_process_attribution.csv
```

使用的 skill：

```text
visipruner-process-performance-breakdown
```

人工 review 重点：

- trace 中必须存在 `visprune.fx_process.*` range；没有 process NVTX 时不能做 strict process attribution。
- `same_input_visipruner_full_eager_process_attribution.csv` 的每行要包含 process id/title、FX nodes/op families、expected/matched kernel families、CUPTI/NVTX timing、validation status。
- 百分比只在相同 metric scope 内计算，例如 process / parent layer；不要除以全请求延迟。

## 7. 全量 input-layer segmented process attribution

目的：

- 用代表 input-layer 的 strict process trace，估计全量 1024 个 input-layer 的 process-wise 性能视图。
- 解决 process-level trace 全量采集数据过大的问题。

方法：

- 读取全量 layer-wise 数据，提取每个 target input-layer 的 feature signature。
- 读取代表层 process-wise 数据，构建代表 input-layer method。
- 基于 phase、q_len、kv_len、workload、retained token/KV regime、dominant kernel family、process set 自行推理 interval 和 representative assignment。
- 对每个 target layer/process/metric 计算：

```text
raw_weight = representative_process_ms * complexity(target) / complexity(representative)
estimated_process_ms = target_layer_measured_ms * raw_weight / sum(raw_weight)
```

- 输出 `observed_fx_op`、`template_scaled`、`fallback_component`、`unknown` 证据来源，并做 layer-level conservation。

参考代码：

```text
autoresearch/experiments/e2_single_request_latency/code/generate_segmented_process_attribution.py
```

当前输出：

```text
autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_full_layer_process_attribution/
  SAME_INPUT_VISIPRUNER_FULL_EAGER_FULL_LAYER_PROCESS_ATTRIBUTION_REPORT.md
  SAME_INPUT_FULL_LAYER_PROCESS_ATTRIBUTION_BREAKDOWN.md
  full_layer_attribution_type_map.csv
  full_layer_template_assignment.csv
  full_layer_process_attribution.csv
  full_layer_process_aggregation.csv
  full_layer_coverage_and_risk.csv
```

当前关键事实：

```text
assignment rows: 1024
observed_fx_op rows: 35 input-layers
template_scaled rows: 989 input-layers
fallback/unknown: 0
metric groups: 2048
max conservation error: about 3e-6 ms
```

使用的 skill：

```text
visipruner-segmented-process-attribution
```

人工 review 重点：

- `full_layer_template_assignment.csv` 是主 review 面：检查每个 target input-layer 的 `feature_match_basis`、`representative_layer_id`、`template_validation_status`。
- `full_layer_attribution_type_map.csv` 用于审查 interval 边界是否有结构或性能理由。
- `full_layer_process_attribution.csv` 必须对每个 `(phase, layer, occurrence, metric)` 守恒到 source layer metric。
- 全局结论如果主要来自 `template_scaled`，只能称为 layer-conserved estimate，不是全量 direct process trace。

## 证据边界与迁移约束

`torch_profile`：

- 可以看热点、shape、Chrome timeline。
- 不能替代算法 schedule、tensor dependency 或 strict process attribution。

`algorithmic_trace`：

- 是 VisiPrune 动态 schedule 和理论 workload 的权威来源。
- 不提供真实 kernel latency。

`dispatch`：

- 是真实 eager ATen op 和 tensor id 证据。
- 不天然提供 graph；dependency/process 需要后处理重建。

`fx`：

- 是 fixed-input replay 的 ATen GraphModule DAG。
- process label 是重建标签，不是官方 module ownership。

`nsys/CUPTI`：

- 是性能时间数据来源。
- 严格归因使用 launch ownership：NVTX CPU range -> CUDA Runtime `correlationId` -> CUPTI kernel。
- 不能用 GPU 时间 overlap 或 `NVTX CPU - CUPTI kernel` 作为归因。

迁移到其他场景时，每一步至少需要重新实现或确认：

- 同等语义的 runtime boundary 和 join keys。
- 同等字段的 layer/input event 表。
- 代表 input-layer 选择规则。
- process reconstruction 的 op/family/shape 证据。
- process-level NVTX handoff。
- launch-owned CUPTI 归因脚本。
- assignment/conservation 校验脚本。
