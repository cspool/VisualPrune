# Segment-based Full-layer FX Process Attribution Draft

本文档记录如何利用 `README-DRAFT.md` 中整理出的“代表 layer 归因方法”，对端到端全部 layer 性能数据做 FX process 归因。它是 `README-DRAFT.md` 的下游 workflow 文档，重点解决一个问题：

```text
FX trace 只有部分代表 layer，但端到端性能数据覆盖全部 layer。如何把端到端 layers 划分为 input-layer 区间，并让每个区间使用对应代表 layer 的归因方法？
```

## 核心判断

`README-DRAFT.md` 的额外产物是代表 layer 归因方法库。每个代表 layer 对应一个 input-layer 区间的归因方法。

本文件的职责是使用这个方法库完成端到端全部 layer 的归因：

```text
representative layer attribution method
  -> assign to one input-layer interval
  -> attribute every layer in that interval
  -> use each target layer's own Nsight/CUPTI/NVTX performance data
```

因此，代表 layer 的作用是提供方法，不是提供耗时。全量归因的数据来源仍然是每个 target layer 自己的真实性能采样。

## 总体目标

目标是构建一个端到端可审计的 attribution map：

```text
end-to-end layer sequence
  -> split into attribution intervals
  -> assign one representative layer method to each interval
  -> use that method's FX process/op template and validation rules
  -> attribute every layer's performance data to FX process/cost_type
  -> compute layer-local percentages
  -> aggregate globally after all layers are attributed
```

最终报告要回答：

- 每个 layer 的延迟如何归到 FX process？
- 每个 FX process 下 compute、memory、runtime、sync、scheduler 分别多少？
- 哪些 layer 使用真实 FX op trace？
- 哪些 layer 使用 template-inferred FX op 结构？
- 哪些 layer 只能 fallback？
- 全局聚合中 observed、inferred、fallback 各占多少？

## 关键对象

### attribution interval

`attribution interval` 是端到端 layer 序列中的连续区间。每个区间必须绑定一个来自 `README-DRAFT.md` 的代表 layer 归因方法。区间内 layer 被认为共享相同或足够接近的 FX process/op structure。

示例：

```text
prefill layer 0
prefill layer 5-18
prefill layer 19-28
prefill layer 29-31
decode layer 0-17
decode layer 18-19
decode layer 20-28
decode layer 29-31
```

区间边界应由结构变化触发，而不是随意切分。

### attribution type

`attribution type` 是端到端区间使用的归因类型。它把一个 input-layer 区间和一个代表 layer 方法绑定起来。

建议字段：

```text
attribution_type_id
variant
phase
layer_interval
boundary_reason
representative_layer_id
representative_layer_method
template_match_basis
process_template
op_family_template
expected_kernel_families
validation_rules
confidence
fallback_policy
```

### process template

`process template` 来自代表 layer 方法。代表 layer 方法本身由 `README-DRAFT.md` 基于 traced FX layer 的 `fx_process_reconstruction.json` 总结得到。

它描述：

```text
process
title
nodes
node_count
fx_op_families
bucket
cost_type_candidates
```

例如：

```text
qkv_projection:
  fx_op_families: GEMM, reshape, transpose
  cost_type_candidates: compute, runtime_launch, scheduler_gap

rope:
  fx_op_families: index, slice, cat, elementwise
  cost_type_candidates: memory, compute

attention_scores:
  fx_op_families: BMM, softmax, elementwise
  cost_type_candidates: compute, memory

visual_process:
  fx_op_families: gather, index, cat, reduce, selection
  cost_type_candidates: memory, compute

mlp:
  fx_op_families: GEMM, activation, elementwise
  cost_type_candidates: compute
```

## 区间边界选择

FX trace layer 如果选在端到端过程中的关键边界，模板迁移会更可靠。

推荐边界条件：

- prefill/decode 切换。
- input shape 或 q/kv length 改变。
- pruning policy 改变。
- visual token 数量发生显著变化。
- backend 改变，例如 eager、FA2、VP-FA/Triton。
- kernel family 结构改变。
- shallow/deep/special visual process 出现或消失。
- layer role 特殊，例如 input-heavy layer、tail layer、post-pruning stable layer。

每个 interval 默认使用一个代表 layer 方法。若一个区间需要多个代表 layer 方法共同约束，必须显式说明组合规则和优先级；否则不要隐式混合多个方法。

## 归因流程

### 1. 建立端到端 layer 序列

从 SAME_INPUT 性能数据中读取所有 layer occurrence：

```text
variant
phase
layer
occurrence
q_len
kv_len
workload_type
NVTX CPU range
CUPTI kernel activity
kernel family
runtime/sync/mem activity
```

这一步覆盖全部端到端 layer，而不是只看 FX sampled layers。

### 2. 读取代表 layer 方法库

从 `README-DRAFT.md` 对应产物读取代表 layer 方法：

```text
representative_layer_id
variant
phase
layer
q_len
kv_len
boundary_role
covered_interval_hint
fx_processes
fx_op_families_by_process
expected_kernel_families
cost_type_candidates
attribution_method_priority
fallback_policy
validation_requirements
confidence
```

然后构建端到端归因可用的方法库：

```text
representative_layer_method_id -> process/op attribution method
```

每个方法必须保留 representative layer，不允许丢失来源。

### 3. 划分端到端 attribution intervals

根据 phase、layer range、shape、backend、pruning state、kernel family 和代表 layer 的 `covered_interval_hint` 划分端到端区间。

每个区间要记录：

```text
interval_id
variant
phase
layer_start
layer_end
boundary_reason
covered_occurrences
representative_layer_id
representative_layer_method_id
```

### 4. 为每个 interval 选择代表 layer 方法

选择规则优先级：

1. exact representative：target layer 自己就是代表 layer。
2. same interval representative：target layer 属于代表 layer 的 declared interval。
3. nearest boundary representative：最近的边界代表 layer，且 shape/backend/policy compatible。
4. component fallback：无法确认代表 layer 方法，只能使用 attn/mlp/outer component。
5. unknown：无法可靠归因。

每个 layer 的 attribution record 必须标记：

```text
attribution_source = observed_fx_op | template_inferred | fallback_component | unknown
```

### 5. 用代表 layer 方法归因 interval 内每个 layer 的性能数据

对 interval 内每个 layer：

1. 使用该 layer 自己的 Nsight/CUPTI/NVTX 性能数据作为 source latency。
2. 使用 interval 绑定的代表 layer 方法决定归因结构。
3. 将 kernel family、runtime API、memory activity、sync/scheduler evidence 映射到代表 layer 方法中的 process/cost_type。
4. 保持 layer 内同指标守恒。
5. 计算 layer 内百分比。

严禁使用代表 layer 的耗时替代当前 target layer 的耗时。

## 归因等级

### observed_fx_op

该 layer 自己有 FX process/op trace，并且性能样本能匹配到这些 ops 或 compatible kernel families。

适合严格结论。

### template_inferred

该 layer 没有自己的 FX trace，但属于某个 attribution interval，并使用该 interval 的 FX process/op template。

适合全局聚合，但必须显示 template source 和 confidence。

### fallback_component

没有可信 FX op template，只能按 measured component，例如 attn/mlp/outer，或粗粒度 kernel family 分配。

只能作为 heuristic attribution。

### unknown

无法可靠匹配到 process 或 component。

必须单独保留，不能强行摊派。

## Template 校验

模板迁移必须用性能侧证据校验。建议检查：

- layer 的 kernel families 是否和 template expected op families 兼容。
- GEMM-heavy process 是否对应 GEMM/GEMV kernel。
- softmax/attention process 是否对应 softmax/BMM/attention kernel。
- visual_process 是否对应 gather/index/cat/reduce/selection/Triton family。
- template interval 内相邻 layer 的 process/cost 分布是否平滑。
- template interval 内是否出现 backend 或 q/kv shape 突变。
- layer 内同指标归因是否守恒。

每个 layer 给出：

```text
template_validation_status = pass | warning | fail
template_confidence = high | medium | low
validation_reason
```

`fail` 的 layer 不应进入 strict global attribution，只能进入 fallback 或 unknown。

## 百分比计算

百分比仍然遵循 `README-DRAFT.md` 的规则：先 layer 内，后全局。

layer 内：

```text
process_metric_pct = process_metric_ms / layer_metric_ms
```

全局：

```text
global_process_metric_pct
  = sum(process_metric_ms over all attributed layers)
    / sum(layer_metric_ms over the same metric scope)
```

全局聚合必须拆出 evidence 来源：

```text
observed_ms
template_inferred_ms
fallback_ms
unknown_ms
```

这样可以区分“全局 process 开销”来自真实 FX trace、模板推断还是 fallback。

## 推荐输出表

### Attribution Type Map

```text
variant
phase
attribution_type_id
layer_interval
boundary_reason
template_source_layers
template_match_basis
expected_processes
expected_op_families
confidence
fallback_policy
```

### Layer Template Assignment

```text
variant
phase
layer
occurrence
q_len
kv_len
attribution_type_id
attribution_source
fx_template_source_layer
template_validation_status
template_confidence
validation_reason
```

### Layer-wise Process Attribution

```text
variant
phase
layer
occurrence
attribution_type_id
attribution_source
process
cost_type
metric
ms
pct_of_layer_metric
fx_op_families
matched_sample_families
attribution_method
fallback_reason
source_layer_metric_ms
```

### Global Process Aggregation

```text
variant
process
cost_type
metric
sum_ms
global_metric_pct
observed_ms
template_inferred_ms
fallback_ms
unknown_ms
observed_pct
template_inferred_pct
fallback_pct
unknown_pct
covered_layers
coverage_note
```

### Coverage and Risk

```text
variant
metric
total_layers
observed_fx_layers
template_inferred_layers
fallback_layers
unknown_layers
observed_ms
template_inferred_ms
fallback_ms
unknown_ms
strict_claim_allowed
risk_note
```

## 解释边界

- 部分 FX trace layers 只提供 process/op structure，不提供其它 layer 的耗时。
- 所有 layer 的耗时必须来自该 layer 自己的性能数据。
- template_inferred 归因可以用于全局趋势分析，但不等同于 observed FX op trace。
- 如果 template 和性能侧 kernel family 不匹配，必须降级为 fallback 或 unknown。
- 如果全局结论主要由 template_inferred 支撑，需要明确说明模板覆盖率和验证通过率。
- 如果全局结论主要由 fallback 支撑，只能称为 heuristic，不应作为严格 process-wise 证据。

## 推荐结论口径

严谨表述：

```text
We attribute all layer-wise performance samples using segment-level FX process templates.
Observed traced layers provide direct FX op evidence.
Untraced layers use interval-specific inferred templates validated by kernel-family evidence.
All reported process percentages are computed within each layer first and aggregated globally only after layer-level conservation checks.
```

中文表述：

```text
我们使用部分关键 FX trace layer 建立端到端区间级归因模板。
每个区间内的未 trace layer 使用自己的 Nsight/CUPTI/NVTX 性能数据，
并借助对应区间的 FX process/op template 进行归因。
报告区分 observed、template-inferred、fallback 和 unknown，
先在 layer 内计算百分比，再在全局聚合后计算同指标百分比。
```
