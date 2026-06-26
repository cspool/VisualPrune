# VisiPrune-Centric Dispatch Filtering Rules

这份规则只服务一个目的：用 dispatch trace 深入理解 **VisiPrune 动态
token 剪枝强相关的 layers**。它不是通用的 ATen 全量采集方案，也不是按
latency/outlier 聚类寻找“慢算子”的入口规则。

旧的泛化规则已经删除。新的入口逻辑是：

```text
selection_trace + layer_trace 先确定 VisiPrune 决策/生效位置
dispatch trace 只围绕这些位置采集真实 ATen 算子和输入 shape
latency/聚类只用于采集后的辅助解释，不用于决定先采哪些 layer
```

## 输入

主输入：

- `algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/layer_trace.csv`
- `algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/selection_trace.csv`

`layer_trace.csv` 给出每次 forward、每层的 `q_len/past_len/kv_len`，用于识别
token 长度变化在哪里真正生效。

`selection_trace.csv` 给出 `value_aware_token_selection` 的调用结果，用于识别
哪些 layer 真正参与 VisiPrune 的 middle selection / deep exit 控制流。

## 基本标识

每个可采集事件用下面两个字段定位：

```text
input_id = forward_id
layer_id = layer_idx
event_id = input{input_id}_layer{layer_id}
```

当前 32-token fresh-forward trace 中：

- `input_id=1` 是 prefill。
- `input_id=2..32` 是 decode forwards。
- `layer_id=0..31` 是 LLaMA decoder layers。

## 过滤原则

### P0: 必采，VisiPrune 决策层

这些 layer 直接执行 VisiPrune 的动态 token 决策逻辑，dispatch 的首要目标是看
这些 layer 内真实发生了哪些 ATen op、输入 shape 如何变化。

| phase | input_id | layer_id | VisiPrune 作用 | 当前 trace 现象 |
|---|---:|---|---|---|
| prefill | 1 | 7-17 | middle selection probe | 调用 selection，但返回 `none` |
| prefill | 1 | 18 | middle token selection | 选出 10 个 visual tokens |
| prefill | 1 | 19-24 | deep exit check | deep check 为 false |
| prefill | 1 | 25 | deep exit check | 第一次 true |
| prefill | 1 | 26 | deep exit check | false，保持已累积 exit 状态 |
| prefill | 1 | 27 | deep exit check | 第二次 true，触发后续视觉 token 移除 |

保留规则：

- `layer_id=7-18` 不折叠。虽然前面很多层 shape 相同，但每层 selection 结果
  可能不同，且 layer 18 是实际选择点。
- `layer_id=19-27` 不折叠。它们共同构成 deep exit 链，`exit_indicator`
  的状态和 true/false 结果是 VisiPrune 机制本身。

### P1: 必采，剪枝生效边界层

这些 layer 未必做出 selection 决策，但它们是 token schedule 从一个状态切到
另一个状态的边界。dispatch 需要证明真实算子输入也随之变化。

| phase | input_id | layer_id | 边界含义 | 当前 trace 现象 |
|---|---:|---:|---|---|
| prefill | 1 | 18 | full visual 状态下最后一个 selection 层 | `q_len=624` |
| prefill | 1 | 19 | middle pruning 后第一个执行层 | `q_len=58` |
| prefill | 1 | 27 | deep exit 决策链最后一个触发层 | `q_len=58` |
| prefill | 1 | 28 | deep visual removal 后第一个执行层 | `q_len=48` |

保留规则：

- `layer_id=19` 必采，因为它是 `624 -> 58` 之后第一个真实执行层。
- `layer_id=28` 必采，因为它是 `58 -> 48` 之后第一个真实执行层。
- `layer_id=18/27` 作为边界前一层保留，用于比较剪枝前后算子输入。

### P2: 次级采集，shallow mask 相关层

shallow 阶段不是 token 删除点，但它属于 VisiPrune 的早期控制逻辑。如果目标是
完整解释 VisiPrune，而不是只解释物理 token 数变化，需要保留代表性事件。

默认保留：

| phase | input_id | layer_id | 作用 |
|---|---:|---:|---|
| prefill | 1 | 0 | shallow 阶段入口 |
| prefill | 1 | 5 | shallow 阶段末层 |
| prefill | 1 | 6 | shallow 与 middle selection 之间的空白边界 |

如果后续确认 shallow mask 的实现路径需要逐层展开，再扩展到 `layer_id=0-5`。

### P3: 次级采集，decode 中剪枝效果的代表层

decode 本身不再执行 middle/deep selection，但它继承 prefill 后的 KV cache
长度，因此可用于解释 VisiPrune 对端到端生成阶段的影响。

默认只保留每个 cache regime 的首尾 decode step，而不是 31 个 decode step 全采：

| regime | input_id | layer_id | 说明 |
|---|---|---|---|
| full-cache affected | 2, 32 | 18 | 仍带 full visual 历史的最后层代表 |
| middle-pruned cache | 2, 32 | 19, 27 | `624 -> 58` 后的 cache 代表 |
| deep-removed cache | 2, 32 | 28, 31 | `58 -> 48` 后的 cache 代表 |

这里的 decode 采集是“剪枝效果验证”，不是“剪枝决策解释”。如果只研究
VisiPrune 的决策机制，可以先跳过 P3。

## 不作为入口规则的内容

下面这些信息可以用于 dispatch 之后的分析，但不应该决定初始采集对象：

- latency outlier
- ATen op 数量聚类
- 相同 shape 的通用去重
- dense Transformer 的完整层覆盖
- “不同权重必须全采”的通用 profiler 规则

原因是本任务的目标不是生成完整后端运行 profile，而是解释 VisiPrune 创新点
相关的动态 token 剪枝路径。

## 默认采集清单

默认 dispatch manifest 应包含：

```text
P0:
  input1_layer7 ... input1_layer18
  input1_layer19 ... input1_layer27

P1:
  input1_layer18
  input1_layer19
  input1_layer27
  input1_layer28

P2:
  input1_layer0
  input1_layer5
  input1_layer6

P3:
  input2_layer18
  input2_layer19
  input2_layer27
  input2_layer28
  input2_layer31
  input32_layer18
  input32_layer19
  input32_layer27
  input32_layer28
  input32_layer31
```

去重后，prefill 默认采集：

```text
input1_layer0
input1_layer5
input1_layer6
input1_layer7 ... input1_layer19
input1_layer20 ... input1_layer28
```

即：

```text
prefill layer_id = 0, 5, 6, 7-28
```

decode 默认采集：

```text
input_id = 2, 32
layer_id = 18, 19, 27, 28, 31
```

## Manifest 字段

dispatch 前应先生成一个 manifest，字段建议如下：

```text
event_id
priority              # P0/P1/P2/P3
input_id
layer_id
phase                 # prefill/decode
q_len
past_len
kv_len
visipruner_role       # middle_probe/middle_select/deep_check/boundary/shallow/decode_effect
selection_result      # none/tensor/true/false/empty
token_state           # full_visual/middle_pruned/deep_removed
reason
repeat_group
repeat_members
keep_default
```

`repeat_group` 只用于解释重复关系，不用于默认删除 P0/P1 事件。

## 重复关系说明

P0/P1 的重复不能按普通 shape 去重：

- `layer_id=7-17` 都是 middle selection probe，但它们共同说明为什么直到
  layer 18 才真正选择 visual tokens。
- `layer_id=19-27` 都处于 `q_len=58`，但 deep exit 的 false/true 序列就是
  VisiPrune 的控制流，不能只留一个代表层。
- `layer_id=28-31` 是 deep removal 后的稳定执行区。默认只保留 layer 28；
  只有需要对比稳定 post-prune 执行时才额外保留 layer 31。
- decode forward 大量重复。默认只保留第一个和最后一个 decode step，用于看
  cache 增长前后在三种 pruning regime 下的输入变化。

## 推荐执行顺序

1. 从 `selection_trace.csv` 读出所有 selection/deep-check event。
2. 从 `layer_trace.csv` 找出 `q_len/kv_len` 发生状态变化的边界。
3. 合并 P0/P1/P2/P3，生成 dispatch manifest。
4. 只对 manifest 中 `keep_default=true` 的事件运行 `TorchDispatchMode`。
5. dispatch 之后再用 latency、ATen op taxonomy、shape 聚类解释这些事件内部
   的差异。

这样得到的 trace 是 VisiPrune-centric 的：它先回答“哪些 layer 和动态 token
剪枝有关”，再回答“这些 layer 内部真实发生了什么算子”。
