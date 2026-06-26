# Fresh Forward VisiPrune Full 32-Token Trace

本目录是一次真实 forward-driven 的 VisiPrune 负载分析结果。

运行配置：

- model: `liuhaotian/llava-v1.5-7b`
- config: `visipruner-full`
- pruning modes: `shallow + middle + deep`
- max new tokens: `32`
- prompt: `Describe the image briefly.`
- image: `/workspace/VisiPrune/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg`
- trace type: algorithmic / theoretical workload trace

这里的结果不是 Nsight/CUDA kernel trace，也不是后端编译结果。它来自真实
`model.generate()`，通过 wrapper 记录每次 forward、每层 shape/token schedule
和 VisiPrune selection 事件，然后按公式计算理论 FLOPs。

## 文件说明

- `algorithmic_trace.json`
  总包 JSON。包含请求信息、模型维度、forward events、layer events、
  selection events 和 FLOP summary。建议先看这个文件。

- `layer_trace.csv`
  每个 model forward、每个 decoder layer 的 shape/token 长度记录。
  用它看 VisiPrune 在每层实际用了多长序列。

- `selection_trace.csv`
  VisiPrune `value_aware_token_selection()` 的调用记录。用它看 middle token
  selection 和 deep exit 是在哪些层发生的。

- `operator_flops.csv`
  基于 `layer_trace.csv` 和模型配置展开出的理论 operator FLOPs。用它看
  `q_proj/k_proj/v_proj/attention/mlp/clip/projector/lm_head` 各自的 FLOPs。

## algorithmic_trace.json 如何读

关键字段：

- `analysis_type`
  当前值为 `forward_driven_algorithmic_trace`，表示它来自真实 forward，而不是
  e2 tracker reconstruction。

- `request`
  请求级信息。当前结果：
  - prompt tokens before image expansion: `49`
  - image tokens: `576`
  - image size: `[1024, 768]`
  - output token count: `33`

- `model_dims`
  LLaVA/LLaMA backbone 的结构参数：
  - hidden size: `4096`
  - intermediate size: `11008`
  - layers: `32`
  - attention heads: `32`
  - key/value heads: `32`
  - vocab size: `32000`

- `forward_events`
  每次 `model.forward` 的记录。当前有 `32` 次：
  - `1` 次 prefill
  - `31` 次 decode

- `layer_events`
  每层 decoder forward 的记录。当前有 `1024` 条：
  - `32` 层 x `32` 次 model forward

- `selection_events`
  VisiPrune selection/deep-exit 的记录。当前有 `21` 条。

- `summary`
  理论 FLOPs 汇总。

## layer_trace.csv 如何读

每一行表示：

```text
+一次 model forward + 一个 decoder layer
```

重要字段：

- `forward_id`: 第几次 model forward。
- `phase`: `prefill` 或 `decode`。
- `layer_idx`: decoder layer 编号，`0-31`。
- `q_len`: 当前层 query token 数。
- `past_len`: 当前层已有 KV cache 长度。
- `kv_len`: 当前 attention 的 KV 长度，等于 `past_len + q_len`。
- `hidden_shape_in`: 进入该层的 hidden state shape。
- `hidden_shape_out`: 离开该层的 hidden state shape。
- `important_visual_tokens_in/out`: VisiPrune 选中的视觉 token 数。
- `exit_indicator_in/out`: deep exit 累计状态。

当前 prefill 的核心 schedule：

| layer range | q_len | 含义 |
|---|---:|---|
| `0-18` | `624` | 原始 multimodal sequence，包含 `576` 个视觉 token |
| `19-27` | `58` | middle pruning 后，只保留 `10` 个视觉 token |
| `28-31` | `48` | deep exit 后，视觉 token 全移除 |

长度关系：

```text
624 = 35 prefix tokens + 576 visual tokens + 13 suffix tokens
58  = 35 prefix tokens + 10 selected visual tokens + 13 suffix tokens
48  = 35 prefix tokens + 13 suffix tokens
```

decode 阶段每次 `q_len=1`，但不同层的 `past_len/kv_len` 跟 prefill 后保留的
token 数有关。

## selection_trace.csv 如何读

这个文件记录 VisiPrune 的核心动态决策。

重要字段：

- `layer_idx`: selection/deep-exit 发生在哪一层。
- `result_type`:
  - `none`: 本层没有选出重要视觉 token。
  - `tensor`: middle selection 返回了视觉 token index。
  - `bool`: deep exit check 返回 True/False。
- `selected_visual_token_count`: middle selection 选中的视觉 token 数。
- `selected_visual_token_indices`: 选中的原始视觉 token 位置。
- `deep_exit`: deep 阶段是否判断可以移除剩余视觉 token。
- `attn_weights_shape`: selection 使用的 attention weights shape。
- `value_states_shape`: selection 使用的 value states shape。

当前结果：

- layer `7-17`: `result_type=none`
- layer `18`: `result_type=tensor`，选中 `10` 个视觉 token
- selected indices:
  `[109, 296, 370, 376, 390, 437, 490, 499, 504, 511]`
- layer `19-24`: deep exit 为 `False`
- layer `25`: deep exit 为 `True`
- layer `26`: deep exit 为 `False`
- layer `27`: deep exit 为 `True`
- layer `28` 起视觉 token 被完全移除

## operator_flops.csv 如何读

这个文件不是 profiler 输出，而是公式展开结果。

每一行表示一个理论 operator 项，例如：

- CLIP vision:
  - `clip_patch_embed`
  - `clip_qkv_proj`
  - `clip_attn_qk_matmul`
  - `clip_attn_sv_matmul`
  - `clip_mlp_fc1`
  - `clip_mlp_fc2`
- multimodal projector:
  - `mm_projector_mlp2x_gelu`
- LLaMA decoder:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `attn_qk_matmul`
  - `attn_sv_matmul`
  - `o_proj`
  - `mlp_gate_proj`
  - `mlp_up_proj`
  - `mlp_down_proj`
- VisiPrune bookkeeping:
  - `visipruner_value_selection_aux`
- LM head:
  - `lm_head_actual_model_path`
  - `lm_head_ideal_last_token_only`

重要字段：

- `phase`: `vision`、`vision_projector`、`prefill`、`decode`。
- `layer_idx`: 对应层编号；vision/projector/lm_head 可能是 `-1` 或无 forward id。
- `q_len`: 当前公式使用的 query length。
- `kv_len`: 当前公式使用的 KV length。
- `op_name`: operator 名称。
- `formula`: 使用的理论公式。
- `flops`: 该项 FLOPs。乘加按 `2 FLOPs` 计算。

例子：

```text
q_proj: 2*B*Q*D*D
attn_qk_matmul: 2*B*H*Q*KV*HEAD_DIM
mlp_gate_proj: 2*B*Q*D*FFN
```

## 当前 FLOPs 汇总

来自 `algorithmic_trace.json -> summary`：

| phase | FLOPs |
|---|---:|
| prefill | `5,222,644,449,280` |
| decode | `424,409,661,440` |
| vision | `381,918,216,192` |
| vision_projector | `24,159,191,040` |

总计：

- full VLM actual model path: `6,044,742,909,952` FLOPs
- ideal last-token-only lm_head: `6,032,422,141,952` FLOPs

主要 FLOPs 来源：

| op | FLOPs |
|---|---:|
| `mlp_gate_proj` | `1,222,987,743,232` |
| `mlp_up_proj` | `1,222,987,743,232` |
| `mlp_down_proj` | `1,222,987,743,232` |
| `q_proj` | `455,065,206,784` |
| `k_proj` | `455,065,206,784` |
| `v_proj` | `455,065,206,784` |
| `o_proj` | `455,065,206,784` |

## 如何和 dense 对比

dense-eager fresh trace 位于：

`../fresh_forward_dense_eager_32tok/`

VisiPrune vs dense 的对比文件位于：

`../fresh_visipruner_vs_dense_32tok.json`

当前对比结果：

- dense-eager full VLM actual model path: `9,275,895,808,000` FLOPs
- VisiPrune full VLM actual model path: `6,044,742,909,952` FLOPs
- saved FLOPs: `3,231,152,898,048`
- saved percentage: `34.83%`

## 复现命令

从 workload_analysis 根目录运行：

```bash
GPU=1 TOKENS=32 /workspace/VisiPrune/workload_analysis/algorithmic_trace/runners/run_full_forward.sh
```

只复现本目录的 VisiPrune trace：

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/algorithmic_trace/tools/visipruner_algorithmic_trace.py \
  --config visipruner-full \
  --max-new-tokens 32 \
  --gpu 1 \
  --tag fresh_forward_visipruner_full_32tok
```
