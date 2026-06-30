# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer13`

本文件按本 layer 的 FX ops 逐个解释 process，并手工给出 tensor 轴/区域图。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder，主要有效输入是 `arg0_1` hidden states、`arg1_1` attention mask、`arg2_1` position ids。
- 为什么需要: 后续所有 ATen 计算都从这些 sampled runtime values 开始。
- 怎么做/计算: #0 `arg0_1` 提供 `[1,624,4096]` hidden rows 给 #8 和 #90；#1 `arg1_1` 提供 `[1,1,624,624]` mask 给 #63；#2 `arg2_1` 供 #36/#39 做 RoPE cos/sin index；#3-#7 无后续用户。

```text
Runtime tensors
                                                       0                                      4096
                                                       ▲                                        ▲
hidden [1,624,4096]                            ──▶    ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘
mask [1,1,624,624]                             ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘
position ids [1,624]                           ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 是 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 需要尺度稳定的 token hidden。
- 怎么做/计算: #8 转 fp32；#9 平方；#10 沿 Hidden 求均值；#11 加 `1e-05`；#12 `rsqrt`；#13 缩放 hidden；#14 转 fp16；#15 提供 norm weight，后续 #16 应用。

```text
RMSNorm over H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
input rows                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       └────────────────────────────────────────┘
rms column                                     ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
normalized rows                                ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: #16-#34 将 normalized hidden 投影为 Q/K/V，并转成 `[1,32,624,128]`。
- 为什么需要: Q/K 产生 attention weights，V 提供被读取内容。
- 怎么做/计算: #16 应用 norm weight；Q 分支 #17-#20/#29-#30 做 view、`mm`、reshape、transpose；K 分支 #21-#24/#31-#32；V 分支 #25-#28/#33-#34。

```text
Projection and head split
                                                       0                                      4096
                                                       ▲                                        ▲
norm hidden                                    ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
Q path                                        ──▶     ┌────────────────────────────────────────┐
                                                       │ Q_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
K path                                        ──▶     ┌────────────────────────────────────────┐
                                                       │ K_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
V path                                        ──▶     ┌────────────────────────────────────────┐
                                                       │ V_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: #35-#54 对 Q/K 做 RoPE 位置旋转。
- 为什么需要: QK score 需要 token 位置信息。
- 怎么做/计算: #35/#38 读取 cos/sin；#36/#39 用 `arg2_1` index；#37/#40 unsqueeze；#41-#47 旋转 Q，#48-#54 旋转 K。旋转过程把 Dh=128 切为两个 64 宽 half，构造 `concat(-right,left)`，再做 `x*cos + rotate*sin`。

```text
RoPE split on Dh=128
                                                       0                       64              128
                                                       ▲                        ▲                ▲
input Q/K                                      ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half                                   ──▶     ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
rotated Q/K                                   ──▶     ┌────────────────────────────────────────┐
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#66 计算标准 attention weights，没有 `fill_`、`sum` 或 `copy_`。
- 为什么需要: attention output 和后面的 Visual value-aware process 都要使用这些 normalized weights。
- 怎么做/计算: #55 转置 K；#56/#57 view Q 为 `[32,624,128]`；#58/#59 view K^T 为 `[32,128,624]`；#60 `bmm`；#61 view `[1,32,624,624]`；#62 scale；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone。

```text
Q_seq x K_seq attention matrix
                                                       K_seq 0                                  624
                                                       ▲                                        ▲
QK scores                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ QK_SCORE_MATRIX                       │
                                                       │ QK_SCORE_MATRIX                       │
                                                       └────────────────────────────────────────┘
softmax weights                                ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       └────────────────────────────────────────┘
```

## Attention-weighted V and hidden reshape

- 是什么: #67-#75 计算 `weights @ V` 并合并 heads。
- 为什么需要: 这一步产生 attention 子层真正的 context hidden。
- 怎么做/计算: #67/#68 将 weights view 为 `[32,624,624]`；#69/#70 将 V view 为 `[32,624,128]`；#71 `bmm`；#72 view；#73 transpose；#74 clone；#75 view 为 `[1,624,4096]`。

```text
Attention output contraction
                                                       0                                       128
                                                       ▲                                        ▲
weights [Q,K]                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_WEIGHT_MATRIX               │
                                                       └────────────────────────────────────────┘
values [K,Dh]                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ VALUE_ROWS                            │
                                                       └────────────────────────────────────────┘
context [Q,Dh]                                 ──▶    ┌────────────────────────────────────────┐
                                                       │ CONTEXT_ROWS                          │
                                                       └────────────────────────────────────────┘
merged [Q,H]                                   ──▶    ┌────────────────────────────────────────┐
                                                       │ MERGED_HEAD_ROWS                      │
                                                       └────────────────────────────────────────┘
```

## Visual-related value-aware process

- 是什么: #76-#85 构造 Visual token span `[35,611)` 的 value-aware delta；该 DAG 没有后续 selection 决策节点。
- 为什么需要: 它把最后 token output 与最后 query attention 加权得到的 Visual value rows 对齐相减，形成 Visual contribution difference。
- 怎么做/计算: #76 取 attention output 最后一行；#77 取最后 query attention row；#78 unsqueeze；#79 与 V head tensor 相乘；#80 permute 到 token-major；#81 clone；#82 view 为 `[1,624,4096]`；#83 slice `[35,611)`；#84 broadcast 最后一行；#85 sub 得到 delta rows，且 #85 无 downstream users。

```text
Visual delta over V=576 rows, H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
last output row                                ──▶    ┌────────────────────────────────────────┐
                                                       │ LAST_OUTPUT_REFERENCE_ROW             │
                                                       └────────────────────────────────────────┘
weighted V rows                                ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       └────────────────────────────────────────┘
delta rows                                     ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       └────────────────────────────────────────┘
decision output                                ──▶    [NO_MATERIALIZED_SELECTION]
```

## Attention output projection and residual

- 是什么: #86-#90 做 attention output projection 和 residual add。
- 为什么需要: 多头 context 需要回到 hidden 表示，并加回输入 residual。
- 怎么做/计算: #86 view 为 `[624,4096]`；#87 读权重；#88 `mm`；#89 reshape；#90 与 `arg0_1` 相加。

```text
Output projection
                                                       0                                      4096
                                                       ▲                                        ▲
attention rows                                  ──▶   ┌────────────────────────────────────────┐
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       └────────────────────────────────────────┘
projected rows                                  ──▶   ┌────────────────────────────────────────┐
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       └────────────────────────────────────────┘
residual rows                                   ──▶   ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: #91-#100 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP projection 需要尺度稳定的 hidden。
- 怎么做/计算: #91 转 fp32；#92 平方；#93 Hidden 均值；#94 加 epsilon；#95 `rsqrt`；#96 缩放；#97 转 fp16；#98 读 norm weight；#99 乘权重；#100 view。

```text
Post-attention RMSNorm
                                                       0                                      4096
                                                       ▲                                        ▲
residual rows                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       └────────────────────────────────────────┘
rms column                                     ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
mlp input rows                                 ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: #101-#114 是 gated MLP 和 final residual。
- 为什么需要: MLP 在 channel 维上提供非线性变换。
- 怎么做/计算: #101/#102 gate projection，#103 reshape，#104 `silu`；#105-#108 up projection；#109 gate/up 相乘；#110 view；#111/#112 down projection；#113 reshape；#114 residual add。

```text
MLP H=4096 -> I=11008 -> H=4096
                                                       0                                      11008
                                                       ▲                                        ▲
mlp input                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
gate/up rows                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ GATE_AND_UP_ROWS                      │
                                                       └────────────────────────────────────────┘
activated rows                                ──▶    ┌────────────────────────────────────────┐
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       └────────────────────────────────────────┘
final hidden                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       └────────────────────────────────────────┘
```

## Layer output

- 是什么: #115 打包输出。
- 为什么需要: 后续层需要 hidden states 和 K/V cache。
- 怎么做/计算: #115 返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`；Visual delta 不进入 output tuple。

```text
Output tuple

hidden_states      #114 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
