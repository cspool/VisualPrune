# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer27`

本文件逐个解释该 layer 的 FX process，并手工绘制 tensor 轴/区域图。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder；主要计算输入是 `arg0_1` hidden states、`arg1_1` attention mask、`arg2_1` position ids。
- 为什么需要: 本 layer 是 58-token 路径，后续 norm、attention、probe delta、MLP 和 output 都从这些输入读取固定值。
- 怎么做/计算: #0 `arg0_1` 提供 `[1,58,4096]` hidden 给 #8 和 #91；#1 `arg1_1` 给 #63；#2 `arg2_1` 给 #36/#39；output tuple 后面还直接返回 `arg6_1`。

```text
Runtime tensors
                                                       0                                      4096
                                                       ▲                                        ▲
hidden [1,58,4096]                             ──▶    ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘
mask [1,1,58,58]                               ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘
position ids [1,58]                            ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对 `arg0_1` 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 前需要稳定 hidden 尺度。
- 怎么做/计算: #8 转 fp32；#9 平方；#10 沿 Hidden 求均值；#11 加 epsilon；#12 `rsqrt`；#13 乘回 hidden；#14 转 fp16；#15 提供 norm weight，供 #16 使用。

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

- 是什么: #16-#34 将 normalized hidden 投影为 Q/K/V，并 reshape 为 `[1,32,58,128]`。
- 为什么需要: Q/K 形成 attention weights，V 被 attention output 和 probe delta 读取。
- 怎么做/计算: #16 应用 norm weight；Q 分支 #17-#20/#29-#30；K 分支 #21-#24/#31-#32；V 分支 #25-#28/#33-#34。每条分支都做 view、`mm`、reshape、transpose。

```text
Projection paths from S=58 x H=4096
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
- 为什么需要: 58-token attention 仍需要位置编码后的 Q/K。
- 怎么做/计算: #35/#38 读取 cos/sin 表；#36/#39 按 `arg2_1` index；#37/#40 unsqueeze；Q 的 #41-#47 和 K 的 #48-#54 都切分 Dh=128 为两半，做 `x*cos + concat(-right,left)*sin`。

```text
RoPE over Dh=128
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

- 是什么: #55-#66 计算标准 `[Q=58,K=58]` attention weights。
- 为什么需要: attention output 和 probe delta 都使用最后 query 的 weights。
- 怎么做/计算: #55 转置 K；#56/#57 view Q 为 `[32,58,128]`；#58/#59 view K^T 为 `[32,128,58]`；#60 `bmm`；#61 view `[1,32,58,58]`；#62 scale；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone。

```text
Attention weights Q=58 x K=58
                                                       K_seq 0                                   58
                                                       ▲                                        ▲
QK scores                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ QK_SCORE_MATRIX                       │
                                                       └────────────────────────────────────────┘
softmax weights                                ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       └────────────────────────────────────────┘
```

## Attention-weighted V and hidden reshape

- 是什么: #67-#75 计算 `weights @ V` 并合并 heads。
- 为什么需要: 产生 attention 子层的 `[1,58,4096]` hidden。
- 怎么做/计算: #67/#68 view weights 为 `[32,58,58]`；#69/#70 view V 为 `[32,58,128]`；#71 `bmm`；#72 view；#73 transpose；#74 clone；#75 view 为 `[1,58,4096]`。

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

- 是什么: #76-#86 构造 probe token `[35,45)` 上的 value-aware delta；没有 norm/threshold/nonzero，因此没有物化选择。
- 为什么需要: 该过程只检查一小段 probe Visual tokens，而不是整个 Visual span，用最后 output row 与 probe weighted-value rows 做差。
- 怎么做/计算: #76 取 attention output 最后一行；#77 取最后 query attention weights；#78 unsqueeze；#79 与 V head tensor 相乘；#80 permute 到 token-major；#81 clone；#82 view 为 `[1,58,4096]`；#83 读取 probe index 常量；#84 `index.Tensor` 选出 token `[35,45)` 的 `P=10` rows；#85 broadcast 最后一行；#86 sub 得到 `[1,10,4096]` probe delta。#86 无 downstream users。

```text
Probe delta over P=10 rows, token [35,45), H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
last output row                                ──▶    ┌────────────────────────────────────────┐
                                                       │ LAST_OUTPUT_REFERENCE_ROW             │
                                                       └────────────────────────────────────────┘
selected probes                                ──▶    ┌────────────────────────────────────────┐
                                                       │ SELECTED_PROBE_ROWS                   │
                                                       └────────────────────────────────────────┘
delta rows                                     ──▶    ┌────────────────────────────────────────┐
                                                       │ PROBE_DELTA_COMPARE                   │
                                                       └────────────────────────────────────────┘
decision output                                ──▶    [NO_MATERIALIZED_SELECTION]
```

## Attention output projection and residual

- 是什么: #87-#91 做 output projection 和 residual add。
- 为什么需要: attention hidden 需要回到模型 hidden 表示，并保留输入 residual。
- 怎么做/计算: #87 view `[58,4096]`；#88 取权重；#89 `mm`；#90 reshape `[1,58,4096]`；#91 与 `arg0_1` 相加。

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

- 是什么: #92-#101 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP projection 需要稳定 hidden 尺度。
- 怎么做/计算: #92 转 fp32；#93 平方；#94 Hidden 均值；#95 加 epsilon；#96 `rsqrt`；#97 缩放；#98 转 fp16；#99 读 norm weight；#100 乘权重；#101 view。

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

- 是什么: #102-#115 是 gated MLP 和 final residual。
- 为什么需要: MLP 在每个 token channel 维上做非线性扩展和压回。
- 怎么做/计算: #102/#103 gate projection；#104 reshape；#105 `silu`；#106-#109 up projection；#110 逐元素乘；#111 view；#112/#113 down projection；#114 reshape；#115 与 #91 residual 相加。

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

- 是什么: #116 打包 layer 输出。
- 为什么需要: 后续流程需要 hidden states、K/V cache 和已有 Visual/control 输入。
- 怎么做/计算: #116 返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, arg6_1, 0)`；#86 probe delta 不进入 output tuple。

```text
Output tuple

hidden_states      #115 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=58,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=58,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=58,Dh=128]
visual output      #6 arg6_1                     ──▶   [PASSTHROUGH_VISUAL_INPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
