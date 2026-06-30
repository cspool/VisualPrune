# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer8`

本文件逐个解释该 layer 的 FX process，并按 observed shapes 手工绘制 tensor 轴/区域图。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder，主要有效输入是 hidden states `arg0_1`、attention mask `arg1_1` 和 position ids `arg2_1`。
- 为什么需要: 后续所有 ATen 节点都从这些 placeholder 读取 sampled runtime values。
- 怎么做/计算: #0 `arg0_1` 提供 `[1,624,4096]` hidden rows 给 #8 和 #90；#1 `arg1_1` 提供 `[1,1,624,624]` mask 给 #63；#2 `arg2_1` 提供 position ids 给 #36/#39；#3-#7 无后续用户。

```text
Runtime tensors
                                                       0                                      4096
                                                       ▲                                        ▲
hidden [1,624,4096]                            ──▶    ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘
                                                       0                                       624
                                                       ▲                                        ▲
mask [1,1,624,624]                             ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘
position_ids [1,624]                           ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对输入 hidden 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 需要稳定尺度的 hidden rows。
- 怎么做/计算: #8 转 fp32，#9 平方，#10 沿 Hidden 求均值，#11 加 epsilon，#12 取 `rsqrt`，#13 乘回 hidden，#14 转 fp16，#15 提供 `[4096]` weight 给 #16。

```text
RMSNorm over H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
input rows       #8                            ──▶    ┌────────────────────────────────────────┐
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       └────────────────────────────────────────┘
rms column       #9-#12                        ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
norm rows        #13-#15                       ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: #16-#34 将 normalized hidden 投影成 Q、K、V，并 reshape 为 `[1,32,624,128]`。
- 为什么需要: Q/K 生成 attention weights，V 提供被读取的 value 内容。
- 怎么做/计算: #16 应用 norm weight；Q 分支 #17-#20/#29-#30 做 view、weight `mm`、reshape 和 transpose；K 分支 #21-#24/#31-#32；V 分支 #25-#28/#33-#34。

```text
Projection paths from S=624 x H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
norm hidden                                      ──▶   ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
Q path #17-#30                                  ──▶   ┌────────────────────────────────────────┐
                                                       │ Q_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
K path #21-#32                                  ──▶   ┌────────────────────────────────────────┐
                                                       │ K_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
V path #25-#34                                  ──▶   ┌────────────────────────────────────────┐
                                                       │ V_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: #35-#54 对 Q/K 做 RoPE 位置旋转。
- 为什么需要: QK score 需要把 token 位置信息混入 Q/K。
- 怎么做/计算: #35/#38 取 cos/sin 表，#36/#39 按 `arg2_1` index，#37/#40 unsqueeze；#41-#47 处理 Q，#48-#54 处理 K，二者都把 Dh=128 切成 64+64，构造 `[-right,left]` 后计算 `x*cos + rotate*sin`。

```text
RoPE half split on Dh=128
                                                       0                       64              128
                                                       ▲                        ▲                ▲
input Q/K                                        ──▶   ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half                                     ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
rotated Q/K                                     ──▶    ┌────────────────────────────────────────┐
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#66 计算标准 QK attention weights；该 stage 没有 Visual clear/fold。
- 为什么需要: 后续 attention output 和 Visual value-aware process 都需要最后的 normalized attention weights。
- 怎么做/计算: #55 转置 K；#56/#57 得到 Q 的 `[32,624,128]`；#58/#59 得到 K^T 的 `[32,128,624]`；#60 `bmm`；#61 view `[1,32,624,624]`；#62 scale；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone。

```text
Q_seq x K_seq attention weights
                                                       K_seq 0                                  624
                                                       ▲                                        ▲
QK scores #60-#63                              ──▶    ┌────────────────────────────────────────┐
                                                       │ QK_SCORE_MATRIX                       │
                                                       │ QK_SCORE_MATRIX                       │
                                                       └────────────────────────────────────────┘
softmax weights #64-#66                        ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       └────────────────────────────────────────┘
```

## Attention-weighted V and hidden reshape

- 是什么: #67-#75 用 attention weights 加权 V 并合并 heads。
- 为什么需要: `weights @ V` 产生 attention 子层的 context hidden。
- 怎么做/计算: #67/#68 将 weights view 为 `[32,624,624]`；#69/#70 将 V view 为 `[32,624,128]`；#71 `bmm` 得到 context；#72 view；#73 transpose；#74 clone；#75 view 为 `[1,624,4096]`。

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

- 是什么: #76-#85 构造 Visual span `[35,611)` 上的 value-aware delta，没有后续 norm/threshold/nonzero 决策。
- 为什么需要: 它把最后输出 token 的 hidden row 与最后 query attention 加权得到的 Visual value rows 对齐比较，形成 Visual token 贡献差异。
- 怎么做/计算: #76 取 attention output 最后一行；#77 取最后 query 的 attention weights；#78 unsqueeze；#79 与 V head tensor 相乘得到 weighted V；#80 permute 到 token-major；#81 clone；#82 view 为 `[1,624,4096]`；#83 slice Visual rows `[35,611)`；#84 broadcast 最后一行；#85 sub 得到 delta rows。#85 在 FX DAG 中没有 downstream users。

```text
Visual token delta, V=576 rows from token [35,611), H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
last output row #76,#84                        ──▶    ┌────────────────────────────────────────┐
                                                       │ LAST_OUTPUT_REFERENCE_ROW             │
                                                       └────────────────────────────────────────┘
weighted V rows #77-#83                        ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       └────────────────────────────────────────┘
delta rows #85                                 ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       └────────────────────────────────────────┘
decision output                                ──▶    [NO_MATERIALIZED_SELECTION]
```

## Attention output projection and residual

- 是什么: #86-#90 对 attention hidden 做 output projection 并与 `arg0_1` 相加。
- 为什么需要: 多头 context 需要回到 hidden 表示，并保留 residual 路径。
- 怎么做/计算: #86 view `[624,4096]`；#87 取权重；#88 `mm`；#89 reshape `[1,624,4096]`；#90 residual add。

```text
Output projection on Q=624 x H=4096
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
- 为什么需要: MLP projection 前需要稳定 token hidden 尺度。
- 怎么做/计算: #91 转 fp32；#92 平方；#93 Hidden 均值；#94 加 epsilon；#95 `rsqrt`；#96 缩放；#97 转 fp16；#98 取 weight；#99 逐元素乘；#100 view。

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

- 是什么: #101-#114 是 gated MLP 与 final residual add。
- 为什么需要: MLP 在每个 token 的 channel 维上做非线性扩展和压回。
- 怎么做/计算: #101/#102 gate projection，#103 reshape，#104 `silu`；#105-#108 up projection；#109 做 gate/up 逐元素乘；#110 view intermediate；#111/#112 down projection；#113 reshape；#114 与 #90 residual 相加。

```text
MLP channel transform H=4096 -> I=11008 -> H=4096
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

- 是什么: #115 打包返回 layer 输出。
- 为什么需要: 下一层需要 hidden states 与 K/V cache；该 layer 的 Visual delta 没有进入返回值。
- 怎么做/计算: #115 返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`。

```text
Output tuple

hidden_states      #114 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
