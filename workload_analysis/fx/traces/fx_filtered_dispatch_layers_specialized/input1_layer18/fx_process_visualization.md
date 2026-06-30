# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer18`

本文件按本 layer 的 FX ops 逐个解释 process，并手工给出 tensor 轴/区域图。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder，主要有效输入是 `arg0_1` hidden states、`arg1_1` attention mask、`arg2_1` position ids。
- 为什么需要: 后续 norm、Q/K/V、RoPE、attention、Visual token selection、MLP 和 output 都依赖这些采样输入。
- 怎么做/计算: #0 `arg0_1` 提供 `[1,624,4096]` hidden rows 给 #8 和 #99；#1 `arg1_1` 进入 #63 mask add；#2 `arg2_1` 进入 #36/#39 position lookup；#3-#7 在该图中无后续用户。

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

- 是什么: #8-#15 对输入 hidden 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 前需要稳定每个 token 的 hidden 尺度。
- 怎么做/计算: #8 转 fp32；#9 平方；#10 沿 Hidden 求均值；#11 加 epsilon；#12 `rsqrt`；#13 缩放 hidden；#14 转 fp16；#15 提供 `[4096]` weight，后续 #16 应用。

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

- 是什么: #16-#34 将 normalized hidden 投影为 Q、K、V，并 reshape 为 `[1,32,624,128]`。
- 为什么需要: Q/K 用于 attention score，V 用于被权重读取，也被后续 Visual value-aware selection 使用。
- 怎么做/计算: #16 应用 norm weight；Q 分支 #17-#20/#29-#30，K 分支 #21-#24/#31-#32，V 分支 #25-#28/#33-#34，三者都执行 view、`mm`、reshape、transpose。

```text
Projection paths
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
- 为什么需要: QK score 需要包含 token 位置信息。
- 怎么做/计算: #35/#38 读取 cos/sin 表；#36/#39 用 `arg2_1` index；#37/#40 unsqueeze；Q 路径 #41-#47 和 K 路径 #48-#54 都按 Dh=128 的两半构造 rotate-half，并计算 `x*cos + rotate_half(x)*sin`。

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

- 是什么: #55-#66 计算标准 attention weights；没有 attention-score-side fill/fold/copy。
- 为什么需要: attention output 和 Visual value-aware selection 都要读取这些 normalized weights。
- 怎么做/计算: #55 转置 K；#56/#57 得到 Q `[32,624,128]`；#58/#59 得到 K^T `[32,128,624]`；#60 `bmm`；#61 view `[1,32,624,624]`；#62 scale；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone。

```text
Q_seq x K_seq attention weights
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
- 为什么需要: 这一步产生 attention 子层的 `[1,624,4096]` context hidden。
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

- 是什么: #76-#94 是带选择输出的 Visual value-aware process：先构造 Visual span `[35,611)` 上的 delta，再按 L2 norm 与阈值选出 token index。
- 为什么需要: 该层需要把最后输出 row 与 Visual token contribution 的差异转成 token 选择结果，用于标出差异超过阈值的 Visual token。
- 怎么做/计算: #76 取 attention output 最后一行；#77 取最后 query 的 attention row；#78 unsqueeze；#79 乘 V head tensor；#80 permute 到 token-major；#81 clone；#82 view 为 `[1,624,4096]`；#83 slice Visual span `[35,611)`；#84 broadcast 最后一行；#85 先做 `last_row - visual_weighted_rows`；#86 再次 broadcast 最后一行；#87 `sub` 得到相对 reference 的 delta；#88 沿 Hidden 做 L2 norm；#89 squeeze batch 维得到每个 Visual token 一个 score；#90 比较 `score > 0.2`；#91 `nonzero` 找到为真的 Visual-span 内部 index；#92 `unbind`；#93 取 index 向量；#94 加 35，将内部 index 映射回原始 token 坐标。

```text
Visual selection over V=576 rows, token span [35,611), H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
last output row       #76,#84,#86              ──▶    ┌────────────────────────────────────────┐
                                                       │ LAST_OUTPUT_REFERENCE_ROW             │
                                                       └────────────────────────────────────────┘
weighted V rows       #77-#83                  ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       └────────────────────────────────────────┘
delta rows            #85,#87                  ──▶    ┌────────────────────────────────────────┐
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       └────────────────────────────────────────┘
score band            #88,#89                  ──▶    ┌────────────────────────────────────────┐
                                                       │ SCORE_PER_VISUAL_TOKEN                │
                                                       └────────────────────────────────────────┘
selected indices      #90-#94                  ──▶    [TOKEN_INDICES_IN_ORIGINAL_SEQUENCE]
```

## Attention output projection and residual

- 是什么: #95-#99 对 attention hidden 做 output projection 并与 `arg0_1` 相加。
- 为什么需要: 多头 context 需要投影回 hidden 表示，并保留 residual。
- 怎么做/计算: #95 view 为 `[624,4096]`；#96 取权重；#97 `mm`；#98 reshape；#99 residual add，输出 `add_tensor_5`。

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

- 是什么: #100-#109 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP projection 需要尺度稳定的 hidden。
- 怎么做/计算: #100 转 fp32；#101 平方；#102 Hidden 均值；#103 加 epsilon；#104 `rsqrt`；#105 缩放；#106 转 fp16；#107 读 norm weight；#108 乘权重；#109 view。

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

- 是什么: #110-#123 是 gated MLP 和 final residual。
- 为什么需要: MLP 在每个 token 的 channel 维提供非线性变换。
- 怎么做/计算: #110/#111 gate projection，#112 reshape，#113 `silu`；#114-#117 up projection；#118 gate/up 相乘；#119 view intermediate；#120/#121 down projection；#122 reshape；#123 与 #99 residual 相加。

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

- 是什么: #124 打包 layer 输出。
- 为什么需要: 后续流程需要 hidden states、K/V cache，以及 Visual selection 输出。
- 怎么做/计算: #124 返回 `(add_tensor_7, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, add_tensor_4, 0)`；其中 `add_tensor_4` 是 #94 得到的原始 token index 向量。

```text
Output tuple

hidden_states      #123 add_tensor_7             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      #94 add_tensor_4              ──▶   [SELECTED_VISUAL_TOKEN_INDICES]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
