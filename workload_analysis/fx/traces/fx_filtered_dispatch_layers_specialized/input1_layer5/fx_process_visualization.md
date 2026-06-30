# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer5`

本文件按 `fx_process_reconstruction.md/json`、`fx_process_nodes.csv` 和
`fx_graph.py` 中的 FX ATen 节点逐个 process 解释。这里的 process 名称只作为
固定输入 FX DAG 上的分组标签使用。

## Runtime FX inputs

- 是什么: 这是 layer5 固定输入 FX 图的 8 个 placeholder，`arg0_1` 是 hidden states，`arg1_1` 是 attention mask，`arg2_1` 是 position ids，其余 placeholder 在该图中没有后续用户。
- 为什么需要: 后续 RMSNorm、Q/K/V projection、RoPE、attention、MLP 和 output tuple 都从这些入口张量读取固定采样值。
- 怎么做/计算: 这个 process 不做数值计算。#0 `arg0_1` 直接提供 `[B=1,S=624,H=4096]` hidden rows 给 #8 RMSNorm 和 #84 residual add；#1 `arg1_1` 提供 `[B=1,1,Q=624,K=624]` mask 给 #63；#2 `arg2_1` 提供 position ids 给 #36/#39 的 cos/sin 查表。

```text
Runtime inputs, displayed by last two meaningful axes

Hidden states S=624 x H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
arg0_1 hidden [1,624,4096]                      ──▶   ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘

Attention mask Q=624 x K=624
                                                       0                                       624
                                                       ▲                                        ▲
arg1_1 mask [1,1,624,624]                      ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘

Position ids S=624
                                                       0                                       624
                                                       ▲                                        ▲
arg2_1 position ids                            ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对输入 hidden rows 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 需要稳定的 hidden 尺度；RMSNorm 先按每个 token 的 Hidden 维求 RMS，再乘以 norm 权重。
- 怎么做/计算: #8 `_to_copy_default` 将 `arg0_1` 转为 fp32；#9 `pow_tensor_scalar` 逐元素平方；#10 `mean_dim` 沿 Hidden 求均值并保留维度；#11 加 `1e-05`；#12 `rsqrt` 取倒数平方根；#13 将缩放因子乘回 fp32 hidden；#14 转回 fp16；#15 `_param_constant0` 提供 `[4096]` RMSNorm 权重，后续 #16 将它乘到归一化结果上。

```text
RMSNorm keeps token axis S=624 and reduces over H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
input hidden      #8 fp32 copy                  ──▶   ┌────────────────────────────────────────┐
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       └────────────────────────────────────────┘
square + mean     #9,#10 reduce H               ──▶   ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
rsqrt + scale     #11-#14                       ──▶   ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: #16-#34 把 RMSNorm 后的 hidden 分别投影为 Q、K、V，并 reshape/transpose 成 `[B,Heads,S,Dh]`。
- 为什么需要: Attention 用 Q 与 K 生成 token 间读取权重，再用权重加权 V；三条 projection 分支对应这三个张量。
- 怎么做/计算: #16 将 #15 权重与 #14 normalized hidden 逐元素相乘。Q 分支 #17 view 为 `[624,4096]`，#18 取 Q 权重，#19 `mm`，#20 回到 `[1,624,4096]`，#29 view 成 `[1,624,32,128]`，#30 transpose 为 `[1,32,624,128]`。K 分支 #21-#24、#31-#32 做同样流程并输出 `transpose_int_1`；V 分支 #25-#28、#33-#34 输出 `transpose_int_2`。

```text
Shared normalized hidden S=624 x H=4096, split into Heads=32 x Dh=128
                                                       0                                      4096
                                                       ▲                                        ▲
norm hidden      #16 shared rows                ──▶   ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
Q projection     #17-#20,#29,#30                ──▶   ┌────────────────────────────────────────┐
                                                       │ Q_MM_THEN_HEAD_SPLIT                  │
                                                       └────────────────────────────────────────┘
K projection     #21-#24,#31,#32                ──▶   ┌────────────────────────────────────────┐
                                                       │ K_MM_THEN_HEAD_SPLIT                  │
                                                       └────────────────────────────────────────┘
V projection     #25-#28,#33,#34                ──▶   ┌────────────────────────────────────────┐
                                                       │ V_MM_THEN_HEAD_SPLIT                  │
                                                       └────────────────────────────────────────┘
head layout      [1,32,624,128]                 ──▶   ┌────────────────────────────────────────┐
                                                       │ HEAD_SPLIT_TOKEN_ROWS                 │
                                                       │ HEAD_SPLIT_TOKEN_ROWS                 │
                                                       └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: #35-#54 对 Q 和 K 注入 RoPE 位置旋转。
- 为什么需要: Q/K projection 只来自内容 hidden；RoPE 用 position ids 查表，把 token 位置编码进 Q/K，使后续 QK score 带有相对位置信息。
- 怎么做/计算: #35/#38 取 cos/sin 表，#36/#39 用 `arg2_1` index 当前 624 个 token 的位置行，#37/#40 unsqueeze 供 head 维广播。Q 路径 #41 计算 `Q*cos`，#42/#43 切分 Dh 的 `[0,64)` 与 `[64,128)`，#44 对右半取负，#45 concat 得到 rotate-half，#46 计算 `rotate_half(Q)*sin`，#47 相加得到 RoPE Q。K 路径 #48-#54 对 K 做同样计算。

```text
RoPE rotates each Dh=128 vector by two half-width slices
                                                       0                       64              128
                                                       ▲                        ▲                ▲
Q or K input      [B,Heads,S,Dh]               ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half       neg(right), left             ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
RoPE output       x*cos + rotate*sin           ──▶    ┌────────────────────────────────────────┐
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#70 先计算标准 QK attention weights，再对 tail query 的 Visual-key 区域做 clear-only 改写。
- 为什么需要: Attention 权重决定每个 query 从哪些 key/value 读取信息；本层 FX DAG 额外把 `q=[611,624)` 对 `k=[35,611)` 的 Visual-key block 用常量清空。这里没有 `sum` 或 `copy_` 节点，因此不能解释为 fold。
- 怎么做/计算: #55 转置 RoPE K 的最后两维；#56-#59 将 Q 和 K^T expand/view 成 `bmm` 输入；#60 `bmm` 得到 `[32,624,624]` scores；#61 view 回 `[1,32,624,624]`；#62 除以 `sqrt(128)=11.313708498984761`；#63 加 attention mask；#64 softmax；#65 转 fp16；#66/#67 slice 出 tail query `q=[611,624)` 与 Visual key `k=[35,611)`；#68 读取常量；#69 `fill_` 原地清空该区域；#70 clone 改写后的 weights 供 attention-output bmm 使用。

```text
Attention weights Q=624 x K=624 with Visual clear-only region

K_seq
             0        35                              611       624
             ▲        ▲                                ▲         ▲
Q_seq    ┌────────┬──────────────────────────────────┬────────┐
0..610   │ KEEP   │ KEEP                             │ KEEP   │
         ├────────┼──────────────────────────────────┼────────┤
611..623 │ KEEP   │ CLEAR_BY_FILL (#66-#69)          │ KEEP   │
         └────────┴──────────────────────────────────┴────────┘
                  ▲                                  ▲
                  └── Visual-key span [35,611) is filled in-place
```

## Attention-weighted V and hidden reshape

- 是什么: #71-#79 用已经 clear-only 改写后的 attention weights 加权 V，并把多头 context 合并回 hidden width。
- 为什么需要: QK softmax 只给出读取权重；真正传入 output projection 的是 `weights @ V` 得到的 context。
- 怎么做/计算: #71/#72 将 #70 weights expand/view 为 `[32,624,624]`；#73/#74 将 V expand/view 为 `[32,624,128]`；#75 `bmm` 沿 K 维求加权和；#76 view 成 `[1,32,624,128]`；#77 transpose 回 token-major；#78 clone 成 contiguous；#79 view 为 `[1,624,4096]`。

```text
Per-head contraction [Q,K] x [K,Dh] -> [Q,Dh], then merge heads
                                                       0                                       128
                                                       ▲                                        ▲
weights [Q,K]     #71-#72                      ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_WEIGHT_MATRIX               │
                                                       │ ATTENTION_WEIGHT_MATRIX               │
                                                       └────────────────────────────────────────┘
values [K,Dh]     #73-#74                      ──▶    ┌────────────────────────────────────────┐
                                                       │ VALUE_ROWS                            │
                                                       │ VALUE_ROWS                            │
                                                       └────────────────────────────────────────┘
context [Q,Dh]    #75-#78                      ──▶    ┌────────────────────────────────────────┐
                                                       │ CONTEXT_ROWS                          │
                                                       │ CONTEXT_ROWS                          │
                                                       └────────────────────────────────────────┘
merged [Q,H]      #79 [1,624,4096]             ──▶    ┌────────────────────────────────────────┐
                                                       │ MERGED_HEAD_ROWS                      │
                                                       │ MERGED_HEAD_ROWS                      │
                                                       └────────────────────────────────────────┘
```

## Attention output projection and residual

- 是什么: #80-#84 对 attention hidden 做 output projection，并与原始 `arg0_1` 做 residual add。
- 为什么需要: 多头 context 需要投影回模型 hidden 表示；residual add 保留 attention 前的 hidden 路径。
- 怎么做/计算: #80 view `[1,624,4096]` 为 `[624,4096]`；#81 读取 output projection 权重；#82 `mm` 投影；#83 reshape 回 `[1,624,4096]`；#84 与 `arg0_1` 逐元素相加得到 attention residual。

```text
Token axis Q=624 x H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
attention hidden  #80 input                    ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       └────────────────────────────────────────┘
projected rows    #81-#83                      ──▶    ┌────────────────────────────────────────┐
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       └────────────────────────────────────────┘
residual output   #84 arg0_1 + projected       ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: #85-#94 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP gate/up/down projection 需要稳定输入尺度；该 norm 把 attention 子层输出调整为 MLP 输入。
- 怎么做/计算: #85 转 fp32；#86 平方；#87 沿 Hidden 均值；#88 加 epsilon；#89 `rsqrt`；#90 缩放 residual hidden；#91 转 fp16；#92 读取 `[4096]` norm 参数；#93 逐元素乘参数；#94 view 为 `[624,4096]` 供 MLP 第一条 projection 使用。

```text
Post-attention RMSNorm over H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
residual hidden   #85 add_tensor_4              ──▶   ┌────────────────────────────────────────┐
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       └────────────────────────────────────────┘
square + mean     #86,#87 reduce H              ──▶   ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
normalized input  #88-#94                       ──▶   ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: #95-#108 是 gated MLP 和 final residual add。
- 为什么需要: Attention 负责 token 间混合，MLP 在每个 token 的 channel 维上扩展到 intermediate width 并引入非线性，再压回 hidden width。
- 怎么做/计算: #95/#96 用 gate 权重对 #94 做 `mm`，#97 得到 `[1,624,11008]`，#98 `silu`；#99-#102 对同一 normalized hidden 做 up projection；#103 将 `silu(gate)` 与 up projection 逐元素相乘；#104 view 为 `[624,11008]`；#105/#106 做 down projection；#107 reshape 回 `[1,624,4096]`；#108 与 #84 attention residual 相加得到 layer output hidden。

```text
MLP channel path per token: H=4096 -> I=11008 -> H=4096
                                                       0                                      11008
                                                       ▲                                        ▲
mlp input         #94 [624,4096]               ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
gate/up rows      #95-#102                     ──▶    ┌────────────────────────────────────────┐
                                                       │ GATE_AND_UP_ROWS                      │
                                                       │ GATE_AND_UP_ROWS                      │
                                                       └────────────────────────────────────────┘
silu * up         #98,#103                     ──▶    ┌────────────────────────────────────────┐
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       └────────────────────────────────────────┘
down + residual   #104-#108 [624,4096]         ──▶    ┌────────────────────────────────────────┐
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       └────────────────────────────────────────┘
```

## Layer output

- 是什么: #109 是 FX output node，把已经计算出的 layer 结果打包返回。
- 为什么需要: 下一层或 generate loop 需要 hidden output、K/V cache，以及该固定路径中的 Visual/control 返回项。
- 怎么做/计算: #109 不做新数值计算，只返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`。其中 `add_tensor_6` 是 MLP final residual hidden，`add_tensor_2` 是 RoPE K cache，`transpose_int_2` 是 V cache；该 layer 没有 materialized visual token output。

```text
Output tuple

hidden_states      #108 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
