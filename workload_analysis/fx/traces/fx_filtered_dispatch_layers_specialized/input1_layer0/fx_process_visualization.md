# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer0`

This companion file explains the reconstructed FX processes from the local
`fx_process_reconstruction.md`, `fx_process_reconstruction.json`,
`fx_process_nodes.csv`, and `fx_graph.py` evidence. The prose below uses the
reconstructed process labels only as grouping labels over the fixed-input FX
ATen DAG.

## Runtime FX inputs

- 是什么: 这是该固定输入 layer 图的入口 process，包含 8 个 placeholder: `arg0_1` 到 `arg7_1`。其中 `arg0_1` 后续进入 input RMSNorm 和 residual add，`arg1_1` 进入 attention mask add，`arg2_1` 用于 RoPE 的 cos/sin index。
- 为什么需要: 后续所有 ATen 节点都从这些 placeholder 读取 sampled runtime values；没有这些入口张量，后面的 norm、projection、attention、MLP 和 output tuple 都无法确定。
- 怎么做/计算: `arg0_1` 不做计算，只向 #8 `_to_copy_default` 和 #90 `add_tensor_4` 提供 hidden states；`arg1_1` 只向 #63 `add_tensor_3` 提供 attention mask；`arg2_1` 只向 #36 `index_tensor` 和 #39 `index_tensor_1` 提供 position ids；`arg3_1` 到 `arg7_1` 在该 FX DAG 中没有用户。

```text
Runtime input tensors, displayed by last two meaningful axes

Hidden states: S=624 x Hidden=4096
                                                           0                                      4096
                                                           ▲                                        ▲
arg0_1 hidden [B=1,S=624,H=4096]                    ──▶   ┌────────────────────────────────────────┐  ◀── users: #8, #90
                                                           │ HIDDEN_STATE_ROWS                     │
                                                           │ HIDDEN_STATE_ROWS                     │
                                                           │ HIDDEN_STATE_ROWS                     │
                                                           └────────────────────────────────────────┘

Attention mask: Q_seq=624 x K_seq=624
                                                           0                                       624
                                                           ▲                                        ▲
arg1_1 mask [B=1,1,Q=624,K=624]                     ──▶   ┌────────────────────────────────────────┐  ◀── user: #63
                                                           │ ATTENTION_MASK_MATRIX                 │
                                                           │ ATTENTION_MASK_MATRIX                 │
                                                           └────────────────────────────────────────┘

Position ids: S=624
                                                           0                                       624
                                                           ▲                                        ▲
arg2_1 position_ids [B=1,S=624]                    ──▶    ┌────────────────────────────────────────┐  ◀── users: #36, #39
                                                           │ POSITION_ID_VECTOR                    │
                                                           └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: 这是 attention 前的 RMSNorm process，节点 #8-#15 将 `arg0_1` 的 hidden rows 归一化并提供给 Q/K/V projection。
- 为什么需要: Q/K/V 线性投影对 hidden scale 敏感。该 process 先把每个 token 的 Hidden 向量归一到稳定 RMS，再乘以 RMSNorm 参数，使后续 projection 在稳定尺度上计算。
- 怎么做/计算: #8 `_to_copy_default` 把 `arg0_1` 转为 fp32；#9 `pow_tensor_scalar` 对 fp32 hidden 逐元素平方；#10 `mean_dim` 沿最后一维 Hidden 求均值并保留维度；#11 `add_tensor` 加 `1e-05`；#12 `rsqrt_default` 取倒数平方根；#13 `mul_tensor` 用该缩放因子乘回 fp32 hidden；#14 `_to_copy_default_1` 转回 fp16；#15 `_param_constant0` 提供 shape `[4096]` 的 RMSNorm 权重，并在下一个 process 的 #16 与归一化 hidden 相乘。

```text
RMSNorm over Hidden axis H=4096; token rows are preserved
                                                           0                                      4096
                                                           ▲                                        ▲
input hidden        #8 arg0_1 -> fp32              ──▶    ┌────────────────────────────────────────┐
                                                           │ INPUT_HIDDEN_ROWS                     │
                                                           │ INPUT_HIDDEN_ROWS                     │
                                                           │ INPUT_HIDDEN_ROWS                     │
                                                           └────────────────────────────────────────┘
square + mean       #9,#10 reduce over Hidden      ──▶    ┌────────────────────────────────────────┐
                                                           │ RMS_REDUCTION_COLUMN                  │
                                                           └────────────────────────────────────────┘
rsqrt + scale       #11,#12,#13                    ──▶    ┌────────────────────────────────────────┐
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           └────────────────────────────────────────┘
fp16 norm output    #14 plus weight #15            ──▶    ┌────────────────────────────────────────┐
                                                           │ RMSNORM_OUTPUT_ROWS                   │
                                                           │ RMSNORM_OUTPUT_ROWS                   │
                                                           └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: 这是把 normalized hidden 转成 Query、Key、Value，并 reshape 成 multi-head layout 的 process，节点 #16-#34。
- 为什么需要: Attention 需要 Q 与 K 计算相关性分数，并用分数归一化后的权重加权 V；因此同一份 hidden 要被三组权重分别投影，再拆成 32 个 head。
- 怎么做/计算: #16 `mul_tensor_1` 将 #15 RMSNorm 权重与 #14 normalized hidden 逐元素相乘；Q 分支 #17 view 为 `[624,4096]`，#18 读取 `[4096,4096]` 权重，#19 `mm` 投影，#20 恢复 `[1,624,4096]`，#29 view 成 `[1,624,32,128]`，#30 transpose 成 `[1,32,624,128]`；K 分支 #21-#24、#31-#32 做同样计算；V 分支 #25-#28、#33-#34 做同样计算。输出是 `transpose_int`(Q), `transpose_int_1`(K), `transpose_int_2`(V)。

```text
Token axis S=624 x Hidden=4096, then split to Heads=32 x Dh=128
                                                           0                                      4096
                                                           ▲                                        ▲
norm hidden        #16 shared input                 ──▶   ┌────────────────────────────────────────┐
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           │ NORMALIZED_HIDDEN_ROWS                │
                                                           └────────────────────────────────────────┘
Q branch           #17-#20,#29,#30                 ──▶    ┌────────────────────────────────────────┐
                                                           │ Q_MM_THEN_HEAD_SPLIT                  │
                                                           └────────────────────────────────────────┘
K branch           #21-#24,#31,#32                 ──▶    ┌────────────────────────────────────────┐
                                                           │ K_MM_THEN_HEAD_SPLIT                  │
                                                           └────────────────────────────────────────┘
V branch           #25-#28,#33,#34                 ──▶    ┌────────────────────────────────────────┐
                                                           │ V_MM_THEN_HEAD_SPLIT                  │
                                                           └────────────────────────────────────────┘
head layout        [B=1,Heads=32,S=624,Dh=128]     ──▶    ┌────────────────────────────────────────┐
                                                           │ HEAD_SPLIT_LAYOUT                     │
                                                           │ HEAD_SPLIT_LAYOUT                     │
                                                           └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: 这是对 Q 和 K 注入 RoPE 位置旋转的 process，节点 #35-#54。
- 为什么需要: Q/K projection 本身只包含 token 内容表示；RoPE 通过 position ids 查表，将位置信息混入 Q/K，使后续 QK 分数包含位置关系。
- 怎么做/计算: #35 和 #38 读取 shape `[624,128]` 的 cos/sin 常量表；#36/#39 用 `arg2_1` position ids index 出当前 token 的 cos/sin 行；#37/#40 unsqueeze 到可广播形状。Q 路径 #41 计算 `Q*cos`，#42/#43 切分 Dh 的 `[0,64)` 和 `[64,128)`，#44 对右半取负，#45 concat 成 rotate-half，#46 计算 `rotate_half(Q)*sin`，#47 相加得到 RoPE Q。K 路径 #48-#54 对 `transpose_int_1` 做同样计算，得到 RoPE K。

```text
RoPE over head dimension Dh=128; token/head axes are preserved
                                                           0                                       128
                                                           ▲                                        ▲
cos/sin lookup     #35-#40 by position_ids         ──▶    ┌────────────────────────────────────────┐
                                                           │ POSITIONAL_COS_SIN_ROWS               │
                                                           │ POSITIONAL_COS_SIN_ROWS               │
                                                           └────────────────────────────────────────┘
left half          slice [0,64)                    ──▶    ┌───────────────────┬────────────────────┐
                                                           │ LEFT_HALF         │ RIGHT_HALF         │
right half neg     slice [64,128) -> neg           ──▶    └───────────────────┴────────────────────┘
rotate half        cat([-right,left])              ──▶    ┌────────────────────────────────────────┐
                                                           │ ROTATE_HALF_Q_OR_K                    │
                                                           └────────────────────────────────────────┘
RoPE output        x*cos + rotate_half(x)*sin      ──▶    ┌────────────────────────────────────────┐
                                                           │ ROTATED_Q_OR_K_ROWS                   │
                                                           │ ROTATED_Q_OR_K_ROWS                   │
                                                           └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: 这是 attention score/weight process，同时包含 Visual-key region 的 fold/clear 改写，节点 #55-#76。
- 为什么需要: 标准 attention 先产生每个 query 对每个 key 的权重；本层还包含 Visual token 区域的特殊处理，将 tail query 对 Visual key span 的质量折叠到代表列并清空其它 Visual-key 列。
- 怎么做/计算: #55 把 RoPE K 的最后两维转置；#56-#59 expand/view Q 和 K^T 为 `bmm` 输入；#60 `bmm` 得到 `[32,624,624]` QK scores；#61 view 回 `[1,32,624,624]`；#62 除以 `11.313708498984761`；#63 加 `arg1_1` mask；#64 softmax；#65 转 fp16。Visual 改写由 #66-#68 slice tail query `q=[611,624)` 和 Visual key `k=[35,611)` 并沿 key 维求和；#69-#72 slice `q=[35,624), k=[35,611)` 并用 `_tensor_constant8` fill 清空；#73-#75 选 tail query 的 `k=35` 列并把 #68 的求和结果 copy 回去；#76 clone 改写后的 weights 供 V 加权使用。

```text
Visual attention-score fold/clear over Q_seq=624 x K_seq=624

K_seq axis
             0        35       36                         611       624
             ▲        ▲        ▲                           ▲         ▲
Q_seq    ┌────────┬────────┬─────────────────────────────┬────────┐
0..34    │ KEEP   │ KEEP   │ KEEP                        │ KEEP   │
         ├────────┼────────┼─────────────────────────────┼────────┤
35..610  │ KEEP   │ CLEAR  │ CLEAR_BY_FILL (#69-#72)     │ KEEP   │
         ├────────┼────────┼─────────────────────────────┼────────┤
611..623 │ KEEP   │ FOLD   │ CLEAR_BY_FILL (#66-#75)     │ KEEP   │
         └────────┴────────┴─────────────────────────────┴────────┘
                    ▲        ▲
                    │        └── k=[36,611) is cleared in-place
                    └── k=35 receives sum over k=[35,611)
```

## Attention-weighted V and hidden reshape

- 是什么: 这是用 attention weights 加权 Value，并将多头 context 合并回 hidden width 的 process，节点 #77-#85。
- 为什么需要: QK softmax 只产生读取系数；模型实际传给 output projection 的是这些系数加权 V 后得到的 context hidden。
- 怎么做/计算: #77-#78 把 #76 weights expand/view 为 `[32,624,624]`；#79-#80 把 #34 V expand/view 为 `[32,624,128]`；#81 `bmm` 沿 K 维做加权求和，输出 per-head context；#82 view 成 `[1,32,624,128]`；#83 transpose token/head 维；#84 clone 成 contiguous；#85 view 成 `[1,624,4096]`。

```text
Attention output contraction: [Q,K] x [K,Dh] -> [Q,Dh]
                                                           0                                       128
                                                           ▲                                        ▲
weights           #77-#78 [Q=624,K=624]           ──▶     ┌────────────────────────────────────────┐
                                                           │ SOFTMAX_WEIGHT_MATRIX                 │
                                                           │ SOFTMAX_WEIGHT_MATRIX                 │
                                                           └────────────────────────────────────────┘
values            #79-#80 [K=624,Dh=128]          ──▶     ┌────────────────────────────────────────┐
                                                           │ VALUE_ROWS                            │
                                                           │ VALUE_ROWS                            │
                                                           └────────────────────────────────────────┘
context           #81-#82 [Q=624,Dh=128]          ──▶     ┌────────────────────────────────────────┐
                                                           │ CONTEXT_ROWS                          │
                                                           │ CONTEXT_ROWS                          │
                                                           └────────────────────────────────────────┘
merged hidden     #83-#85 [Q=624,H=4096]          ──▶     ┌────────────────────────────────────────┐
                                                           │ MERGED_HEAD_ROWS                      │
                                                           │ MERGED_HEAD_ROWS                      │
                                                           │ MERGED_HEAD_ROWS                      │
                                                           └────────────────────────────────────────┘
```

## Attention output projection and residual

- 是什么: 这是 attention 子层的 output projection 与 residual add process，节点 #86-#90。
- 为什么需要: 多头 context 需要通过输出线性层回到模型 hidden 表示，并与进入 attention 前的 residual hidden 相加，形成 attention 子层输出。
- 怎么做/计算: #86 将 #85 view 成 `[624,4096]`；#87 读取 shape `[4096,4096]` 的 output projection 权重；#88 `mm` 做线性投影；#89 unsafe_view 回 `[1,624,4096]`；#90 将投影结果与 `arg0_1` residual 逐元素相加，输出 `add_tensor_4`。

```text
Token axis Q=624 x Hidden=4096
                                                           0                                      4096
                                                           ▲                                        ▲
attention hidden   #86 input                       ──▶    ┌────────────────────────────────────────┐
                                                           │ ATTENTION_HIDDEN_ROWS                 │
                                                           │ ATTENTION_HIDDEN_ROWS                 │
                                                           └────────────────────────────────────────┘
linear projected   #87-#89                         ──▶    ┌────────────────────────────────────────┐
                                                           │ OUTPUT_PROJECTION_ROWS                │
                                                           │ OUTPUT_PROJECTION_ROWS                │
                                                           └────────────────────────────────────────┘
residual add       #90 arg0_1 + projected          ──▶    ┌────────────────────────────────────────┐
                                                           │ ATTENTION_RESIDUAL_ROWS               │
                                                           │ ATTENTION_RESIDUAL_ROWS               │
                                                           └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: 这是 MLP 前的 RMSNorm process，节点 #91-#100，对 attention residual 后的 hidden rows 再做一次逐 token normalization。
- 为什么需要: MLP 的 gate/up/down projection 需要稳定尺度的输入；该 process 把 attention 子层输出调整为适合 MLP 的 hidden 表示。
- 怎么做/计算: #91 把 `add_tensor_4` 转 fp32；#92 平方；#93 沿 Hidden 均值；#94 加 epsilon；#95 `rsqrt`；#96 乘回 fp32 hidden；#97 转回 fp16；#98 读取 shape `[4096]` 的 RMSNorm 参数；#99 逐元素乘参数；#100 view 为 `[624,4096]` 供 MLP gate projection 使用，#99 同时供 #105 的 up projection view 使用。

```text
Post-attention RMSNorm over Hidden axis H=4096
                                                           0                                      4096
                                                           ▲                                        ▲
residual hidden    #91 add_tensor_4 -> fp32        ──▶    ┌────────────────────────────────────────┐
                                                           │ RESIDUAL_HIDDEN_ROWS                  │
                                                           │ RESIDUAL_HIDDEN_ROWS                  │
                                                           └────────────────────────────────────────┘
square + mean      #92-#93 reduce Hidden          ──▶     ┌────────────────────────────────────────┐
                                                           │ RMS_REDUCTION_COLUMN                  │
                                                           └────────────────────────────────────────┘
rsqrt + scale      #94-#99                         ──▶    ┌────────────────────────────────────────┐
                                                           │ NORMALIZED_MLP_INPUT_ROWS             │
                                                           │ NORMALIZED_MLP_INPUT_ROWS             │
                                                           └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: 这是 gated MLP 与 final residual add process，节点 #101-#114。
- 为什么需要: Attention 混合 token 间信息，MLP 在每个 token 的 channel/hidden 维上做非线性扩展和压回，补充表达能力，并通过 residual 保留 attention 子层输出。
- 怎么做/计算: Gate 分支 #101-#103 读取 `[4096,11008]` 权重并对 #100 normalized hidden 做 `mm`，#104 对结果做 SiLU；Up 分支 #105-#108 对同一 normalized hidden 做第二个 `[4096,11008]` projection；#109 将 SiLU gate 与 up projection 逐元素相乘；#110 view 成 `[624,11008]`；#111-#113 用 down projection 权重把 intermediate 压回 `[1,624,4096]`；#114 将 down output 与 #90 attention residual `add_tensor_4` 相加，得到 layer hidden output。

```text
MLP keeps token axis S=624; channel axis H=4096 -> I=11008 -> H=4096
                                                           0                                      11008
                                                           ▲                                        ▲
mlp input          #100/#105 [S=624,H=4096]        ──▶    ┌────────────────────────────────────────┐
                                                           │ NORMALIZED_MLP_INPUT_ROWS             │
                                                           │ NORMALIZED_MLP_INPUT_ROWS             │
                                                           └────────────────────────────────────────┘
gate/up            #101-#108                       ──▶    ┌────────────────────────────────────────┐
                                                           │ GATE_AND_UP_ROWS                      │
                                                           │ GATE_AND_UP_ROWS                      │
                                                           └────────────────────────────────────────┘
silu(gate) * up    #104,#109                       ──▶    ┌────────────────────────────────────────┐
                                                           │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                           │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                           └────────────────────────────────────────┘
down + residual    #110-#114 [S=624,H=4096]        ──▶    ┌────────────────────────────────────────┐
                                                           │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                           │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                           └────────────────────────────────────────┘
```

## Layer output

- 是什么: 这是 FX output node 的返回打包 process，节点 #115。
- 为什么需要: 下一层或 generate loop 需要当前层 hidden output、K/V cache，以及 VisiPrune 控制输出；该 process 把这些已计算值组织为固定返回结构。
- 怎么做/计算: #115 不做新的张量计算，只返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`。其中 `add_tensor_6` 是 MLP final residual hidden，`add_tensor_2` 是 RoPE K，`transpose_int_2` 是 V head layout，`None` 表示该层没有 materialized important visual token output，`0` 是 control scalar。

```text
Layer output tuple

hidden_states      #114 add_tensor_6               ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2                ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2             ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      literal None                    ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                       ──▶   [EXIT_INDICATOR_SCALAR]
```
