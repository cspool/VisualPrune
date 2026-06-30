# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer28`

本文件逐个解释该 layer 的 FX process，并手工画出对应 tensor 轴/区域图。依据来自同目录的
`fx_process_reconstruction.md/json`、`fx_process_nodes.csv` 和 `fx_graph.py`。

## Runtime FX inputs

- 是什么: #0-#7 是该固定输入 FX DAG 的 placeholder。`arg0_1` 是 `[1,48,4096]` hidden states，`arg1_1` 是 `[1,1,48,48]` attention mask，`arg2_1` 是 position ids。
- 为什么需要: 所有后续 ATen 节点都从这些入口读取本次采样的 layer 输入；它们决定这条 FX 图上的 norm、attention 和 MLP 计算。
- 怎么做/计算: 这个 process 不做数值计算。`arg0_1` 进入 #8 `_to_copy_default` 和 #80 residual add；`arg1_1` 进入 #63 mask add；`arg2_1` 进入 #36/#39 的 cos/sin index。

```text
Runtime inputs

                                                       0                                      4096
                                                       ▲                                        ▲
arg0_1 hidden [B=1,S=48,H=4096]               ──▶    ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘

                                                       0                                       48
                                                       ▲                                        ▲
arg1_1 mask [B=1,1,Q=48,K=48]                ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘

                                                       0                                       48
                                                       ▲                                        ▲
arg2_1 position ids                             ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对 `arg0_1` 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 前需要稳定每个 token 的 hidden 尺度。
- 怎么做/计算: #8 把 hidden 转 fp32；#9 平方；#10 沿 Hidden 求均值；#11 加 epsilon；#12 `rsqrt`；#13 乘回 hidden；#14 转 fp16；#15 提供 `[4096]` norm weight，供 #16 与 normalized hidden 相乘。

```text
RMSNorm over H=4096, preserving S=48
                                                       0                                      4096
                                                       ▲                                        ▲
input rows       #8 fp32 copy                  ──▶    ┌────────────────────────────────────────┐
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       │ INPUT_HIDDEN_ROWS                     │
                                                       └────────────────────────────────────────┘
rms column       #9,#10 reduce H               ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
norm rows        #11-#15                        ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: #16-#34 将 normalized hidden 投影为 Q、K、V，并转成 `[1,32,48,128]` head layout。
- 为什么需要: Attention 需要 Q/K 计算权重、V 提供被读取的 value 内容。
- 怎么做/计算: #16 应用 RMSNorm 权重。Q 分支 #17 view、#18 取权重、#19 `mm`、#20 reshape、#29 view、#30 transpose；K 分支 #21-#24/#31-#32；V 分支 #25-#28/#33-#34。三条分支共享同一个 normalized hidden 输入，但写入不同 Q/K/V 张量。

```text
S=48 x H=4096 projected to Heads=32 x Dh=128
                                                       0                                      4096
                                                       ▲                                        ▲
norm hidden      #16                           ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       │ NORMALIZED_HIDDEN_ROWS                │
                                                       └────────────────────────────────────────┘
Q path           #17-#20,#29,#30               ──▶    ┌────────────────────────────────────────┐
                                                       │ Q_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
K path           #21-#24,#31,#32               ──▶    ┌────────────────────────────────────────┐
                                                       │ K_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
V path           #25-#28,#33,#34               ──▶    ┌────────────────────────────────────────┐
                                                       │ V_HEAD_SPLIT_ROWS                     │
                                                       └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: #35-#54 用 position ids 查出的 cos/sin 对 Q 和 K 做 RoPE 旋转。
- 为什么需要: 后续 QK score 需要包含 token 位置信息，而不只是内容投影。
- 怎么做/计算: #35/#38 读取 cos/sin 表；#36/#39 用 `arg2_1` index 当前 token；#37/#40 unsqueeze 供广播。Q 的 #41-#47 和 K 的 #48-#54 都执行 `x*cos + concat(-x_right, x_left)*sin`，其中 Dh=128 被切为两个 64 宽 half。

```text
RoPE over Dh=128
                                                       0                       64              128
                                                       ▲                        ▲                ▲
input Q/K        before rotate                 ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half      #42-#46 / #49-#53             ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
rotated Q/K      #47 / #54                     ──▶    ┌────────────────────────────────────────┐
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#66 是标准 attention score/weight 计算；该 layer 没有 Visual clear/fold/copy 改写。
- 为什么需要: 每个 query token 需要对所有 key token 形成归一化读取权重，供后续加权 V。
- 怎么做/计算: #55 转置 K 的最后两维；#56/#57 将 Q 变为 `[32,48,128]`；#58/#59 将 K^T 变为 `[32,128,48]`；#60 `bmm` 得到 `[32,48,48]`；#61 view 为 `[1,32,48,48]`；#62 除以 `sqrt(128)`；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone 后作为 attention weights。

```text
Attention contraction and normalized weights

Q x K^T
                                                       K_seq 0                                  48
                                                       ▲                                        ▲
Q_seq 0..623                                     ──▶   ┌────────────────────────────────────────┐
                                                       │ QK_SCORE_MATRIX                       │
                                                       │ QK_SCORE_MATRIX                       │
                                                       │ QK_SCORE_MATRIX                       │
                                                       └────────────────────────────────────────┘
mask + softmax                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       │ NORMALIZED_ATTENTION_WEIGHTS          │
                                                       └────────────────────────────────────────┘
```

## Attention-weighted V and hidden reshape

- 是什么: #67-#75 用 attention weights 加权 V，并把多头 context 合并回 `[1,48,4096]`。
- 为什么需要: Attention weights 本身不是 hidden output；`weights @ V` 才是 attention 子层的上下文表示。
- 怎么做/计算: #67/#68 把 #66 weights 变为 `[32,48,48]`；#69/#70 把 V 变为 `[32,48,128]`；#71 `bmm` 沿 K 维求和；#72 view 为 `[1,32,48,128]`；#73 transpose；#74 clone；#75 view 为 `[1,48,4096]`。

```text
[Q,K] x [K,Dh] -> [Q,Dh], then heads merge to H=4096
                                                       0                                       128
                                                       ▲                                        ▲
weights [Q,K]   #67-#68                       ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_WEIGHT_MATRIX               │
                                                       │ ATTENTION_WEIGHT_MATRIX               │
                                                       └────────────────────────────────────────┘
values [K,Dh]   #69-#70                       ──▶    ┌────────────────────────────────────────┐
                                                       │ VALUE_ROWS                            │
                                                       │ VALUE_ROWS                            │
                                                       └────────────────────────────────────────┘
context [Q,Dh]  #71-#74                       ──▶    ┌────────────────────────────────────────┐
                                                       │ CONTEXT_ROWS                          │
                                                       │ CONTEXT_ROWS                          │
                                                       └────────────────────────────────────────┘
merged [Q,H]    #75                            ──▶    ┌────────────────────────────────────────┐
                                                       │ MERGED_HEAD_ROWS                      │
                                                       │ MERGED_HEAD_ROWS                      │
                                                       └────────────────────────────────────────┘
```

## Attention output projection and residual

- 是什么: #76-#80 做 attention output projection 并加 residual。
- 为什么需要: 多头 context 要经过输出线性层回到模型 hidden 表示，并与 attention 前 hidden 相加。
- 怎么做/计算: #76 view 为 `[48,4096]`；#77 读取 projection 权重；#78 `mm`；#79 reshape 为 `[1,48,4096]`；#80 与 `arg0_1` 相加。

```text
Q=48 x H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
attention rows   #76 input                    ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       └────────────────────────────────────────┘
projected rows   #77-#79                      ──▶    ┌────────────────────────────────────────┐
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       └────────────────────────────────────────┘
residual rows    #80                           ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: #81-#90 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP 线性投影前需要稳定尺度的 token hidden。
- 怎么做/计算: #81 转 fp32；#82 平方；#83 沿 Hidden 均值；#84 加 epsilon；#85 `rsqrt`；#86 乘回 residual hidden；#87 转 fp16；#88 读取 norm weight；#89 逐元素乘；#90 view 成 `[48,4096]`。

```text
RMSNorm before MLP
                                                       0                                      4096
                                                       ▲                                        ▲
residual rows    #81                          ──▶    ┌────────────────────────────────────────┐
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       └────────────────────────────────────────┘
rms column       #82,#83                      ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
mlp input rows   #84-#90                      ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: #91-#104 是 gated MLP 和最终 residual add。
- 为什么需要: MLP 对每个 token 的 channel 维做非线性变换，补充 attention 子层后的表达能力。
- 怎么做/计算: #91/#92 做 gate projection；#93 reshape 到 `[1,48,11008]`；#94 `silu`；#95-#98 做 up projection；#99 将 gate activation 与 up projection 相乘；#100 view `[48,11008]`；#101/#102 down projection；#103 reshape 回 hidden；#104 与 #80 residual 相加。

```text
Per-token channel transform H=4096 -> I=11008 -> H=4096
                                                       0                                      11008
                                                       ▲                                        ▲
mlp input       #90 [48,4096]                ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
gate/up         #91-#98                       ──▶    ┌────────────────────────────────────────┐
                                                       │ GATE_AND_UP_ROWS                      │
                                                       │ GATE_AND_UP_ROWS                      │
                                                       └────────────────────────────────────────┘
activated       #94,#99                       ──▶    ┌────────────────────────────────────────┐
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       └────────────────────────────────────────┘
final hidden    #100-#104                     ──▶    ┌────────────────────────────────────────┐
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       └────────────────────────────────────────┘
```

## Layer output

- 是什么: #105 只打包返回该 layer 已经算好的输出。
- 为什么需要: 后续层需要 hidden states 和 K/V cache；该层没有 materialized Visual token output。
- 怎么做/计算: #105 返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`，其中 `add_tensor_6` 是 #104 final hidden，`add_tensor_2` 是 #54 RoPE K，`transpose_int_2` 是 #34 V。

```text
Output tuple

hidden_states      #104 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=48,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=48,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=48,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
