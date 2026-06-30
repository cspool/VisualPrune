# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer7`

本文件逐个解释该 layer 的 FX process，并为每个 process 手工给出 tensor 轴/区域图。依据来自同目录的
`fx_process_reconstruction.md/json`、`fx_process_nodes.csv` 和 `fx_graph.py`。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder；`arg0_1` 是 hidden states，`arg1_1` 是 attention mask，`arg2_1` 是 position ids。
- 为什么需要: 后续 norm、projection、RoPE、attention、Visual delta、MLP 和 output 都从这些入口读取本次采样值。
- 怎么做/计算: 该 process 不做数值计算。#0 `arg0_1` 供 #8 RMSNorm 与 #90 residual add 使用；#1 `arg1_1` 供 #63 mask add；#2 `arg2_1` 供 #36/#39 cos/sin 查表；#3-#7 在该图中无后续用户。

```text
Runtime inputs

                                                       0                                      4096
                                                       ▲                                        ▲
arg0_1 hidden [B=1,S=624,H=4096]               ──▶    ┌────────────────────────────────────────┐
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       │ HIDDEN_STATE_ROWS                     │
                                                       └────────────────────────────────────────┘

                                                       0                                       624
                                                       ▲                                        ▲
arg1_1 mask [B=1,1,Q=624,K=624]                ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       │ ATTENTION_MASK_MATRIX                 │
                                                       └────────────────────────────────────────┘

                                                       0                                       624
                                                       ▲                                        ▲
arg2_1 position ids                             ──▶    ┌────────────────────────────────────────┐
                                                       │ POSITION_ID_VECTOR                    │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对 `arg0_1` 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 之前需要稳定每个 token hidden 向量的尺度。
- 怎么做/计算: #8 转 fp32；#9 对 hidden 逐元素平方；#10 沿 Hidden 求均值；#11 加 `1e-05`；#12 `rsqrt`；#13 乘回 fp32 hidden；#14 转 fp16；#15 提供 `[4096]` norm weight，后续 #16 将权重乘到 normalized hidden 上。

```text
RMSNorm over H=4096, preserving S=624
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

- 是什么: #16-#34 把 normalized hidden 投影为 Q/K/V，并 reshape/transpose 到 `[1,32,624,128]`。
- 为什么需要: Q/K 产生 attention weights，V 提供被读取的 value 内容。
- 怎么做/计算: #16 应用 RMSNorm weight。Q 分支 #17 view、#18 取权重、#19 `mm`、#20 reshape、#29 view、#30 transpose；K 分支 #21-#24/#31-#32；V 分支 #25-#28/#33-#34。三个分支共享 `[624,4096]` 输入，输出三个 head-split 张量。

```text
S=624 x H=4096 projected to Heads=32 x Dh=128
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

- 是什么: #35-#54 用 position ids 查出的 cos/sin 对 Q 和 K 做 RoPE。
- 为什么需要: 后续 QK score 需要包含 token 位置信息。
- 怎么做/计算: #35/#38 读取 cos/sin 表；#36/#39 用 `arg2_1` index 当前 624 个 token；#37/#40 unsqueeze。Q 路径 #41-#47 和 K 路径 #48-#54 都把 Dh=128 切成 `[0,64)` 与 `[64,128)`，构造 `concat(-right,left)`，再计算 `x*cos + rotate_half(x)*sin`。

```text
RoPE over Dh=128
                                                       0                       64              128
                                                       ▲                        ▲                ▲
input Q/K        before rotate                 ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half      neg(right), left              ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
rotated Q/K      #47 / #54                     ──▶    ┌────────────────────────────────────────┐
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       │ ROTATED_Q_OR_K_ROWS                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#66 是标准 QK score 到 attention weights 的过程；本 layer 的 attention score 阶段没有 `fill_`、`sum`、`copy_` 改写。
- 为什么需要: 每个 query token 需要对所有 key token 形成归一化读取权重，供后续加权 V 和 Visual value-aware process 使用。
- 怎么做/计算: #55 转置 K；#56/#57 将 Q 变成 `[32,624,128]`；#58/#59 将 K^T 变成 `[32,128,624]`；#60 `bmm` 得到 `[32,624,624]`；#61 view 为 `[1,32,624,624]`；#62 scale；#63 加 mask；#64 softmax；#65 转 fp16；#66 clone，随后同时被 attention output 和 Visual process 使用。

```text
QK score and softmax weights

                                                       K_seq 0                                  624
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

- 是什么: #67-#75 将 attention weights 乘以 V，得到 `[1,624,4096]` attention hidden。
- 为什么需要: Attention weights 是读取系数；`weights @ V` 才是每个 token 的 context hidden。
- 怎么做/计算: #67/#68 把 #66 weights view 为 `[32,624,624]`；#69/#70 把 V view 为 `[32,624,128]`；#71 `bmm` 沿 K 维加权求和；#72 view；#73 transpose；#74 clone；#75 合并 heads 得到 `[1,624,4096]`。

```text
[Q,K] x [K,Dh] -> [Q,Dh], then heads merge
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
merged [Q,H]    #75 [1,624,4096]              ──▶    ┌────────────────────────────────────────┐
                                                       │ MERGED_HEAD_ROWS                      │
                                                       │ MERGED_HEAD_ROWS                      │
                                                       └────────────────────────────────────────┘
```

## Visual-related value-aware process

- 是什么: #76-#85 构造 Visual token span 上的 value-aware delta；它没有 `norm`、threshold 或 `nonzero`，因此 FX DAG 中没有物化选择决策。
- 为什么需要: 该过程把最后一个 output row 与由最后 query attention 加权得到的 Visual token contribution 做差，用于表示 Visual span 对最后输出的 value 贡献差异。
- 怎么做/计算: #76 从 #75 attention output 取最后 token row；#77 从 #66 attention weights 取最后 query 的 attention row；#78 unsqueeze 到 value head 维；#79 与 V head tensor `transpose_int_2` 相乘，得到每个 key/value token 的 weighted V contribution；#80 permute 成 token-major；#81 clone；#82 view 为 `[1,624,4096]`；#83 slice Visual span `token=[35,611)`；#84 将最后 output row unsqueeze 成 `[1,1,4096]`；#85 `sub` 计算最后 row 与 Visual contribution rows 的差。#85 的结果没有 downstream users。

```text
Visual delta construction over V=576 rows and H=4096

Visual token axis V=576, span [35,611)                 Hidden dimension
                                                       0                                      4096
                                                       ▲                                        ▲
last output row   #76,#84 broadcast             ──▶   ┌────────────────────────────────────────┐
                                                       │ LAST_OUTPUT_REFERENCE_ROW             │
                                                       └────────────────────────────────────────┘
weighted V rows   #77-#83                       ──▶   ┌────────────────────────────────────────┐
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       │ VISUAL_WEIGHTED_VALUE_ROWS            │
                                                       └────────────────────────────────────────┘
delta rows        #85 sub                       ──▶   ┌────────────────────────────────────────┐
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       │ VISUAL_DELTA_ROWS                     │
                                                       └────────────────────────────────────────┘
decision output   no downstream users           ──▶   [NO_MATERIALIZED_SELECTION]
```

## Attention output projection and residual

- 是什么: #86-#90 对 attention hidden 做 output projection 并加 residual。
- 为什么需要: 多头 context 需要投影回模型 hidden 表示，同时保留 attention 前 residual 路径。
- 怎么做/计算: #86 view `[1,624,4096]` 为 `[624,4096]`；#87 读取 projection 权重；#88 `mm`；#89 reshape 回 `[1,624,4096]`；#90 与 `arg0_1` 相加。

```text
Q=624 x H=4096
                                                       0                                      4096
                                                       ▲                                        ▲
attention rows   #86 input                    ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       │ ATTENTION_HIDDEN_ROWS                 │
                                                       └────────────────────────────────────────┘
projected rows   #87-#89                      ──▶    ┌────────────────────────────────────────┐
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       │ OUTPUT_PROJECTION_ROWS                │
                                                       └────────────────────────────────────────┘
residual rows    #90                           ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       │ ATTENTION_RESIDUAL_ROWS               │
                                                       └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: #91-#100 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: MLP projection 前需要稳定 hidden 尺度。
- 怎么做/计算: #91 转 fp32；#92 平方；#93 沿 Hidden 求均值；#94 加 epsilon；#95 `rsqrt`；#96 乘回 hidden；#97 转 fp16；#98 读取 norm weight；#99 逐元素乘；#100 view 为 `[624,4096]`。

```text
RMSNorm before MLP
                                                       0                                      4096
                                                       ▲                                        ▲
residual rows    #91                          ──▶    ┌────────────────────────────────────────┐
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       │ RESIDUAL_HIDDEN_ROWS                  │
                                                       └────────────────────────────────────────┘
rms column       #92,#93                      ──▶    ┌────────────────────────────────────────┐
                                                       │ RMS_REDUCTION_COLUMN                  │
                                                       └────────────────────────────────────────┘
mlp input rows   #94-#100                     ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: #101-#114 是 gated MLP 和 final residual add。
- 为什么需要: 在每个 token 的 channel 维上做非线性扩展和压回，补充 attention 后的表达。
- 怎么做/计算: #101/#102 gate projection；#103 reshape；#104 `silu`；#105-#108 up projection；#109 逐元素乘 `silu(gate)` 和 up；#110 view `[624,11008]`；#111/#112 down projection；#113 reshape `[1,624,4096]`；#114 与 #90 residual 相加。

```text
Per-token MLP H=4096 -> I=11008 -> H=4096
                                                       0                                      11008
                                                       ▲                                        ▲
mlp input       #100 [624,4096]               ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       │ NORMALIZED_MLP_INPUT_ROWS             │
                                                       └────────────────────────────────────────┘
gate/up         #101-#108                     ──▶    ┌────────────────────────────────────────┐
                                                       │ GATE_AND_UP_ROWS                      │
                                                       │ GATE_AND_UP_ROWS                      │
                                                       └────────────────────────────────────────┘
activated       #104,#109                     ──▶    ┌────────────────────────────────────────┐
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       │ ACTIVATED_INTERMEDIATE_ROWS           │
                                                       └────────────────────────────────────────┘
final hidden    #110-#114                     ──▶    ┌────────────────────────────────────────┐
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       │ FINAL_RESIDUAL_HIDDEN_ROWS            │
                                                       └────────────────────────────────────────┘
```

## Layer output

- 是什么: #115 打包 layer 输出。
- 为什么需要: 后续层或 generate loop 需要 hidden states、K/V cache 和该路径的 Visual/control 输出。
- 怎么做/计算: #115 返回 `(add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)`；#85 的 Visual delta 没有进入 output tuple。

```text
Output tuple

hidden_states      #114 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROWS: B=1,S=624,H=4096]
dynamic_cache.K    #54 add_tensor_2              ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,S=624,Dh=128]
dynamic_cache.V    #34 transpose_int_2           ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,S=624,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
