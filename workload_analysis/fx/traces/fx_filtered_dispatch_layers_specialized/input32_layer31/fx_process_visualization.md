# FX Process Explanation And Tensor Visualization

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input32_layer31`

本文件逐个解释该 decode layer 的 FX process，并手工绘制 tensor 轴/区域图。

## Runtime FX inputs

- 是什么: #0-#7 是固定输入 FX DAG 的 placeholder；主要有效输入是当前 token hidden `arg0_1`、attention mask `arg1_1`、position ids `arg2_1`。
- 为什么需要: 这是单 query decode 路径，后续计算只处理当前 token，但 attention 会读已有 K/V cache 与当前 token 拼接后的 key/value。
- 怎么做/计算: #0 `arg0_1` 提供 `[1,1,4096]` hidden 给 #8 和 #83；#1 `arg1_1` 提供 `[1,1,1,655]` mask 给 #67；#2 `arg2_1` 供 #36/#39 RoPE 查表；其余 placeholder 在该图中不直接参与数值计算。

```text
Decode runtime tensors
                                                       0                                      4096
                                                       ▲                                        ▲
current hidden [1,1,4096]                      ──▶    ┌────────────────────────────────────────┐
                                                       │ CURRENT_TOKEN_HIDDEN                  │
                                                       └────────────────────────────────────────┘
attention mask [1,1,1,655]                     ──▶    ┌────────────────────────────────────────┐
                                                       │ SINGLE_QUERY_MASK_OVER_K              │
                                                       └────────────────────────────────────────┘
position ids [1,1]                             ──▶    ┌────────────────────────────────────────┐
                                                       │ CURRENT_POSITION_ID                   │
                                                       └────────────────────────────────────────┘
```

## Input RMSNorm

- 是什么: #8-#15 对当前 token hidden 做 attention 前 RMSNorm。
- 为什么需要: Q/K/V projection 前需要稳定 hidden 尺度。
- 怎么做/计算: #8 转 fp32；#9 平方；#10 沿 Hidden 求均值；#11 加 epsilon；#12 `rsqrt`；#13 乘回 hidden；#14 转 fp16；#15 提供 norm weight，#16 应用。

```text
RMSNorm over H=4096 for one token
                                                       0                                      4096
                                                       ▲                                        ▲
input row                                       ──▶    ┌────────────────────────────────────────┐
                                                       │ CURRENT_TOKEN_HIDDEN                  │
                                                       └────────────────────────────────────────┘
rms scalar                                      ──▶    [RMS_REDUCTION_SCALAR]
normalized row                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_CURRENT_TOKEN              │
                                                       └────────────────────────────────────────┘
```

## Q/K/V projection and head reshape

- 是什么: #16-#34 将当前 token normalized hidden 投影为 Q/K/V，并 reshape 成 `[1,32,1,128]`。
- 为什么需要: 当前 query 用 Q 读取 cache K/V；当前 token 的 K/V 也要追加到 cache。
- 怎么做/计算: #16 乘 norm weight。Q 分支 #17-#20/#29-#30，K 分支 #21-#24/#31-#32，V 分支 #25-#28/#33-#34，均执行 view、`mm`、reshape、transpose。

```text
One-token projection H=4096 -> Heads=32 x Dh=128
                                                       0                                      4096
                                                       ▲                                        ▲
norm hidden                                    ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_CURRENT_TOKEN              │
                                                       └────────────────────────────────────────┘
Q current                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ Q_HEAD_ROW                            │
                                                       └────────────────────────────────────────┘
K current                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ K_HEAD_ROW_TO_APPEND                  │
                                                       └────────────────────────────────────────┘
V current                                      ──▶    ┌────────────────────────────────────────┐
                                                       │ V_HEAD_ROW_TO_APPEND                  │
                                                       └────────────────────────────────────────┘
```

## RoPE position embedding

- 是什么: #35-#54 对当前 token Q/K 做 RoPE 旋转。
- 为什么需要: 当前 query/key 需要带上当前位置编码，才能与 cache key 对齐计算。
- 怎么做/计算: #35/#38 读取 cos/sin 表；#36/#39 按 `arg2_1` 取当前 position 行；#37/#40 unsqueeze；#41-#47 旋转 Q；#48-#54 旋转当前 K。Dh=128 被切成两个 64 宽 half。

```text
RoPE on current Q/K Dh=128
                                                       0                       64              128
                                                       ▲                        ▲                ▲
input Q/K                                      ──▶    ┌──────────────────────┬─────────────────┐
                                                       │ LEFT_HALF            │ RIGHT_HALF      │
                                                       └──────────────────────┴─────────────────┘
rotate-half                                   ──▶     ┌──────────────────────┬─────────────────┐
                                                       │ NEG_RIGHT_HALF       │ LEFT_HALF       │
                                                       └──────────────────────┴─────────────────┘
rotated current Q/K                           ──▶     ┌────────────────────────────────────────┐
                                                       │ ROTATED_CURRENT_ROW                   │
                                                       └────────────────────────────────────────┘
```

## QK scores, mask, softmax

- 是什么: #55-#70 先把 cached K/V 与当前 K/V concat，再计算单 query 对 655 个 key 的 attention weights。
- 为什么需要: decode 时当前 query 只新增一个 token，但必须能读取历史 cache 中的所有 key/value。
- 怎么做/计算: #55 读取 cached K，#56 将 cached K 与 #54 当前 RoPE K 沿 token/cache 维 concat 为 `cat_default_2`，长度 655；#57 读取 cached V，#58 将 cached V 与 #34 当前 V concat 为 `cat_default_3`；#59 转置 concat K；#60/#61 view 当前 Q 为 `[32,1,128]`；#62/#63 view K^T 为 `[32,128,655]`；#64 `bmm` 得到 `[32,1,655]`；#65 view `[1,32,1,655]`；#66 scale；#67 加 mask；#68 softmax；#69 转 fp16；#70 clone。

```text
Decode attention: Q=1 attends over K_total=655

K_seq total
             cached keys 0..653                         current key 654
             ▲                                           ▲
concat K  ┌─────────────────────────────────────────────┬──────┐
          │ CACHED_KEY_ROWS                              │ CUR  │
          └─────────────────────────────────────────────┴──────┘
Q x K^T   ┌────────────────────────────────────────────────────┐
          │ SINGLE_QUERY_SCORE_ROW                             │
          └────────────────────────────────────────────────────┘
softmax   ┌────────────────────────────────────────────────────┐
          │ SINGLE_QUERY_WEIGHT_ROW                            │
          └────────────────────────────────────────────────────┘
```

## Attention-weighted V and hidden reshape

- 是什么: #71-#78 用单 query weights 加权 concat V，得到当前 token context hidden。
- 为什么需要: 当前 token 输出要从历史+当前 value rows 中读取信息。
- 怎么做/计算: #71/#72 view weights 为 `[32,1,655]`；#73/#74 view concat V 为 `[32,655,128]`；#75 `bmm` 得到 `[32,1,128]`；#76 view；#77 transpose；#78 view 为 `[1,1,4096]`。

```text
Single-query value read
                                                       0                                       128
                                                       ▲                                        ▲
weights [1,655]                                ──▶    ┌────────────────────────────────────────┐
                                                       │ SINGLE_QUERY_WEIGHT_ROW               │
                                                       └────────────────────────────────────────┘
values [655,Dh]                                ──▶    ┌────────────────────────────────────────┐
                                                       │ CONCAT_VALUE_ROWS                     │
                                                       └────────────────────────────────────────┘
context [1,Dh]                                 ──▶    ┌────────────────────────────────────────┐
                                                       │ CURRENT_CONTEXT_ROW                   │
                                                       └────────────────────────────────────────┘
merged [1,H]                                   ──▶    ┌────────────────────────────────────────┐
                                                       │ MERGED_CURRENT_HIDDEN                 │
                                                       └────────────────────────────────────────┘
```

## Attention output projection and residual

- 是什么: #79-#83 对当前 token attention hidden 做 output projection 并加 residual。
- 为什么需要: context 需要回到 hidden 表示并与输入 current hidden 相加。
- 怎么做/计算: #79 view `[1,4096]`；#80 取 projection weight；#81 `mm`；#82 reshape `[1,1,4096]`；#83 与 `arg0_1` 相加。

```text
Current-token output projection
                                                       0                                      4096
                                                       ▲                                        ▲
attention row                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_CURRENT_ROW                 │
                                                       └────────────────────────────────────────┘
projected row                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ OUTPUT_PROJECTION_ROW                 │
                                                       └────────────────────────────────────────┘
residual row                                   ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROW                │
                                                       └────────────────────────────────────────┘
```

## Post-attention RMSNorm

- 是什么: #84-#93 对 attention residual 做 MLP 前 RMSNorm。
- 为什么需要: 当前 token 的 MLP projection 需要稳定 hidden 尺度。
- 怎么做/计算: #84 转 fp32；#85 平方；#86 Hidden 均值；#87 加 epsilon；#88 `rsqrt`；#89 缩放；#90 转 fp16；#91 读取 norm weight；#92 乘权重；#93 view。

```text
Post-attention RMSNorm for one token
                                                       0                                      4096
                                                       ▲                                        ▲
residual row                                   ──▶    ┌────────────────────────────────────────┐
                                                       │ ATTENTION_RESIDUAL_ROW                │
                                                       └────────────────────────────────────────┘
rms scalar                                     ──▶    [RMS_REDUCTION_SCALAR]
mlp input row                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROW              │
                                                       └────────────────────────────────────────┘
```

## MLP and final residual

- 是什么: #94-#107 是单 token gated MLP 和 final residual。
- 为什么需要: MLP 对当前 token hidden 做 channel 非线性变换。
- 怎么做/计算: #94/#95 gate projection；#96 reshape；#97 `silu`；#98-#101 up projection；#102 gate/up 相乘；#103 view `[1,11008]`；#104/#105 down projection；#106 reshape；#107 与 #83 residual 相加。

```text
MLP H=4096 -> I=11008 -> H=4096
                                                       0                                      11008
                                                       ▲                                        ▲
mlp input row                                  ──▶    ┌────────────────────────────────────────┐
                                                       │ NORMALIZED_MLP_INPUT_ROW              │
                                                       └────────────────────────────────────────┘
gate/up row                                   ──▶     ┌────────────────────────────────────────┐
                                                       │ GATE_AND_UP_ROW                       │
                                                       └────────────────────────────────────────┘
activated row                                 ──▶     ┌────────────────────────────────────────┐
                                                       │ ACTIVATED_INTERMEDIATE_ROW            │
                                                       └────────────────────────────────────────┘
final hidden                                  ──▶     ┌────────────────────────────────────────┐
                                                       │ FINAL_CURRENT_HIDDEN                  │
                                                       └────────────────────────────────────────┘
```

## Layer output

- 是什么: #108 打包当前 token hidden 和更新后的 K/V cache。
- 为什么需要: decode 下一步需要新的 hidden 输出，以及包含当前 token 的 K/V cache。
- 怎么做/计算: #108 返回 `(add_tensor_6, {'dynamic_cache_layer': (cat_default_2, cat_default_3)}, None, 0)`；`cat_default_2` 和 `cat_default_3` 是 #56/#58 拼接出的更新后 cache。

```text
Output tuple

hidden_states      #107 add_tensor_6             ──▶   [OUTPUT_HIDDEN_ROW: B=1,S=1,H=4096]
dynamic_cache.K    #56 cat_default_2             ──▶   [CACHE_KEY_ROWS: B=1,Heads=32,K=655,Dh=128]
dynamic_cache.V    #58 cat_default_3             ──▶   [CACHE_VALUE_ROWS: B=1,Heads=32,K=655,Dh=128]
visual output      literal None                  ──▶   [NO_VISUAL_TOKEN_OUTPUT]
exit indicator     literal 0                     ──▶   [EXIT_INDICATOR_SCALAR]
```
