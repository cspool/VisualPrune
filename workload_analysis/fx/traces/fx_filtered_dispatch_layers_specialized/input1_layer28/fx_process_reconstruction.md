# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer28`
GraphModule: `fx_graph_module.pt`

Source: this file is reconstructed by loading `fx_graph_module.pt` and iterating `GraphModule.graph.nodes`.
FX provides the graph DAG and node metadata; the process labels below are reconstruction labels over that DAG.

## Stage Summary

| stage | node range | node count | external inputs | external outputs |
| --- | ---: | ---: | --- | --- |
| Runtime FX inputs | 0-7 | 8 | - | `arg0_1`, `arg1_1`, `arg2_1`, `arg6_1` |
| Input RMSNorm | 8-15 | 8 | `arg0_1` | `_param_constant0`, `_to_copy_default_1` |
| Q/K/V projection and head reshape | 16-34 | 19 | `_param_constant0`, `_to_copy_default_1` | `transpose_int`, `transpose_int_1`, `transpose_int_2` |
| RoPE position embedding | 35-54 | 20 | `arg2_1`, `transpose_int`, `transpose_int_1` | `add_tensor_1`, `add_tensor_2` |
| QK scores, mask, softmax | 55-66 | 12 | `add_tensor_1`, `add_tensor_2`, `arg1_1` | `clone_default` |
| Attention-weighted V and hidden reshape | 67-75 | 9 | `clone_default`, `transpose_int_2` | `view_default_12` |
| Attention output projection and residual | 76-80 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 81-90 | 10 | `add_tensor_4` | `mul_tensor_7`, `view_default_14` |
| MLP and final residual | 91-104 | 14 | `add_tensor_4`, `mul_tensor_7`, `view_default_14` | `add_tensor_6` |
| Layer output | 105-105 | 1 | `add_tensor_2`, `add_tensor_6`, `arg6_1`, `transpose_int_2` | - |

## Process Code

### Runtime FX inputs

```python
# placeholder arg0_1
# placeholder arg1_1
# placeholder arg2_1
# placeholder arg3_1
# placeholder arg4_1
# placeholder arg5_1
# placeholder arg6_1
# placeholder arg7_1
```

**解释与可视化**

是什么：这一段是固定输入样本暴露给 FX GraphModule 的入口节点。`arg0_1` 是本层 hidden states，`arg1_1` 是 attention mask，`arg2_1` 是 position ids，`arg6_1` 是进入该层时已有的 `important_vis_tokens`；`arg3_1`、`arg4_1`、`arg5_1`、`arg7_1` 在这个固定 DAG 中没有用户节点。

为什么需要：后续每个计算段都依赖这些入口张量来固定本次 layer28 的实际形状：hidden 行数是 48，Hidden 宽度是 4096，attention 的 Q/K 方阵是 48 x 48，RoPE 通过 48 个 position ids 查表。这个目录没有单独的 `visual_process` stage，`important_vis_tokens` 在本层只作为输出元组的一部分透传。

怎么做/计算：placeholder 本身不做数值计算，只建立数据依赖。`arg0_1` 被 `_to_copy_default` 读入 input RMSNorm，并在 `add_tensor_4` 中作为 attention residual 的左操作数；`arg1_1` 只在 `add_tensor_3` 中加到 scaled QK logits；`arg2_1` 被 `index_tensor` 和 `index_tensor_1` 用作 cos/sin 表索引；`arg6_1` 只被 `output` 读取，作为已有 Visual token/index 信息透传。

```text
Tensor: InputHidden X, shape [B=1, S=48, H=4096]
Formula: X[s,h] is forwarded to RMSNorm and residual add
Token axis S=0..47 (compressed)                      Hidden axis H=0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
input hidden rows                                ──▶   +----------------------------------------+  ◀── row examples: X[0,0], X[47,4095]
                                                       | HIDDEN_ROW_s                           |
                                                       | HIDDEN_ROW_s+1 .. HIDDEN_ROW_46        |
                                                       | HIDDEN_ROW_47                          |
                                                       +----------------------------------------+

Tensor: AttentionMask M, shape [B=1, HeadBroadcast=1, Q=48, K=48]
Formula: masked_logits[q,k] = scaled_qk[q,k] + M[q,k]
Q_seq axis 0..47                                      K_seq axis 0..47
                                                       0                                      47
                                                       ▲                                       ▲
mask grid                                      ──▶     +----------------------------------------+  ◀── examples: M[0,0], M[47,47]
                                                       | MASK_ROW_q_OVER_KEYS                   |
                                                       | MASK_ROWS_1_TO_46                      |
                                                       | MASK_ROW_47_OVER_KEYS                  |
                                                       +----------------------------------------+

Tensor: PositionIds P, shape [B=1, S=48]
Formula: cos_sin_row[s,:] = table[P[s],:]
S=0                     S=47
▲                        ▲
[ P0 | P1 | ... | P47 ]  ──▶ used for RoPE table lookup

Tensor: ImportantVisTokens I, shape [P=10]
Formula: I is passthrough in this fixed DAG
P=0              P=9
▲                 ▲
[ I0 | ... | I9 ]  ──▶ returned unchanged
```

### Input RMSNorm

```python
_to_copy_default = aten._to_copy.default(arg0_1, dtype=torch.float32)
pow_tensor_scalar = aten.pow.Tensor_Scalar(_to_copy_default, 2)
mean_dim = aten.mean.dim(pow_tensor_scalar, [-1], True)
add_tensor = aten.add.Tensor(mean_dim, 1e-05)
rsqrt_default = aten.rsqrt.default(add_tensor)
mul_tensor = aten.mul.Tensor(_to_copy_default, rsqrt_default)
_to_copy_default_1 = aten._to_copy.default(mul_tensor, dtype=torch.float16)
_param_constant0 = self._param_constant0
```

**解释与可视化**

是什么：这一段对输入 hidden states 的每个 token 行独立做 RMSNorm，得到供 Q/K/V projection 使用的 fp16 normalized hidden，并读取长度为 4096 的 input norm 权重。

为什么需要：Q/K/V 的线性投影对 Hidden 维的尺度敏感。RMSNorm 先把每个 token 行按 Hidden 维均方根归一化，再交给后续投影，使 attention 分支输入的数值尺度稳定。

怎么做/计算：`_to_copy_default` 把 `arg0_1` 转成 fp32，给归一化保留更高精度；`pow_tensor_scalar` 对每个 Hidden 元素平方；`mean_dim` 在最后一维 Hidden 上求均值并保留 `[B,S,1]` 形状；`add_tensor` 加 `1e-05` 防止零除；`rsqrt_default` 得到每行的 inverse RMS；`mul_tensor` 把该 `[B,S,1]` 缩放广播回 `[B,S,H]` 并逐元素乘输入；`_to_copy_default_1` 转回 fp16；`_param_constant0` 是 `[4096]` 的 norm weight，下一段用它和 normalized hidden 逐 Hidden 维相乘。

```text
Tensor: InputRows X_fp32, shape [S=48, H=4096]
Formula: X_fp32[s,h] = float32(InputHidden[s,h])
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
input rows                                      ──▶    +----------------------------------------+  ◀── examples: X[0,0], X[47,4095]
                                                       | INPUT_ROW_0                            |
                                                       | INPUT_ROWS_1_TO_46                     |
                                                       | INPUT_ROW_47                           |
                                                       +----------------------------------------+

Tensor: RowRMSScale R, shape [S=48, 1]
Formula: R[s] = rsqrt(mean_h(X_fp32[s,h]^2) + 1e-05)
S axis 0..47                                           scalar column
                                                       +----------+
row scale                                      ──▶     | R_0      |  ◀── reduced from INPUT_ROW_0 over H=0..4095
                                                       | R_1..46  |
                                                       | R_47     |
                                                       +----------+

Tensor: NormRows N, shape [S=48, H=4096]
Formula: N[s,h] = fp16(X_fp32[s,h] * R[s])
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
normalized rows                                ──▶     +----------------------------------------+  ◀── same S/H coordinates as input
                                                       | NORM_ROW_0                             |
                                                       | NORM_ROWS_1_TO_46                      |
                                                       | NORM_ROW_47                            |
                                                       +----------------------------------------+
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [48, 4096])
_tensor_constant184 = self._tensor_constant184
mm_default = aten.mm.default(view_default, _tensor_constant184)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 48, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [48, 4096])
_tensor_constant185 = self._tensor_constant185
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant185)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 48, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [48, 4096])
_tensor_constant186 = self._tensor_constant186
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant186)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 48, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 48, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 48, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 48, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段把 input RMSNorm 输出映射成 Q、K、V 三个 attention 分支，并把每个 `[S,4096]` 投影结果重新解释为 32 个 head、每个 head 128 维。

为什么需要：attention 的 Q/K 点积需要按 head 计算相似度，V 则提供随后被 attention 权重加权求和的内容。拆成 `[Heads=32, Dh=128]` 后，QK score 和 value 汇聚都能在每个 head 内独立进行。

怎么做/计算：`mul_tensor_1` 将 `[4096]` 的 `_param_constant0` 广播到 normalized hidden 上，得到投影前的加权 hidden；Q 分支用 `view_default` 把它展成 `[48,4096]`，`mm_default` 乘 `_tensor_constant184 [4096,4096]`，再 `_unsafe_view_default` 恢复 `[1,48,4096]`；K/V 分支分别通过 `view_default_1/mm_default_1/_unsafe_view_default_1` 和 `view_default_2/mm_default_2/_unsafe_view_default_2` 做相同结构的投影。随后 `view_default_3/4/5` 把三个结果改成 `[1,48,32,128]`，`transpose_int/1/2` 交换 token/head 轴，输出 `[1,32,48,128]` 的 Q、K、V。

```text
Tensor: WeightedNormRows A, shape [S=48, H=4096]
Formula: A[s,h] = NormRows[s,h] * NormWeight[h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
shared input                                   ──▶     +----------------------------------------+  ◀── reused by Q, K, and V projections
                                                       | WEIGHTED_NORM_ROW_0                    |
                                                       | WEIGHTED_NORM_ROWS_1_TO_46             |
                                                       | WEIGHTED_NORM_ROW_47                   |
                                                       +----------------------------------------+

Tensor: ProjectedRows Q2D/K2D/V2D, each shape [S=48, H=4096]
Formula: Q2D = A @ Wq, K2D = A @ Wk, V2D = A @ Wv
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
linear projection                              ──▶     +----------------------------------------+  ◀── three aligned S x H products
                                                       | Q2D_ROW_s / K2D_ROW_s / V2D_ROW_s      |
                                                       | ROWS_1_TO_46                           |
                                                       | ROW_47                                 |
                                                       +----------------------------------------+

Tensor: HeadSplit Q/K/V, each shape [B=1, Heads=32, S=48, Dh=128]
Formula: HeadSplit[b,head,s,dh] = ProjectedRows[s, head*128 + dh]
For one representative head h, show last two axes S x Dh.
S axis 0..47                                           Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
head rows                                      ──▶     +----------------------------------------+  ◀── examples: Q[h,0,0], K[h,47,127], V[h,35,64]
                                                       | HEAD_h_TOKEN_0_DH_ROW                  |
                                                       | HEAD_h_TOKEN_ROWS_1_TO_46              |
                                                       | HEAD_h_TOKEN_47_DH_ROW                 |
                                                       +----------------------------------------+
```

### RoPE position embedding

```python
_tensor_constant187 = self._tensor_constant187
index_tensor = aten.index.Tensor(_tensor_constant187, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant188 = self._tensor_constant188
index_tensor_1 = aten.index.Tensor(_tensor_constant188, [arg2_1])
unsqueeze_default_1 = aten.unsqueeze.default(index_tensor_1, 1)
mul_tensor_2 = aten.mul.Tensor(transpose_int, unsqueeze_default)
slice_tensor = aten.slice.Tensor(transpose_int, 3, 0, 64)
slice_tensor_1 = aten.slice.Tensor(transpose_int, 3, 64, 9223372036854775807)
neg_default = aten.neg.default(slice_tensor_1)
cat_default = aten.cat.default([neg_default, slice_tensor], -1)
mul_tensor_3 = aten.mul.Tensor(cat_default, unsqueeze_default_1)
add_tensor_1 = aten.add.Tensor(mul_tensor_2, mul_tensor_3)
mul_tensor_4 = aten.mul.Tensor(transpose_int_1, unsqueeze_default)
slice_tensor_2 = aten.slice.Tensor(transpose_int_1, 3, 0, 64)
slice_tensor_3 = aten.slice.Tensor(transpose_int_1, 3, 64, 9223372036854775807)
neg_default_1 = aten.neg.default(slice_tensor_3)
cat_default_1 = aten.cat.default([neg_default_1, slice_tensor_2], -1)
mul_tensor_5 = aten.mul.Tensor(cat_default_1, unsqueeze_default_1)
add_tensor_2 = aten.add.Tensor(mul_tensor_4, mul_tensor_5)
```

**解释与可视化**

是什么：这一段对 Q 和 K 的每个 head 向量应用 RoPE 位置旋转，输出带位置信息的 rotated Q 和 rotated K。V 不参与 RoPE，保持上一段的 value head 布局。

为什么需要：后续 QK 点积只比较向量本身，如果不把 token 位置编码进 Q/K，attention score 无法表达相对位置关系。RoPE 通过按 position ids 查 cos/sin 表，并在 Dh 轴两半之间旋转，把位置信息注入 Q/K。

怎么做/计算：`index_tensor` 用 `arg2_1 [1,48]` 从 `_tensor_constant187 [624,128]` 取每个 token 的 cos 行，`unsqueeze_default` 插入 head 广播维；`index_tensor_1` 和 `unsqueeze_default_1` 对 sin 表做相同查表。Q 分支先用 `mul_tensor_2` 计算 `Q * cos`；`slice_tensor` 取 Dh `0..63`，`slice_tensor_1` 取 Dh `64..127`，`neg_default` 对右半取负，`cat_default` 拼成 `[-right, left]` 的 rotate-half；`mul_tensor_3` 乘 sin，`add_tensor_1` 得到 rotated Q。K 分支用 `mul_tensor_4`、`slice_tensor_2/3`、`neg_default_1`、`cat_default_1`、`mul_tensor_5`、`add_tensor_2` 做同样的旋转，输出 rotated K 并同时进入 cache。

```text
Tensor: QKHeadRows X, shape [Heads=32, S=48, Dh=128]
Formula: X is either Q or K before RoPE; show one head/token row over Dh
Dh axis 0..127, split into two halves
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
input x ──▶ +--------------------------+-+--------------------------+  ◀── examples: x[...,0], x[...,64], x[...,127]
            | LEFT_HALF                | | RIGHT_HALF               |
            +--------------------------+-+--------------------------+

Tensor: RotateHalf(X), shape [Heads=32, S=48, Dh=128]
Formula: RotateHalf(X)[...,0:64] = -X[...,64:128]; RotateHalf(X)[...,64:128] = X[...,0:64]
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
rot x  ──▶  +--------------------------+-+--------------------------+  ◀── left/right halves exchange roles
            | NEG_RIGHT_HALF           | | LEFT_HALF                |
            +--------------------------+-+--------------------------+

Tensor: RoPERows Y, shape [Heads=32, S=48, Dh=128]
Formula: Y[head,s,dh] = X[head,s,dh] * Cos[P[s],dh] + RotateHalf(X)[head,s,dh] * Sin[P[s],dh]
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
output ──▶  +--------------------------+-+--------------------------+  ◀── same Dh coordinates after rotation
            | ROTATED_LEFT_HALF        | | ROTATED_RIGHT_HALF       |
            +--------------------------+-+--------------------------+
```

### QK scores, mask, softmax

```python
transpose_int_3 = aten.transpose.int(add_tensor_2, 2, 3)
expand_default = aten.expand.default(add_tensor_1, [1, 32, 48, 128])
view_default_6 = aten.view.default(expand_default, [32, 48, 128])
expand_default_1 = aten.expand.default(transpose_int_3, [1, 32, 128, 48])
view_default_7 = aten.view.default(expand_default_1, [32, 128, 48])
bmm_default = aten.bmm.default(view_default_6, view_default_7)
view_default_8 = aten.view.default(bmm_default, [1, 32, 48, 48])
div_tensor = aten.div.Tensor(view_default_8, 11.313708498984761)
add_tensor_3 = aten.add.Tensor(div_tensor, arg1_1)
_softmax_default = aten._softmax.default(add_tensor_3, -1, True)
_to_copy_default_2 = aten._to_copy.default(_softmax_default, dtype=torch.float16)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段计算标准 scaled dot-product attention 的 QK logits、加 mask，并沿 K 轴 softmax 得到 attention weights。该 DAG 中这里没有 `fill_`、`sum`、`copy_` 这类 Visual attention 区域改写节点。

为什么需要：下一段要用 `[Q,K]` 权重对 V 的 K-token 行加权求和，因此这里必须先把每个 head 内每个 query token 对所有 key token 的权重归一化出来。

怎么做/计算：`transpose_int_3` 把 rotated K 的最后两维从 `[S=48,Dh=128]` 转成 `[Dh=128,K=48]`；`expand_default/view_default_6` 将 rotated Q 变成 `[32,48,128]`，`expand_default_1/view_default_7` 将转置后的 K 变成 `[32,128,48]`；`bmm_default` 在每个 head 内做 `[Q,Dh] x [Dh,K]`，得到 `[32,48,48]` QK score；`view_default_8` 恢复 `[1,32,48,48]`；`div_tensor` 除以 `11.313708498984761` 即 `sqrt(128)`；`add_tensor_3` 加上 `[1,1,48,48]` mask 并广播到 32 个 head；`_softmax_default` 沿最后一维 K 归一化；`_to_copy_default_2` 转 fp16；`clone_default` 生成后续 value 汇聚使用的 attention weights。

```text
Tensor: QKLogits L, shape [Heads=32, Q=48, K=48]
Formula: L[head,q,k] = dot(RoPE_Q[head,q,:], RoPE_K[head,k,:]) / sqrt(128) + Mask[q,k]
For one representative head, show last two axes Q x K.
Q_seq axis 0..47                                      K_seq axis 0..47
                                                       0                                      47
                                                       ▲                                       ▲
masked logits                                 ──▶     +----------------------------------------+  ◀── row q contains scores over all K
                                                       | LOGIT_ROW_Q0                           |
                                                       | LOGIT_ROWS_Q1_TO_Q46                   |
                                                       | LOGIT_ROW_Q47                          |
                                                       +----------------------------------------+

Tensor: AttentionWeights W, shape [Heads=32, Q=48, K=48]
Formula: W[head,q,k] = softmax_k(L[head,q,k])
Q_seq axis 0..47                                      K_seq axis 0..47
                                                       0                                      47
                                                       ▲                                       ▲
weights                                       ──▶     +----------------------------------------+  ◀── each row sums over K after mask
                                                       | WEIGHT_ROW_Q0                          |
                                                       | WEIGHT_ROWS_Q1_TO_Q46                  |
                                                       | WEIGHT_ROW_Q47                         |
                                                       +----------------------------------------+
examples: W[head=0,q=0,k=0], W[head=0,q=35,k=10], W[head=31,q=47,k=47]
```

### Attention-weighted V and hidden reshape

```python
expand_default_2 = aten.expand.default(clone_default, [1, 32, 48, 48])
view_default_9 = aten.view.default(expand_default_2, [32, 48, 48])
expand_default_3 = aten.expand.default(transpose_int_2, [1, 32, 48, 128])
view_default_10 = aten.view.default(expand_default_3, [32, 48, 128])
bmm_default_1 = aten.bmm.default(view_default_9, view_default_10)
view_default_11 = aten.view.default(bmm_default_1, [1, 32, 48, 128])
transpose_int_4 = aten.transpose.int(view_default_11, 1, 2)
clone_default_1 = aten.clone.default(transpose_int_4, memory_format=torch.contiguous_format)
view_default_12 = aten.view.default(clone_default_1, [1, 48, 4096])
```

**解释与可视化**

是什么：这一段用 attention weights 对 V 做 value 汇聚，得到每个 head 的 context，再把 head 轴和 Dh 轴合并回 Hidden 维。

为什么需要：QK softmax 只给出“看哪些 key token”的权重；真正传给输出投影的是这些权重对 value 内容的加权和。合并多头后，attention 分支才能重新回到 `[B,S,4096]` 的 hidden 宽度。

怎么做/计算：`expand_default_2/view_default_9` 将 attention weights 变成 `[32,48,48]`；`expand_default_3/view_default_10` 将 V head tensor 变成 `[32,48,128]`；`bmm_default_1` 在每个 head 内计算 `[Q,K] x [K,Dh] -> [Q,Dh]`，得到每个 query token 的 value 加权和；`view_default_11` 恢复 batch/head 形状 `[1,32,48,128]`；`transpose_int_4` 改成 `[1,48,32,128]`；`clone_default_1` 使内存连续；`view_default_12` 把 `32*128` 合并为 Hidden，输出 `[1,48,4096]`。

```text
Tensor: AttentionWeights W, shape [Heads=32, Q=48, K=48]
Formula: W[head,q,k] comes from softmax over K
Q_seq axis 0..47                                      K_seq axis 0..47
                                                       0                                      47
                                                       ▲                                       ▲
weights                                       ──▶     +----------------------------------------+  ◀── one row selects a weighted mix of V rows
                                                       | WEIGHT_ROW_Q0                          |
                                                       | WEIGHT_ROWS_Q1_TO_Q46                  |
                                                       | WEIGHT_ROW_Q47                         |
                                                       +----------------------------------------+

Tensor: ValueRows V, shape [Heads=32, K=48, Dh=128]
Formula: V[head,k,dh] is the value content for each key token
K_seq axis 0..47                                      Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
value rows                                    ──▶     +----------------------------------------+  ◀── same K axis consumed by W
                                                       | VALUE_ROW_K0                           |
                                                       | VALUE_ROWS_K1_TO_K46                   |
                                                       | VALUE_ROW_K47                          |
                                                       +----------------------------------------+

Tensor: ContextRows C, shape [Heads=32, Q=48, Dh=128]
Formula: C[head,q,dh] = sum_k W[head,q,k] * V[head,k,dh]
Q_seq axis 0..47                                      Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
context                                       ──▶     +----------------------------------------+  ◀── examples: C[0,0,0], C[31,47,127]
                                                       | CONTEXT_ROW_Q0                         |
                                                       | CONTEXT_ROWS_Q1_TO_Q46                 |
                                                       | CONTEXT_ROW_Q47                        |
                                                       +----------------------------------------+

Tensor: MergedContext O, shape [B=1, S=48, H=4096]
Formula: O[s, head*128 + dh] = C[head,s,dh]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
merged output                                  ──▶     +----------------------------------------+  ◀── heads are concatenated along H
                                                       | MERGED_CONTEXT_ROW_0                   |
                                                       | MERGED_CONTEXT_ROWS_1_TO_46            |
                                                       | MERGED_CONTEXT_ROW_47                  |
                                                       +----------------------------------------+
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [48, 4096])
_tensor_constant189 = self._tensor_constant189
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant189)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 48, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把合并后的 attention context 投影回模型 hidden space，然后和原始输入 hidden 在同一 `[S,H]` 坐标上逐元素相加，形成 attention residual 输出。

为什么需要：多头 context 合并后仍需要通过输出投影重新混合通道；残差连接保留进入该层的原始 hidden 信息，使后续 RMSNorm/MLP 在 attention 更新后的表示上继续计算。

怎么做/计算：`view_default_13` 把 `[1,48,4096]` context 展为 `[48,4096]`；`mm_default_3` 乘 `_tensor_constant189 [4096,4096]` 得到输出投影结果；`_unsafe_view_default_3` 恢复 `[1,48,4096]`；`add_tensor_4` 将投影结果和 `arg0_1` 在 token/Hidden 坐标上逐元素相加，输出 post-attention residual。

```text
Tensor: ProjectedAttention P, shape [S=48, H=4096]
Formula: P[s,h] = MergedContext[s,:] @ Wo[:,h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
projection                                    ──▶     +----------------------------------------+  ◀── output projection rows
                                                       | PROJECTED_ATTENTION_ROW_0              |
                                                       | PROJECTED_ATTENTION_ROWS_1_TO_46       |
                                                       | PROJECTED_ATTENTION_ROW_47             |
                                                       +----------------------------------------+

Tensor: ResidualInput X, shape [S=48, H=4096]
Formula: X[s,h] is the original layer input on the same coordinates
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual source                                ──▶     +----------------------------------------+  ◀── aligned with P row-for-row
                                                       | INPUT_ROW_0                            |
                                                       | INPUT_ROWS_1_TO_46                     |
                                                       | INPUT_ROW_47                           |
                                                       +----------------------------------------+

Tensor: AttentionResidual A, shape [S=48, H=4096]
Formula: A[s,h] = X[s,h] + P[s,h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual output                                ──▶     +----------------------------------------+  ◀── examples: A[0,0], A[35,2048], A[47,4095]
                                                       | ATTENTION_RESIDUAL_ROW_0               |
                                                       | ATTENTION_RESIDUAL_ROWS_1_TO_46        |
                                                       | ATTENTION_RESIDUAL_ROW_47              |
                                                       +----------------------------------------+
```

### Post-attention RMSNorm

```python
_to_copy_default_3 = aten._to_copy.default(add_tensor_4, dtype=torch.float32)
pow_tensor_scalar_1 = aten.pow.Tensor_Scalar(_to_copy_default_3, 2)
mean_dim_1 = aten.mean.dim(pow_tensor_scalar_1, [-1], True)
add_tensor_5 = aten.add.Tensor(mean_dim_1, 1e-05)
rsqrt_default_1 = aten.rsqrt.default(add_tensor_5)
mul_tensor_6 = aten.mul.Tensor(_to_copy_default_3, rsqrt_default_1)
_to_copy_default_4 = aten._to_copy.default(mul_tensor_6, dtype=torch.float16)
_param_constant5 = self._param_constant5
mul_tensor_7 = aten.mul.Tensor(_param_constant5, _to_copy_default_4)
view_default_14 = aten.view.default(mul_tensor_7, [48, 4096])
```

**解释与可视化**

是什么：这一段对 attention residual 再做一次 RMSNorm，输出供 MLP gate/up projection 共享的 normalized rows。

为什么需要：attention residual 经过一次残差加法后尺度会变化。MLP 前的 RMSNorm 将每个 token 行重新按 Hidden 维标准化，使后续 gate/up/down projection 的输入尺度稳定。

怎么做/计算：`_to_copy_default_3` 把 `add_tensor_4` 转 fp32；`pow_tensor_scalar_1` 对每个 Hidden 元素平方；`mean_dim_1` 沿 Hidden 求均值并保留 `[B,S,1]`；`add_tensor_5` 加 `1e-05`；`rsqrt_default_1` 得到 inverse RMS；`mul_tensor_6` 将 inverse RMS 广播回 `[B,S,H]` 逐元素缩放；`_to_copy_default_4` 转 fp16；`_param_constant5` 是 `[4096]` 的 post-attention norm weight；`mul_tensor_7` 逐 Hidden 维应用权重；`view_default_14` 把结果展为 `[48,4096]` 供 gate projection 使用，同时同一 normalized tensor 也会被 `view_default_15` 供 up projection 使用。

```text
Tensor: AttentionResidual A_fp32, shape [S=48, H=4096]
Formula: A_fp32[s,h] = float32(AttentionResidual[s,h])
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual rows                                  ──▶     +----------------------------------------+  ◀── reduced over H for each token
                                                       | ATTENTION_RESIDUAL_ROW_0               |
                                                       | ATTENTION_RESIDUAL_ROWS_1_TO_46        |
                                                       | ATTENTION_RESIDUAL_ROW_47              |
                                                       +----------------------------------------+

Tensor: MLPScale R2, shape [S=48, 1]
Formula: R2[s] = rsqrt(mean_h(A_fp32[s,h]^2) + 1e-05)
S axis 0..47                                           scalar column
                                                       +----------+
row scale                                      ──▶     | R2_0     |  ◀── one scale per token row
                                                       | R2_1..46 |
                                                       | R2_47    |
                                                       +----------+

Tensor: MLPInput U, shape [S=48, H=4096]
Formula: U[s,h] = fp16(A_fp32[s,h] * R2[s]) * MLPNormWeight[h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
mlp input                                     ──▶      +----------------------------------------+  ◀── examples: U[0,0], U[35,1024], U[47,4095]
                                                       | MLP_INPUT_ROW_0                        |
                                                       | MLP_INPUT_ROWS_1_TO_46                 |
                                                       | MLP_INPUT_ROW_47                       |
                                                       +----------------------------------------+
```

### MLP and final residual

```python
_tensor_constant190 = self._tensor_constant190
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant190)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 48, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [48, 4096])
_tensor_constant191 = self._tensor_constant191
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant191)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 48, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [48, 11008])
_tensor_constant192 = self._tensor_constant192
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant192)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 48, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP：同一 MLP 输入分成 gate 和 up 两个 `[S,11008]` 分支，gate 经过 SiLU 后与 up 逐元素相乘，再 down project 回 `[S,4096]` 并加回 attention residual。

为什么需要：attention 负责跨 token 汇聚，MLP 负责每个 token 内的通道非线性变换。gate/up 乘法提供可控的中间通道激活，down projection 回到 Hidden 宽度后通过 residual add 继续传给下一层。

怎么做/计算：`_tensor_constant190 [4096,11008]` 是 gate projection 权重，`mm_default_4` 将 `view_default_14 [48,4096]` 投到 `[48,11008]`，`_unsafe_view_default_4` 恢复 batch，`silu_default` 对 gate 分支逐元素做 SiLU；`view_default_15` 复用同一个 `mul_tensor_7` 形成 up 分支输入，`mm_default_5` 乘 `_tensor_constant191 [4096,11008]` 得到 up activations；`mul_tensor_8` 将 `silu(gate)` 和 up 在 `[S,11008]` 上逐元素相乘；`view_default_16` 展成 `[48,11008]`；`mm_default_6` 乘 `_tensor_constant192 [11008,4096]` down project 回 Hidden；`_unsafe_view_default_6` 恢复 `[1,48,4096]`；`add_tensor_6` 与 `add_tensor_4` 在相同 `[S,H]` 坐标逐元素相加。

```text
Tensor: MLPInput U, shape [S=48, H=4096]
Formula: U is the normalized post-attention residual
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
shared mlp input                               ──▶     +----------------------------------------+  ◀── reused by gate and up projections
                                                       | MLP_INPUT_ROW_0                        |
                                                       | MLP_INPUT_ROWS_1_TO_46                 |
                                                       | MLP_INPUT_ROW_47                       |
                                                       +----------------------------------------+

Tensor: GatedIntermediate G, shape [S=48, I=11008]
Formula: G[s,i] = SiLU(U[s,:] @ W_gate[:,i]) * (U[s,:] @ W_up[:,i])
S axis 0..47                                           Intermediate axis I=0..11007
                                                       0                                     11007
                                                       ▲                                       ▲
gated rows                                    ──▶      +----------------------------------------+  ◀── examples: G[0,0], G[47,11007]
                                                       | SILU_GATE_TIMES_UP_ROW_0               |
                                                       | SILU_GATE_TIMES_UP_ROWS_1_TO_46        |
                                                       | SILU_GATE_TIMES_UP_ROW_47              |
                                                       +----------------------------------------+

Tensor: MLPDown D, shape [S=48, H=4096]
Formula: D[s,h] = G[s,:] @ W_down[:,h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
down projection                                ──▶     +----------------------------------------+  ◀── returns to Hidden width
                                                       | MLP_DOWN_ROW_0                         |
                                                       | MLP_DOWN_ROWS_1_TO_46                  |
                                                       | MLP_DOWN_ROW_47                        |
                                                       +----------------------------------------+

Tensor: FinalHidden Y, shape [S=48, H=4096]
Formula: Y[s,h] = AttentionResidual[s,h] + D[s,h]
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
final residual                                 ──▶     +----------------------------------------+  ◀── examples: Y[0,0], Y[35,2048], Y[47,4095]
                                                       | FINAL_HIDDEN_ROW_0                     |
                                                       | FINAL_HIDDEN_ROWS_1_TO_46              |
                                                       | FINAL_HIDDEN_ROW_47                    |
                                                       +----------------------------------------+
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, arg6_1, 2)
```

**解释与可视化**

是什么：这一段是输出打包节点，不再做新的数值运算。返回值包含 final hidden、当前层 K/V cache、透传的 `important_vis_tokens`，以及固定控制标量 `2`。

为什么需要：`add_tensor_6` 是 layer28 的主 hidden 输出，供下一层继续计算；`add_tensor_2` 和 `transpose_int_2` 分别是当前层 rotated K 与 V，用于后续 cache 复用；`arg6_1` 保存进入本层时已有的 Visual token/index 信息。本层 FX DAG 没有观测到 value-aware Visual token 选择或 attention 区域改写，只在 output 中保留该输入。

怎么做/计算：`output` 节点把 `add_tensor_6` 放在返回元组第一项；把 `add_tensor_2` 和 `transpose_int_2` 放入 `{'dynamic_cache_layer': (...)}`；把 `arg6_1` 原样放到第三项；最后返回标量 `2`。这些值都已由前面节点产生或由 placeholder 提供，`return` 本身只组织结构。

```text
Tensor: ReturnHidden Y, shape [B=1, S=48, H=4096]
Formula: Y is the final residual hidden from MLP
S axis 0..47                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
return item 0                                  ──▶     +----------------------------------------+  ◀── consumed by the next layer
                                                       | FINAL_HIDDEN_ROW_0                     |
                                                       | FINAL_HIDDEN_ROWS_1_TO_46              |
                                                       | FINAL_HIDDEN_ROW_47                    |
                                                       +----------------------------------------+

Tensor: DynamicCache K/V, shapes K=[B=1, Heads=32, S=48, Dh=128], V=[B=1, Heads=32, S=48, Dh=128]
Formula: cache = (RoPE_K, ValueRows)
For one representative head, show S x Dh.
S axis 0..47                                           Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
cache K/V                                     ──▶     +----------------------------------------+  ◀── examples: K[31,47,127], V[28,47,64]
                                                       | CACHE_ROW_TOKEN_0                      |
                                                       | CACHE_ROWS_TOKEN_1_TO_46               |
                                                       | CACHE_ROW_TOKEN_47                     |
                                                       +----------------------------------------+

Tensor: ImportantVisTokens I, shape [P=10]
Formula: I_out[p] = I_in[p]
P=0              P=9
▲                 ▲
[ I0 | ... | I9 ]  ──▶ returned unchanged

Tensor: ControlScalar C, shape []
Formula: C = 2
[ CONTROL_SCALAR_2 ] ──▶ returned as final tuple item
```

## Node Table

| index | stage | name | op | target | args | users |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `inputs` | `arg0_1` | `placeholder` | `arg0_1` | - | `_to_copy_default`, `add_tensor_4` |
| 1 | `inputs` | `arg1_1` | `placeholder` | `arg1_1` | - | `add_tensor_3` |
| 2 | `inputs` | `arg2_1` | `placeholder` | `arg2_1` | - | `index_tensor`, `index_tensor_1` |
| 3 | `inputs` | `arg3_1` | `placeholder` | `arg3_1` | - | - |
| 4 | `inputs` | `arg4_1` | `placeholder` | `arg4_1` | - | - |
| 5 | `inputs` | `arg5_1` | `placeholder` | `arg5_1` | - | - |
| 6 | `inputs` | `arg6_1` | `placeholder` | `arg6_1` | - | `output` |
| 7 | `inputs` | `arg7_1` | `placeholder` | `arg7_1` | - | - |
| 8 | `input_rmsnorm` | `_to_copy_default` | `call_function` | `aten._to_copy.default` | `arg0_1` | `mul_tensor`, `pow_tensor_scalar` |
| 9 | `input_rmsnorm` | `pow_tensor_scalar` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default` | `mean_dim` |
| 10 | `input_rmsnorm` | `mean_dim` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar` | `add_tensor` |
| 11 | `input_rmsnorm` | `add_tensor` | `call_function` | `aten.add.Tensor` | `mean_dim` | `rsqrt_default` |
| 12 | `input_rmsnorm` | `rsqrt_default` | `call_function` | `aten.rsqrt.default` | `add_tensor` | `mul_tensor` |
| 13 | `input_rmsnorm` | `mul_tensor` | `call_function` | `aten.mul.Tensor` | `_to_copy_default`, `rsqrt_default` | `_to_copy_default_1` |
| 14 | `input_rmsnorm` | `_to_copy_default_1` | `call_function` | `aten._to_copy.default` | `mul_tensor` | `mul_tensor_1` |
| 15 | `input_rmsnorm` | `_param_constant0` | `get_attr` | `_param_constant0` | - | `mul_tensor_1` |
| 16 | `qkv_projection` | `mul_tensor_1` | `call_function` | `aten.mul.Tensor` | `_param_constant0`, `_to_copy_default_1` | `view_default`, `view_default_1`, `view_default_2` |
| 17 | `qkv_projection` | `view_default` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default` |
| 18 | `qkv_projection` | `_tensor_constant184` | `get_attr` | `_tensor_constant184` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant184` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant185` | `get_attr` | `_tensor_constant185` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant185` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant186` | `get_attr` | `_tensor_constant186` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant186` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `output` |
| 35 | `rope` | `_tensor_constant187` | `get_attr` | `_tensor_constant187` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant187`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant188` | `get_attr` | `_tensor_constant188` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant188`, `arg2_1` | `unsqueeze_default_1` |
| 40 | `rope` | `unsqueeze_default_1` | `call_function` | `aten.unsqueeze.default` | `index_tensor_1` | `mul_tensor_3`, `mul_tensor_5` |
| 41 | `rope` | `mul_tensor_2` | `call_function` | `aten.mul.Tensor` | `transpose_int`, `unsqueeze_default` | `add_tensor_1` |
| 42 | `rope` | `slice_tensor` | `call_function` | `aten.slice.Tensor` | `transpose_int` | `cat_default` |
| 43 | `rope` | `slice_tensor_1` | `call_function` | `aten.slice.Tensor` | `transpose_int` | `neg_default` |
| 44 | `rope` | `neg_default` | `call_function` | `aten.neg.default` | `slice_tensor_1` | `cat_default` |
| 45 | `rope` | `cat_default` | `call_function` | `aten.cat.default` | `neg_default`, `slice_tensor` | `mul_tensor_3` |
| 46 | `rope` | `mul_tensor_3` | `call_function` | `aten.mul.Tensor` | `cat_default`, `unsqueeze_default_1` | `add_tensor_1` |
| 47 | `rope` | `add_tensor_1` | `call_function` | `aten.add.Tensor` | `mul_tensor_2`, `mul_tensor_3` | `expand_default` |
| 48 | `rope` | `mul_tensor_4` | `call_function` | `aten.mul.Tensor` | `transpose_int_1`, `unsqueeze_default` | `add_tensor_2` |
| 49 | `rope` | `slice_tensor_2` | `call_function` | `aten.slice.Tensor` | `transpose_int_1` | `cat_default_1` |
| 50 | `rope` | `slice_tensor_3` | `call_function` | `aten.slice.Tensor` | `transpose_int_1` | `neg_default_1` |
| 51 | `rope` | `neg_default_1` | `call_function` | `aten.neg.default` | `slice_tensor_3` | `cat_default_1` |
| 52 | `rope` | `cat_default_1` | `call_function` | `aten.cat.default` | `neg_default_1`, `slice_tensor_2` | `mul_tensor_5` |
| 53 | `rope` | `mul_tensor_5` | `call_function` | `aten.mul.Tensor` | `cat_default_1`, `unsqueeze_default_1` | `add_tensor_2` |
| 54 | `rope` | `add_tensor_2` | `call_function` | `aten.add.Tensor` | `mul_tensor_4`, `mul_tensor_5` | `output`, `transpose_int_3` |
| 55 | `attention_scores` | `transpose_int_3` | `call_function` | `aten.transpose.int` | `add_tensor_2` | `expand_default_1` |
| 56 | `attention_scores` | `expand_default` | `call_function` | `aten.expand.default` | `add_tensor_1` | `view_default_6` |
| 57 | `attention_scores` | `view_default_6` | `call_function` | `aten.view.default` | `expand_default` | `bmm_default` |
| 58 | `attention_scores` | `expand_default_1` | `call_function` | `aten.expand.default` | `transpose_int_3` | `view_default_7` |
| 59 | `attention_scores` | `view_default_7` | `call_function` | `aten.view.default` | `expand_default_1` | `bmm_default` |
| 60 | `attention_scores` | `bmm_default` | `call_function` | `aten.bmm.default` | `view_default_6`, `view_default_7` | `view_default_8` |
| 61 | `attention_scores` | `view_default_8` | `call_function` | `aten.view.default` | `bmm_default` | `div_tensor` |
| 62 | `attention_scores` | `div_tensor` | `call_function` | `aten.div.Tensor` | `view_default_8` | `add_tensor_3` |
| 63 | `attention_scores` | `add_tensor_3` | `call_function` | `aten.add.Tensor` | `div_tensor`, `arg1_1` | `_softmax_default` |
| 64 | `attention_scores` | `_softmax_default` | `call_function` | `aten._softmax.default` | `add_tensor_3` | `_to_copy_default_2` |
| 65 | `attention_scores` | `_to_copy_default_2` | `call_function` | `aten._to_copy.default` | `_softmax_default` | `clone_default` |
| 66 | `attention_scores` | `clone_default` | `call_function` | `aten.clone.default` | `_to_copy_default_2` | `expand_default_2` |
| 67 | `attention_output` | `expand_default_2` | `call_function` | `aten.expand.default` | `clone_default` | `view_default_9` |
| 68 | `attention_output` | `view_default_9` | `call_function` | `aten.view.default` | `expand_default_2` | `bmm_default_1` |
| 69 | `attention_output` | `expand_default_3` | `call_function` | `aten.expand.default` | `transpose_int_2` | `view_default_10` |
| 70 | `attention_output` | `view_default_10` | `call_function` | `aten.view.default` | `expand_default_3` | `bmm_default_1` |
| 71 | `attention_output` | `bmm_default_1` | `call_function` | `aten.bmm.default` | `view_default_9`, `view_default_10` | `view_default_11` |
| 72 | `attention_output` | `view_default_11` | `call_function` | `aten.view.default` | `bmm_default_1` | `transpose_int_4` |
| 73 | `attention_output` | `transpose_int_4` | `call_function` | `aten.transpose.int` | `view_default_11` | `clone_default_1` |
| 74 | `attention_output` | `clone_default_1` | `call_function` | `aten.clone.default` | `transpose_int_4` | `view_default_12` |
| 75 | `attention_output` | `view_default_12` | `call_function` | `aten.view.default` | `clone_default_1` | `view_default_13` |
| 76 | `output_projection` | `view_default_13` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 77 | `output_projection` | `_tensor_constant189` | `get_attr` | `_tensor_constant189` | - | `mm_default_3` |
| 78 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant189` | `_unsafe_view_default_3` |
| 79 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 80 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 81 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_6`, `pow_tensor_scalar_1` |
| 82 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 83 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 84 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 85 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_6` |
| 86 | `post_attention_rmsnorm` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 87 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_6` | `mul_tensor_7` |
| 88 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_7` |
| 89 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_14`, `view_default_15` |
| 90 | `post_attention_rmsnorm` | `view_default_14` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_4` |
| 91 | `mlp` | `_tensor_constant190` | `get_attr` | `_tensor_constant190` | - | `mm_default_4` |
| 92 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant190` | `_unsafe_view_default_4` |
| 93 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 94 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 95 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 96 | `mlp` | `_tensor_constant191` | `get_attr` | `_tensor_constant191` | - | `mm_default_5` |
| 97 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant191` | `_unsafe_view_default_5` |
| 98 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 99 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 100 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 101 | `mlp` | `_tensor_constant192` | `get_attr` | `_tensor_constant192` | - | `mm_default_6` |
| 102 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant192` | `_unsafe_view_default_6` |
| 103 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 104 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 105 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2`, `arg6_1` | - |
