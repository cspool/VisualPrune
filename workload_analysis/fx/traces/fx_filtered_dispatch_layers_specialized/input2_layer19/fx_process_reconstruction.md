# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input2_layer19`
GraphModule: `fx_graph_module.pt`

Source: this file is reconstructed by loading `fx_graph_module.pt` and iterating `GraphModule.graph.nodes`.
FX provides the graph DAG and node metadata; the process labels below are reconstruction labels over that DAG.

## Stage Summary

| stage | node range | node count | external inputs | external outputs |
| --- | ---: | ---: | --- | --- |
| Runtime FX inputs | 0-7 | 8 | - | `arg0_1`, `arg1_1`, `arg2_1` |
| Input RMSNorm | 8-15 | 8 | `arg0_1` | `_param_constant0`, `_to_copy_default_1` |
| Q/K/V projection and head reshape | 16-34 | 19 | `_param_constant0`, `_to_copy_default_1` | `transpose_int`, `transpose_int_1`, `transpose_int_2` |
| RoPE position embedding | 35-54 | 20 | `arg2_1`, `transpose_int`, `transpose_int_1` | `add_tensor_1`, `add_tensor_2` |
| QK scores, mask, softmax | 55-70 | 16 | `add_tensor_1`, `add_tensor_2`, `arg1_1`, `transpose_int_2` | `cat_default_2`, `cat_default_3`, `clone_default` |
| Attention-weighted V and hidden reshape | 71-78 | 8 | `cat_default_3`, `clone_default` | `view_default_12` |
| Attention output projection and residual | 79-83 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 84-93 | 10 | `add_tensor_4` | `mul_tensor_7`, `view_default_14` |
| MLP and final residual | 94-107 | 14 | `add_tensor_4`, `mul_tensor_7`, `view_default_14` | `add_tensor_6` |
| Layer output | 108-108 | 1 | `add_tensor_6`, `cat_default_2`, `cat_default_3` | - |

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

是什么：这一段是单步 decode 固定样本暴露给 FX GraphModule 的入口节点。`arg0_1` 是当前 token 的 hidden state，`arg1_1` 是当前 query 对 59 个 key 位置的 attention mask，`arg2_1` 是当前 token 的 position id；`arg3_1` 到 `arg7_1` 在这个固定 DAG 中没有用户节点，其中 metadata 里 `important_vis_tokens` 为 null。

为什么需要：decode 时本层只处理一个新 token，但它需要和历史 cache 以及当前新 key/value 一起计算 attention。`arg0_1` 提供这一个 token 的内容，`arg1_1` 限制它可以 attend 的 K 位置，`arg2_1` 让 Q/K 使用当前位置的 RoPE 表项。本 trace 没有单独的 `visual_process` stage，也没有 value-aware Visual token 选择节点。

怎么做/计算：placeholder 不做数值计算，只建立依赖。`arg0_1` 被 `_to_copy_default` 用于 input RMSNorm，并在 `add_tensor_4` 中作为 attention residual 的左操作数；`arg1_1` 只在 `add_tensor_3` 中加到 scaled QK logits；`arg2_1` 被 `index_tensor` 和 `index_tensor_1` 用作 cos/sin 表索引；`arg6_1` 对应的 important Visual token 输入为 null，且没有下游 FX 用户。

```text
Tensor: InputHidden X, shape [B=1, S=1, H=4096]
Formula: X[0,h] is forwarded to RMSNorm and residual add
Token axis S=0..0                                     Hidden axis H=0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
current token                                  ──▶     +----------------------------------------+  ◀── examples: X[0,0], X[0,4095]
                                                       | CURRENT_TOKEN_HIDDEN_ROW               |
                                                       +----------------------------------------+

Tensor: AttentionMask M, shape [B=1, HeadBroadcast=1, Q=1, K=59]
Formula: masked_logits[0,k] = scaled_qk[0,k] + M[0,k]
Q_seq axis 0..0                                       K_seq axis 0..58
                                                       0                                      58
                                                       ▲                                       ▲
decode mask                                   ──▶     +----------------------------------------+  ◀── examples: M[0,0], M[0,58]
                                                       | MASK_ROW_FOR_CURRENT_QUERY             |
                                                       +----------------------------------------+

Tensor: PositionIds P, shape [B=1, S=1]
Formula: cos_sin_row[0,:] = table[P[0],:]
S=0
▲
[ P0 ]  ──▶ used for RoPE table lookup

Tensor: ImportantVisTokens I
Formula: I is null and has no FX users in this fixed DAG
[ NULL_IMPORTANT_VIS_TOKENS ]  ──▶ no Visual-token selection process is observed
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

是什么：这一段对当前 token 的 hidden row 做 RMSNorm，输出供 Q/K/V projection 使用的 fp16 normalized row，并读取长度为 4096 的 input norm 权重。

为什么需要：即使 decode 只处理一个 token，Q/K/V projection 前仍需要按 Hidden 维稳定这个 token row 的数值尺度，避免投影输入幅度随 residual 状态漂移。

怎么做/计算：`_to_copy_default` 把 `arg0_1 [1,1,4096]` 转成 fp32；`pow_tensor_scalar` 对当前 row 的每个 Hidden 元素平方；`mean_dim` 沿 Hidden 维求均值并保留 `[1,1,1]`；`add_tensor` 加 `1e-05`；`rsqrt_default` 得到这个 token row 的 inverse RMS；`mul_tensor` 将该标量广播回 Hidden 维并逐元素缩放；`_to_copy_default_1` 转回 fp16；`_param_constant0 [4096]` 作为下一段逐 Hidden 维相乘的 norm weight。

```text
Tensor: CurrentRow X_fp32, shape [S=1, H=4096]
Formula: X_fp32[0,h] = float32(InputHidden[0,h])
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
input row                                     ──▶     +----------------------------------------+  ◀── examples: X[0,0], X[0,4095]
                                                       | CURRENT_INPUT_ROW                      |
                                                       +----------------------------------------+

Tensor: RowRMSScale R, shape [S=1, 1]
Formula: R[0] = rsqrt(mean_h(X_fp32[0,h]^2) + 1e-05)
S=0                                                    scalar column
                                                       +----------+
row scale                                      ──▶     | R_0      |  ◀── reduced from H=0..4095
                                                       +----------+

Tensor: NormRow N, shape [S=1, H=4096]
Formula: N[0,h] = fp16(X_fp32[0,h] * R[0])
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
normalized row                                ──▶     +----------------------------------------+  ◀── examples: N[0,0], N[0,2048], N[0,4095]
                                                       | CURRENT_NORM_ROW                       |
                                                       +----------------------------------------+
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant248 = self._tensor_constant248
mm_default = aten.mm.default(view_default, _tensor_constant248)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 1, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant249 = self._tensor_constant249
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant249)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 1, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant250 = self._tensor_constant250
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant250)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 1, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 1, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 1, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 1, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段从当前 normalized hidden 生成当前 token 的 Q、K、V，并把每个 `[1,4096]` 投影结果拆成 32 个 128 维 head。

为什么需要：单步 decode 只产生一个新 query，但同时要产生当前 token 的新 K/V。新 Q 用来查询历史 58 个 cache 位置加当前 key；新 K/V 会追加进 cache，供当前和后续 token 使用。

怎么做/计算：`mul_tensor_1` 将 `_param_constant0 [4096]` 广播到 normalized row 上，得到投影前的加权 hidden；Q 分支 `view_default -> mm_default -> _unsafe_view_default` 使用 `_tensor_constant248 [4096,4096]` 得到 `[1,1,4096]`；K 分支用 `_tensor_constant249`，V 分支用 `_tensor_constant250` 做同样投影。之后 `view_default_3/4/5` 把三个输出改为 `[1,1,32,128]`，`transpose_int/1/2` 交换 token/head 轴，得到 `[1,32,1,128]` 的当前 token Q、K、V。

```text
Tensor: WeightedNormRow A, shape [S=1, H=4096]
Formula: A[0,h] = NormRow[0,h] * NormWeight[h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
shared input                                  ──▶     +----------------------------------------+  ◀── reused by Q, K, and V projections
                                                       | WEIGHTED_CURRENT_ROW                   |
                                                       +----------------------------------------+

Tensor: ProjectedRows Q2D/K2D/V2D, each shape [S=1, H=4096]
Formula: Q2D = A @ Wq, K2D = A @ Wk, V2D = A @ Wv
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
linear projection                              ──▶     +----------------------------------------+  ◀── three aligned current-token rows
                                                       | Q2D_ROW / K2D_ROW / V2D_ROW            |
                                                       +----------------------------------------+

Tensor: HeadSplit Q/K/V, each shape [B=1, Heads=32, S=1, Dh=128]
Formula: HeadSplit[b,head,0,dh] = ProjectedRows[0, head*128 + dh]
For one representative head h, show S x Dh.
S axis 0..0                                           Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
head row                                      ──▶     +----------------------------------------+  ◀── examples: Q[h,0,0], K[h,0,127], V[h,0,64]
                                                       | CURRENT_TOKEN_HEAD_h_DH_ROW            |
                                                       +----------------------------------------+
```

### RoPE position embedding

```python
_tensor_constant251 = self._tensor_constant251
index_tensor = aten.index.Tensor(_tensor_constant251, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant252 = self._tensor_constant252
index_tensor_1 = aten.index.Tensor(_tensor_constant252, [arg2_1])
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

是什么：这一段给当前 token 的 Q/K 应用 RoPE，输出带当前位置编码的 rotated Q `add_tensor_1` 和 rotated K `add_tensor_2`。V 不参与 RoPE，保留上一段的 value head。

为什么需要：decode 当前 token 会同时作为 query 查询历史 keys，并作为新 key 追加进 cache。新 query 和新 key 都必须包含当前位置的 RoPE 信息，否则它与历史 cache 中已编码的位置无法正确比较。

怎么做/计算：`index_tensor` 用 `arg2_1 [1,1]` 从 `_tensor_constant251 [625,128]` 取当前 position 的 cos row，`unsqueeze_default` 插入 head 广播维；`index_tensor_1` 和 `unsqueeze_default_1` 对 sin 表做同样索引。Q 分支中，`mul_tensor_2` 计算 `Q * cos`；`slice_tensor` 取 Dh `0..63`，`slice_tensor_1` 取 Dh `64..127`，`neg_default` 对右半取负，`cat_default` 拼成 `[-right, left]`，`mul_tensor_3` 乘 sin，`add_tensor_1` 得到 rotated Q。K 分支用 `mul_tensor_4`、`slice_tensor_2/3`、`neg_default_1`、`cat_default_1`、`mul_tensor_5`、`add_tensor_2` 对当前 K 做相同旋转。

```text
Tensor: QKHeadRow X, shape [Heads=32, S=1, Dh=128]
Formula: X is either current-token Q or current-token K before RoPE
Dh axis 0..127, split into two halves
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
input x ──▶ +--------------------------+-+--------------------------+  ◀── examples: x[...,0], x[...,64], x[...,127]
            | LEFT_HALF                | | RIGHT_HALF               |
            +--------------------------+-+--------------------------+

Tensor: RotateHalf(X), shape [Heads=32, S=1, Dh=128]
Formula: RotateHalf(X)[...,0:64] = -X[...,64:128]; RotateHalf(X)[...,64:128] = X[...,0:64]
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
rot x  ──▶  +--------------------------+-+--------------------------+  ◀── right half is negated and moved left
            | NEG_RIGHT_HALF           | | LEFT_HALF                |
            +--------------------------+-+--------------------------+

Tensor: RoPERow Y, shape [Heads=32, S=1, Dh=128]
Formula: Y[head,0,dh] = X[head,0,dh] * Cos[P0,dh] + RotateHalf(X)[head,0,dh] * Sin[P0,dh]
            0                         63 64                       127
            ▲                          ▲ ▲                         ▲
output ──▶  +--------------------------+-+--------------------------+  ◀── same Dh coordinates after position rotation
            | ROTATED_LEFT_HALF        | | ROTATED_RIGHT_HALF       |
            +--------------------------+-+--------------------------+
```

### QK scores, mask, softmax

```python
_tensor_constant2 = self._tensor_constant2
cat_default_2 = aten.cat.default([_tensor_constant2, add_tensor_2], -2)
_tensor_constant3 = self._tensor_constant3
cat_default_3 = aten.cat.default([_tensor_constant3, transpose_int_2], -2)
transpose_int_3 = aten.transpose.int(cat_default_2, 2, 3)
expand_default = aten.expand.default(add_tensor_1, [1, 32, 1, 128])
view_default_6 = aten.view.default(expand_default, [32, 1, 128])
expand_default_1 = aten.expand.default(transpose_int_3, [1, 32, 128, 59])
view_default_7 = aten.view.default(expand_default_1, [32, 128, 59])
bmm_default = aten.bmm.default(view_default_6, view_default_7)
view_default_8 = aten.view.default(bmm_default, [1, 32, 1, 59])
div_tensor = aten.div.Tensor(view_default_8, 11.313708498984761)
add_tensor_3 = aten.add.Tensor(div_tensor, arg1_1)
_softmax_default = aten._softmax.default(add_tensor_3, -1, True)
_to_copy_default_2 = aten._to_copy.default(_softmax_default, dtype=torch.float16)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段先把历史 K/V cache 与当前 token 的新 K/V 拼接，再计算当前 query 对 59 个 key 位置的 masked softmax attention weights。这里的 K 轴由历史 58 行和当前新 key 第 58 行组成。

为什么需要：decode attention 不能只看当前 token 的 K/V，它需要访问所有历史上下文。`cat_default_2` 是更新后的 K cache，既用于当前 QK score，也会返回给后续 token；`cat_default_3` 是更新后的 V cache，供下一段 value 汇聚和最终输出 cache 使用。

怎么做/计算：`_tensor_constant2 [1,32,58,128]` 是历史 K cache，`cat_default_2` 将它与当前 rotated K `add_tensor_2 [1,32,1,128]` 沿倒数第二维拼接成 `[1,32,59,128]`；`_tensor_constant3 [1,32,58,128]` 是历史 V cache，`cat_default_3` 将它与当前 V `transpose_int_2 [1,32,1,128]` 拼成 `[1,32,59,128]`。`transpose_int_3` 把 K cache 改成 `[1,32,128,59]`；`expand_default/view_default_6` 将 current rotated Q 变成 `[32,1,128]`；`expand_default_1/view_default_7` 将 K 变成 `[32,128,59]`；`bmm_default` 得到每个 head 的 `[1,59]` QK score；`view_default_8` 恢复 `[1,32,1,59]`；`div_tensor` 除以 `sqrt(128)`；`add_tensor_3` 加上 mask `[1,1,1,59]`；`_softmax_default` 沿 K=59 归一化；`_to_copy_default_2` 转 fp16；`clone_default` 作为当前 query 的 attention weights 进入下一段。

```text
Tensor: UpdatedKeyCache K_all, shape [Heads=32, K=59, Dh=128]
Formula: K_all[head,0:58,:] = K_cache_old[head,0:58,:]; K_all[head,58,:] = RoPE_K_current[head,0,:]
For one representative head, show K x Dh.
K_seq axis 0..58                                       Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
key cache                                     ──▶     +----------------------------------------+  ◀── K=0..57 are cached
                                                       | CACHED_KEY_ROWS_0_TO_57                |
                                                       +----------------------------------------+
current key                                   ──▶     +----------------------------------------+  ◀── K=58 is appended current key
                                                       | CURRENT_KEY_ROW_58                     |
                                                       +----------------------------------------+

Tensor: UpdatedValueCache V_all, shape [Heads=32, K=59, Dh=128]
Formula: V_all[head,0:58,:] = V_cache_old[head,0:58,:]; V_all[head,58,:] = V_current[head,0,:]
K_seq axis 0..58                                       Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
value cache                                   ──▶     +----------------------------------------+  ◀── same K coordinates as K_all
                                                       | CACHED_VALUE_ROWS_0_TO_57              |
                                                       +----------------------------------------+
current value                                 ──▶     +----------------------------------------+  ◀── K=58 is appended current value
                                                       | CURRENT_VALUE_ROW_58                   |
                                                       +----------------------------------------+

Tensor: DecodeLogits L, shape [Heads=32, Q=1, K=59]
Formula: L[head,0,k] = dot(RoPE_Q_current[head,0,:], K_all[head,k,:]) / sqrt(128) + Mask[0,k]
Q_seq axis 0..0                                       K_seq axis 0..58
                                                       0                                      58
                                                       ▲                                       ▲
masked logits                                 ──▶     +----------------------------------------+  ◀── one current query scores all cached+current keys
                                                       | CURRENT_QUERY_LOGITS_OVER_K            |
                                                       +----------------------------------------+

Tensor: DecodeWeights W, shape [Heads=32, Q=1, K=59]
Formula: W[head,0,k] = softmax_k(L[head,0,k])
Q_seq axis 0..0                                       K_seq axis 0..58
                                                       0                                      58
                                                       ▲                                       ▲
weights                                       ──▶     +----------------------------------------+  ◀── examples: W[0,0,0], W[0,0,58], W[31,0,58]
                                                       | SOFTMAX_WEIGHTS_OVER_59_KEYS           |
                                                       +----------------------------------------+
```

### Attention-weighted V and hidden reshape

```python
expand_default_2 = aten.expand.default(clone_default, [1, 32, 1, 59])
view_default_9 = aten.view.default(expand_default_2, [32, 1, 59])
expand_default_3 = aten.expand.default(cat_default_3, [1, 32, 59, 128])
view_default_10 = aten.view.default(expand_default_3, [32, 59, 128])
bmm_default_1 = aten.bmm.default(view_default_9, view_default_10)
view_default_11 = aten.view.default(bmm_default_1, [1, 32, 1, 128])
transpose_int_4 = aten.transpose.int(view_default_11, 1, 2)
view_default_12 = aten.view.default(transpose_int_4, [1, 1, 4096])
```

**解释与可视化**

是什么：这一段用当前 query 的 `[1,59]` attention weights 对更新后的 V cache 做加权求和，得到当前 token 在每个 head 上的 context，再合并回 `[1,1,4096]`。

为什么需要：QK softmax 只决定当前 token 应该从哪些 cache 位置取信息；真正的 attention 输出来自这些权重对历史 value rows 和当前 value row 的加权汇聚。

怎么做/计算：`expand_default_2/view_default_9` 将 `clone_default` 变成 `[32,1,59]`；`expand_default_3/view_default_10` 将 `cat_default_3` 变成 `[32,59,128]`；`bmm_default_1` 在每个 head 内执行 `[Q=1,K=59] x [K=59,Dh=128] -> [Q=1,Dh=128]`；`view_default_11` 恢复 `[1,32,1,128]`；`transpose_int_4` 改成 `[1,1,32,128]`；`view_default_12` 合并 `32*128` 为 Hidden，得到当前 token attention context `[1,1,4096]`。

```text
Tensor: DecodeWeights W, shape [Heads=32, Q=1, K=59]
Formula: W[head,0,k] is the masked softmax row over K
Q_seq axis 0..0                                       K_seq axis 0..58
                                                       0                                      58
                                                       ▲                                       ▲
weights                                       ──▶     +----------------------------------------+  ◀── one row mixes all V cache positions
                                                       | SOFTMAX_WEIGHTS_OVER_59_KEYS           |
                                                       +----------------------------------------+

Tensor: UpdatedValueCache V_all, shape [Heads=32, K=59, Dh=128]
Formula: V_all includes cached rows 0..57 and current row 58
K_seq axis 0..58                                       Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
value rows                                    ──▶     +----------------------------------------+  ◀── K axis consumed by W
                                                       | VALUE_ROWS_0_TO_57_CACHED              |
                                                       +----------------------------------------+
                                                       +----------------------------------------+  ◀── current row at K=58
                                                       | VALUE_ROW_58_CURRENT                   |
                                                       +----------------------------------------+

Tensor: CurrentContext C, shape [Heads=32, Q=1, Dh=128]
Formula: C[head,0,dh] = sum_k W[head,0,k] * V_all[head,k,dh]
Q_seq axis 0..0                                       Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
context                                       ──▶     +----------------------------------------+  ◀── examples: C[0,0,0], C[31,0,127]
                                                       | CURRENT_CONTEXT_DH_ROW                 |
                                                       +----------------------------------------+

Tensor: MergedContext O, shape [B=1, S=1, H=4096]
Formula: O[0, head*128 + dh] = C[head,0,dh]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
merged output                                 ──▶     +----------------------------------------+  ◀── heads are concatenated along H
                                                       | CURRENT_MERGED_CONTEXT_ROW             |
                                                       +----------------------------------------+
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [1, 4096])
_tensor_constant253 = self._tensor_constant253
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant253)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 1, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把当前 token 的 merged attention context 通过输出投影映射回 hidden space，并与当前 token 原始输入 hidden 在同一 Hidden 坐标上逐元素相加。

为什么需要：多头 context 合并后仍需要输出投影来重新混合通道；残差连接保留进入该层的当前 token hidden，使后续 RMSNorm/MLP 在 attention 更新后的表示上继续计算。

怎么做/计算：`view_default_13` 把 `view_default_12 [1,1,4096]` 展成 `[1,4096]`；`mm_default_3` 乘 `_tensor_constant253 [4096,4096]` 得到输出投影结果；`_unsafe_view_default_3` 恢复 `[1,1,4096]`；`add_tensor_4` 将投影结果和 `arg0_1` 在当前 token 的 Hidden 维上逐元素相加。

```text
Tensor: ProjectedAttention P, shape [S=1, H=4096]
Formula: P[0,h] = MergedContext[0,:] @ Wo[:,h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
projection                                    ──▶     +----------------------------------------+  ◀── output projection row
                                                       | PROJECTED_CURRENT_CONTEXT              |
                                                       +----------------------------------------+

Tensor: ResidualInput X, shape [S=1, H=4096]
Formula: X[0,h] is the original current-token layer input
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual source                               ──▶     +----------------------------------------+  ◀── aligned with P over Hidden
                                                       | CURRENT_INPUT_HIDDEN_ROW               |
                                                       +----------------------------------------+

Tensor: AttentionResidual A, shape [S=1, H=4096]
Formula: A[0,h] = X[0,h] + P[0,h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual output                               ──▶     +----------------------------------------+  ◀── examples: A[0,0], A[0,2048], A[0,4095]
                                                       | CURRENT_ATTENTION_RESIDUAL             |
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
view_default_14 = aten.view.default(mul_tensor_7, [1, 4096])
```

**解释与可视化**

是什么：这一段对当前 token 的 attention residual 再做 RMSNorm，生成供 MLP gate/up projection 共享的 normalized row。

为什么需要：attention residual 经过残差加法后尺度会变化。MLP 前的 RMSNorm 重新按 Hidden 维标准化当前 token row，使 gate/up/down projection 的输入尺度稳定。

怎么做/计算：`_to_copy_default_3` 把 `add_tensor_4 [1,1,4096]` 转 fp32；`pow_tensor_scalar_1` 对 Hidden 元素平方；`mean_dim_1` 沿 Hidden 求均值并保留 `[1,1,1]`；`add_tensor_5` 加 `1e-05`；`rsqrt_default_1` 得到 inverse RMS；`mul_tensor_6` 将该标量广播回 Hidden 并逐元素缩放；`_to_copy_default_4` 转 fp16；`_param_constant5 [4096]` 是 post-attention norm weight；`mul_tensor_7` 应用该权重；`view_default_14` 展成 `[1,4096]` 供 gate projection 使用，同一 `mul_tensor_7` 也通过 `view_default_15` 供 up projection 使用。

```text
Tensor: AttentionResidual A_fp32, shape [S=1, H=4096]
Formula: A_fp32[0,h] = float32(AttentionResidual[0,h])
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
residual row                                  ──▶     +----------------------------------------+  ◀── reduced over H=0..4095
                                                       | CURRENT_ATTENTION_RESIDUAL             |
                                                       +----------------------------------------+

Tensor: MLPScale R2, shape [S=1, 1]
Formula: R2[0] = rsqrt(mean_h(A_fp32[0,h]^2) + 1e-05)
S=0                                                    scalar column
                                                       +----------+
row scale                                      ──▶     | R2_0     |  ◀── one scale for the current token
                                                       +----------+

Tensor: MLPInput U, shape [S=1, H=4096]
Formula: U[0,h] = fp16(A_fp32[0,h] * R2[0]) * MLPNormWeight[h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
mlp input                                     ──▶     +----------------------------------------+  ◀── examples: U[0,0], U[0,1024], U[0,4095]
                                                       | CURRENT_MLP_INPUT_ROW                  |
                                                       +----------------------------------------+
```

### MLP and final residual

```python
_tensor_constant254 = self._tensor_constant254
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant254)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 1, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [1, 4096])
_tensor_constant255 = self._tensor_constant255
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant255)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 1, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [1, 11008])
_tensor_constant256 = self._tensor_constant256
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant256)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 1, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行当前 token 的 gated MLP：同一 MLP 输入分成 gate 和 up 两个 `[1,11008]` 分支，gate 经过 SiLU 后与 up 逐元素相乘，再 down project 回 `[1,4096]`，最后加回 attention residual。

为什么需要：attention 分支已经完成跨 cache token 的信息汇聚；MLP 分支对当前 token 做通道内非线性变换。gate/up 乘法控制中间通道激活，down projection 回到 Hidden 宽度后通过 residual add 形成本层最终 hidden。

怎么做/计算：`_tensor_constant254 [4096,11008]` 是 gate projection 权重，`mm_default_4` 将 `view_default_14 [1,4096]` 投到 `[1,11008]`，`_unsafe_view_default_4` 恢复 `[1,1,11008]`，`silu_default` 对 gate 分支逐元素激活；`view_default_15` 复用同一个 normalized tensor，`mm_default_5` 乘 `_tensor_constant255 [4096,11008]` 得到 up 分支；`mul_tensor_8` 将 `silu(gate)` 与 up 在 intermediate 轴逐元素相乘；`view_default_16` 展成 `[1,11008]`；`mm_default_6` 乘 `_tensor_constant256 [11008,4096]` down project 回 Hidden；`_unsafe_view_default_6` 恢复 `[1,1,4096]`；`add_tensor_6` 与 `add_tensor_4` 在 Hidden 坐标逐元素相加。

```text
Tensor: MLPInput U, shape [S=1, H=4096]
Formula: U is the normalized post-attention residual
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
shared mlp input                              ──▶     +----------------------------------------+  ◀── reused by gate and up projections
                                                       | CURRENT_MLP_INPUT_ROW                  |
                                                       +----------------------------------------+

Tensor: GatedIntermediate G, shape [S=1, I=11008]
Formula: G[0,i] = SiLU(U[0,:] @ W_gate[:,i]) * (U[0,:] @ W_up[:,i])
S axis 0..0                                           Intermediate axis I=0..11007
                                                       0                                     11007
                                                       ▲                                       ▲
gated row                                    ──▶      +----------------------------------------+  ◀── examples: G[0,0], G[0,11007]
                                                       | CURRENT_SILU_GATE_TIMES_UP             |
                                                       +----------------------------------------+

Tensor: MLPDown D, shape [S=1, H=4096]
Formula: D[0,h] = G[0,:] @ W_down[:,h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
down projection                               ──▶     +----------------------------------------+  ◀── returns to Hidden width
                                                       | CURRENT_MLP_DOWN_ROW                   |
                                                       +----------------------------------------+

Tensor: FinalHidden Y, shape [S=1, H=4096]
Formula: Y[0,h] = AttentionResidual[0,h] + D[0,h]
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
final residual                                ──▶     +----------------------------------------+  ◀── examples: Y[0,0], Y[0,2048], Y[0,4095]
                                                       | CURRENT_FINAL_HIDDEN_ROW               |
                                                       +----------------------------------------+
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (cat_default_2, cat_default_3)}, None, 0)
```

**解释与可视化**

是什么：这一段是输出打包节点，不再做新的数值计算。返回值包含当前 token final hidden、更新后的 K/V cache、Visual 输出占位 `None`，以及固定控制标量 `0`。

为什么需要：`add_tensor_6` 是 layer19 对当前 decode token 的主 hidden 输出；`cat_default_2/cat_default_3` 已经把当前 K/V 追加到历史 cache，供后续 token 继续解码；由于本 trace 的 `important_vis_tokens` 为 null 且没有 Visual process，第三项返回 `None`。

怎么做/计算：`output` 节点把 `add_tensor_6` 放入返回元组第一项；把 `cat_default_2` 和 `cat_default_3` 放入 `{'dynamic_cache_layer': (...)}`；第三项直接是 `None`；最后返回标量 `0`。这些值都已由前面节点产生，`return` 本身只组织结构。

```text
Tensor: ReturnHidden Y, shape [B=1, S=1, H=4096]
Formula: Y is the final hidden row from the MLP residual
S axis 0..0                                           H axis 0..4095
                                                       0                                      4095
                                                       ▲                                        ▲
return item 0                                 ──▶     +----------------------------------------+  ◀── consumed by the next layer/token path
                                                       | CURRENT_FINAL_HIDDEN_ROW               |
                                                       +----------------------------------------+

Tensor: DynamicCache K/V, shapes K=[B=1, Heads=32, K=59, Dh=128], V=[B=1, Heads=32, K=59, Dh=128]
Formula: cache_out = (UpdatedKeyCache, UpdatedValueCache)
For one representative head, show K x Dh.
K_seq axis 0..58                                       Dh axis 0..127
                                                       0                                      127
                                                       ▲                                       ▲
cache K/V                                     ──▶     +----------------------------------------+  ◀── rows 0..57 are old cache
                                                       | CACHE_ROWS_0_TO_57                     |
                                                       +----------------------------------------+
current row                                   ──▶     +----------------------------------------+  ◀── row 58 appended in this layer call
                                                       | CACHE_ROW_58_CURRENT                   |
                                                       +----------------------------------------+

Tensor: VisualOutput Vout, shape []
Formula: Vout = None
[ NO_VISUAL_OUTPUT ]  ──▶ no value-aware Visual process in this fixed DAG

Tensor: ControlScalar C, shape []
Formula: C = 0
[ CONTROL_SCALAR_0 ] ──▶ returned as final tuple item
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
| 6 | `inputs` | `arg6_1` | `placeholder` | `arg6_1` | - | - |
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
| 18 | `qkv_projection` | `_tensor_constant248` | `get_attr` | `_tensor_constant248` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant248` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant249` | `get_attr` | `_tensor_constant249` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant249` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant250` | `get_attr` | `_tensor_constant250` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant250` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `cat_default_3` |
| 35 | `rope` | `_tensor_constant251` | `get_attr` | `_tensor_constant251` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant251`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant252` | `get_attr` | `_tensor_constant252` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant252`, `arg2_1` | `unsqueeze_default_1` |
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
| 54 | `rope` | `add_tensor_2` | `call_function` | `aten.add.Tensor` | `mul_tensor_4`, `mul_tensor_5` | `cat_default_2` |
| 55 | `attention_scores` | `_tensor_constant2` | `get_attr` | `_tensor_constant2` | - | `cat_default_2` |
| 56 | `attention_scores` | `cat_default_2` | `call_function` | `aten.cat.default` | `_tensor_constant2`, `add_tensor_2` | `output`, `transpose_int_3` |
| 57 | `attention_scores` | `_tensor_constant3` | `get_attr` | `_tensor_constant3` | - | `cat_default_3` |
| 58 | `attention_scores` | `cat_default_3` | `call_function` | `aten.cat.default` | `_tensor_constant3`, `transpose_int_2` | `expand_default_3`, `output` |
| 59 | `attention_scores` | `transpose_int_3` | `call_function` | `aten.transpose.int` | `cat_default_2` | `expand_default_1` |
| 60 | `attention_scores` | `expand_default` | `call_function` | `aten.expand.default` | `add_tensor_1` | `view_default_6` |
| 61 | `attention_scores` | `view_default_6` | `call_function` | `aten.view.default` | `expand_default` | `bmm_default` |
| 62 | `attention_scores` | `expand_default_1` | `call_function` | `aten.expand.default` | `transpose_int_3` | `view_default_7` |
| 63 | `attention_scores` | `view_default_7` | `call_function` | `aten.view.default` | `expand_default_1` | `bmm_default` |
| 64 | `attention_scores` | `bmm_default` | `call_function` | `aten.bmm.default` | `view_default_6`, `view_default_7` | `view_default_8` |
| 65 | `attention_scores` | `view_default_8` | `call_function` | `aten.view.default` | `bmm_default` | `div_tensor` |
| 66 | `attention_scores` | `div_tensor` | `call_function` | `aten.div.Tensor` | `view_default_8` | `add_tensor_3` |
| 67 | `attention_scores` | `add_tensor_3` | `call_function` | `aten.add.Tensor` | `div_tensor`, `arg1_1` | `_softmax_default` |
| 68 | `attention_scores` | `_softmax_default` | `call_function` | `aten._softmax.default` | `add_tensor_3` | `_to_copy_default_2` |
| 69 | `attention_scores` | `_to_copy_default_2` | `call_function` | `aten._to_copy.default` | `_softmax_default` | `clone_default` |
| 70 | `attention_scores` | `clone_default` | `call_function` | `aten.clone.default` | `_to_copy_default_2` | `expand_default_2` |
| 71 | `attention_output` | `expand_default_2` | `call_function` | `aten.expand.default` | `clone_default` | `view_default_9` |
| 72 | `attention_output` | `view_default_9` | `call_function` | `aten.view.default` | `expand_default_2` | `bmm_default_1` |
| 73 | `attention_output` | `expand_default_3` | `call_function` | `aten.expand.default` | `cat_default_3` | `view_default_10` |
| 74 | `attention_output` | `view_default_10` | `call_function` | `aten.view.default` | `expand_default_3` | `bmm_default_1` |
| 75 | `attention_output` | `bmm_default_1` | `call_function` | `aten.bmm.default` | `view_default_9`, `view_default_10` | `view_default_11` |
| 76 | `attention_output` | `view_default_11` | `call_function` | `aten.view.default` | `bmm_default_1` | `transpose_int_4` |
| 77 | `attention_output` | `transpose_int_4` | `call_function` | `aten.transpose.int` | `view_default_11` | `view_default_12` |
| 78 | `attention_output` | `view_default_12` | `call_function` | `aten.view.default` | `transpose_int_4` | `view_default_13` |
| 79 | `output_projection` | `view_default_13` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 80 | `output_projection` | `_tensor_constant253` | `get_attr` | `_tensor_constant253` | - | `mm_default_3` |
| 81 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant253` | `_unsafe_view_default_3` |
| 82 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 83 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 84 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_6`, `pow_tensor_scalar_1` |
| 85 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 86 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 87 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 88 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_6` |
| 89 | `post_attention_rmsnorm` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 90 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_6` | `mul_tensor_7` |
| 91 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_7` |
| 92 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_14`, `view_default_15` |
| 93 | `post_attention_rmsnorm` | `view_default_14` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_4` |
| 94 | `mlp` | `_tensor_constant254` | `get_attr` | `_tensor_constant254` | - | `mm_default_4` |
| 95 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant254` | `_unsafe_view_default_4` |
| 96 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 97 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 98 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 99 | `mlp` | `_tensor_constant255` | `get_attr` | `_tensor_constant255` | - | `mm_default_5` |
| 100 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant255` | `_unsafe_view_default_5` |
| 101 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 102 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 103 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 104 | `mlp` | `_tensor_constant256` | `get_attr` | `_tensor_constant256` | - | `mm_default_6` |
| 105 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant256` | `_unsafe_view_default_6` |
| 106 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 107 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 108 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `cat_default_2`, `cat_default_3` | - |
