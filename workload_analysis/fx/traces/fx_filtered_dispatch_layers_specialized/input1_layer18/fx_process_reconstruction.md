# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer18`
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
| QK scores, mask, softmax | 55-66 | 12 | `add_tensor_1`, `add_tensor_2`, `arg1_1` | `clone_default` |
| Attention-weighted V and hidden reshape | 67-75 | 9 | `clone_default`, `transpose_int_2` | `view_default_12` |
| Visual-related value-aware process | 76-94 | 19 | `clone_default`, `transpose_int_2`, `view_default_12` | `add_tensor_4` |
| Attention output projection and residual | 95-99 | 5 | `arg0_1`, `view_default_12` | `add_tensor_5` |
| Post-attention RMSNorm | 100-109 | 10 | `add_tensor_5` | `mul_tensor_8`, `view_default_15` |
| MLP and final residual | 110-123 | 14 | `add_tensor_5`, `mul_tensor_8`, `view_default_15` | `add_tensor_7` |
| Layer output | 124-124 | 1 | `add_tensor_2`, `add_tensor_4`, `add_tensor_7`, `transpose_int_2` | - |

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

是什么：这一段是固定输入样本的 FX placeholder；8 个 placeholder 中，后续计算实际使用 `arg0_1`、`arg1_1`、`arg2_1`。

为什么需要：这个 layer 的后续 ATen DAG 需要从 placeholder 取得 hidden states、attention mask 和 position ids；其余 placeholder 在这个固定 DAG 中没有 users。

怎么做/计算：placeholder 不做数值计算。`arg0_1` 作为 `[B=1,S=624,H=4096]` hidden 输入，进入 `_to_copy_default` 做 RMSNorm，也在 output projection 后参与残差 `add_tensor_5`；`arg1_1` 是 `[Q=624,K=624]` mask，后面被 `add_tensor_3 = div_tensor + arg1_1` 加到 QK logits；`arg2_1` 是 token position id 向量，后面作为 `aten.index.Tensor` 的 index 从 RoPE cos/sin 表取行。

```text
Runtime inputs

Tensor: Input hidden H_in [B=1, S=624, H=4096]
Formula: H_in = sampled layer hidden input; H_norm and H_attn both derive from this same tensor
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | hidden row 0, H examples 0 and 4095    |
          | token rows 1 .. 622                    |
S=623     | hidden row 623, H examples 0 and 4095  |
          +----------------------------------------+

Tensor: Attention mask M [Q=624, K=624]
Formula: masked logits use L_masked[q,k] = L_scaled[q,k] + M[q,k]
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)
          K=0                                  K=623
Q=0       +----------------------------------------+
          | mask row 0                             |
          | mask rows 1 .. 622                     |
Q=623     | mask row 623                           |
          +----------------------------------------+

Tensor: Position ids P [S=624]
Formula: P[s] selects the RoPE cos/sin table row for token coordinate s
S axis: 0 .. 623
S=0       [pos0 | pos1 | ... | pos622 | pos623]

examples: H_in[0,0,0], M[623,623], P[35].
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

是什么：这一段对输入 hidden states 做 RMSNorm，输出 fp16 normalized hidden 并读取 input norm 权重。

为什么需要：Q/K/V 投影前要按 token 行的 Hidden RMS 尺度归一化。

怎么做/计算：`_to_copy_default` 把 `arg0_1` 转 fp32；`pow_tensor_scalar` 对每个 token row 的 Hidden 元素平方；`mean_dim` 沿最后一维 Hidden 求均值并保留维度；`add_tensor` 加 `1e-05`；`rsqrt_default` 得到 `1/sqrt(mean(x^2)+eps)`；`mul_tensor` 把原 fp32 hidden 逐元素乘该缩放；`_to_copy_default_1` 转回 fp16；`_param_constant0` 是下一段投影前使用的 RMSNorm weight。

```text
Input RMSNorm [S=624, H=4096]

Tensor: Squared hidden H_sq [S=624, H=4096]
Formula: H_sq[s,h] = float32(H_in[s,h])^2
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | hidden row 0 squared over H            | ----> | r0 |
          | token rows 1 .. 622                    |       | .. |
S=623     | hidden row 623 squared over H          | ----> | r623 |
          +----------------------------------------+       +----+

Tensor: Row RMS scale R_in [S=624, 1]
Formula: R_in[s] = 1 / sqrt(mean_h(H_sq[s,h]) + 1e-05)
S axis: 0 .. 623
S=0       +------+
          | r0   |
          | ..   |
S=623     | r623 |
          +------+

Tensor: Normalized hidden H_norm [S=624, H=4096]
Formula: H_norm[s,h] = fp16(float32(H_in[s,h]) * R_in[s])
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | normalized row 0                       |
          | normalized rows 1 .. 622               |
S=623     | normalized row 623                     |
          +----------------------------------------+

examples: H_norm[0,0], H_norm[35,2048], H_norm[623,4095].
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant85 = self._tensor_constant85
mm_default = aten.mm.default(view_default, _tensor_constant85)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant86 = self._tensor_constant86
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant86)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant87 = self._tensor_constant87
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant87)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 624, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 624, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 624, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 624, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段从 normalized hidden 生成 Q、K、V，并把 hidden 维拆成 32 个 128 维 head。

为什么需要：后续 attention 用 Q/K 做分数，用 V 提供被权重汇聚的 value 内容。

怎么做/计算：`mul_tensor_1` 用 `_param_constant0` 对 `_to_copy_default_1` 做 Hidden 维逐元素缩放；Q/K/V 三个分支分别把 `mul_tensor_1` `view` 成 `[624,4096]`，再用 `_tensor_constant85/86/87` 做 `mm` 得到 `[624,4096]` 投影，随后 `_unsafe_view` 恢复 batch 维；`view_default_3/4/5` 把 Hidden 维拆成 `[Heads=32,Dh=128]`；`transpose_int/1/2` 把布局从 `[B,S,Heads,Dh]` 转为 `[B,Heads,S,Dh]`，供 attention 使用。

```text
Q/K/V projection

Tensor: Norm-weighted hidden H_proj [S=624, H=4096]
Formula: H_proj[s,h] = H_norm[s,h] * gamma_in[h]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | norm-weighted hidden row 0             |
          | norm-weighted rows 1 .. 622            |
S=623     | norm-weighted hidden row 623           |
          +----------------------------------------+

Tensor: Dense Q/K/V rows Q_dense, K_dense, V_dense [S=624, H=4096]
Formula: Q_dense=H_proj W_Q; K_dense=H_proj W_K; V_dense=H_proj W_V
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | Q/K/V projected row 0                  |
          | projected rows 1 .. 622                |
S=623     | Q/K/V projected row 623                |
          +----------------------------------------+

Tensor: Head-split Q/K/V Q_h, K_h, V_h [Heads=32, S=624, Dh=128]
Formula: Q_h[h,s,d]=Q_dense[s,h*128+d]; K_h/V_h use the same head split
S axis: 0 .. 623, Dh axis: 0 .. 127 for each head h (S height compressed)
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | head h token row 0                     |
          | head h token rows 1 .. 622             |
S=623     | head h token row 623                   |
          +----------------------------------------+

examples: Q_h[0,0,0], K_h[31,611,127], V_h[18,623,64].
```

### RoPE position embedding

```python
_tensor_constant88 = self._tensor_constant88
index_tensor = aten.index.Tensor(_tensor_constant88, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant89 = self._tensor_constant89
index_tensor_1 = aten.index.Tensor(_tensor_constant89, [arg2_1])
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

是什么：这一段给 Q/K 应用 RoPE，得到带位置旋转的 Q `add_tensor_1` 和 K `add_tensor_2`。

为什么需要：QK 点积要包含 token 位置信息，RoPE 用 cos/sin 表按 position ids 旋转每个 head 向量。

怎么做/计算：`index_tensor/index_tensor_1` 用 `arg2_1` 从 `_tensor_constant88/89` 取 cos/sin 行，并 `unsqueeze` 到可广播到 head 维；Q 分支用 `mul_tensor_2 = transpose_int * cos`，再通过 `slice_tensor` 取 Dh `0..63`、`slice_tensor_1` 取 Dh `64..127`，`neg_default` 对右半取负，`cat_default` 拼成 rotate-half，`mul_tensor_3 = rotate_half(Q) * sin`，最后 `add_tensor_1 = mul_tensor_2 + mul_tensor_3`；K 分支对 `transpose_int_1` 执行同一计算，得到 `add_tensor_2`。

```text
RoPE Dh regions

Tensor: Pre-rotation Q/K half regions X_qk [Heads=32, S=624, Dh=128]
Formula: X_qk is Q_h or K_h; LEFT=X_qk[...,0:64], RIGHT=X_qk[...,64:128]
Dh axis: 0 .. 127, split at 64
          Dh=0              Dh=63 Dh=64             Dh=127
Q/K       +---------------------+-----------------------+
          | LEFT_HALF           | RIGHT_HALF            |
          +---------------------+-----------------------+

Tensor: Rotate-half Q/K Rot(X_qk) [Heads=32, S=624, Dh=128]
Formula: Rot(X_qk) = concat(-RIGHT, LEFT)
Dh axis: 0 .. 127, split at 64
          Dh=0              Dh=63 Dh=64             Dh=127
          +---------------------+-----------------------+
          | -RIGHT_HALF         | LEFT_HALF             |
          +---------------------+-----------------------+

Tensor: RoPE Q/K Q_rope, K_rope [Heads=32, S=624, Dh=128]
Formula: X_rope[h,s,d] = X_qk[h,s,d] * cos[P[s],d] + Rot(X_qk)[h,s,d] * sin[P[s],d]
Dh axis: 0 .. 127, S axis: 0 .. 623 (S height compressed)
          Dh=0                                  Dh=127
S=0       +----------------------------------------+
          | RoPE output row 0                      |
          | rows 1 .. 622                          |
S=623     | RoPE output row 623                    |
          +----------------------------------------+

examples: Q_rope[0,0,0], K_rope[12,35,64], K_rope[31,623,127].
```

### QK scores, mask, softmax

```python
transpose_int_3 = aten.transpose.int(add_tensor_2, 2, 3)
expand_default = aten.expand.default(add_tensor_1, [1, 32, 624, 128])
view_default_6 = aten.view.default(expand_default, [32, 624, 128])
expand_default_1 = aten.expand.default(transpose_int_3, [1, 32, 128, 624])
view_default_7 = aten.view.default(expand_default_1, [32, 128, 624])
bmm_default = aten.bmm.default(view_default_6, view_default_7)
view_default_8 = aten.view.default(bmm_default, [1, 32, 624, 624])
div_tensor = aten.div.Tensor(view_default_8, 11.313708498984761)
add_tensor_3 = aten.add.Tensor(div_tensor, arg1_1)
_softmax_default = aten._softmax.default(add_tensor_3, -1, True)
_to_copy_default_2 = aten._to_copy.default(_softmax_default, dtype=torch.float16)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段是标准 QK score、mask、softmax 计算，没有 `fill_`、`sum` 或 `copy_` 的区域改写。

为什么需要：下一段 attention-weighted V 需要 `[Q,K]` 权重。

怎么做/计算：`transpose_int_3` 把 rotated K `add_tensor_2` 的最后两维转成 `[Dh,K]`；`add_tensor_1` 和 `transpose_int_3` 经 `expand/view` 成 `[32,624,128]` 与 `[32,128,624]`；`bmm_default` 对每个 head 计算 `Q @ K^T` 得到 `[32,624,624]`；`view_default_8` 恢复 `[B,Heads,Q,K]`；`div_tensor` 除以 `sqrt(128)`；`add_tensor_3` 加 `arg1_1` mask；`_softmax_default` 沿 K 维归一化；`_to_copy_default_2` 转 fp16；`clone_default` 复制出后续 attention output 和 visual process 共用的权重。

```text
Plain attention weights [Heads=32, Q=624, K=624]

Tensor: QK logits L_raw [Heads=32, Q=624, K=624]
Formula: L_raw[h,q,k] = dot(Q_rope[h,q,:], K_rope[h,k,:])
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)
          K=0                                  K=623
Q=0       +----------------------------------------+
          | raw QK logits row 0                    |
          | raw QK rows 1 .. 622                   |
Q=623     | raw QK logits row 623                  |
          +----------------------------------------+

Tensor: Attention weights A [Heads=32, Q=624, K=624]
Formula: A[h,q,k] = softmax_k(L_raw[h,q,k] / sqrt(128) + M[q,k])
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)
          K=0                                  K=623
Q=0       +----------------------------------------+
          | softmax weights row 0                  |
          | softmax rows 1 .. 622                  |
Q=623     | softmax weights row 623                |
          +----------------------------------------+

examples: A[0,0,0], A[0,611,35], A[31,623,623].
```

### Attention-weighted V and hidden reshape

```python
expand_default_2 = aten.expand.default(clone_default, [1, 32, 624, 624])
view_default_9 = aten.view.default(expand_default_2, [32, 624, 624])
expand_default_3 = aten.expand.default(transpose_int_2, [1, 32, 624, 128])
view_default_10 = aten.view.default(expand_default_3, [32, 624, 128])
bmm_default_1 = aten.bmm.default(view_default_9, view_default_10)
view_default_11 = aten.view.default(bmm_default_1, [1, 32, 624, 128])
transpose_int_4 = aten.transpose.int(view_default_11, 1, 2)
clone_default_1 = aten.clone.default(transpose_int_4, memory_format=torch.contiguous_format)
view_default_12 = aten.view.default(clone_default_1, [1, 624, 4096])
```

**解释与可视化**

是什么：这一段用 attention weights 对 V 做加权求和，并合并多头 context。

为什么需要：attention 输出是权重对 value 内容的汇聚，不只是 QK 权重本身。

怎么做/计算：`clone_default` 经 `expand_default_2/view_default_9` 变成 `[32,624,624]` 权重矩阵；V `transpose_int_2` 经 `expand_default_3/view_default_10` 变成 `[32,624,128]` value rows；`bmm_default_1` 对每个 head 计算 `[Q,K] @ [K,Dh] -> [Q,Dh]`；`view_default_11` 恢复 batch/head；`transpose_int_4` 把 layout 转成 `[B,Q,Heads,Dh]`；`clone_default_1` 连续化；`view_default_12` 把 `Heads*Dh` 合并成 `[1,624,4096]` attention context。

```text
Attention-weighted V

Tensor: Attention weights A [Heads=32, Q=624, K=624]
Formula: A[h,q,k] = softmax_k(L_raw[h,q,k] / sqrt(128) + M[q,k])
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)
          K=0                                  K=623
Q=0       +----------------------------------------+
          | attention weight row 0                 |
          | rows 1 .. 622                          |
Q=623     | attention weight row 623               |
          +----------------------------------------+

Tensor: Value rows V_h [Heads=32, K=624, Dh=128]
Formula: V_h[h,k,d] = V_dense[k,h*128+d]
K axis: 0 .. 623, Dh axis: 0 .. 127 (K height compressed)
          Dh=0                                 Dh=127
K=0       +----------------------------------------+
          | V row 0                                |
          | V rows 1 .. 622                        |
K=623     | V row 623                              |
          +----------------------------------------+

Tensor: Per-head context C [Heads=32, Q=624, Dh=128]
Formula: C[h,q,d] = sum_k A[h,q,k] * V_h[h,k,d]
Q axis: 0 .. 623, Dh axis: 0 .. 127 (Q height compressed)
          Dh=0                                 Dh=127
Q=0       +----------------------------------------+
          | context row 0                          |
          | context rows 1 .. 622                  |
Q=623     | context row 623                        |
          +----------------------------------------+

Tensor: Merged attention hidden C_merge [B=1, S=624, H=4096]
Formula: C_merge[0,q,h*128+d] = C[h,q,d]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | merged attention row 0                 |
          | merged rows 1 .. 622                   |
S=623     | merged attention row 623               |
          +----------------------------------------+

examples: C[0,0,0], C[31,623,127], C_merge[0,623,4095].
```

### Visual-related value-aware process

```python
select_int = aten.select.int(view_default_12, 1, -1)
select_int_1 = aten.select.int(clone_default, 2, -1)
unsqueeze_default_2 = aten.unsqueeze.default(select_int_1, 3)
mul_tensor_6 = aten.mul.Tensor(unsqueeze_default_2, transpose_int_2)
permute_default = aten.permute.default(mul_tensor_6, [0, 2, 1, 3])
clone_default_2 = aten.clone.default(permute_default, memory_format=torch.contiguous_format)
view_default_13 = aten.view.default(clone_default_2, [1, 624, -1])
slice_tensor_4 = aten.slice.Tensor(view_default_13, 1, 35, 611)
unsqueeze_default_3 = aten.unsqueeze.default(select_int, 1)
sub_tensor = aten.sub.Tensor(unsqueeze_default_3, slice_tensor_4)
unsqueeze_default_4 = aten.unsqueeze.default(select_int, 1)
sub_tensor_1 = aten.sub.Tensor(sub_tensor, unsqueeze_default_4)
linalg_vector_norm_default = aten.linalg_vector_norm.default(sub_tensor_1, 2, [-1])
squeeze_dim = aten.squeeze.dim(linalg_vector_norm_default, 0)
gt_scalar = aten.gt.Scalar(squeeze_dim, 0.2)
nonzero_default = aten.nonzero.default(gt_scalar)
unbind_int = aten.unbind.int(nonzero_default, 1)
getitem = _operator.getitem(unbind_int, 0)
add_tensor_4 = aten.add.Tensor(getitem, 35)
```

**解释与可视化**

是什么：这一段从最后一个 query 的 value-aware contribution 中计算 Visual token 分数，并输出超过阈值的原始 sequence token index。

为什么需要：该固定 FX DAG 在本层不只是构造 Visual delta，还把 Visual span `S=35..610` 内满足阈值条件的 token offset 转回原始 sequence 坐标，作为 layer output 的第三项返回。

怎么做/计算：`select_int` 从 `view_default_12` 的 sequence 维选 `-1`，得到 last output row `[B=1,H=4096]`；`select_int_1` 从 `clone_default` 的 Q 维选 `-1`，得到最后 query 对所有 K 的 attention weights `[B=1,Heads=32,K=624]`；`unsqueeze_default_2` 在最后加一维，使权重可与 `transpose_int_2` 的 V `[B,Heads,K,Dh]` 相乘；`mul_tensor_6` 逐 K/Dh 得到 last-query value contribution；`permute_default` 转成 token-major `[B,K,Heads,Dh]`；`clone_default_2` 连续化；`view_default_13` 把 head/Dh 合并为 `[1,624,4096]` contribution rows；`slice_tensor_4` 取 sequence `35..610`，得到 576 行 Visual contribution；`unsqueeze_default_3` 把 last output row 变成 `[1,1,4096]` reference；`sub_tensor` 对每个 Visual row 计算 `reference - contribution`；`unsqueeze_default_4` 再次扩展同一个 reference；`sub_tensor_1` 按 ops 继续计算 `(reference - contribution) - reference`；`linalg_vector_norm_default` 沿 Hidden 维对 `sub_tensor_1` 求 L2 norm；`squeeze_dim` 去掉 batch 维得到 Visual score band；`gt_scalar` 将每个 score 与阈值 `0.2` 比较；`nonzero_default` 找出为真的 Visual offset；`unbind_int/getitem` 取 offset 向量；`add_tensor_4` 给 offset 加 35，变回原始 sequence index。

```text
Visual selection from value-aware rows

Visual token axis V=576 maps to sequence S=35 .. 610.
Hidden axis H=0 .. 4095 (H width compressed)

Tensor: Last output reference R_last [B=1, H=4096]
Formula: R_last[h] = C_merge[0,623,h]
H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
ref       +----------------------------------------+
          | C_merge last row, examples H0/H4095    |
          +----------------------------------------+

Tensor: Last-query attention A_last [Heads=32, K=624]
Formula: A_last[h,k] = A[h,623,k]
K axis: 0 .. 623, Heads axis: 0 .. 31 (K width compressed)
          K=0                                  K=623
head 0    +----------------------------------------+
          | last-query weights for head 0          |
          | heads 1 .. 30                          |
head 31   | last-query weights for head 31         |
          +----------------------------------------+

Tensor: Token contribution T_contrib [S=624, H=4096]
Formula: T_contrib[k,h*128+d] = A_last[h,k] * V_h[h,k,d]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | weighted V contribution row 0          |
          | contribution rows 1 .. 622             |
S=623     | weighted V contribution row 623        |
          +----------------------------------------+

Tensor: Visual contribution V_contrib [V=576, H=4096]
Formula: V_contrib[v,h] = T_contrib[v+35,h]
V axis: 0 .. 575 maps S=35 .. 610, H axis: 0 .. 4095
          H=0                                  H=4095
V=0/S=35  +----------------------------------------+
          | contribution from attn[Q=623,K=35]     |
          | Visual rows S=36 .. 609                |
V=575     | contribution from attn[Q=623,K=610]    |
          +----------------------------------------+

Tensor: Reference-minus-Visual delta D_ref [V=576, H=4096]
Formula: D_ref[v,h] = R_last[h] - V_contrib[v,h]
V axis: 0 .. 575, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
V=0       +----------------------------------------+
          | REF_MINUS_VISUAL row 0                 |
          | REF_MINUS_VISUAL rows 1 .. 574         |
V=575     | REF_MINUS_VISUAL row 575               |
          +----------------------------------------+

Tensor: Score source D_score [V=576, H=4096]
Formula: D_score[v,h] = D_ref[v,h] - R_last[h]
V axis: 0 .. 575, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
V=0       +----------------------------------------+
          | SCORE_SOURCE row 0                     |
          | SCORE_SOURCE rows 1 .. 574             |
V=575     | SCORE_SOURCE row 575                   |
          +----------------------------------------+

Tensor: Score band S_score [V=576]
Formula: S_score[v] = ||D_score[v,:]||_2
V axis: 0 .. 575
V=0       +----------------------------------------+ V=575
          | score[0] ... score[288] ... score[575] |
          +----------------------------------------+

Tensor: Selection mask M_sel [V=576]
Formula: M_sel[v] = S_score[v] > 0.2
V axis: 0 .. 575
V=0       +----------------------------------------+ V=575
          | bool[0] ... bool[288] ... bool[575]    |
          +----------------------------------------+

Tensor: Selected sequence ids S_selected [P]
Formula: S_selected[p] = 35 + v_p where M_sel[v_p] = True
P axis: 0 .. P-1, values are original sequence coordinates
          +----------------------------------------+
          | 35+v0, 35+v1, ... selected token ids   |
          +----------------------------------------+

examples: if v=0 is selected -> S=35; if v=575 is selected -> S=610.
```

### Attention output projection and residual

```python
view_default_14 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant90 = self._tensor_constant90
mm_default_3 = aten.mm.default(view_default_14, _tensor_constant90)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_5 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把 attention context 过输出投影，并与原始 hidden 做残差加法。

为什么需要：输出投影把多头 context 映射回 hidden space，残差连接保留层输入。

怎么做/计算：`view_default_14` 把 `view_default_12` 展平为 `[624,4096]`；`mm_default_3` 与 `_tensor_constant90` 做输出投影，仍是 `[624,4096]`；`_unsafe_view_default_3` 恢复 `[1,624,4096]`；`add_tensor_5` 在相同 S/H 坐标上执行 `arg0_1 + projected_context`，形成 post-attention residual。

```text
Output projection + residual [S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095

Tensor: Attention hidden C_merge [S=624, H=4096]
Formula: C_merge[0,q,h*128+d] = C[h,q,d]
          H=0                                  H=4095
S=0       +----------------------------------------+
          | attention context row 0                |
          | context rows 1 .. 622                  |
S=623     | attention context row 623              |
          +----------------------------------------+

Tensor: Projected attention O_attn [S=624, H=4096]
Formula: O_attn = C_merge W_O
          H=0                                  H=4095
S=0       +----------------------------------------+
          | projected context rows 0 .. 623        |
          | projected rows 1 .. 622                |
S=623     | projected context row 623              |
          +----------------------------------------+

Tensor: Post-attention residual H_attn [S=624, H=4096]
Formula: H_attn[s,h] = H_in[s,h] + O_attn[s,h]
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_in rows 0 .. 623                     |
          | + O_attn on same S/H coords            |
S=623     | residual row 623                       |
          +----------------------------------------+

examples: H_attn[0,0], H_attn[611,2048], H_attn[623,4095].
```

### Post-attention RMSNorm

```python
_to_copy_default_3 = aten._to_copy.default(add_tensor_5, dtype=torch.float32)
pow_tensor_scalar_1 = aten.pow.Tensor_Scalar(_to_copy_default_3, 2)
mean_dim_1 = aten.mean.dim(pow_tensor_scalar_1, [-1], True)
add_tensor_6 = aten.add.Tensor(mean_dim_1, 1e-05)
rsqrt_default_1 = aten.rsqrt.default(add_tensor_6)
mul_tensor_7 = aten.mul.Tensor(_to_copy_default_3, rsqrt_default_1)
_to_copy_default_4 = aten._to_copy.default(mul_tensor_7, dtype=torch.float16)
_param_constant5 = self._param_constant5
mul_tensor_8 = aten.mul.Tensor(_param_constant5, _to_copy_default_4)
view_default_15 = aten.view.default(mul_tensor_8, [624, 4096])
```

**解释与可视化**

是什么：这一段对 post-attention residual 做 RMSNorm，并生成 MLP 输入。

为什么需要：MLP projection 前需要归一化 token rows。

怎么做/计算：`_to_copy_default_3` 把 `add_tensor_5` 转 fp32；`pow_tensor_scalar_1` 对 Hidden 元素平方；`mean_dim_1` 沿 Hidden 维求均值；`add_tensor_6` 加 `1e-05`；`rsqrt_default_1` 得到行级 scale；`mul_tensor_7` 把 fp32 residual 乘 scale；`_to_copy_default_4` 转 fp16；`mul_tensor_8` 应用 `_param_constant5` norm weight；`view_default_15` 把结果展平成 `[624,4096]` 作为 MLP gate 分支输入，`mul_tensor_8` 还被 `view_default_16` 用作 up 分支输入。

```text
Post-attention RMSNorm [S=624, H=4096]

Tensor: Squared residual H_attn_sq [S=624, H=4096]
Formula: H_attn_sq[s,h] = float32(H_attn[s,h])^2
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | H_attn row 0 squared                   | ----> | r0 |
          | rows 1 .. 622                          |       | .. |
S=623     | H_attn row 623 squared                 | ----> | r623 |
          +----------------------------------------+       +----+

Tensor: Post-attention RMS scale R_post [S=624, 1]
Formula: R_post[s] = 1 / sqrt(mean_h(H_attn_sq[s,h]) + 1e-05)
S axis: 0 .. 623
S=0       +------+
          | r0   |
          | ..   |
S=623     | r623 |
          +------+

Tensor: MLP input H_mlp [S=624, H=4096]
Formula: H_mlp[s,h] = fp16(float32(H_attn[s,h]) * R_post[s]) * gamma_post[h]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | normalized residual row 0              |
          | normalized rows 1 .. 622               |
S=623     | normalized residual row 623            |
          +----------------------------------------+

examples: H_mlp[0,0], H_mlp[35,1024], H_mlp[623,4095].
```

### MLP and final residual

```python
_tensor_constant91 = self._tensor_constant91
mm_default_4 = aten.mm.default(view_default_15, _tensor_constant91)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_16 = aten.view.default(mul_tensor_8, [624, 4096])
_tensor_constant92 = self._tensor_constant92
mm_default_5 = aten.mm.default(view_default_16, _tensor_constant92)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_9 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_17 = aten.view.default(mul_tensor_9, [624, 11008])
_tensor_constant93 = self._tensor_constant93
mm_default_6 = aten.mm.default(view_default_17, _tensor_constant93)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_7 = aten.add.Tensor(add_tensor_5, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP，并把 down projection 结果加回 attention residual。

为什么需要：MLP 为每个 token 做非线性通道变换，residual add 保持主干 hidden 传递。

怎么做/计算：`mm_default_4` 使用 `view_default_15` 和 `_tensor_constant91` 产生 gate projection `[624,11008]`，`_unsafe_view_default_4` 恢复 batch 维，`silu_default` 对 gate 做 SiLU；`view_default_16` 从同一个 `mul_tensor_8` 得到 up 分支输入，`mm_default_5` 与 `_tensor_constant92` 产生 up projection `[624,11008]`；`mul_tensor_9` 在同一 S/I 坐标上计算 `silu(gate) * up`；`view_default_17` 展平后，`mm_default_6` 与 `_tensor_constant93` down project 回 `[624,4096]`；`_unsafe_view_default_6` 恢复 batch 维；`add_tensor_7` 与 `add_tensor_5` 在同一 S/H 坐标上做最终 residual add。

```text
MLP [S=624, H=4096] -> [S=624, I=11008] -> [S=624, H=4096]

Tensor: MLP input H_mlp [S=624, H=4096]
Formula: H_mlp[s,h] = fp16(float32(H_attn[s,h]) * R_post[s]) * gamma_post[h]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | MLP input row 0                        |
          | MLP input rows 1 .. 622                |
S=623     | MLP input row 623                      |
          +----------------------------------------+

Tensor: Gated intermediate G [S=624, I=11008]
Formula: G[s,i] = silu((H_mlp W_gate)[s,i]) * (H_mlp W_up)[s,i]
S axis: 0 .. 623, I axis: 0 .. 11007 (I width compressed)
          I=0                                  I=11007
S=0       +----------------------------------------+
          | gated intermediate row 0               |
          | gated rows 1 .. 622                    |
S=623     | gated intermediate row 623             |
          +----------------------------------------+

Tensor: Layer hidden H_out [S=624, H=4096]
Formula: H_out[s,h] = H_attn[s,h] + (G W_down)[s,h]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | final residual row 0                   |
          | final rows 1 .. 622                    |
S=623     | final residual row 623                 |
          +----------------------------------------+

examples: G[0,0], G[623,11007], H_out[623,4095].
```

### Layer output

```python
return (add_tensor_7, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, add_tensor_4, 0)
```

**解释与可视化**

是什么：这一段返回 final hidden、当前层 K/V cache、Visual selected token indices 和控制标量 `0`。

为什么需要：`add_tensor_7` 供下一层继续计算；K/V cache 供后续解码复用；`add_tensor_4` 是本层 Visual process 输出的原始 sequence token 坐标。

怎么做/计算：`return` 不创建新 tensor，只把 `add_tensor_7` 作为 hidden output，把 `add_tensor_2` 作为当前 layer 的 K cache，把 `transpose_int_2` 作为当前 layer 的 V cache，把 `add_tensor_4` 作为 Visual selected token ids，再返回控制标量 `0`。

```text
Layer output

Tensor: Layer hidden output H_out [B=1, S=624, H=4096]
Formula: tuple[0] = H_out
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | final hidden row 0                     |
          | rows 1 .. 622                          |
S=623     | final hidden row 623                   |
          +----------------------------------------+

Tensor: Cache key K_cache [Heads=32, S=624, Dh=128]
Formula: K_cache = K_rope, returned in dynamic_cache_layer[0]
S axis: 0 .. 623, Dh axis: 0 .. 127 for each head h
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | rotated K row 0                        |
          | rotated K rows 1 .. 622                |
S=623     | rotated K row 623                      |
          +----------------------------------------+

Tensor: Cache value V_cache [Heads=32, S=624, Dh=128]
Formula: V_cache = V_h, returned in dynamic_cache_layer[1]
S axis: 0 .. 623, Dh axis: 0 .. 127 for each head h
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | V row 0                                |
          | V rows 1 .. 622                        |
S=623     | V row 623                              |
          +----------------------------------------+

Tensor: Visual selected ids S_selected [P]
Formula: tuple[2] = S_selected, with each id in original sequence range 35 .. 610
P axis: 0 .. P-1
          +----------------------------------------+
          | selected_id[0] ... selected_id[P-1]    |
          +----------------------------------------+

examples: K_cache[31,623,127], V_cache[18,611,64], S_selected[p].
```

## Node Table

| index | stage | name | op | target | args | users |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `inputs` | `arg0_1` | `placeholder` | `arg0_1` | - | `_to_copy_default`, `add_tensor_5` |
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
| 18 | `qkv_projection` | `_tensor_constant85` | `get_attr` | `_tensor_constant85` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant85` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant86` | `get_attr` | `_tensor_constant86` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant86` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant87` | `get_attr` | `_tensor_constant87` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant87` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `mul_tensor_6`, `output` |
| 35 | `rope` | `_tensor_constant88` | `get_attr` | `_tensor_constant88` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant88`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant89` | `get_attr` | `_tensor_constant89` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant89`, `arg2_1` | `unsqueeze_default_1` |
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
| 66 | `attention_scores` | `clone_default` | `call_function` | `aten.clone.default` | `_to_copy_default_2` | `expand_default_2`, `select_int_1` |
| 67 | `attention_output` | `expand_default_2` | `call_function` | `aten.expand.default` | `clone_default` | `view_default_9` |
| 68 | `attention_output` | `view_default_9` | `call_function` | `aten.view.default` | `expand_default_2` | `bmm_default_1` |
| 69 | `attention_output` | `expand_default_3` | `call_function` | `aten.expand.default` | `transpose_int_2` | `view_default_10` |
| 70 | `attention_output` | `view_default_10` | `call_function` | `aten.view.default` | `expand_default_3` | `bmm_default_1` |
| 71 | `attention_output` | `bmm_default_1` | `call_function` | `aten.bmm.default` | `view_default_9`, `view_default_10` | `view_default_11` |
| 72 | `attention_output` | `view_default_11` | `call_function` | `aten.view.default` | `bmm_default_1` | `transpose_int_4` |
| 73 | `attention_output` | `transpose_int_4` | `call_function` | `aten.transpose.int` | `view_default_11` | `clone_default_1` |
| 74 | `attention_output` | `clone_default_1` | `call_function` | `aten.clone.default` | `transpose_int_4` | `view_default_12` |
| 75 | `attention_output` | `view_default_12` | `call_function` | `aten.view.default` | `clone_default_1` | `select_int`, `view_default_14` |
| 76 | `visual_process` | `select_int` | `call_function` | `aten.select.int` | `view_default_12` | `unsqueeze_default_3`, `unsqueeze_default_4` |
| 77 | `visual_process` | `select_int_1` | `call_function` | `aten.select.int` | `clone_default` | `unsqueeze_default_2` |
| 78 | `visual_process` | `unsqueeze_default_2` | `call_function` | `aten.unsqueeze.default` | `select_int_1` | `mul_tensor_6` |
| 79 | `visual_process` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `unsqueeze_default_2`, `transpose_int_2` | `permute_default` |
| 80 | `visual_process` | `permute_default` | `call_function` | `aten.permute.default` | `mul_tensor_6` | `clone_default_2` |
| 81 | `visual_process` | `clone_default_2` | `call_function` | `aten.clone.default` | `permute_default` | `view_default_13` |
| 82 | `visual_process` | `view_default_13` | `call_function` | `aten.view.default` | `clone_default_2` | `slice_tensor_4` |
| 83 | `visual_process` | `slice_tensor_4` | `call_function` | `aten.slice.Tensor` | `view_default_13` | `sub_tensor` |
| 84 | `visual_process` | `unsqueeze_default_3` | `call_function` | `aten.unsqueeze.default` | `select_int` | `sub_tensor` |
| 85 | `visual_process` | `sub_tensor` | `call_function` | `aten.sub.Tensor` | `unsqueeze_default_3`, `slice_tensor_4` | `sub_tensor_1` |
| 86 | `visual_process` | `unsqueeze_default_4` | `call_function` | `aten.unsqueeze.default` | `select_int` | `sub_tensor_1` |
| 87 | `visual_process` | `sub_tensor_1` | `call_function` | `aten.sub.Tensor` | `sub_tensor`, `unsqueeze_default_4` | `linalg_vector_norm_default` |
| 88 | `visual_process` | `linalg_vector_norm_default` | `call_function` | `aten.linalg_vector_norm.default` | `sub_tensor_1` | `squeeze_dim` |
| 89 | `visual_process` | `squeeze_dim` | `call_function` | `aten.squeeze.dim` | `linalg_vector_norm_default` | `gt_scalar` |
| 90 | `visual_process` | `gt_scalar` | `call_function` | `aten.gt.Scalar` | `squeeze_dim` | `nonzero_default` |
| 91 | `visual_process` | `nonzero_default` | `call_function` | `aten.nonzero.default` | `gt_scalar` | `unbind_int` |
| 92 | `visual_process` | `unbind_int` | `call_function` | `aten.unbind.int` | `nonzero_default` | `getitem` |
| 93 | `visual_process` | `getitem` | `call_function` | `_operator.getitem` | `unbind_int` | `add_tensor_4` |
| 94 | `visual_process` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `getitem` | `output` |
| 95 | `output_projection` | `view_default_14` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 96 | `output_projection` | `_tensor_constant90` | `get_attr` | `_tensor_constant90` | - | `mm_default_3` |
| 97 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant90` | `_unsafe_view_default_3` |
| 98 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_5` |
| 99 | `output_projection` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_7` |
| 100 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_5` | `mul_tensor_7`, `pow_tensor_scalar_1` |
| 101 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 102 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_6` |
| 103 | `post_attention_rmsnorm` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 104 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_6` | `mul_tensor_7` |
| 105 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 106 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_7` | `mul_tensor_8` |
| 107 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_8` |
| 108 | `post_attention_rmsnorm` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_15`, `view_default_16` |
| 109 | `post_attention_rmsnorm` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_4` |
| 110 | `mlp` | `_tensor_constant91` | `get_attr` | `_tensor_constant91` | - | `mm_default_4` |
| 111 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant91` | `_unsafe_view_default_4` |
| 112 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 113 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_9` |
| 114 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_5` |
| 115 | `mlp` | `_tensor_constant92` | `get_attr` | `_tensor_constant92` | - | `mm_default_5` |
| 116 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant92` | `_unsafe_view_default_5` |
| 117 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_9` |
| 118 | `mlp` | `mul_tensor_9` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_17` |
| 119 | `mlp` | `view_default_17` | `call_function` | `aten.view.default` | `mul_tensor_9` | `mm_default_6` |
| 120 | `mlp` | `_tensor_constant93` | `get_attr` | `_tensor_constant93` | - | `mm_default_6` |
| 121 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_17`, `_tensor_constant93` | `_unsafe_view_default_6` |
| 122 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_7` |
| 123 | `mlp` | `add_tensor_7` | `call_function` | `aten.add.Tensor` | `add_tensor_5`, `_unsafe_view_default_6` | `output` |
| 124 | `layer_output` | `output` | `output` | `output` | `add_tensor_7`, `add_tensor_2`, `transpose_int_2`, `add_tensor_4` | - |
