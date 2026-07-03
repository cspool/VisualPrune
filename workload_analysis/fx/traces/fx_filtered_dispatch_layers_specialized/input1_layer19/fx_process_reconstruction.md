# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer19`
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
| Visual-related value-aware process | 76-86 | 11 | `clone_default`, `transpose_int_2`, `view_default_12` | - |
| Attention output projection and residual | 87-91 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 92-101 | 10 | `add_tensor_4` | `mul_tensor_8`, `view_default_15` |
| MLP and final residual | 102-115 | 14 | `add_tensor_4`, `mul_tensor_8`, `view_default_15` | `add_tensor_6` |
| Layer output | 116-116 | 1 | `add_tensor_2`, `add_tensor_6`, `arg6_1`, `transpose_int_2` | - |

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

是什么：这一段是固定样本进入 FX DAG 的 placeholder。后续实际参与计算的是 `arg0_1`、`arg1_1`、`arg2_1`；`arg6_1` 只在 layer output 中透传。

为什么需要：本层需要 hidden states 作为主干输入，需要 attention mask 修正 QK logits，需要 position ids 做 RoPE 查表；已有的 Visual token/index 信息需要随本层输出继续传给后续流程。

怎么做/计算：placeholder 本身不做数值计算。`arg0_1` 被 `_to_copy_default` 读入输入 RMSNorm，并在 `add_tensor_4` 中作为 attention residual；`arg1_1` 被 `add_tensor_3` 加到缩放后的 QK logits；`arg2_1` 被 `index_tensor` 和 `index_tensor_1` 用于 cos/sin 位置表索引；`arg6_1` 没有参与中间计算，只被 `output` 节点打包返回。`arg3_1`、`arg4_1`、`arg5_1`、`arg7_1` 在这个固定输入图中没有 user。

```text
Tensor: Input hidden H_in [B=1, S=58, H=4096]
Formula: H_in[0,s,h] is the sampled layer hidden input consumed by RMSNorm and residual paths
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_IN row 0; examples h=0,2048,4095     |
          | H_IN rows 1 .. 56                      |
S=57      | H_IN row 57; examples h=0,2048,4095    |
          +----------------------------------------+

Tensor: Attention mask M [Q=58, K=58]
Formula: masked logits later use L_mask[q,k] = L_scaled[q,k] + M[q,k]
Q axis: 0 .. 57, K axis: 0 .. 57 (square compressed)
          K=0                                  K=57
Q=0       +----------------------------------------+
          | MASK row 0; examples k=0,29,57         |
          | MASK rows 1 .. 56                      |
Q=57      | MASK row 57; examples k=0,29,57        |
          +----------------------------------------+

Tensor: Position ids P [S=58]
Formula: cos/sin rows later use C_pos[s,d] = C_table[P[s],d] and S_pos[s,d] = S_table[P[s],d]
S axis: 0 .. 57
          +----------------------------------------+
          | P[0] ... P[35] ... P[57]               |
          +----------------------------------------+

Tensor: Passthrough visual ids V_prev [P_prev]
Formula: V_prev is returned unchanged as output tuple element 2
P_prev axis: 0 .. end
          +----------------------------------------+
          | V_PREV[p] examples p=0,mid,end         |
          +----------------------------------------+
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

是什么：这一段对输入 hidden states 做 RMSNorm，得到 fp16 的 normalized hidden，同时读取 input RMSNorm 的权重向量。

为什么需要：Q/K/V 投影前要按 token 行的 Hidden RMS 尺度归一化。

怎么做/计算：`_to_copy_default` 把输入 hidden 转成 fp32；`pow_tensor_scalar` 对每个 hidden 元素平方；`mean_dim` 在最后一维 Hidden 上求均值并保留 `[B,S,1]` 形状；`add_tensor` 加上 `1e-05`；`rsqrt_default` 得到每个 token 的 RMS 缩放因子；`mul_tensor` 把 fp32 hidden 与该缩放因子广播相乘；`_to_copy_default_1` 转回 fp16；`_param_constant0` 读取 shape `[4096]` 的权重，供下一段与 normalized hidden 逐 Hidden 维相乘。

```text
Tensor: Input hidden H_in [S=58, H=4096]
Formula: H32[s,h] = fp32(H_in[0,s,h])
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H32 row 0; examples h=0,2048,4095      |
          | H32 rows 1 .. 56                       |
S=57      | H32 row 57; examples h=0,2048,4095     |
          +----------------------------------------+

Tensor: RMS scale R_in [S=58, 1]
Formula: R_in[s,0] = rsqrt(mean_h(H32[s,h]^2) + 1e-05)
S axis: 0 .. 57
          +------+
S=0       | R0   |
          | ...  |
S=57      | R57  |
          +------+

Tensor: Normalized hidden H_norm [S=58, H=4096]
Formula: H_norm[s,h] = fp16(H32[s,h] * R_in[s,0])
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_NORM row 0; examples h=0,2048,4095   |
          | H_NORM rows 1 .. 56                    |
S=57      | H_NORM row 57; examples h=0,2048,4095  |
          +----------------------------------------+

Tensor: Input RMSNorm weight Gamma_in [H=4096]
Formula: Gamma_in[h] is loaded here and applied in the next process as Gamma_in[h] * H_norm[s,h]
H axis: 0 .. 4095
          +----------------------------------------+
          | GAMMA examples h=0,2048,4095           |
          +----------------------------------------+
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [58, 4096])
_tensor_constant94 = self._tensor_constant94
mm_default = aten.mm.default(view_default, _tensor_constant94)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 58, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [58, 4096])
_tensor_constant95 = self._tensor_constant95
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant95)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 58, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [58, 4096])
_tensor_constant96 = self._tensor_constant96
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant96)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 58, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 58, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 58, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 58, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段从 normalized hidden 生成 Q、K、V，并把 hidden 维拆成 32 个 128 维 head。

为什么需要：后续 attention 用 Q/K 做分数，用 V 提供被权重汇聚的 value 内容。

怎么做/计算：`mul_tensor_1` 把上一段的 fp16 normalized hidden 与 `_param_constant0` 在 Hidden 维逐元素相乘；`view_default`、`view_default_1`、`view_default_2` 都把结果看成 `[58,4096]`；Q/K/V 三个分支分别通过 `mm_default`、`mm_default_1`、`mm_default_2` 与 `_tensor_constant94/95/96` 做矩阵乘，得到三个 `[58,4096]` dense 投影；`_unsafe_view_default*` 恢复 batch 维为 `[1,58,4096]`；`view_default_3/4/5` 把 4096 拆成 `[32,128]`；`transpose_int/1/2` 把形状变成 `[1,32,58,128]`，即 head 在 sequence 维之前。

```text
Tensor: Weighted normalized hidden H_qkv [S=58, H=4096]
Formula: H_qkv[s,h] = Gamma_in[h] * H_norm[s,h]
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_QKV row 0; examples h=0,2048,4095    |
          | H_QKV rows 1 .. 56                     |
S=57      | H_QKV row 57; examples h=0,2048,4095   |
          +----------------------------------------+

Tensor: Dense projections Q_dense, K_dense, V_dense [S=58, H=4096]
Formula: Q_dense[s,o] = sum_h H_qkv[s,h] * W_Q[h,o]; K_dense and V_dense use W_K and W_V
S axis: 0 .. 57, projected H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | DENSE row 0; examples o=0,2048,4095    |
          | DENSE rows 1 .. 56                     |
S=57      | DENSE row 57; examples o=0,2048,4095   |
          +----------------------------------------+

Tensor: Head-split Q_h, K_h, V_h [Heads=32, S=58, Dh=128]
Formula: Q_h[a,s,d] = Q_dense[s, a*128 + d], where d is the within-head offset 0..127; K_h and V_h follow the same mapping
Shown for one head a; S axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | HEAD row 0; examples d=0,64,127        |
          | HEAD rows 1 .. 56                      |
S=57      | HEAD row 57; examples d=0,64,127       |
          +----------------------------------------+
```

### RoPE position embedding

```python
_tensor_constant97 = self._tensor_constant97
index_tensor = aten.index.Tensor(_tensor_constant97, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant98 = self._tensor_constant98
index_tensor_1 = aten.index.Tensor(_tensor_constant98, [arg2_1])
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

怎么做/计算：`index_tensor` 用 `arg2_1` 从 `_tensor_constant97` 取 cos 行，`unsqueeze_default` 插入 head 广播维；`index_tensor_1` 用同样的 position ids 从 `_tensor_constant98` 取 sin 行，`unsqueeze_default_1` 插入 head 广播维。Q 分支先用 `mul_tensor_2` 计算 Q 与 cos 的逐元素乘；`slice_tensor` 取 Dh `0..63`，`slice_tensor_1` 取 Dh `64..127`，`neg_default` 对右半取负，`cat_default` 拼成 rotate-half；`mul_tensor_3` 乘 sin，`add_tensor_1` 得到 RoPE 后的 Q。K 分支用 `mul_tensor_4`、`slice_tensor_2/3`、`neg_default_1`、`cat_default_1`、`mul_tensor_5`、`add_tensor_2` 做同样计算。

```text
Tensor: Position tables C_pos and S_pos [S=58, Dh=128]
Formula: C_pos[s,d] = CosTable[P[s],d], S_pos[s,d] = SinTable[P[s],d]
S axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | C_POS/S_POS row 0; examples d=0,64,127 |
          | C_POS/S_POS rows 1 .. 56               |
S=57      | C_POS/S_POS row 57; examples d=0,64,127|
          +----------------------------------------+

Tensor: Rotate-half view Rot(X_h) [Heads=32, S=58, Dh=128]
Formula: Rot(X_h)[a,s,0..63] = -X_h[a,s,64..127], Rot(X_h)[a,s,64..127] = X_h[a,s,0..63]
Shown for X=Q or K; Dh axis: 0 .. 127, split at 64
          Dh=0              Dh=63 Dh=64             Dh=127
S row     +---------------------+-----------------------+
          | NEG_RIGHT_HALF      | LEFT_HALF             |
          +---------------------+-----------------------+

Tensor: RoPE Q/K Q_rot, K_rot [Heads=32, S=58, Dh=128]
Formula: X_rot[a,s,d] = X_h[a,s,d] * C_pos[s,d] + Rot(X_h)[a,s,d] * S_pos[s,d], for X in {Q,K}
Shown for one head a; S axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | X_ROT row 0; examples d=0,64,127       |
          | X_ROT rows 1 .. 56                     |
S=57      | X_ROT row 57; examples d=0,64,127      |
          +----------------------------------------+
```

### QK scores, mask, softmax

```python
transpose_int_3 = aten.transpose.int(add_tensor_2, 2, 3)
expand_default = aten.expand.default(add_tensor_1, [1, 32, 58, 128])
view_default_6 = aten.view.default(expand_default, [32, 58, 128])
expand_default_1 = aten.expand.default(transpose_int_3, [1, 32, 128, 58])
view_default_7 = aten.view.default(expand_default_1, [32, 128, 58])
bmm_default = aten.bmm.default(view_default_6, view_default_7)
view_default_8 = aten.view.default(bmm_default, [1, 32, 58, 58])
div_tensor = aten.div.Tensor(view_default_8, 11.313708498984761)
add_tensor_3 = aten.add.Tensor(div_tensor, arg1_1)
_softmax_default = aten._softmax.default(add_tensor_3, -1, True)
_to_copy_default_2 = aten._to_copy.default(_softmax_default, dtype=torch.float16)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段是标准 QK score、mask、softmax 计算，没有区域改写。

为什么需要：下一段 attention-weighted V 需要 `[Q,K]` 权重。

怎么做/计算：`transpose_int_3` 把 RoPE K 的最后两维从 `[S,Dh]` 转成 `[Dh,S]`；`expand_default/view_default_6` 把 RoPE Q 变成 `[32,58,128]`；`expand_default_1/view_default_7` 把转置 K 变成 `[32,128,58]`；`bmm_default` 对每个 head 做 QK 矩阵乘得到 `[32,58,58]`；`view_default_8` 恢复 `[1,32,58,58]`；`div_tensor` 除以 `11.313708498984761`，即 `sqrt(128)`；`add_tensor_3` 加 attention mask；`_softmax_default` 在 K 维归一化；`_to_copy_default_2` 转 fp16；`clone_default` 复制成后续 attention output 和 visual_process 共同读取的 attention weights。

```text
Tensor: Raw attention logits L_raw [Heads=32, Q=58, K=58]
Formula: L_raw[a,q,k] = sum_d Q_rot[a,q,d] * K_rot[a,k,d]
Shown for one head a; Q axis: 0 .. 57, K axis: 0 .. 57 (square compressed)
          K=0                                  K=57
Q=0       +----------------------------------------+
          | L_RAW row 0; examples k=0,29,57        |
          | L_RAW rows 1 .. 56                     |
Q=57      | L_RAW row 57; examples k=0,29,57       |
          +----------------------------------------+

Tensor: Attention weights A [Heads=32, Q=58, K=58]
Formula: A[a,q,k] = softmax_k(L_raw[a,q,k] / sqrt(128) + M[q,k])
Shown for one head a; Q axis: 0 .. 57, K axis: 0 .. 57 (same square)
          K=0                                  K=57
Q=0       +----------------------------------------+
          | A row 0; examples k=0,29,57            |
          | A rows 1 .. 56                         |
Q=57      | A row 57; examples k=0,29,57           |
          +----------------------------------------+
```

### Attention-weighted V and hidden reshape

```python
expand_default_2 = aten.expand.default(clone_default, [1, 32, 58, 58])
view_default_9 = aten.view.default(expand_default_2, [32, 58, 58])
expand_default_3 = aten.expand.default(transpose_int_2, [1, 32, 58, 128])
view_default_10 = aten.view.default(expand_default_3, [32, 58, 128])
bmm_default_1 = aten.bmm.default(view_default_9, view_default_10)
view_default_11 = aten.view.default(bmm_default_1, [1, 32, 58, 128])
transpose_int_4 = aten.transpose.int(view_default_11, 1, 2)
clone_default_1 = aten.clone.default(transpose_int_4, memory_format=torch.contiguous_format)
view_default_12 = aten.view.default(clone_default_1, [1, 58, 4096])
```

**解释与可视化**

是什么：这一段用 attention weights 对 V 做加权求和，并合并多头 context。

为什么需要：attention 输出是权重对 value 内容的汇聚，不只是 QK 权重本身。

怎么做/计算：`expand_default_2/view_default_9` 把 attention weights 视作 `[32,58,58]`；`expand_default_3/view_default_10` 把 V heads 视作 `[32,58,128]`；`bmm_default_1` 对每个 head 做 `[Q,K] x [K,Dh] -> [Q,Dh]`；`view_default_11` 恢复 batch/head；`transpose_int_4` 把 `[B,Heads,S,Dh]` 转成 `[B,S,Heads,Dh]`；`clone_default_1` 连续化；`view_default_12` 把 `Heads*Dh` 合并成 `[1,58,4096]`。

```text
Tensor: Attention weights A [Heads=32, Q=58, K=58]
Formula: A[a,q,k] comes from softmax_k(L_raw[a,q,k] / sqrt(128) + M[q,k])
Shown for one head a; Q axis: 0 .. 57, K axis: 0 .. 57 (square compressed)
          K=0                                  K=57
Q=0       +----------------------------------------+
          | A row 0; examples k=0,29,57            |
          | A rows 1 .. 56                         |
Q=57      | A row 57; examples k=0,29,57           |
          +----------------------------------------+

Tensor: Value heads V_h [Heads=32, K=58, Dh=128]
Formula: V_h[a,k,d] = V_dense[k, a*128 + d]
Shown for one head a; K axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
K=0       +----------------------------------------+
          | V_H row 0; examples d=0,64,127         |
          | V_H rows 1 .. 56                       |
K=57      | V_H row 57; examples d=0,64,127        |
          +----------------------------------------+

Tensor: Per-head context C_h [Heads=32, Q=58, Dh=128]
Formula: C_h[a,q,d] = sum_k A[a,q,k] * V_h[a,k,d]
Shown for one head a; Q axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
Q=0       +----------------------------------------+
          | C_H row 0; examples d=0,64,127         |
          | C_H rows 1 .. 56                       |
Q=57      | C_H row 57; examples d=0,64,127        |
          +----------------------------------------+

Tensor: Merged context C_merge [B=1, S=58, H=4096]
Formula: C_merge[0,s,a*128+d] = C_h[a,s,d]
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | C_MERGE row 0; examples h=0,2048,4095  |
          | C_MERGE rows 1 .. 56                   |
S=57      | C_MERGE row 57; examples h=0,2048,4095 |
          +----------------------------------------+
```

### Visual-related value-aware process

```python
select_int = aten.select.int(view_default_12, 1, -1)
select_int_1 = aten.select.int(clone_default, 2, -1)
unsqueeze_default_2 = aten.unsqueeze.default(select_int_1, 3)
mul_tensor_6 = aten.mul.Tensor(unsqueeze_default_2, transpose_int_2)
permute_default = aten.permute.default(mul_tensor_6, [0, 2, 1, 3])
clone_default_2 = aten.clone.default(permute_default, memory_format=torch.contiguous_format)
view_default_13 = aten.view.default(clone_default_2, [1, 58, -1])
_tensor_constant99 = self._tensor_constant99
index_tensor_2 = aten.index.Tensor(view_default_13, [None, _tensor_constant99])
unsqueeze_default_3 = aten.unsqueeze.default(select_int, 1)
sub_tensor = aten.sub.Tensor(unsqueeze_default_3, index_tensor_2)
```

**解释与可视化**

是什么：这一段是 value-aware probe 差分构造。它从最后一个 query 的 attention weights 和 V heads 重建每个 token 对最后 query 的 value contribution，然后按 10 个 probe token 行取子集，与最后一行 merged context 做差。这个差分在本 FX DAG 中没有外部输出和后续 user。

为什么需要：它把最后 query 视角下的 token contribution 限定到 10 个 probe 行，形成 `[10,4096]` 的 reference-minus-probe 区域。该层 return 仍然透传已有 `arg6_1`，因此这里的 probe 差分是被构造出来但未继续用于本层返回值的中间结果。

怎么做/计算：`select_int` 从 `view_default_12` 选择 sequence 维最后一行，得到最后 token 的 merged context reference；`select_int_1` 从 `clone_default` 选择 `Q=-1` 的 attention weights，得到每个 head 对全部 K token 的最后 query 权重；`unsqueeze_default_2` 给该权重补出 Dh 广播维；`mul_tensor_6` 将最后 query 的权重逐 head、逐 key token 乘到 `transpose_int_2` 的 V heads 上；`permute_default` 把 token 轴移到 head 轴之前；`clone_default_2` 使该布局连续；`view_default_13` 合并 head 和 Dh，得到 `[1,58,4096]` 的 token-major contribution；`_tensor_constant99` 是 shape `[10]` 的 int64 索引，在 `fx_graph.py` 中同一索引向量呈现为 `arange(35,45)`，即 token 35..44；`index_tensor_2` 从 token 轴取这 10 行；`unsqueeze_default_3` 把 reference row 扩成 `[1,1,4096]` 以便广播；`sub_tensor` 计算 reference row 减去 selected probe contribution rows。`sub_tensor` 在 Node Table 中没有 user，所以该计算结果不进入后续 output projection 或 return。

```text
Tensor: Last-query attention A_last [Heads=32, K=58]
Formula: A_last[a,k] = A[a,57,k]
Shown as K rows for one head band; K axis: 0 .. 57
          +----------------------------------------+
K=0       | A_LAST key 0                           |
          | A_LAST keys 1 .. 56                    |
K=57      | A_LAST key 57                          |
          +----------------------------------------+

Tensor: Last-query value contribution V_last [S=58, H=4096]
Formula: V_last[s,a*128+d] = A_last[a,s] * V_h[a,s,d]
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | OTHER_TOKEN_CONTRIB rows 0 .. 34       |
S=35      | SELECTED_PROBE_SOURCE rows 35 .. 44    |
          | OTHER_TOKEN_CONTRIB rows 45 .. 56      |
S=57      | LAST_TOKEN_CONTRIB row 57              |
          +----------------------------------------+
          examples: V_last[35,0], V_last[40,2048], V_last[44,4095]

Tensor: Probe contribution P_probe [P=10, H=4096]
Formula: P_probe[p,h] = V_last[35+p,h], so p=0 maps to S=35 and p=9 maps to S=44
P axis: 0 .. 9, H axis: 0 .. 4095 (same H width as V_last)
          H=0                                  H=4095
P=0       +----------------------------------------+
          | PROBE row 0 from S=35; h=0,2048,4095   |
          | PROBE rows 1 .. 8 from S=36 .. 43      |
P=9       | PROBE row 9 from S=44; h=0,2048,4095   |
          +----------------------------------------+

Tensor: Reference context row R_last [H=4096]
Formula: R_last[h] = C_merge[0,57,h]
H axis: 0 .. 4095 (same H width as P_probe)
          H=0                                  H=4095
          +----------------------------------------+
          | R_LAST examples h=0,2048,4095          |
          +----------------------------------------+

Tensor: Probe delta D_probe [P=10, H=4096]
Formula: D_probe[p,h] = R_last[h] - P_probe[p,h]
P axis: 0 .. 9, H axis: 0 .. 4095 (same H width as P_probe)
          H=0                                  H=4095
P=0       +----------------------------------------+
          | DELTA row 0 = R_LAST - PROBE[S=35]     |
          | DELTA rows 1 .. 8 = R_LAST - PROBE     |
P=9       | DELTA row 9 = R_LAST - PROBE[S=44]     |
          +----------------------------------------+
```

### Attention output projection and residual

```python
view_default_14 = aten.view.default(view_default_12, [58, 4096])
_tensor_constant100 = self._tensor_constant100
mm_default_3 = aten.mm.default(view_default_14, _tensor_constant100)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 58, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把 attention context 过输出投影，并与原始 hidden 做残差加法。

为什么需要：输出投影把多头 context 映射回 hidden space，残差连接保留层输入。

怎么做/计算：`view_default_14` 把 merged context 看成 `[58,4096]`；`_tensor_constant100` 是输出投影矩阵；`mm_default_3` 做矩阵乘得到 `[58,4096]`；`_unsafe_view_default_3` 恢复 `[1,58,4096]`；`add_tensor_4` 把投影结果与原始 `arg0_1` 在相同 `[B,S,H]` 坐标逐元素相加。

```text
Tensor: Merged context C_merge [S=58, H=4096]
Formula: C_merge[s,a*128+d] = C_h[a,s,d]
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | C_MERGE row 0; examples h=0,2048,4095  |
          | C_MERGE rows 1 .. 56                   |
S=57      | C_MERGE row 57; examples h=0,2048,4095 |
          +----------------------------------------+

Tensor: Projected attention O_attn [S=58, H=4096]
Formula: O_attn[s,h] = sum_j C_merge[s,j] * W_O[j,h]
S axis: 0 .. 57, H axis: 0 .. 4095 (same H width)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | O_ATTN row 0; examples h=0,2048,4095   |
          | O_ATTN rows 1 .. 56                    |
S=57      | O_ATTN row 57; examples h=0,2048,4095  |
          +----------------------------------------+

Tensor: Attention residual H_attn [B=1, S=58, H=4096]
Formula: H_attn[0,s,h] = H_in[0,s,h] + O_attn[s,h]
S axis: 0 .. 57, H axis: 0 .. 4095 (same H width)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_ATTN row 0; examples h=0,2048,4095   |
          | H_ATTN rows 1 .. 56                    |
S=57      | H_ATTN row 57; examples h=0,2048,4095  |
          +----------------------------------------+
```

### Post-attention RMSNorm

```python
_to_copy_default_3 = aten._to_copy.default(add_tensor_4, dtype=torch.float32)
pow_tensor_scalar_1 = aten.pow.Tensor_Scalar(_to_copy_default_3, 2)
mean_dim_1 = aten.mean.dim(pow_tensor_scalar_1, [-1], True)
add_tensor_5 = aten.add.Tensor(mean_dim_1, 1e-05)
rsqrt_default_1 = aten.rsqrt.default(add_tensor_5)
mul_tensor_7 = aten.mul.Tensor(_to_copy_default_3, rsqrt_default_1)
_to_copy_default_4 = aten._to_copy.default(mul_tensor_7, dtype=torch.float16)
_param_constant5 = self._param_constant5
mul_tensor_8 = aten.mul.Tensor(_param_constant5, _to_copy_default_4)
view_default_15 = aten.view.default(mul_tensor_8, [58, 4096])
```

**解释与可视化**

是什么：这一段对 post-attention residual 做 RMSNorm，并生成 MLP 输入。

为什么需要：MLP projection 前需要归一化 token rows。

怎么做/计算：`_to_copy_default_3` 把 attention residual 转 fp32；`pow_tensor_scalar_1` 对每个元素平方；`mean_dim_1` 沿 Hidden 求均值并保留 `[B,S,1]`；`add_tensor_5` 加 `1e-05`；`rsqrt_default_1` 得到每个 token 的 RMS 缩放；`mul_tensor_7` 应用缩放；`_to_copy_default_4` 转 fp16；`_param_constant5` 读取 post-attention RMSNorm 权重；`mul_tensor_8` 逐 Hidden 维乘权重；`view_default_15` 把结果看成 `[58,4096]` 供 MLP gate 分支使用，`mul_tensor_8` 也被下一段再次 view 给 up 分支。

```text
Tensor: Attention residual H_attn [S=58, H=4096]
Formula: H32_post[s,h] = fp32(H_attn[0,s,h])
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H32_POST row 0; examples h=0,2048,4095 |
          | H32_POST rows 1 .. 56                  |
S=57      | H32_POST row 57; examples h=0,2048,4095|
          +----------------------------------------+

Tensor: Post-attention RMS scale R_post [S=58, 1]
Formula: R_post[s,0] = rsqrt(mean_h(H32_post[s,h]^2) + 1e-05)
S axis: 0 .. 57
          +------+
S=0       | R0   |
          | ...  |
S=57      | R57  |
          +------+

Tensor: MLP input H_mlp [S=58, H=4096]
Formula: H_mlp[s,h] = Gamma_post[h] * fp16(H32_post[s,h] * R_post[s,0])
S axis: 0 .. 57, H axis: 0 .. 4095 (same H width)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_MLP row 0; examples h=0,2048,4095    |
          | H_MLP rows 1 .. 56                     |
S=57      | H_MLP row 57; examples h=0,2048,4095   |
          +----------------------------------------+
```

### MLP and final residual

```python
_tensor_constant101 = self._tensor_constant101
mm_default_4 = aten.mm.default(view_default_15, _tensor_constant101)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 58, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_16 = aten.view.default(mul_tensor_8, [58, 4096])
_tensor_constant102 = self._tensor_constant102
mm_default_5 = aten.mm.default(view_default_16, _tensor_constant102)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 58, 11008])
mul_tensor_9 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_17 = aten.view.default(mul_tensor_9, [58, 11008])
_tensor_constant103 = self._tensor_constant103
mm_default_6 = aten.mm.default(view_default_17, _tensor_constant103)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 58, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP，并把 down projection 结果加回 attention residual。

为什么需要：MLP 为每个 token 做非线性通道变换，residual add 保持主干 hidden 传递。

怎么做/计算：`_tensor_constant101` 是 gate projection 权重，`mm_default_4` 把 `view_default_15` 投影到 `[58,11008]`，`_unsafe_view_default_4` 恢复 batch 维，`silu_default` 做 SiLU 激活；`view_default_16` 再把同一个 `mul_tensor_8` 看成 `[58,4096]`，`mm_default_5` 与 `_tensor_constant102` 得到 up projection `[58,11008]`，`_unsafe_view_default_5` 恢复 batch 维；`mul_tensor_9` 将 SiLU gate 与 up projection 逐元素相乘；`view_default_17` 看成 `[58,11008]`；`mm_default_6` 与 `_tensor_constant103` 做 down projection 回 `[58,4096]`；`_unsafe_view_default_6` 恢复 `[1,58,4096]`；`add_tensor_6` 与 `add_tensor_4` 逐元素相加，得到本层 final hidden。

```text
Tensor: MLP input H_mlp [S=58, H=4096]
Formula: H_mlp[s,h] is the post-attention normalized hidden consumed by both gate and up branches
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_MLP row 0; examples h=0,2048,4095    |
          | H_MLP rows 1 .. 56                     |
S=57      | H_MLP row 57; examples h=0,2048,4095   |
          +----------------------------------------+

Tensor: Gated intermediate M_int [S=58, I=11008]
Formula: M_int[s,i] = silu(sum_h H_mlp[s,h] * W_gate[h,i]) * sum_h H_mlp[s,h] * W_up[h,i]
S axis: 0 .. 57, I axis: 0 .. 11007 (I width compressed)
          I=0                                  I=11007
S=0       +----------------------------------------+
          | M_INT row 0; examples i=0,5504,11007   |
          | M_INT rows 1 .. 56                     |
S=57      | M_INT row 57; examples i=0,5504,11007  |
          +----------------------------------------+

Tensor: Down-projected MLP output O_mlp [S=58, H=4096]
Formula: O_mlp[s,h] = sum_i M_int[s,i] * W_down[i,h]
S axis: 0 .. 57, H axis: 0 .. 4095 (same H width)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | O_MLP row 0; examples h=0,2048,4095    |
          | O_MLP rows 1 .. 56                     |
S=57      | O_MLP row 57; examples h=0,2048,4095   |
          +----------------------------------------+

Tensor: Final layer hidden H_out [B=1, S=58, H=4096]
Formula: H_out[0,s,h] = H_attn[0,s,h] + O_mlp[s,h]
S axis: 0 .. 57, H axis: 0 .. 4095 (same H width)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_OUT row 0; examples h=0,2048,4095    |
          | H_OUT rows 1 .. 56                     |
S=57      | H_OUT row 57; examples h=0,2048,4095   |
          +----------------------------------------+
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, arg6_1, 0)
```

**解释与可视化**

是什么：这一段是输出打包节点，返回 final hidden、当前层 K/V cache、透传的 Visual token/index 信息和控制标量 `0`。

为什么需要：final hidden 供下一层继续计算；K/V cache 供后续解码复用；Visual token/index 信息在本层没有更新，但需要继续传递；控制标量保持固定返回协议。

怎么做/计算：`return` 只把 `add_tensor_6` 放入 tuple 第 0 项；把 `add_tensor_2` 和 `transpose_int_2` 放入 `dynamic_cache_layer` 的 K/V；把 `arg6_1` 原样放入 tuple 第 2 项；最后返回常量 `0`。这一段没有新增矩阵乘、归一化、索引或比较计算。

```text
Tensor: Output hidden tuple item 0 H_out [B=1, S=58, H=4096]
Formula: Return[0] = H_out
S axis: 0 .. 57, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | H_OUT row 0; examples h=0,2048,4095    |
          | H_OUT rows 1 .. 56                     |
S=57      | H_OUT row 57; examples h=0,2048,4095   |
          +----------------------------------------+

Tensor: Cache K/V tuple item 1 K_cache and V_cache [Heads=32, S=58, Dh=128]
Formula: Return[1].K = K_rot, Return[1].V = V_h
Shown for one head a; S axis: 0 .. 57, Dh axis: 0 .. 127
          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | K_CACHE/V_CACHE row 0; d=0,64,127      |
          | K_CACHE/V_CACHE rows 1 .. 56           |
S=57      | K_CACHE/V_CACHE row 57; d=0,64,127     |
          +----------------------------------------+

Tensor: Passthrough visual ids tuple item 2 V_prev [P_prev]
Formula: Return[2] = V_prev, unchanged from runtime input
P_prev axis: 0 .. end
          +----------------------------------------+
          | V_PREV[p] examples p=0,mid,end         |
          +----------------------------------------+

Tensor: Control scalar tuple item 3 Z []
Formula: Return[3] = 0
          +---+
          | 0 |
          +---+
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
| 18 | `qkv_projection` | `_tensor_constant94` | `get_attr` | `_tensor_constant94` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant94` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant95` | `get_attr` | `_tensor_constant95` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant95` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant96` | `get_attr` | `_tensor_constant96` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant96` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `mul_tensor_6`, `output` |
| 35 | `rope` | `_tensor_constant97` | `get_attr` | `_tensor_constant97` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant97`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant98` | `get_attr` | `_tensor_constant98` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant98`, `arg2_1` | `unsqueeze_default_1` |
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
| 76 | `visual_process` | `select_int` | `call_function` | `aten.select.int` | `view_default_12` | `unsqueeze_default_3` |
| 77 | `visual_process` | `select_int_1` | `call_function` | `aten.select.int` | `clone_default` | `unsqueeze_default_2` |
| 78 | `visual_process` | `unsqueeze_default_2` | `call_function` | `aten.unsqueeze.default` | `select_int_1` | `mul_tensor_6` |
| 79 | `visual_process` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `unsqueeze_default_2`, `transpose_int_2` | `permute_default` |
| 80 | `visual_process` | `permute_default` | `call_function` | `aten.permute.default` | `mul_tensor_6` | `clone_default_2` |
| 81 | `visual_process` | `clone_default_2` | `call_function` | `aten.clone.default` | `permute_default` | `view_default_13` |
| 82 | `visual_process` | `view_default_13` | `call_function` | `aten.view.default` | `clone_default_2` | `index_tensor_2` |
| 83 | `visual_process` | `_tensor_constant99` | `get_attr` | `_tensor_constant99` | - | `index_tensor_2` |
| 84 | `visual_process` | `index_tensor_2` | `call_function` | `aten.index.Tensor` | `view_default_13`, `_tensor_constant99` | `sub_tensor` |
| 85 | `visual_process` | `unsqueeze_default_3` | `call_function` | `aten.unsqueeze.default` | `select_int` | `sub_tensor` |
| 86 | `visual_process` | `sub_tensor` | `call_function` | `aten.sub.Tensor` | `unsqueeze_default_3`, `index_tensor_2` | - |
| 87 | `output_projection` | `view_default_14` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 88 | `output_projection` | `_tensor_constant100` | `get_attr` | `_tensor_constant100` | - | `mm_default_3` |
| 89 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant100` | `_unsafe_view_default_3` |
| 90 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 91 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 92 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_7`, `pow_tensor_scalar_1` |
| 93 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 94 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 95 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 96 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_7` |
| 97 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 98 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_7` | `mul_tensor_8` |
| 99 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_8` |
| 100 | `post_attention_rmsnorm` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_15`, `view_default_16` |
| 101 | `post_attention_rmsnorm` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_4` |
| 102 | `mlp` | `_tensor_constant101` | `get_attr` | `_tensor_constant101` | - | `mm_default_4` |
| 103 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant101` | `_unsafe_view_default_4` |
| 104 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 105 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_9` |
| 106 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_5` |
| 107 | `mlp` | `_tensor_constant102` | `get_attr` | `_tensor_constant102` | - | `mm_default_5` |
| 108 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant102` | `_unsafe_view_default_5` |
| 109 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_9` |
| 110 | `mlp` | `mul_tensor_9` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_17` |
| 111 | `mlp` | `view_default_17` | `call_function` | `aten.view.default` | `mul_tensor_9` | `mm_default_6` |
| 112 | `mlp` | `_tensor_constant103` | `get_attr` | `_tensor_constant103` | - | `mm_default_6` |
| 113 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_17`, `_tensor_constant103` | `_unsafe_view_default_6` |
| 114 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 115 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 116 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2`, `arg6_1` | - |
