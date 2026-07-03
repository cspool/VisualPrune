# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer0`
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
| QK scores, mask, softmax | 55-76 | 22 | `add_tensor_1`, `add_tensor_2`, `arg1_1` | `clone_default` |
| Attention-weighted V and hidden reshape | 77-85 | 9 | `clone_default`, `transpose_int_2` | `view_default_12` |
| Attention output projection and residual | 86-90 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 91-100 | 10 | `add_tensor_4` | `mul_tensor_7`, `view_default_14` |
| MLP and final residual | 101-114 | 14 | `add_tensor_4`, `mul_tensor_7`, `view_default_14` | `add_tensor_6` |
| Layer output | 115-115 | 1 | `add_tensor_2`, `add_tensor_6`, `transpose_int_2` | - |

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

是什么：这一段是 FX 固定输入入口，`arg0_1`、`arg1_1`、`arg2_1` 被后续计算实际使用，其他 placeholder 在该固定样本的 DAG 中没有用户。

为什么需要：后续 attention、RoPE、RMSNorm 和残差计算都要从这些运行时输入取得 hidden states、attention mask 和 position ids。

怎么做/计算：placeholder 本身不做数值计算。`arg0_1` 作为 `[B=1, S=624, H=4096]` hidden states 输入给 `_to_copy_default` 和残差 `add_tensor_4`；`arg1_1` 作为 attention mask 输入给 `add_tensor_3`；`arg2_1` 作为 token position ids 输入给 RoPE 的 `index.Tensor`。

```text
Runtime inputs, displayed as tensor axes

arg0_1 hidden states [B=1, S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                H=4095
S=0       +--------------------------------------+
          | hidden[0,0,0], hidden[0,0,1], ...    |
          | token rows 1 .. 622                  |
S=623     | hidden row 623, H examples 0 and 4095 |
          +--------------------------------------+

arg1_1 attention mask [B=1, Heads/1, Q=624, K=624]
Q axis: 0 .. 623, K axis: 0 .. 623

          K=0                                K=623
Q=0       +--------------------------------------+
          | mask[0,0,0,0], mask[0,0,0,1], ...    |
          | causal / allowed-position region     |
Q=623     | mask row 623, K examples 0 and 623    |
          +--------------------------------------+

arg2_1 position ids
token axis: 0 .. 623; examples: pos[0], pos[35], pos[611], pos[623]
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

是什么：这一段对输入 hidden states 做 RMSNorm 前半段和权重准备，输出 `_to_copy_default_1` 给 Q/K/V 投影使用，并读取 RMSNorm 权重 `_param_constant0`。

为什么需要：投影前要把每个 token 的 Hidden 维归一化，避免不同 token 行的幅值差异直接进入 Q/K/V 线性层。

怎么做/计算：`_to_copy_default` 把 `arg0_1` 转成 fp32；`pow_tensor_scalar` 对每个 hidden 元素平方；`mean_dim` 沿最后一维 Hidden 求均值并保留维度，得到每个 token 一个 RMS 标量；`add_tensor` 加 `1e-05`；`rsqrt_default` 取倒数平方根；`mul_tensor` 把原 fp32 hidden 逐元素乘以该缩放；`_to_copy_default_1` 转回 fp16；`_param_constant0` 提供 4096 维缩放权重，后续 `mul_tensor_1` 会把它乘到归一化结果上。

```text
Input RMSNorm over [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                      H=4095       RMS col
S=0       +--------------------------------------------+        +------+
          | x[0,0]^2, x[0,1]^2, ..., x[0,4095]^2       |  mean  | r[0] |
          | rows 1 .. 622                              | -----> | ...  |
S=623     | x[623,0]^2, ..., x[623,4095]^2             |        | r[623] |
          +--------------------------------------------+        +------+

region contents:
- hidden region: fp32 values from `_to_copy_default`
- RMS col: mean over H, plus eps, then rsqrt
- output examples: norm[0,0] = x[0,0] * rsqrt(mean(x[0,:]^2)+eps)
                   norm[623,4095] uses the same row-623 RMS scale
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant3 = self._tensor_constant3
mm_default = aten.mm.default(view_default, _tensor_constant3)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant4 = self._tensor_constant4
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant4)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant5 = self._tensor_constant5
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant5)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 624, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 624, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 624, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 624, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段把归一化 hidden states 分别投影成 Q、K、V，并把每个 `[S, H]` 结果 reshape 成多头布局 `[B=1, Heads=32, S=624, Dh=128]`。

为什么需要：attention 需要用 Q 和 K 计算 token-token 分数，用 V 提供被 attention 加权汇聚的内容；head reshape 把 4096 维 hidden 拆成 32 个 128 维 head。

怎么做/计算：`mul_tensor_1` 先把 RMSNorm 权重 `_param_constant0` 乘到 `_to_copy_default_1`；三条分支各自 `view` 成 `[624,4096]`，再与对应权重矩阵 `_tensor_constant3/4/5` 做 `mm`；`_unsafe_view_default*` 把结果恢复成 `[1,624,4096]`；`view_default_3/4/5` 把 Hidden 轴拆成 `[32,128]`；`transpose_int/1/2` 交换 sequence 和 head 维，得到后续 attention 使用的 Q、K、V。

```text
Projection input [S=624, H=4096] -> Q/K/V heads [Heads=32, S=624, Dh=128]
S axis: 0 .. 623, H axis: 0 .. 4095; Dh axis: 0 .. 127

input normalized hidden
          H=0                                  H=4095
S=0       +----------------------------------------+
          | norm[0,0] ... norm[0,4095]             |
          | rows 1 .. 622                          |
S=623     | norm[623,0] ... norm[623,4095]         |
          +----------------------------------------+
                    |
                    | three mm branches with Wq/Wk/Wv
                    v
Q/K/V output per head, displayed for head axis 0 .. 31

head=0    Dh=0                              Dh=127
S=0       +--------------------------------------+
          | q/k/v[h0, token0, dim0..127]         |
          | token rows 1 .. 622                  |
S=623     | q/k/v[h0, token623, dim0..127]       |
          +--------------------------------------+

head=31 has the same S=0..623 and Dh=0..127 axes.
Example elements: Q[0,0,0,0], K[0,31,611,127], V[0,15,623,64].
```

### RoPE position embedding

```python
_tensor_constant6 = self._tensor_constant6
index_tensor = aten.index.Tensor(_tensor_constant6, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant7 = self._tensor_constant7
index_tensor_1 = aten.index.Tensor(_tensor_constant7, [arg2_1])
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

是什么：这一段按 `arg2_1` 的 position ids 取 cos/sin 表，对 Q 和 K 的每个 head 向量做 RoPE 旋转，输出旋转后的 `add_tensor_1` 和 `add_tensor_2`。

为什么需要：QK 点积需要携带 token 位置信息；RoPE 把位置编码注入每个 head 的 128 维向量里。

怎么做/计算：`index_tensor/index_tensor_1` 从 cos/sin 常量表按 position ids 取 `[624,128]` 行，再 `unsqueeze` 到可广播形状；Q 分支先做 `transpose_int * cos`，再把 Dh 轴切成 `0..63` 与 `64..127`，对右半取负并与左半拼接得到 rotate-half，然后乘 `sin`，最后两项相加得 `add_tensor_1`；K 分支对 `transpose_int_1` 做同样操作，得到 `add_tensor_2`。

```text
RoPE on Q and K [Heads=32, S=624, Dh=128]
Dh axis split: 0 .. 63 | 64 .. 127

for each head h=0..31 and token s=0..623

          Dh=0             Dh=63 Dh=64            Dh=127
Q/K row   +--------------------+----------------------+
          | left half x[0:64]  | right half x[64:128] |
          +--------------------+----------------------+
                    | rotate-half region
                    v
rot-half  +--------------------+----------------------+
          | -right half        | left half            |
          +--------------------+----------------------+

output = x * cos[pos[s], d] + rot_half(x) * sin[pos[s], d]
examples: out[h=0,s=0,d=0], out[h=0,s=0,d=64],
          out[h=31,s=623,d=127]
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
slice_tensor_4 = aten.slice.Tensor(_to_copy_default_2, 2, 611, 9223372036854775807)
slice_tensor_5 = aten.slice.Tensor(slice_tensor_4, 3, 35, 611)
sum_dim_int_list = aten.sum.dim_IntList(slice_tensor_5, [-1])
slice_tensor_6 = aten.slice.Tensor(_to_copy_default_2, 2, 35, 9223372036854775807)
slice_tensor_7 = aten.slice.Tensor(slice_tensor_6, 3, 35, 611)
_tensor_constant8 = self._tensor_constant8
fill__tensor = aten.fill_.Tensor(slice_tensor_7, _tensor_constant8)
slice_tensor_8 = aten.slice.Tensor(_to_copy_default_2, 2, 611, 9223372036854775807)
select_int = aten.select.int(slice_tensor_8, 3, 35)
copy__default = aten.copy_.default(select_int, sum_dim_int_list)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段先计算标准 attention 权重，再对 Visual-key 区域做 fold/clear：把尾部 query 对 Visual token keys 的概率质量汇总到 key=35，并清空较大范围内的 Visual-key block。

为什么需要：attention 权重是后续加权 V 的系数；这里的 `slice/sum/fill_/copy_` 体现了该固定输入下的 Visual token attention 调整。

怎么做/计算：`transpose_int_3` 把 K 的 Dh 和 K_seq 维转置；Q/K 分别 `expand -> view` 成 `[32,624,128]` 与 `[32,128,624]`；`bmm_default` 得到每个 head 的 `[Q=624,K=624]` 分数；`div_tensor` 按 `sqrt(128)` 缩放；`add_tensor_3` 加 mask；`_softmax_default` 沿 K 轴归一化并转 fp16。随后 `slice_tensor_4/5` 取 `Q=611..623, K=35..610`，`sum_dim_int_list` 对 Visual-key 区求和；`slice_tensor_6/7` 取 `Q=35..623, K=35..610`，`fill__tensor` 用常量清空/覆盖该块；`select_int` 取 `Q=611..623, K=35` 列，`copy__default` 把 fold 后的质量写入这个代表 key 列；最后 `clone_default` 复制调整后的权重供下一段使用。

```text
Attention weights [Heads=32, Q=624, K=624]
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)

K=0        K=35                         K=610 K=611      K=623
+----------+--------------------------------+-------------+
| KEEP     | KEEP                           | KEEP        | Q=0
|          |                                |             |
+----------+--------------------------------+-------------+ Q=35
| KEEP     | CLEAR_BY_FILL                  | KEEP        |
|          | examples: w[h,35,35],          |             |
|          | w[h,100,200], w[h,610,610]     |             |
+----------+--------------------------------+-------------+ Q=611
| KEEP     | FOLD_SOURCE then CLEAR_BY_FILL | KEEP        |
|          | sum K=35..610 -> K=35          |             |
+----------+--------------------------------+-------------+ Q=623

Fold target column:
for Q=611..623, K=35 receives sum of original K=35..610.
Example elements: source w[h,611,35], w[h,611,610];
target w[h,611,35]; cleared w[h,100,100].
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

是什么：这一段用调整后的 attention 权重乘 V，得到每个 query token 的上下文向量，并把多头结果合并回 `[B=1,S=624,H=4096]`。

为什么需要：QK 权重只给出 token 间的分配比例，真正的 attention 输出要把这些权重作用到 V 的内容向量上。

怎么做/计算：`clone_default` 被 `expand/view` 成 `[32,624,624]`；`transpose_int_2` 的 V 被 `expand/view` 成 `[32,624,128]`；`bmm_default_1` 执行 `[Q,K] x [K,Dh] -> [Q,Dh]`；`view_default_11` 恢复 batch/head 维；`transpose_int_4` 把布局变回 `[B,S,Heads,Dh]`；`clone_default_1` 保证 contiguous；`view_default_12` 把 `32*128` 合并为 4096 Hidden。

```text
Attention contraction per head

weights [Q=624, K=624]          V [K=624, Dh=128]        context [Q=624, Dh=128]
K=0                    K=623    Dh=0          Dh=127     Dh=0          Dh=127
Q=0 +----------------------+     +------------------+     +------------------+
    | w[0,0], ..., w[0,623]| x   | v[0,0..127]      | --> | ctx[0,0..127]    |
    | rows 1 .. 622        |     | rows 1 .. 622    |     | rows 1 .. 622    |
Q=623| w[623,0..623]       |     | v[623,0..127]    |     | ctx[623,0..127]  |
    +----------------------+     +------------------+     +------------------+

merged hidden [S=624, H=4096]
H axis 0 .. 4095 = head0 Dh0..127, ..., head31 Dh0..127.
Example elements: ctx[h=0,q=0,d=0], merged[q=623,hid=4095].
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant9 = self._tensor_constant9
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant9)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把 attention context 通过输出投影矩阵映射回 hidden space，并与原始 `arg0_1` 做残差相加。

为什么需要：多头 attention 合并后的向量还需要经过输出线性层，再通过 residual path 保留层输入信息。

怎么做/计算：`view_default_13` 把 `[1,624,4096]` 展平成 `[624,4096]`；`mm_default_3` 乘输出投影 `_tensor_constant9`；`_unsafe_view_default_3` 恢复 `[1,624,4096]`；`add_tensor_4` 与 `arg0_1` 逐元素相加，得到 post-attention hidden。

```text
Output projection and residual [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                      H=4095
context   +--------------------------------------------+
rows      | proj(ctx)[s,h] examples: [0,0], [623,4095] |
          +--------------------------------------------+
                         +
input     +--------------------------------------------+
rows      | arg0_1[s,h] same S/H coordinates            |
          +--------------------------------------------+
                         =
output    +--------------------------------------------+
rows      | add_tensor_4[s,h] post-attention residual   |
          +--------------------------------------------+
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
view_default_14 = aten.view.default(mul_tensor_7, [624, 4096])
```

**解释与可视化**

是什么：这一段对 attention residual 后的 hidden 做第二个 RMSNorm，并准备 MLP 输入。

为什么需要：MLP 前通常需要重新归一化，使 gate/up projection 接收到稳定尺度的 token 行。

怎么做/计算：`_to_copy_default_3` 把 `add_tensor_4` 转 fp32；`pow_tensor_scalar_1` 平方；`mean_dim_1` 沿 Hidden 求均值；`add_tensor_5` 加 epsilon；`rsqrt_default_1` 得到缩放；`mul_tensor_6` 应用缩放；`_to_copy_default_4` 转 fp16；`_param_constant5` 是 post-attention norm 权重；`mul_tensor_7` 应用权重；`view_default_14` 展平成 `[624,4096]` 供 MLP 的 `mm` 使用。

```text
Post-attention RMSNorm [S=624, H=4096]

          H=0                                      H=4095       RMS col
S=0       +--------------------------------------------+        +------+
          | add_tensor_4[0,0..4095]^2                  | mean   | r[0] |
          | rows 1 .. 622                              | -----> | ...  |
S=623     | add_tensor_4[623,0..4095]^2                |        | r[623] |
          +--------------------------------------------+        +------+

region contents:
- input region: post-attention residual hidden
- RMS col: row-wise Hidden reduction, eps add, rsqrt
- output examples: mlp_in[0,0], mlp_in[611,2048], mlp_in[623,4095]
```

### MLP and final residual

```python
_tensor_constant10 = self._tensor_constant10
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant10)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [624, 4096])
_tensor_constant11 = self._tensor_constant11
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant11)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [624, 11008])
_tensor_constant12 = self._tensor_constant12
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant12)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 Llama-style gated MLP：gate projection、SiLU、up projection、逐元素门控、down projection，然后与 attention residual 再相加。

为什么需要：attention 负责 token 间信息汇聚，MLP 负责对每个 token 行做非线性通道变换，并通过 residual 叠回主干 hidden。

怎么做/计算：`mm_default_4` 把 `[624,4096]` 投影到 gate `[624,11008]`，`silu_default` 做激活；`mm_default_5` 产生 up `[624,11008]`；`mul_tensor_8` 把 gate 激活和 up 逐元素相乘；`mm_default_6` down project 回 `[624,4096]`；`add_tensor_6` 把 down 结果加回 `add_tensor_4`。

```text
MLP per token, axes keep S rows aligned

input / residual [S=624, H=4096]       intermediate [S=624, I=11008]
H=0                         H=4095    I=0                         I=11007
S=0 +-----------------------------+    +----------------------------------+
    | mlp_in[0,0..4095]           | -> | gate=silu(mm), up=mm, gate*up    |
    | rows 1 .. 622               |    | examples: inter[0,0], inter[0,42]|
S=623| mlp_in[623,0..4095]        |    | inter[623,11007]                 |
    +-----------------------------+    +----------------------------------+
                                                |
                                                v down projection
output [S=624, H=4096]
H=0                         H=4095
S=0 +-----------------------------+
    | final = residual + down_out |
    | rows 1 .. 622               |
S=623| final hidden row 623        |
    +-----------------------------+
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)
```

**解释与可视化**

是什么：这一段把本层已经算好的结果打包返回：最终 hidden、当前层 dynamic cache 中的 K/V、Visual token 输出占位 `None`，以及控制标量 `0`。

为什么需要：上层调用者需要 final hidden 继续进入下一层；KV cache 需要保存 RoPE 后 K `add_tensor_2` 和 V `transpose_int_2`，供后续解码步复用。

怎么做/计算：`return` 不再做新的张量数值计算，只把 `add_tensor_6`、`add_tensor_2`、`transpose_int_2` 放入返回 tuple/dict。`add_tensor_2` 来自 RoPE K，`transpose_int_2` 来自 V head reshape。

```text
Layer output package

hidden output [B=1, S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095

          H=0                                      H=4095
S=0       +--------------------------------------------+
          | final hidden row 0, H examples 0 and 4095   |
          | rows 1 .. 622                              |
S=623     | final hidden row 623, H examples 0 and 4095 |
          +--------------------------------------------+

dynamic_cache_layer:
K cache add_tensor_2     [B=1, Heads=32, S=624, Dh=128]
V cache transpose_int_2  [B=1, Heads=32, S=624, Dh=128]
Dh axis examples: d=0, d=64, d=127; token examples: s=0, s=35, s=623.
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
| 18 | `qkv_projection` | `_tensor_constant3` | `get_attr` | `_tensor_constant3` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant3` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant4` | `get_attr` | `_tensor_constant4` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant4` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant5` | `get_attr` | `_tensor_constant5` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant5` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `output` |
| 35 | `rope` | `_tensor_constant6` | `get_attr` | `_tensor_constant6` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant6`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant7` | `get_attr` | `_tensor_constant7` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant7`, `arg2_1` | `unsqueeze_default_1` |
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
| 65 | `attention_scores` | `_to_copy_default_2` | `call_function` | `aten._to_copy.default` | `_softmax_default` | `clone_default`, `slice_tensor_4`, `slice_tensor_6`, `slice_tensor_8` |
| 66 | `attention_scores` | `slice_tensor_4` | `call_function` | `aten.slice.Tensor` | `_to_copy_default_2` | `slice_tensor_5` |
| 67 | `attention_scores` | `slice_tensor_5` | `call_function` | `aten.slice.Tensor` | `slice_tensor_4` | `sum_dim_int_list` |
| 68 | `attention_scores` | `sum_dim_int_list` | `call_function` | `aten.sum.dim_IntList` | `slice_tensor_5` | `copy__default` |
| 69 | `attention_scores` | `slice_tensor_6` | `call_function` | `aten.slice.Tensor` | `_to_copy_default_2` | `slice_tensor_7` |
| 70 | `attention_scores` | `slice_tensor_7` | `call_function` | `aten.slice.Tensor` | `slice_tensor_6` | `fill__tensor` |
| 71 | `attention_scores` | `_tensor_constant8` | `get_attr` | `_tensor_constant8` | - | `fill__tensor` |
| 72 | `attention_scores` | `fill__tensor` | `call_function` | `aten.fill_.Tensor` | `slice_tensor_7`, `_tensor_constant8` | - |
| 73 | `attention_scores` | `slice_tensor_8` | `call_function` | `aten.slice.Tensor` | `_to_copy_default_2` | `select_int` |
| 74 | `attention_scores` | `select_int` | `call_function` | `aten.select.int` | `slice_tensor_8` | `copy__default` |
| 75 | `attention_scores` | `copy__default` | `call_function` | `aten.copy_.default` | `select_int`, `sum_dim_int_list` | - |
| 76 | `attention_scores` | `clone_default` | `call_function` | `aten.clone.default` | `_to_copy_default_2` | `expand_default_2` |
| 77 | `attention_output` | `expand_default_2` | `call_function` | `aten.expand.default` | `clone_default` | `view_default_9` |
| 78 | `attention_output` | `view_default_9` | `call_function` | `aten.view.default` | `expand_default_2` | `bmm_default_1` |
| 79 | `attention_output` | `expand_default_3` | `call_function` | `aten.expand.default` | `transpose_int_2` | `view_default_10` |
| 80 | `attention_output` | `view_default_10` | `call_function` | `aten.view.default` | `expand_default_3` | `bmm_default_1` |
| 81 | `attention_output` | `bmm_default_1` | `call_function` | `aten.bmm.default` | `view_default_9`, `view_default_10` | `view_default_11` |
| 82 | `attention_output` | `view_default_11` | `call_function` | `aten.view.default` | `bmm_default_1` | `transpose_int_4` |
| 83 | `attention_output` | `transpose_int_4` | `call_function` | `aten.transpose.int` | `view_default_11` | `clone_default_1` |
| 84 | `attention_output` | `clone_default_1` | `call_function` | `aten.clone.default` | `transpose_int_4` | `view_default_12` |
| 85 | `attention_output` | `view_default_12` | `call_function` | `aten.view.default` | `clone_default_1` | `view_default_13` |
| 86 | `output_projection` | `view_default_13` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 87 | `output_projection` | `_tensor_constant9` | `get_attr` | `_tensor_constant9` | - | `mm_default_3` |
| 88 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant9` | `_unsafe_view_default_3` |
| 89 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 90 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 91 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_6`, `pow_tensor_scalar_1` |
| 92 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 93 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 94 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 95 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_6` |
| 96 | `post_attention_rmsnorm` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 97 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_6` | `mul_tensor_7` |
| 98 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_7` |
| 99 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_14`, `view_default_15` |
| 100 | `post_attention_rmsnorm` | `view_default_14` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_4` |
| 101 | `mlp` | `_tensor_constant10` | `get_attr` | `_tensor_constant10` | - | `mm_default_4` |
| 102 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant10` | `_unsafe_view_default_4` |
| 103 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 104 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 105 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 106 | `mlp` | `_tensor_constant11` | `get_attr` | `_tensor_constant11` | - | `mm_default_5` |
| 107 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant11` | `_unsafe_view_default_5` |
| 108 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 109 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 110 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 111 | `mlp` | `_tensor_constant12` | `get_attr` | `_tensor_constant12` | - | `mm_default_6` |
| 112 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant12` | `_unsafe_view_default_6` |
| 113 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 114 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 115 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2` | - |
