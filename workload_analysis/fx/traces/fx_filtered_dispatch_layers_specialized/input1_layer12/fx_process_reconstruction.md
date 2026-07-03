# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer12`
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
| Visual-related value-aware process | 76-85 | 10 | `clone_default`, `transpose_int_2`, `view_default_12` | - |
| Attention output projection and residual | 86-90 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 91-100 | 10 | `add_tensor_4` | `mul_tensor_8`, `view_default_15` |
| MLP and final residual | 101-114 | 14 | `add_tensor_4`, `mul_tensor_8`, `view_default_15` | `add_tensor_6` |
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

是什么：这一段是固定样本的 FX placeholder，后续实际使用 `arg0_1`、`arg1_1`、`arg2_1`。

为什么需要：`arg0_1` 提供 hidden states，`arg1_1` 提供 attention mask，`arg2_1` 提供 RoPE position ids。

怎么做/计算：placeholder 不做数值计算。`arg0_1` 进入输入 RMSNorm 和残差；`arg1_1` 在 QK score 后加到 logits；`arg2_1` 被两个 `index.Tensor` 用于 cos/sin 查表。

```text
Runtime inputs

arg0_1 hidden [B=1, S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | hidden row 0, H examples 0 and 4095    |
          | token rows 1 .. 622                    |
S=623     | hidden row 623, H examples 0 and 4095  |
          +----------------------------------------+

arg1_1 mask [Q=624, K=624], axes 0..623 on Q and K.
arg2_1 position ids, token axis 0..623.
examples: hidden[0,0,0], mask[623,623], pos[35].
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

怎么做/计算：`_to_copy_default` 转 fp32；`pow_tensor_scalar` 对 hidden 元素平方；`mean_dim` 沿 Hidden 维求平均；`add_tensor` 加 eps；`rsqrt_default` 得到缩放；`mul_tensor` 应用缩放；`_to_copy_default_1` 转 fp16；`_param_constant0` 留给下一段逐 Hidden 维相乘。

```text
Input RMSNorm [S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | hidden row 0 squared over H            | ----> | r0 |
          | token rows 1 .. 622                    |       | .. |
S=623     | hidden row 623 squared over H          | ----> | r623 |
          +----------------------------------------+       +----+

region contents: fp32 hidden, square, Hidden mean, eps+rsqrt, fp16 norm.
examples: norm[0,0], norm[35,2048], norm[623,4095].
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant31 = self._tensor_constant31
mm_default = aten.mm.default(view_default, _tensor_constant31)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant32 = self._tensor_constant32
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant32)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant33 = self._tensor_constant33
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant33)
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

怎么做/计算：`mul_tensor_1` 乘 input norm 权重；Q/K/V 三个分支分别 `view -> mm -> _unsafe_view`，使用 `_tensor_constant31/32/33`；再 `view` 成 `[1,624,32,128]`，`transpose` 到 `[1,32,624,128]`。

```text
Q/K/V projection

input [S=624, H=4096]                output [Heads=32, S=624, Dh=128]
S axis 0..623, H axis 0..4095        head axis 0..31, Dh axis 0..127

          H=0                                  H=4095
S=0       +----------------------------------------+
          | normalized hidden row 0                |
          | rows 1 .. 622                          | -- Wq/Wk/Wv -->
S=623     | normalized hidden row 623              |
          +----------------------------------------+

          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | head h row 0, dim examples             |
          | token rows 1 .. 622                    |
S=623     | head h row 623                         |
          +----------------------------------------+

examples: Q[0,0,0,0], K[0,31,611,127], V[0,12,623,64].
```

### RoPE position embedding

```python
_tensor_constant34 = self._tensor_constant34
index_tensor = aten.index.Tensor(_tensor_constant34, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant35 = self._tensor_constant35
index_tensor_1 = aten.index.Tensor(_tensor_constant35, [arg2_1])
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

怎么做/计算：`index_tensor/index_tensor_1` 用 `arg2_1` 从 `_tensor_constant34/35` 取 cos/sin；Q 分支计算 `transpose_int*cos`，再把 Dh 轴 `0..63` 和 `64..127` 切开，右半取负并拼回左半形成 rotate-half，乘 sin 后相加；K 分支同样生成 `add_tensor_2`。

```text
RoPE Dh regions
Dh axis: 0 .. 127, split at 64

          Dh=0              Dh=63 Dh=64             Dh=127
Q/K       +---------------------+-----------------------+
          | LEFT_HALF           | RIGHT_HALF            |
          +---------------------+-----------------------+
rot_half  +---------------------+-----------------------+
          | -RIGHT_HALF         | LEFT_HALF             |
          +---------------------+-----------------------+

out[h,s,d] = x[h,s,d] * cos[pos[s],d] + rot_half[h,s,d] * sin[pos[s],d]
examples: out[0,0,0], out[12,35,64], out[31,623,127].
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

怎么做/计算：`transpose_int_3` 把 K 转成 `[Dh,K]`；Q/K 经 `expand/view` 成 `[32,624,128]` 和 `[32,128,624]`；`bmm_default` 得到 `[32,624,624]`；`view_default_8` 恢复 batch/head；`div_tensor` 除以 `sqrt(128)`；`add_tensor_3` 加 mask；`_softmax_default` 沿 K 归一化；`_to_copy_default_2` 转 fp16；`clone_default` 复制权重。

```text
Plain attention weights [Heads=32, Q=624, K=624]
Q axis 0 .. 623, K axis 0 .. 623 (square compressed)

          K=0                                  K=623
Q=0       +----------------------------------------+
          | masked softmax row 0                   |
          | normal attention rows 1 .. 622         |
Q=623     | masked softmax row 623                 |
          +----------------------------------------+

region contents: scaled QK logits + mask, softmax over K.
examples: w[0,0,0], w[0,611,35], w[31,623,623].
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

怎么做/计算：`clone_default` 展成 `[32,624,624]`；V `transpose_int_2` 展成 `[32,624,128]`；`bmm_default_1` 计算 `[Q,K] x [K,Dh] -> [Q,Dh]`；`view_default_11` 和 `transpose_int_4` 恢复 `[B,S,Heads,Dh]`；`clone_default_1` 连续化；`view_default_12` 合并成 `[1,624,4096]`。

```text
Attention-weighted V

weights [Q=624, K=624]     V [K=624, Dh=128]       context [Q=624, Dh=128]
K axis 0..623              Dh axis 0..127          Dh axis 0..127
Q=0  +------------------+  +-------------------+   +-------------------+
     | weight row 0     |x | V rows 0..623     |-> | context row 0     |
     | rows 1 .. 622    |  | value region      |   | rows 1 .. 622     |
Q=623| weight row 623   |  | V row 623         |   | context row 623   |
     +------------------+  +-------------------+   +-------------------+

merged hidden [S=624, H=4096], H axis 0 .. 4095.
examples: ctx[0,0,0], ctx[31,623,127], merged[623,4095].
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
```

**解释与可视化**

是什么：这一段构造 Visual-span 的 delta 张量，但没有继续做阈值、`nonzero` 或 index 输出；因此它只在该 FX DAG 中形成 Visual value-aware 差分中间量。

为什么需要：它把最后一个 query 的 attention-to-V 贡献重建成 token rows，并与最后输出 row 作 subtraction，用于表示 Visual token contribution 与最终 last-token context 的差异。

怎么做/计算：`select_int` 从 `view_default_12` 选最后一个 sequence row，得到 last output row `[B=1,H=4096]`；`select_int_1` 从 `clone_default` 选 `Q=-1` 的 attention 权重；`unsqueeze_default_2` 扩展权重；`mul_tensor_6` 做 last-query weight 乘 V；`permute_default` 调整到 token-major；`clone_default_2` 连续化；`view_default_13` 合并 head/Dh 成 `[1,624,4096]`；`slice_tensor_4` 取 Visual token span `S=35..610`；`unsqueeze_default_3` 扩展 last output row；`sub_tensor` 计算 reference minus Visual contribution rows。

```text
Visual delta construction

Visual token axis V=576 maps to sequence S=35 .. 610.
Hidden axis H=0 .. 4095 (H width compressed)

last output reference [1, H]
          H=0                                  H=4095
ref       +----------------------------------------+
          | select_int last row, examples H0/H4095 |
          +----------------------------------------+

Visual contribution rows [V=576, H=4096]
          H=0                                  H=4095
V=0/S=35  +----------------------------------------+
          | contribution from attn[Q=623,K=35]     |
          | Visual rows S=36 .. 609                |
V=575     | contribution from attn[Q=623,K=610]    |
          +----------------------------------------+

delta rows = reference - contribution
          H=0                                  H=4095
V=0       +----------------------------------------+
          | sub_tensor[0,0,H] examples H0/H4095    |
          | Visual delta rows 1 .. 574             |
V=575     | sub_tensor[0,575,H] examples H0/H4095  |
          +----------------------------------------+
```

### Attention output projection and residual

```python
view_default_14 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant36 = self._tensor_constant36
mm_default_3 = aten.mm.default(view_default_14, _tensor_constant36)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把 attention context 过输出投影，并与原始 hidden 做残差加法。

为什么需要：输出投影把多头 context 映射回 hidden space，残差连接保留层输入。

怎么做/计算：`view_default_14` 展平 context；`mm_default_3` 乘 `_tensor_constant36`；`_unsafe_view_default_3` 恢复 `[1,624,4096]`；`add_tensor_4` 与 `arg0_1` 在相同 S/H 坐标逐元素相加。

```text
Output projection + residual [S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095

          H=0                                  H=4095
proj      +----------------------------------------+
          | projected context rows 0 .. 623        |
input     +----------------------------------------+
          | arg0_1 rows 0 .. 623                   |
output    +----------------------------------------+
          | add_tensor_4 rows 0 .. 623             |
          +----------------------------------------+

examples: add_tensor_4[0,0], add_tensor_4[611,2048], add_tensor_4[623,4095].
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
view_default_15 = aten.view.default(mul_tensor_8, [624, 4096])
```

**解释与可视化**

是什么：这一段对 post-attention residual 做 RMSNorm，并生成 MLP 输入。

为什么需要：MLP projection 前需要归一化 token rows。

怎么做/计算：`_to_copy_default_3` 转 fp32；`pow_tensor_scalar_1` 平方；`mean_dim_1` 沿 Hidden 求均值；`add_tensor_5` 加 eps；`rsqrt_default_1` 取倒数平方根；`mul_tensor_7` 缩放；`_to_copy_default_4` 转 fp16；`mul_tensor_8` 应用 `_param_constant5`；`view_default_15` 得到 `[624,4096]`。

```text
Post-attention RMSNorm [S=624, H=4096]

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | add_tensor_4 row 0 squared             | ----> | r0 |
          | rows 1 .. 622                          |       | .. |
S=623     | add_tensor_4 row 623 squared           | ----> | r623 |
          +----------------------------------------+       +----+

region contents: residual hidden, Hidden-axis square/mean, eps+rsqrt, norm weight.
examples: mlp_in[0,0], mlp_in[35,1024], mlp_in[623,4095].
```

### MLP and final residual

```python
_tensor_constant37 = self._tensor_constant37
mm_default_4 = aten.mm.default(view_default_15, _tensor_constant37)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_16 = aten.view.default(mul_tensor_8, [624, 4096])
_tensor_constant38 = self._tensor_constant38
mm_default_5 = aten.mm.default(view_default_16, _tensor_constant38)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_9 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_17 = aten.view.default(mul_tensor_9, [624, 11008])
_tensor_constant39 = self._tensor_constant39
mm_default_6 = aten.mm.default(view_default_17, _tensor_constant39)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP，并把 down projection 结果加回 attention residual。

为什么需要：MLP 为每个 token 做非线性通道变换，residual add 保持主干 hidden 传递。

怎么做/计算：`mm_default_4` 产生 gate `[624,11008]`，`silu_default` 激活；`mm_default_5` 产生 up `[624,11008]`；`mul_tensor_9` gate/up 逐元素乘；`mm_default_6` down project 回 `[624,4096]`；`add_tensor_6` 加回 `add_tensor_4`。

```text
MLP [S=624, H=4096] -> [S=624, I=11008] -> [S=624, H=4096]

H axis 0 .. 4095                         I axis 0 .. 11007
S=0  +-------------------------------+   +-------------------------------+
     | mlp input row 0               |-> | silu(gate) * up row 0         |
     | rows 1 .. 622                 |   | rows 1 .. 622                 |
S=623| mlp input row 623             |   | gated row 623                 |
     +-------------------------------+   +-------------------------------+
                         down projection
S=0  +-------------------------------+
     | final residual row 0          |
     | rows 1 .. 622                 |
S=623| final residual row 623        |
     +-------------------------------+

examples: gate[0,0], up[623,11007], final[623,4095].
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)
```

**解释与可视化**

是什么：这一段返回 final hidden、当前层 K/V cache、Visual 输出占位 `None` 和控制标量 `0`。

为什么需要：`add_tensor_6` 供下一层继续计算；K/V cache 供后续解码复用。该层 Visual process 只构造了未返回的 `sub_tensor` 中间量，所以返回 tuple 的第三项仍是 `None`。

怎么做/计算：`return` 只打包 `add_tensor_6`、`add_tensor_2` 和 `transpose_int_2`；没有新张量计算。

```text
Layer output

hidden add_tensor_6 [B=1, S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095
          H=0                                  H=4095
S=0       +----------------------------------------+
          | final hidden row 0                     |
          | rows 1 .. 622                          |
S=623     | final hidden row 623                   |
          +----------------------------------------+

dynamic_cache_layer:
K add_tensor_2    [B=1, Heads=32, S=624, Dh=128]
V transpose_int_2 [B=1, Heads=32, S=624, Dh=128]
examples: K[0,0,0], K[31,623,127], V[12,611,64].
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
| 18 | `qkv_projection` | `_tensor_constant31` | `get_attr` | `_tensor_constant31` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant31` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant32` | `get_attr` | `_tensor_constant32` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant32` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant33` | `get_attr` | `_tensor_constant33` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant33` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `mul_tensor_6`, `output` |
| 35 | `rope` | `_tensor_constant34` | `get_attr` | `_tensor_constant34` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant34`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant35` | `get_attr` | `_tensor_constant35` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant35`, `arg2_1` | `unsqueeze_default_1` |
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
| 82 | `visual_process` | `view_default_13` | `call_function` | `aten.view.default` | `clone_default_2` | `slice_tensor_4` |
| 83 | `visual_process` | `slice_tensor_4` | `call_function` | `aten.slice.Tensor` | `view_default_13` | `sub_tensor` |
| 84 | `visual_process` | `unsqueeze_default_3` | `call_function` | `aten.unsqueeze.default` | `select_int` | `sub_tensor` |
| 85 | `visual_process` | `sub_tensor` | `call_function` | `aten.sub.Tensor` | `unsqueeze_default_3`, `slice_tensor_4` | - |
| 86 | `output_projection` | `view_default_14` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 87 | `output_projection` | `_tensor_constant36` | `get_attr` | `_tensor_constant36` | - | `mm_default_3` |
| 88 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant36` | `_unsafe_view_default_3` |
| 89 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 90 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 91 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_7`, `pow_tensor_scalar_1` |
| 92 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 93 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 94 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 95 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_7` |
| 96 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 97 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_7` | `mul_tensor_8` |
| 98 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_8` |
| 99 | `post_attention_rmsnorm` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_15`, `view_default_16` |
| 100 | `post_attention_rmsnorm` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_4` |
| 101 | `mlp` | `_tensor_constant37` | `get_attr` | `_tensor_constant37` | - | `mm_default_4` |
| 102 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant37` | `_unsafe_view_default_4` |
| 103 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 104 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_9` |
| 105 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_5` |
| 106 | `mlp` | `_tensor_constant38` | `get_attr` | `_tensor_constant38` | - | `mm_default_5` |
| 107 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant38` | `_unsafe_view_default_5` |
| 108 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_9` |
| 109 | `mlp` | `mul_tensor_9` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_17` |
| 110 | `mlp` | `view_default_17` | `call_function` | `aten.view.default` | `mul_tensor_9` | `mm_default_6` |
| 111 | `mlp` | `_tensor_constant39` | `get_attr` | `_tensor_constant39` | - | `mm_default_6` |
| 112 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_17`, `_tensor_constant39` | `_unsafe_view_default_6` |
| 113 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 114 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 115 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2` | - |
