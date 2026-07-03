# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer6`
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
| Attention output projection and residual | 76-80 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 81-90 | 10 | `add_tensor_4` | `mul_tensor_7`, `view_default_14` |
| MLP and final residual | 91-104 | 14 | `add_tensor_4`, `mul_tensor_7`, `view_default_14` | `add_tensor_6` |
| Layer output | 105-105 | 1 | `add_tensor_2`, `add_tensor_6`, `transpose_int_2` | - |

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

是什么：这一段是固定 FX DAG 的运行时输入入口，实际被后续节点使用的是 hidden states `arg0_1`、attention mask `arg1_1` 和 position ids `arg2_1`。

为什么需要：layer 的归一化、attention mask、RoPE 位置查表都依赖这些输入。

怎么做/计算：placeholder 节点不做数值计算。`arg0_1` 供 `_to_copy_default` 和残差 `add_tensor_4` 使用；`arg1_1` 在 `add_tensor_3` 中加到 QK 分数；`arg2_1` 传给两个 `index.Tensor` 节点取 cos/sin。

```text
Runtime input regions

arg0_1 hidden [B=1, S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | hidden row 0, H examples 0 and 4095    |
          | token rows 1 .. 622                    |
S=623     | hidden row 623, H examples 0 and 4095  |
          +----------------------------------------+

arg1_1 mask [Q=624, K=624], Q axis 0 .. 623, K axis 0 .. 623
arg2_1 positions, token axis 0 .. 623
examples: hidden[0,0,0], mask[0,0], pos[623].
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

是什么：这一段对输入 hidden states 做 RMSNorm，并读取输入 norm 的 4096 维权重。

为什么需要：Q/K/V 线性投影前需要按每个 token 行的 RMS 尺度归一化。

怎么做/计算：`_to_copy_default` 把 `arg0_1` 转 fp32；`pow_tensor_scalar` 平方；`mean_dim` 沿 Hidden 维求均值；`add_tensor` 加 epsilon；`rsqrt_default` 取倒数平方根；`mul_tensor` 逐元素缩放；`_to_copy_default_1` 转 fp16；`_param_constant0` 在下一段乘到归一化结果上。

```text
Input RMSNorm [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | x[0,h]^2 over H                        | ----> | r0 |
          | token rows 1 .. 622                    |       | .. |
S=623     | x[623,h]^2 over H                      | ----> | r623 |
          +----------------------------------------+       +----+

region contents: hidden fp32 values, squared values, row RMS scale, fp16 norm output.
examples: norm[0,0], norm[611,2048], norm[623,4095].
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant203 = self._tensor_constant203
mm_default = aten.mm.default(view_default, _tensor_constant203)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant204 = self._tensor_constant204
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant204)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant205 = self._tensor_constant205
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant205)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 624, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 624, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 624, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 624, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段将 normalized hidden 投影成 Q、K、V，并 reshape 成 `[B=1, Heads=32, S=624, Dh=128]`。

为什么需要：Q/K 用于生成 attention 权重，V 用于被权重加权求和；head reshape 把 4096 hidden 维拆成 32 个 128 维子空间。

怎么做/计算：`mul_tensor_1` 应用 input norm 权重；三个分支各自 `view -> mm -> _unsafe_view`，分别用 `_tensor_constant203/204/205` 生成 Q/K/V；`view_default_3/4/5` 把 `[1,624,4096]` 改成 `[1,624,32,128]`；`transpose_int/1/2` 变为 `[1,32,624,128]`。

```text
Q/K/V projection

input [S=624, H=4096]                   head output [Heads=32, S=624, Dh=128]
S axis 0 .. 623, H axis 0 .. 4095       head axis 0 .. 31, Dh axis 0 .. 127

          H=0                                  H=4095
S=0       +----------------------------------------+
          | norm row 0                            |
          | rows 1 .. 622                         | -- three mm branches -->
S=623     | norm row 623                          |
          +----------------------------------------+

head h region
          Dh=0                                  Dh=127
S=0       +----------------------------------------+
          | q/k/v[h,0,d] examples                 |
          | token rows 1 .. 622                    |
S=623     | q/k/v[h,623,d] examples               |
          +----------------------------------------+

examples: Q[0,0,0,0], K[0,31,611,127], V[0,6,623,64].
```

### RoPE position embedding

```python
_tensor_constant206 = self._tensor_constant206
index_tensor = aten.index.Tensor(_tensor_constant206, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant207 = self._tensor_constant207
index_tensor_1 = aten.index.Tensor(_tensor_constant207, [arg2_1])
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

是什么：这一段给 Q 和 K 加 RoPE 位置旋转，输出旋转后的 Q `add_tensor_1` 与 K `add_tensor_2`。

为什么需要：attention 分数需要包含 token 顺序信息，RoPE 通过 cos/sin 旋转把 position ids 融入 head 向量。

怎么做/计算：`index_tensor` 和 `index_tensor_1` 用 `arg2_1` 从 cos/sin 表取位置行；`unsqueeze` 用于广播；Q 分支将 Dh 轴切成 `0..63`、`64..127`，对右半取负并与左半拼接，再计算 `x*cos + rotate_half(x)*sin`；K 分支同样生成 `add_tensor_2`。

```text
RoPE over Dh dimension
Dh axis: 0 .. 127, split at 64

          Dh=0              Dh=63 Dh=64             Dh=127
Q/K       +---------------------+-----------------------+
          | LEFT_HALF           | RIGHT_HALF            |
          +---------------------+-----------------------+
rotated   +---------------------+-----------------------+
          | -RIGHT_HALF         | LEFT_HALF             |
          +---------------------+-----------------------+

element formula: out[h,s,d] = x[h,s,d] * cos[pos[s],d] + rot[h,s,d] * sin[pos[s],d]
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

是什么：这一段计算标准 attention score、加 mask、softmax，并复制 fp16 权重；没有额外的 Visual clear/fold 写入。

为什么需要：得到 `[Q,K]` attention 权重后，下一段才能按权重汇聚 V。

怎么做/计算：`transpose_int_3` 把 K 变成 `[Dh,K]`；Q/K 经 `expand/view` 分别成为 `[32,624,128]` 和 `[32,128,624]`；`bmm_default` 做 QK 矩阵乘得到 `[32,624,624]`；`view_default_8` 恢复 batch/head；`div_tensor` 除以 `sqrt(128)`；`add_tensor_3` 加 attention mask；`_softmax_default` 沿 K 轴归一化；`_to_copy_default_2` 转 fp16；`clone_default` 复制权重。

```text
Plain attention weights [Heads=32, Q=624, K=624]
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)

K=0                                      K=623
Q=0       +----------------------------------------+
          | softmax weights for query row 0        |
          | normal masked attention region         |
          | examples: w[h,35,35], w[h,611,200]     |
Q=623     | softmax weights for query row 623      |
          +----------------------------------------+

region contents: scaled QK score + mask, then softmax over K.
examples: w[head=0,Q=0,K=0], w[head=31,Q=623,K=623].
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

是什么：这一段执行 attention weights 与 V 的 contraction，并把多头 context 合并回 hidden。

为什么需要：softmax 权重本身不是输出，必须对 V 的 token 内容做加权和。

怎么做/计算：`clone_default` 变成 `[32,624,624]`；V `transpose_int_2` 变成 `[32,624,128]`；`bmm_default_1` 计算 `[Q,K] x [K,Dh] -> [Q,Dh]`；`view_default_11` 恢复 `[1,32,624,128]`；`transpose_int_4` 改成 `[1,624,32,128]`；`clone_default_1` 连续化；`view_default_12` 合并成 `[1,624,4096]`。

```text
Attention-weighted V

weights [Q=624, K=624]      V [K=624, Dh=128]      context [Q=624, Dh=128]
K axis 0 .. 623             Dh axis 0 .. 127       Dh axis 0 .. 127
Q=0   +------------------+  +------------------+   +------------------+
      | weight row 0     |x | V rows 0..623    |-> | context row 0    |
      | rows 1 .. 622    |  | value region     |   | rows 1 .. 622    |
Q=623 | weight row 623   |  | V row 623        |   | context row 623  |
      +------------------+  +------------------+   +------------------+

merged hidden [S=624, H=4096], H axis 0 .. 4095.
examples: ctx[0,0,0], ctx[31,623,127], merged[623,4095].
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant208 = self._tensor_constant208
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant208)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把 attention context 过输出投影，并与原输入 hidden 做残差相加。

为什么需要：输出投影把多头 context 映射回模型 hidden 空间，残差保留进入本层的表示。

怎么做/计算：`view_default_13` 展平为 `[624,4096]`；`mm_default_3` 乘 `_tensor_constant208`；`_unsafe_view_default_3` 恢复 `[1,624,4096]`；`add_tensor_4` 与 `arg0_1` 按相同 S/H 坐标逐元素相加。

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
mul_tensor_6 = aten.mul.Tensor(_to_copy_default_3, rsqrt_default_1)
_to_copy_default_4 = aten._to_copy.default(mul_tensor_6, dtype=torch.float16)
_param_constant5 = self._param_constant5
mul_tensor_7 = aten.mul.Tensor(_param_constant5, _to_copy_default_4)
view_default_14 = aten.view.default(mul_tensor_7, [624, 4096])
```

**解释与可视化**

是什么：这一段对 post-attention residual hidden 做 RMSNorm，得到 MLP 输入。

为什么需要：MLP 的 gate/up projection 需要归一化后的 token rows。

怎么做/计算：`_to_copy_default_3` 转 fp32；`pow_tensor_scalar_1` 平方；`mean_dim_1` 沿 Hidden 求均值；`add_tensor_5` 加 eps；`rsqrt_default_1` 取倒数平方根；`mul_tensor_6` 缩放；`_to_copy_default_4` 转 fp16；`mul_tensor_7` 乘 `_param_constant5`；`view_default_14` 展平成 `[624,4096]`。

```text
Post-attention RMSNorm [S=624, H=4096]

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | add_tensor_4 row 0 squared             | ----> | r0 |
          | rows 1 .. 622                          |       | .. |
S=623     | add_tensor_4 row 623 squared           | ----> | r623 |
          +----------------------------------------+       +----+

region contents: residual hidden, Hidden-axis reduction, eps+rsqrt, norm weight.
examples: mlp_in[0,0], mlp_in[35,1024], mlp_in[623,4095].
```

### MLP and final residual

```python
_tensor_constant209 = self._tensor_constant209
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant209)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [624, 4096])
_tensor_constant210 = self._tensor_constant210
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant210)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [624, 11008])
_tensor_constant211 = self._tensor_constant211
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant211)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP，再把 down projection 结果加回 attention residual。

为什么需要：MLP 提供逐 token 的非线性通道变换，residual add 维持主干 hidden 流。

怎么做/计算：`mm_default_4` 得到 gate `[624,11008]`，`silu_default` 激活；`mm_default_5` 得到 up `[624,11008]`；`mul_tensor_8` 做 gate/up 逐元素乘；`mm_default_6` down project 回 `[624,4096]`；`add_tensor_6` 加回 `add_tensor_4`。

```text
MLP and residual
S axis 0 .. 623, H axis 0 .. 4095, I axis 0 .. 11007

input [S,H]                         intermediate [S,I]
S=0  +---------------------------+  +---------------------------+
     | mlp input row 0           |->| silu(gate) * up row 0     |
     | rows 1 .. 622             |  | rows 1 .. 622             |
S=623| mlp input row 623         |  | gated row 623             |
     +---------------------------+  +---------------------------+
                 down project to H=0..4095
S=0  +---------------------------+
     | final residual row 0      |
     | rows 1 .. 622             |
S=623| final residual row 623    |
     +---------------------------+

examples: gate[0,0], up[623,11007], final[623,4095].
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)
```

**解释与可视化**

是什么：这一段打包返回 final hidden、当前层 K/V cache、Visual 输出占位 `None` 和控制标量 `0`。

为什么需要：final hidden 继续传给下一层；K/V cache 用于后续解码步复用当前层 attention 的 key/value。

怎么做/计算：`return` 不做新计算，只返回 `add_tensor_6`，以及 `dynamic_cache_layer` 中的 RoPE K `add_tensor_2` 和 V `transpose_int_2`。

```text
Layer output tensors

add_tensor_6 [B=1, S=624, H=4096]
S axis 0 .. 623, H axis 0 .. 4095
          H=0                                  H=4095
S=0       +----------------------------------------+
          | final hidden row 0                     |
          | rows 1 .. 622                          |
S=623     | final hidden row 623                   |
          +----------------------------------------+

cache:
K add_tensor_2    [B=1, Heads=32, S=624, Dh=128]
V transpose_int_2 [B=1, Heads=32, S=624, Dh=128]
Dh axis 0 .. 127; examples: K[0,0,0], K[31,623,127], V[6,611,64].
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
| 18 | `qkv_projection` | `_tensor_constant203` | `get_attr` | `_tensor_constant203` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant203` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant204` | `get_attr` | `_tensor_constant204` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant204` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant205` | `get_attr` | `_tensor_constant205` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant205` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `output` |
| 35 | `rope` | `_tensor_constant206` | `get_attr` | `_tensor_constant206` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant206`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant207` | `get_attr` | `_tensor_constant207` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant207`, `arg2_1` | `unsqueeze_default_1` |
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
| 77 | `output_projection` | `_tensor_constant208` | `get_attr` | `_tensor_constant208` | - | `mm_default_3` |
| 78 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant208` | `_unsafe_view_default_3` |
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
| 91 | `mlp` | `_tensor_constant209` | `get_attr` | `_tensor_constant209` | - | `mm_default_4` |
| 92 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant209` | `_unsafe_view_default_4` |
| 93 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 94 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 95 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 96 | `mlp` | `_tensor_constant210` | `get_attr` | `_tensor_constant210` | - | `mm_default_5` |
| 97 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant210` | `_unsafe_view_default_5` |
| 98 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 99 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 100 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 101 | `mlp` | `_tensor_constant211` | `get_attr` | `_tensor_constant211` | - | `mm_default_6` |
| 102 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant211` | `_unsafe_view_default_6` |
| 103 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 104 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 105 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2` | - |
