# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer5`
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
| QK scores, mask, softmax | 55-70 | 16 | `add_tensor_1`, `add_tensor_2`, `arg1_1` | `clone_default` |
| Attention-weighted V and hidden reshape | 71-79 | 9 | `clone_default`, `transpose_int_2` | `view_default_12` |
| Attention output projection and residual | 80-84 | 5 | `arg0_1`, `view_default_12` | `add_tensor_4` |
| Post-attention RMSNorm | 85-94 | 10 | `add_tensor_4` | `mul_tensor_7`, `view_default_14` |
| MLP and final residual | 95-108 | 14 | `add_tensor_4`, `mul_tensor_7`, `view_default_14` | `add_tensor_6` |
| Layer output | 109-109 | 1 | `add_tensor_2`, `add_tensor_6`, `transpose_int_2` | - |

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

是什么：这一段提供固定样本的 FX placeholder。该层实际使用 `arg0_1` hidden states、`arg1_1` attention mask 和 `arg2_1` position ids。

为什么需要：后续 RMSNorm、attention mask add、RoPE 查表都从这些运行时输入开始。

怎么做/计算：placeholder 不做数值计算。`arg0_1` 被 `_to_copy_default` 和残差 `add_tensor_4` 使用；`arg1_1` 被 `add_tensor_3` 加到 QK 分数上；`arg2_1` 被两个 `index.Tensor` 用来取 cos/sin 位置表。

```text
Input tensors

arg0_1 hidden [B=1, S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | hidden row 0, H examples 0 and 4095    |
          | token rows 1 .. 622                    |
S=623     | hidden row 623, H examples 0 and 4095  |
          +----------------------------------------+

arg1_1 mask [Q=624, K=624]
Q axis: 0 .. 623, K axis: 0 .. 623
examples: mask[0,0], mask[611,35], mask[623,623]

arg2_1 position ids, token axis 0 .. 623
examples: pos[0], pos[35], pos[611], pos[623]
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

是什么：这一段把输入 hidden states 做输入 RMSNorm，并读取 RMSNorm 权重。

为什么需要：Q/K/V 投影前需要每个 token 行在 Hidden 维上有稳定尺度。

怎么做/计算：`_to_copy_default` 转 fp32；`pow_tensor_scalar` 对 Hidden 元素平方；`mean_dim` 沿 `H` 求每行均值；`add_tensor` 加 `1e-05`；`rsqrt_default` 得到缩放因子；`mul_tensor` 把原 fp32 hidden 乘缩放；`_to_copy_default_1` 转回 fp16；`_param_constant0` 是 4096 维权重，下一段用它做逐 hidden 维缩放。

```text
Input RMSNorm [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                  H=4095     RMS
S=0       +----------------------------------------+      +----+
          | square hidden row 0                    | ---> | r0 |
          | rows 1 .. 622                          |      | .. |
S=623     | square hidden row 623                  | ---> | r623 |
          +----------------------------------------+      +----+

region contents: fp32 hidden values, row-wise square, Hidden mean, eps+rsqrt.
examples: norm[0,0], norm[35,2048], norm[623,4095].
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant193 = self._tensor_constant193
mm_default = aten.mm.default(view_default, _tensor_constant193)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant194 = self._tensor_constant194
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant194)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant195 = self._tensor_constant195
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant195)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 624, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 624, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 624, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 624, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段从归一化 hidden 生成 Q、K、V，并把 4096 hidden 维拆成 32 个 head、每个 head 128 维。

为什么需要：attention 的分数来自 QK contraction，输出内容来自 V；多头布局让每个 head 在独立的 `Dh=128` 空间里计算。

怎么做/计算：`mul_tensor_1` 把 `_param_constant0` 乘到归一化 hidden 上；三个 `view -> mm -> _unsafe_view` 分支分别用 `_tensor_constant193/194/195` 生成 Q/K/V `[1,624,4096]`；随后每个分支 `view` 为 `[1,624,32,128]`，再 `transpose` 为 `[1,32,624,128]`。

```text
Q/K/V projection

input [S=624, H=4096]                     output per head [S=624, Dh=128]
S axis 0 .. 623, H axis 0 .. 4095         head axis 0 .. 31, Dh axis 0 .. 127

          H=0                         H=4095
S=0       +-------------------------------+
          | norm row 0                    |
          | rows 1 .. 622                 | --mm--> Q/K/V heads
S=623     | norm row 623                  |
          +-------------------------------+

head=0 view
          Dh=0                         Dh=127
S=0       +-------------------------------+
          | q/k/v row 0, dim examples     |
          | token rows 1 .. 622           |
S=623     | q/k/v row 623                 |
          +-------------------------------+

examples: Q[0,0,0,0], K[0,31,611,127], V[0,5,623,64].
```

### RoPE position embedding

```python
_tensor_constant196 = self._tensor_constant196
index_tensor = aten.index.Tensor(_tensor_constant196, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant197 = self._tensor_constant197
index_tensor_1 = aten.index.Tensor(_tensor_constant197, [arg2_1])
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

是什么：这一段对 Q 和 K 应用 RoPE 位置旋转，输出 `add_tensor_1` 和 `add_tensor_2`。

为什么需要：QK 分数需要感知 token 位置，RoPE 把 `arg2_1` 的 position ids 注入每个 head 向量。

怎么做/计算：`index_tensor/index_tensor_1` 分别按 `arg2_1` 从 cos/sin 表取 `[624,128]`；`unsqueeze` 让其可广播到 heads；Q 分支把 Dh 轴切成 `0..63` 与 `64..127`，右半取负后与左半拼成 rotate-half，再计算 `Q*cos + rotate_half(Q)*sin`；K 分支对 `transpose_int_1` 做同样计算。

```text
RoPE region over Dh axis
Dh axis: 0 .. 127, split into 0 .. 63 and 64 .. 127

          Dh=0              Dh=63 Dh=64             Dh=127
Q/K       +---------------------+-----------------------+
          | LEFT_HALF           | RIGHT_HALF            |
          +---------------------+-----------------------+
rot_half  +---------------------+-----------------------+
          | -RIGHT_HALF         | LEFT_HALF             |
          +---------------------+-----------------------+

output element = x[h,s,d] * cos[pos[s],d] + rot_half[h,s,d] * sin[pos[s],d]
examples: out[h=0,s=0,d=0], out[h=12,s=35,d=64], out[h=31,s=623,d=127].
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
_tensor_constant198 = self._tensor_constant198
fill__tensor = aten.fill_.Tensor(slice_tensor_5, _tensor_constant198)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段计算 attention 权重，然后对尾部 query 的 Visual-key 区域做 clear-only 覆盖。

为什么需要：softmax 后的 `[Q,K]` 权重决定后续 V 汇聚；`fill_` 表示该固定样本中 `Q=611..623, K=35..610` 的权重块被常量覆盖。

怎么做/计算：`transpose_int_3` 把 K 转成 `[Dh,K]`；Q/K 经 `expand/view` 变为 `[32,624,128]` 和 `[32,128,624]`；`bmm_default` 得到 `[32,624,624]`；`view_default_8` 恢复 batch/head；`div_tensor` 按 `sqrt(128)` 缩放；`add_tensor_3` 加 mask；`_softmax_default` 沿 K 轴归一化；`_to_copy_default_2` 转 fp16。随后 `slice_tensor_4` 取 `Q=611..623`，`slice_tensor_5` 取其中 `K=35..610`，`fill__tensor` 用 `_tensor_constant198` 覆盖该矩形区域；`clone_default` 复制覆盖后的权重给 V contraction。

```text
Attention weights [Heads=32, Q=624, K=624]
Q axis: 0 .. 623, K axis: 0 .. 623 (square compressed)

K=0        K=35                         K=610 K=611      K=623
+----------+--------------------------------+-------------+
| KEEP     | KEEP                           | KEEP        | Q=0
|          |                                |             |
+----------+--------------------------------+-------------+ Q=611
| KEEP     | CLEAR_BY_FILL                  | KEEP        |
|          | examples: w[h,611,35],         |             |
|          | w[h,620,200], w[h,623,610]     |             |
+----------+--------------------------------+-------------+ Q=623

Region contents:
- KEEP: normal masked-softmax weights.
- CLEAR_BY_FILL: `fill_` writes the observed constant into Q=611..623, K=35..610.
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

是什么：这一段把 attention 权重与 V 相乘，得到每个 query 的 context，再合并回 hidden 维。

为什么需要：QK softmax 只是权重，模型需要用这些权重对 V 内容向量求加权和。

怎么做/计算：`clone_default` 经 `expand/view` 变成 `[32,624,624]`；V `transpose_int_2` 经 `expand/view` 变成 `[32,624,128]`；`bmm_default_1` 计算 `[Q,K] x [K,Dh] -> [Q,Dh]`；`view_default_11` 和 `transpose_int_4` 恢复 `[B,S,Heads,Dh]`；`clone_default_1` 连续化；`view_default_12` 合并为 `[1,624,4096]`。

```text
Per-head contraction

weights [Q=624, K=624]      V [K=624, Dh=128]      context [Q=624, Dh=128]
K axis 0 .. 623             Dh axis 0 .. 127       Dh axis 0 .. 127
Q=0   +------------------+  +------------------+   +------------------+
      | weight row 0     |x | V rows 0..623    |-> | ctx row 0        |
      | rows 1 .. 622    |  | examples V[0,d]  |   | rows 1 .. 622    |
Q=623 | weight row 623   |  | V[623,d]         |   | ctx row 623      |
      +------------------+  +------------------+   +------------------+

Merged output: [S=624, H=4096], H=0..4095 = 32 heads x Dh=0..127.
examples: ctx[h=0,q=0,d=0], ctx[h=31,q=623,d=127], merged[623,4095].
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant199 = self._tensor_constant199
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant199)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段执行 attention output projection 并与层输入做 residual add。

为什么需要：attention context 需要通过输出矩阵回到 hidden 表示空间，残差连接保留原输入。

怎么做/计算：`view_default_13` 展平成 `[624,4096]`；`mm_default_3` 乘 `_tensor_constant199`；`_unsafe_view_default_3` 恢复 `[1,624,4096]`；`add_tensor_4` 与 `arg0_1` 在同一 S/H 坐标逐元素相加。

```text
Projection + residual [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                  H=4095
proj      +----------------------------------------+
          | projected attention rows 0 .. 623      |
          +----------------------------------------+
input     +----------------------------------------+
          | original arg0_1 rows 0 .. 623          |
          +----------------------------------------+
output    +----------------------------------------+
          | add_tensor_4 = input + projected       |
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

是什么：这一段对 attention residual 后的 hidden 做第二次 RMSNorm，输出 MLP 输入。

为什么需要：MLP gate/up projection 前需要稳定的 row-wise hidden 尺度。

怎么做/计算：`_to_copy_default_3` 转 fp32；`pow_tensor_scalar_1` 平方；`mean_dim_1` 沿 Hidden 求均值；`add_tensor_5` 加 eps；`rsqrt_default_1` 取倒数平方根；`mul_tensor_6` 逐元素缩放；`_to_copy_default_4` 转 fp16；`mul_tensor_7` 乘 `_param_constant5`；`view_default_14` 得到 `[624,4096]`。

```text
Post-attention RMSNorm [S=624, H=4096]
S axis: 0 .. 623, H axis: 0 .. 4095 (H width compressed)

          H=0                                  H=4095     RMS
S=0       +----------------------------------------+      +----+
          | post-attn row 0 squared                | ---> | r0 |
          | rows 1 .. 622                          |      | .. |
S=623     | post-attn row 623 squared              | ---> | r623 |
          +----------------------------------------+      +----+

region contents: hidden row values, row-wise square/mean, eps+rsqrt, norm weight.
examples: mlp_in[0,0], mlp_in[35,1024], mlp_in[623,4095].
```

### MLP and final residual

```python
_tensor_constant200 = self._tensor_constant200
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant200)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [624, 4096])
_tensor_constant201 = self._tensor_constant201
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant201)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [624, 11008])
_tensor_constant202 = self._tensor_constant202
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant202)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行 gated MLP 并把 down projection 结果加回 attention residual。

为什么需要：MLP 对每个 token 的 hidden 通道做非线性扩展、门控和压回，补充 attention 之外的逐 token 变换。

怎么做/计算：`mm_default_4` 产生 gate `[624,11008]`，`silu_default` 激活；`mm_default_5` 产生 up `[624,11008]`；`mul_tensor_8` 做 gate/up 逐元素乘；`view_default_16` 展平中间维；`mm_default_6` down project 回 `[624,4096]`；`add_tensor_6` 与 `add_tensor_4` 相加。

```text
MLP [S=624, H=4096] -> [S=624, I=11008] -> [S=624, H=4096]

H axis: 0 .. 4095                         I axis: 0 .. 11007
S=0  +-------------------------------+    +-------------------------------+
     | mlp input row 0               | -> | silu(gate) * up row 0         |
     | rows 1 .. 622                 |    | rows 1 .. 622                 |
S=623| mlp input row 623             |    | gated row 623                 |
     +-------------------------------+    +-------------------------------+
                         down projection
S=0  +-------------------------------+
     | final = attention residual + down_out |
     | rows 1 .. 622                 |
S=623| final hidden row 623          |
     +-------------------------------+

examples: inter[0,0], inter[623,11007], final[623,4095].
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)
```

**解释与可视化**

是什么：这一段返回本层 final hidden、dynamic cache 的 K/V、Visual 输出占位 `None` 和控制标量 `0`。

为什么需要：`add_tensor_6` 继续送入下一层；`add_tensor_2` 和 `transpose_int_2` 作为当前层 K/V cache 供后续解码使用。

怎么做/计算：`return` 只打包已计算值，不新增张量计算。`add_tensor_6` 来自 MLP residual，`add_tensor_2` 是 RoPE 后 K，`transpose_int_2` 是 V head layout。

```text
Returned tensors

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
examples: K[h=0,s=0,d=0], K[h=31,s=623,d=127], V[h=5,s=611,d=64].
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
| 18 | `qkv_projection` | `_tensor_constant193` | `get_attr` | `_tensor_constant193` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant193` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant194` | `get_attr` | `_tensor_constant194` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant194` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant195` | `get_attr` | `_tensor_constant195` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant195` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `output` |
| 35 | `rope` | `_tensor_constant196` | `get_attr` | `_tensor_constant196` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant196`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant197` | `get_attr` | `_tensor_constant197` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant197`, `arg2_1` | `unsqueeze_default_1` |
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
| 65 | `attention_scores` | `_to_copy_default_2` | `call_function` | `aten._to_copy.default` | `_softmax_default` | `clone_default`, `slice_tensor_4` |
| 66 | `attention_scores` | `slice_tensor_4` | `call_function` | `aten.slice.Tensor` | `_to_copy_default_2` | `slice_tensor_5` |
| 67 | `attention_scores` | `slice_tensor_5` | `call_function` | `aten.slice.Tensor` | `slice_tensor_4` | `fill__tensor` |
| 68 | `attention_scores` | `_tensor_constant198` | `get_attr` | `_tensor_constant198` | - | `fill__tensor` |
| 69 | `attention_scores` | `fill__tensor` | `call_function` | `aten.fill_.Tensor` | `slice_tensor_5`, `_tensor_constant198` | - |
| 70 | `attention_scores` | `clone_default` | `call_function` | `aten.clone.default` | `_to_copy_default_2` | `expand_default_2` |
| 71 | `attention_output` | `expand_default_2` | `call_function` | `aten.expand.default` | `clone_default` | `view_default_9` |
| 72 | `attention_output` | `view_default_9` | `call_function` | `aten.view.default` | `expand_default_2` | `bmm_default_1` |
| 73 | `attention_output` | `expand_default_3` | `call_function` | `aten.expand.default` | `transpose_int_2` | `view_default_10` |
| 74 | `attention_output` | `view_default_10` | `call_function` | `aten.view.default` | `expand_default_3` | `bmm_default_1` |
| 75 | `attention_output` | `bmm_default_1` | `call_function` | `aten.bmm.default` | `view_default_9`, `view_default_10` | `view_default_11` |
| 76 | `attention_output` | `view_default_11` | `call_function` | `aten.view.default` | `bmm_default_1` | `transpose_int_4` |
| 77 | `attention_output` | `transpose_int_4` | `call_function` | `aten.transpose.int` | `view_default_11` | `clone_default_1` |
| 78 | `attention_output` | `clone_default_1` | `call_function` | `aten.clone.default` | `transpose_int_4` | `view_default_12` |
| 79 | `attention_output` | `view_default_12` | `call_function` | `aten.view.default` | `clone_default_1` | `view_default_13` |
| 80 | `output_projection` | `view_default_13` | `call_function` | `aten.view.default` | `view_default_12` | `mm_default_3` |
| 81 | `output_projection` | `_tensor_constant199` | `get_attr` | `_tensor_constant199` | - | `mm_default_3` |
| 82 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant199` | `_unsafe_view_default_3` |
| 83 | `output_projection` | `_unsafe_view_default_3` | `call_function` | `aten._unsafe_view.default` | `mm_default_3` | `add_tensor_4` |
| 84 | `output_projection` | `add_tensor_4` | `call_function` | `aten.add.Tensor` | `arg0_1`, `_unsafe_view_default_3` | `_to_copy_default_3`, `add_tensor_6` |
| 85 | `post_attention_rmsnorm` | `_to_copy_default_3` | `call_function` | `aten._to_copy.default` | `add_tensor_4` | `mul_tensor_6`, `pow_tensor_scalar_1` |
| 86 | `post_attention_rmsnorm` | `pow_tensor_scalar_1` | `call_function` | `aten.pow.Tensor_Scalar` | `_to_copy_default_3` | `mean_dim_1` |
| 87 | `post_attention_rmsnorm` | `mean_dim_1` | `call_function` | `aten.mean.dim` | `pow_tensor_scalar_1` | `add_tensor_5` |
| 88 | `post_attention_rmsnorm` | `add_tensor_5` | `call_function` | `aten.add.Tensor` | `mean_dim_1` | `rsqrt_default_1` |
| 89 | `post_attention_rmsnorm` | `rsqrt_default_1` | `call_function` | `aten.rsqrt.default` | `add_tensor_5` | `mul_tensor_6` |
| 90 | `post_attention_rmsnorm` | `mul_tensor_6` | `call_function` | `aten.mul.Tensor` | `_to_copy_default_3`, `rsqrt_default_1` | `_to_copy_default_4` |
| 91 | `post_attention_rmsnorm` | `_to_copy_default_4` | `call_function` | `aten._to_copy.default` | `mul_tensor_6` | `mul_tensor_7` |
| 92 | `post_attention_rmsnorm` | `_param_constant5` | `get_attr` | `_param_constant5` | - | `mul_tensor_7` |
| 93 | `post_attention_rmsnorm` | `mul_tensor_7` | `call_function` | `aten.mul.Tensor` | `_param_constant5`, `_to_copy_default_4` | `view_default_14`, `view_default_15` |
| 94 | `post_attention_rmsnorm` | `view_default_14` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_4` |
| 95 | `mlp` | `_tensor_constant200` | `get_attr` | `_tensor_constant200` | - | `mm_default_4` |
| 96 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant200` | `_unsafe_view_default_4` |
| 97 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 98 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 99 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 100 | `mlp` | `_tensor_constant201` | `get_attr` | `_tensor_constant201` | - | `mm_default_5` |
| 101 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant201` | `_unsafe_view_default_5` |
| 102 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 103 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 104 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 105 | `mlp` | `_tensor_constant202` | `get_attr` | `_tensor_constant202` | - | `mm_default_6` |
| 106 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant202` | `_unsafe_view_default_6` |
| 107 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 108 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 109 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2` | - |
