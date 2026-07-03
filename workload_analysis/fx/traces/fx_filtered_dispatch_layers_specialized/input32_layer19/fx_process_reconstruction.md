# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input32_layer19`
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

是什么：这一段是单步 decode 固定样本的 FX placeholder，后续实际使用 `arg0_1`、`arg1_1`、`arg2_1`。

为什么需要：`arg0_1` 提供当前 token hidden state，`arg1_1` 提供当前 query 对 cache keys 的 attention mask，`arg2_1` 提供当前 token 的 RoPE position id。

怎么做/计算：placeholder 不做数值计算。`arg0_1` 进入 RMSNorm 和残差；`arg1_1` 加到 QK logits；`arg2_1` 被 `index.Tensor` 用于 cos/sin 查表。

```text
Runtime inputs

arg0_1 hidden [B=1, S=1, H=4096]
S axis: 0 .. 0, H axis: 0 .. 4095 (H width compressed)
          H=0                                  H=4095
S=0       +----------------------------------------+
          | current token hidden row, H0/H4095     |
          +----------------------------------------+

arg1_1 mask [Q=1, K=89], axes Q=0..0 and K=0..88.
arg2_1 position id for the current token, token axis 0..0.
examples: hidden[0,0,0], mask[0,88], pos[0].
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

是什么：这一段对当前 token hidden state 做 RMSNorm，输出 fp16 normalized hidden 并读取 input norm 权重。

为什么需要：Q/K/V 投影前仍要按 Hidden 维归一化当前 token。

怎么做/计算：`_to_copy_default` 转 fp32；`pow_tensor_scalar` 对当前 row 每个 Hidden 元素平方；`mean_dim` 沿 Hidden 求均值；`add_tensor` 加 eps；`rsqrt_default` 得到缩放；`mul_tensor` 应用缩放；`_to_copy_default_1` 转 fp16；`_param_constant0` 供投影前逐 Hidden 维相乘。

```text
Input RMSNorm [S=1, H=4096]
S axis 0 .. 0, H axis 0 .. 4095

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | current row squared over Hidden        | ----> | r0 |
          +----------------------------------------+       +----+

region contents: fp32 hidden, square, Hidden mean, eps+rsqrt, fp16 norm.
examples: norm[0,0], norm[0,2048], norm[0,4095].
```

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant293 = self._tensor_constant293
mm_default = aten.mm.default(view_default, _tensor_constant293)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 1, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant294 = self._tensor_constant294
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant294)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 1, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [1, 4096])
_tensor_constant295 = self._tensor_constant295
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant295)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 1, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 1, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 1, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 1, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

**解释与可视化**

是什么：这一段从当前 normalized hidden 生成当前 token 的 Q、K、V，并拆成 32 个 128 维 head。

为什么需要：单步 decode 只产生一个新 query，但当前 K/V 还会被追加进 cache。

怎么做/计算：`mul_tensor_1` 乘 input norm 权重；Q/K/V 三个分支分别 `view -> mm -> _unsafe_view`，使用 `_tensor_constant293/294/295`；再 `view` 成 `[1,1,32,128]`，`transpose` 到 `[1,32,1,128]`。

```text
Q/K/V projection

input [S=1, H=4096]                  output [Heads=32, S=1, Dh=128]
S axis 0..0, H axis 0..4095          head axis 0..31, Dh axis 0..127

          H=0                                  H=4095
S=0       +----------------------------------------+
          | normalized current token row           |
          +----------------------------------------+

          Dh=0                                 Dh=127
S=0       +----------------------------------------+
          | current token head h row               |
          +----------------------------------------+

examples: Q[0,0,0,0], K[0,31,0,127], V[0,19,0,64].
```

### RoPE position embedding

```python
_tensor_constant296 = self._tensor_constant296
index_tensor = aten.index.Tensor(_tensor_constant296, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant297 = self._tensor_constant297
index_tensor_1 = aten.index.Tensor(_tensor_constant297, [arg2_1])
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

是什么：这一段给当前 token 的 Q/K 应用 RoPE，输出 rotated Q `add_tensor_1` 和 rotated K `add_tensor_2`。

为什么需要：新 query 与追加到 cache 的新 key 都需要包含当前位置编码。

怎么做/计算：`index_tensor/index_tensor_1` 用 `arg2_1` 从 `_tensor_constant296/297` 取 cos/sin；Q 分支计算 `transpose_int*cos`，把 Dh 轴 `0..63` 和 `64..127` 切开，右半取负并拼回左半形成 rotate-half，乘 sin 后相加；K 分支同样生成 `add_tensor_2`。

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

out[h,0,d] = x[h,0,d] * cos[pos[0],d] + rot_half[h,0,d] * sin[pos[0],d]
examples: out[0,0,0], out[12,0,64], out[31,0,127].
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
expand_default_1 = aten.expand.default(transpose_int_3, [1, 32, 128, 89])
view_default_7 = aten.view.default(expand_default_1, [32, 128, 89])
bmm_default = aten.bmm.default(view_default_6, view_default_7)
view_default_8 = aten.view.default(bmm_default, [1, 32, 1, 89])
div_tensor = aten.div.Tensor(view_default_8, 11.313708498984761)
add_tensor_3 = aten.add.Tensor(div_tensor, arg1_1)
_softmax_default = aten._softmax.default(add_tensor_3, -1, True)
_to_copy_default_2 = aten._to_copy.default(_softmax_default, dtype=torch.float16)
clone_default = aten.clone.default(_to_copy_default_2)
```

**解释与可视化**

是什么：这一段把已有 K cache 与当前 K concat 成长度 89 的 K 轴，然后计算当前 query 对所有 keys 的 attention weights。

为什么需要：decode 时当前 token 要 attend 到历史 88 个 cache keys 和当前新 key；同时 `cat_default_2` 会作为更新后的 K cache 返回。

怎么做/计算：`_tensor_constant2` 是已有 K cache，`cat_default_2` 将它和当前 `add_tensor_2` 沿 token/cache 轴拼接；`_tensor_constant3` 是已有 V cache，`cat_default_3` 将它和当前 `transpose_int_2` 拼接；`transpose_int_3` 把 K 变成 `[Dh,K]`；Q/K 经 `expand/view` 成 `[32,1,128]` 和 `[32,128,89]`；`bmm_default` 得到 `[32,1,89]`；`view_default_8` 恢复 `[1,32,1,89]`；`div_tensor` 缩放；`add_tensor_3` 加 mask；`_softmax_default` 沿 K=89 归一化；`_to_copy_default_2` 转 fp16；`clone_default` 复制权重。

```text
Decode QK with cache concat

K axis after concat: 0 .. 88 = cached 0 .. 87 + current 88.
Q axis: 0 .. 0.

K cache / current K [K=89, Dh=128] (K height compressed)
          Dh=0                                  Dh=127
K=0       +----------------------------------------+
          | cached key rows 0 .. 87                |
K=88      | current key add_tensor_2               |
          +----------------------------------------+

attention weights [Q=1, K=89] (K width compressed)
          K=0                                  K=88
Q=0       +----------------------------------------+
          | softmax over cached keys + current key |
          +----------------------------------------+

examples: k_cache[0,0], k_current[88,127], weight[0,88].
```

### Attention-weighted V and hidden reshape

```python
expand_default_2 = aten.expand.default(clone_default, [1, 32, 1, 89])
view_default_9 = aten.view.default(expand_default_2, [32, 1, 89])
expand_default_3 = aten.expand.default(cat_default_3, [1, 32, 89, 128])
view_default_10 = aten.view.default(expand_default_3, [32, 89, 128])
bmm_default_1 = aten.bmm.default(view_default_9, view_default_10)
view_default_11 = aten.view.default(bmm_default_1, [1, 32, 1, 128])
transpose_int_4 = aten.transpose.int(view_default_11, 1, 2)
view_default_12 = aten.view.default(transpose_int_4, [1, 1, 4096])
```

**解释与可视化**

是什么：这一段用当前 query 的 `[1,89]` attention weights 对 concat 后的 V cache 做加权求和，并合并多头 context。

为什么需要：当前 token 的 attention 输出要汇聚历史 value cache 和当前 value。

怎么做/计算：`clone_default` 展成 `[32,1,89]`；`cat_default_3` 展成 `[32,89,128]`；`bmm_default_1` 计算 `[1,K] x [K,Dh] -> [1,Dh]`；`view_default_11` 恢复 batch/head；`transpose_int_4` 转回 token-major；`view_default_12` 合并 heads 得到 `[1,1,4096]`。

```text
Attention-weighted V

weights [Q=1, K=89]        V [K=89, Dh=128]        context [Q=1, Dh=128]
K axis 0..88               Dh axis 0..127          Dh axis 0..127
Q=0  +------------------+  +-------------------+   +-------------------+
     | weights over K   |x | cached V + current |-> | current context   |
     +------------------+  +-------------------+   +-------------------+

merged hidden [S=1, H=4096], H axis 0 .. 4095.
examples: ctx[0,0,0], ctx[31,0,127], merged[0,4095].
```

### Attention output projection and residual

```python
view_default_13 = aten.view.default(view_default_12, [1, 4096])
_tensor_constant298 = self._tensor_constant298
mm_default_3 = aten.mm.default(view_default_13, _tensor_constant298)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 1, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
```

**解释与可视化**

是什么：这一段把当前 token context 过输出投影，并与当前 token 原始 hidden 做残差加法。

为什么需要：输出投影把多头 context 映射回 hidden space，残差连接保留当前 token 输入。

怎么做/计算：`view_default_13` 展平 context；`mm_default_3` 乘 `_tensor_constant298`；`_unsafe_view_default_3` 恢复 `[1,1,4096]`；`add_tensor_4` 与 `arg0_1` 在同一 H 坐标逐元素相加。

```text
Output projection + residual [S=1, H=4096]
S axis 0 .. 0, H axis 0 .. 4095

          H=0                                  H=4095
proj      +----------------------------------------+
          | projected current context              |
input     +----------------------------------------+
          | arg0_1 current hidden                  |
output    +----------------------------------------+
          | add_tensor_4 current hidden            |
          +----------------------------------------+

examples: add_tensor_4[0,0], add_tensor_4[0,2048], add_tensor_4[0,4095].
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

是什么：这一段对当前 token post-attention residual 做 RMSNorm，并生成 MLP 输入。

为什么需要：MLP projection 前需要归一化当前 token row。

怎么做/计算：`_to_copy_default_3` 转 fp32；`pow_tensor_scalar_1` 平方；`mean_dim_1` 沿 Hidden 求均值；`add_tensor_5` 加 eps；`rsqrt_default_1` 取倒数平方根；`mul_tensor_6` 缩放；`_to_copy_default_4` 转 fp16；`mul_tensor_7` 应用 `_param_constant5`；`view_default_14` 得到 `[1,4096]`。

```text
Post-attention RMSNorm [S=1, H=4096]
S axis 0 .. 0, H axis 0 .. 4095

          H=0                                  H=4095      RMS
S=0       +----------------------------------------+       +----+
          | add_tensor_4 current row squared       | ----> | r0 |
          +----------------------------------------+       +----+

region contents: residual hidden, Hidden-axis square/mean, eps+rsqrt, norm weight.
examples: mlp_in[0,0], mlp_in[0,1024], mlp_in[0,4095].
```

### MLP and final residual

```python
_tensor_constant299 = self._tensor_constant299
mm_default_4 = aten.mm.default(view_default_14, _tensor_constant299)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 1, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_15 = aten.view.default(mul_tensor_7, [1, 4096])
_tensor_constant300 = self._tensor_constant300
mm_default_5 = aten.mm.default(view_default_15, _tensor_constant300)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 1, 11008])
mul_tensor_8 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_16 = aten.view.default(mul_tensor_8, [1, 11008])
_tensor_constant301 = self._tensor_constant301
mm_default_6 = aten.mm.default(view_default_16, _tensor_constant301)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 1, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

**解释与可视化**

是什么：这一段执行当前 token 的 gated MLP，并把 down projection 结果加回 attention residual。

为什么需要：MLP 为当前 token 做非线性通道变换，residual add 保持主干 hidden 传递。

怎么做/计算：`mm_default_4` 使用 `_tensor_constant299` 产生 gate `[1,11008]`，`silu_default` 激活；`mm_default_5` 使用 `_tensor_constant300` 产生 up `[1,11008]`；`mul_tensor_8` gate/up 逐元素乘；`mm_default_6` 使用 `_tensor_constant301` down project 回 `[1,4096]`；`add_tensor_6` 加回 `add_tensor_4`。

```text
MLP [S=1, H=4096] -> [S=1, I=11008] -> [S=1, H=4096]

H axis 0 .. 4095                         I axis 0 .. 11007
S=0  +-------------------------------+   +-------------------------------+
     | mlp input current row         |-> | silu(gate) * up current row   |
     +-------------------------------+   +-------------------------------+
                         down projection
S=0  +-------------------------------+
     | final residual current row    |
     +-------------------------------+

examples: gate[0,0], up[0,11007], final[0,4095].
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (cat_default_2, cat_default_3)}, None, 0)
```

**解释与可视化**

是什么：这一段返回当前 token final hidden、更新后的 K/V cache、Visual 输出占位 `None` 和控制标量 `0`。

为什么需要：`add_tensor_6` 是本层 decode 输出；`cat_default_2/cat_default_3` 把当前 K/V 追加到 cache，供后续 token 复用。

怎么做/计算：`return` 只打包 `add_tensor_6`、`cat_default_2`、`cat_default_3`、`None` 和 `0`；没有新张量计算。

```text
Layer output

hidden add_tensor_6 [B=1, S=1, H=4096]
S axis 0 .. 0, H axis 0 .. 4095
          H=0                                  H=4095
S=0       +----------------------------------------+
          | final current token hidden             |
          +----------------------------------------+

dynamic_cache_layer:
K cat_default_2 [B=1, Heads=32, K=89, Dh=128]
V cat_default_3 [B=1, Heads=32, K=89, Dh=128]
examples: K[31,88,127], V[19,88,64], hidden[0,0,4095].
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
| 18 | `qkv_projection` | `_tensor_constant293` | `get_attr` | `_tensor_constant293` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant293` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant294` | `get_attr` | `_tensor_constant294` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant294` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant295` | `get_attr` | `_tensor_constant295` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant295` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `cat_default_3` |
| 35 | `rope` | `_tensor_constant296` | `get_attr` | `_tensor_constant296` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant296`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant297` | `get_attr` | `_tensor_constant297` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant297`, `arg2_1` | `unsqueeze_default_1` |
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
| 80 | `output_projection` | `_tensor_constant298` | `get_attr` | `_tensor_constant298` | - | `mm_default_3` |
| 81 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_13`, `_tensor_constant298` | `_unsafe_view_default_3` |
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
| 94 | `mlp` | `_tensor_constant299` | `get_attr` | `_tensor_constant299` | - | `mm_default_4` |
| 95 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant299` | `_unsafe_view_default_4` |
| 96 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 97 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_8` |
| 98 | `mlp` | `view_default_15` | `call_function` | `aten.view.default` | `mul_tensor_7` | `mm_default_5` |
| 99 | `mlp` | `_tensor_constant300` | `get_attr` | `_tensor_constant300` | - | `mm_default_5` |
| 100 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant300` | `_unsafe_view_default_5` |
| 101 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_8` |
| 102 | `mlp` | `mul_tensor_8` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_16` |
| 103 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_6` |
| 104 | `mlp` | `_tensor_constant301` | `get_attr` | `_tensor_constant301` | - | `mm_default_6` |
| 105 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant301` | `_unsafe_view_default_6` |
| 106 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 107 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 108 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `cat_default_2`, `cat_default_3` | - |
