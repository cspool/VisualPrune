# FX Layer Process Reconstruction

Trace directory: `workload_analysis/fx/traces/fx_filtered_dispatch_layers_specialized/input1_layer17`
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

### Q/K/V projection and head reshape

```python
mul_tensor_1 = aten.mul.Tensor(_param_constant0, _to_copy_default_1)
view_default = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant76 = self._tensor_constant76
mm_default = aten.mm.default(view_default, _tensor_constant76)
_unsafe_view_default = aten._unsafe_view.default(mm_default, [1, 624, 4096])
view_default_1 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant77 = self._tensor_constant77
mm_default_1 = aten.mm.default(view_default_1, _tensor_constant77)
_unsafe_view_default_1 = aten._unsafe_view.default(mm_default_1, [1, 624, 4096])
view_default_2 = aten.view.default(mul_tensor_1, [624, 4096])
_tensor_constant78 = self._tensor_constant78
mm_default_2 = aten.mm.default(view_default_2, _tensor_constant78)
_unsafe_view_default_2 = aten._unsafe_view.default(mm_default_2, [1, 624, 4096])
view_default_3 = aten.view.default(_unsafe_view_default, [1, 624, 32, 128])
transpose_int = aten.transpose.int(view_default_3, 1, 2)
view_default_4 = aten.view.default(_unsafe_view_default_1, [1, 624, 32, 128])
transpose_int_1 = aten.transpose.int(view_default_4, 1, 2)
view_default_5 = aten.view.default(_unsafe_view_default_2, [1, 624, 32, 128])
transpose_int_2 = aten.transpose.int(view_default_5, 1, 2)
```

### RoPE position embedding

```python
_tensor_constant79 = self._tensor_constant79
index_tensor = aten.index.Tensor(_tensor_constant79, [arg2_1])
unsqueeze_default = aten.unsqueeze.default(index_tensor, 1)
_tensor_constant80 = self._tensor_constant80
index_tensor_1 = aten.index.Tensor(_tensor_constant80, [arg2_1])
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

### Attention output projection and residual

```python
view_default_14 = aten.view.default(view_default_12, [624, 4096])
_tensor_constant81 = self._tensor_constant81
mm_default_3 = aten.mm.default(view_default_14, _tensor_constant81)
_unsafe_view_default_3 = aten._unsafe_view.default(mm_default_3, [1, 624, 4096])
add_tensor_4 = aten.add.Tensor(arg0_1, _unsafe_view_default_3)
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

### MLP and final residual

```python
_tensor_constant82 = self._tensor_constant82
mm_default_4 = aten.mm.default(view_default_15, _tensor_constant82)
_unsafe_view_default_4 = aten._unsafe_view.default(mm_default_4, [1, 624, 11008])
silu_default = aten.silu.default(_unsafe_view_default_4)
view_default_16 = aten.view.default(mul_tensor_8, [624, 4096])
_tensor_constant83 = self._tensor_constant83
mm_default_5 = aten.mm.default(view_default_16, _tensor_constant83)
_unsafe_view_default_5 = aten._unsafe_view.default(mm_default_5, [1, 624, 11008])
mul_tensor_9 = aten.mul.Tensor(silu_default, _unsafe_view_default_5)
view_default_17 = aten.view.default(mul_tensor_9, [624, 11008])
_tensor_constant84 = self._tensor_constant84
mm_default_6 = aten.mm.default(view_default_17, _tensor_constant84)
_unsafe_view_default_6 = aten._unsafe_view.default(mm_default_6, [1, 624, 4096])
add_tensor_6 = aten.add.Tensor(add_tensor_4, _unsafe_view_default_6)
```

### Layer output

```python
return (add_tensor_6, {'dynamic_cache_layer': (add_tensor_2, transpose_int_2)}, None, 0)
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
| 18 | `qkv_projection` | `_tensor_constant76` | `get_attr` | `_tensor_constant76` | - | `mm_default` |
| 19 | `qkv_projection` | `mm_default` | `call_function` | `aten.mm.default` | `view_default`, `_tensor_constant76` | `_unsafe_view_default` |
| 20 | `qkv_projection` | `_unsafe_view_default` | `call_function` | `aten._unsafe_view.default` | `mm_default` | `view_default_3` |
| 21 | `qkv_projection` | `view_default_1` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_1` |
| 22 | `qkv_projection` | `_tensor_constant77` | `get_attr` | `_tensor_constant77` | - | `mm_default_1` |
| 23 | `qkv_projection` | `mm_default_1` | `call_function` | `aten.mm.default` | `view_default_1`, `_tensor_constant77` | `_unsafe_view_default_1` |
| 24 | `qkv_projection` | `_unsafe_view_default_1` | `call_function` | `aten._unsafe_view.default` | `mm_default_1` | `view_default_4` |
| 25 | `qkv_projection` | `view_default_2` | `call_function` | `aten.view.default` | `mul_tensor_1` | `mm_default_2` |
| 26 | `qkv_projection` | `_tensor_constant78` | `get_attr` | `_tensor_constant78` | - | `mm_default_2` |
| 27 | `qkv_projection` | `mm_default_2` | `call_function` | `aten.mm.default` | `view_default_2`, `_tensor_constant78` | `_unsafe_view_default_2` |
| 28 | `qkv_projection` | `_unsafe_view_default_2` | `call_function` | `aten._unsafe_view.default` | `mm_default_2` | `view_default_5` |
| 29 | `qkv_projection` | `view_default_3` | `call_function` | `aten.view.default` | `_unsafe_view_default` | `transpose_int` |
| 30 | `qkv_projection` | `transpose_int` | `call_function` | `aten.transpose.int` | `view_default_3` | `mul_tensor_2`, `slice_tensor`, `slice_tensor_1` |
| 31 | `qkv_projection` | `view_default_4` | `call_function` | `aten.view.default` | `_unsafe_view_default_1` | `transpose_int_1` |
| 32 | `qkv_projection` | `transpose_int_1` | `call_function` | `aten.transpose.int` | `view_default_4` | `mul_tensor_4`, `slice_tensor_2`, `slice_tensor_3` |
| 33 | `qkv_projection` | `view_default_5` | `call_function` | `aten.view.default` | `_unsafe_view_default_2` | `transpose_int_2` |
| 34 | `qkv_projection` | `transpose_int_2` | `call_function` | `aten.transpose.int` | `view_default_5` | `expand_default_3`, `mul_tensor_6`, `output` |
| 35 | `rope` | `_tensor_constant79` | `get_attr` | `_tensor_constant79` | - | `index_tensor` |
| 36 | `rope` | `index_tensor` | `call_function` | `aten.index.Tensor` | `_tensor_constant79`, `arg2_1` | `unsqueeze_default` |
| 37 | `rope` | `unsqueeze_default` | `call_function` | `aten.unsqueeze.default` | `index_tensor` | `mul_tensor_2`, `mul_tensor_4` |
| 38 | `rope` | `_tensor_constant80` | `get_attr` | `_tensor_constant80` | - | `index_tensor_1` |
| 39 | `rope` | `index_tensor_1` | `call_function` | `aten.index.Tensor` | `_tensor_constant80`, `arg2_1` | `unsqueeze_default_1` |
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
| 87 | `output_projection` | `_tensor_constant81` | `get_attr` | `_tensor_constant81` | - | `mm_default_3` |
| 88 | `output_projection` | `mm_default_3` | `call_function` | `aten.mm.default` | `view_default_14`, `_tensor_constant81` | `_unsafe_view_default_3` |
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
| 101 | `mlp` | `_tensor_constant82` | `get_attr` | `_tensor_constant82` | - | `mm_default_4` |
| 102 | `mlp` | `mm_default_4` | `call_function` | `aten.mm.default` | `view_default_15`, `_tensor_constant82` | `_unsafe_view_default_4` |
| 103 | `mlp` | `_unsafe_view_default_4` | `call_function` | `aten._unsafe_view.default` | `mm_default_4` | `silu_default` |
| 104 | `mlp` | `silu_default` | `call_function` | `aten.silu.default` | `_unsafe_view_default_4` | `mul_tensor_9` |
| 105 | `mlp` | `view_default_16` | `call_function` | `aten.view.default` | `mul_tensor_8` | `mm_default_5` |
| 106 | `mlp` | `_tensor_constant83` | `get_attr` | `_tensor_constant83` | - | `mm_default_5` |
| 107 | `mlp` | `mm_default_5` | `call_function` | `aten.mm.default` | `view_default_16`, `_tensor_constant83` | `_unsafe_view_default_5` |
| 108 | `mlp` | `_unsafe_view_default_5` | `call_function` | `aten._unsafe_view.default` | `mm_default_5` | `mul_tensor_9` |
| 109 | `mlp` | `mul_tensor_9` | `call_function` | `aten.mul.Tensor` | `silu_default`, `_unsafe_view_default_5` | `view_default_17` |
| 110 | `mlp` | `view_default_17` | `call_function` | `aten.view.default` | `mul_tensor_9` | `mm_default_6` |
| 111 | `mlp` | `_tensor_constant84` | `get_attr` | `_tensor_constant84` | - | `mm_default_6` |
| 112 | `mlp` | `mm_default_6` | `call_function` | `aten.mm.default` | `view_default_17`, `_tensor_constant84` | `_unsafe_view_default_6` |
| 113 | `mlp` | `_unsafe_view_default_6` | `call_function` | `aten._unsafe_view.default` | `mm_default_6` | `add_tensor_6` |
| 114 | `mlp` | `add_tensor_6` | `call_function` | `aten.add.Tensor` | `add_tensor_4`, `_unsafe_view_default_6` | `output` |
| 115 | `layer_output` | `output` | `output` | `output` | `add_tensor_6`, `add_tensor_2`, `transpose_int_2` | - |
