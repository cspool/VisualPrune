# input2_layer27 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002605` | `t00002606` |
| 2 | `pow.Tensor_Scalar` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002606` | `t00002607` |
| 3 | `mean.dim` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002607` | `t00002608` |
| 4 | `add.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002608` | `t00002609` |
| 5 | `rsqrt.default` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002609` | `t00002610` |
| 6 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002606, t00002610` | `t00002611` |
| 7 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002611` | `t00002612` |
| 8 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002279, t00002612` | `t00002613` |
| 9 | `linear.default` | `model.layers.27.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002613, t00002281` | `t00002614` |
| 10 | `linear.default` | `model.layers.27.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002613, t00002283` | `t00002615` |
| 11 | `linear.default` | `model.layers.27.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002613, t00002285` | `t00002616` |
| 12 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002614` | `t00002617` |
| 13 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002617` | `t00002618` |
| 14 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002615` | `t00002619` |
| 15 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002619` | `t00002620` |
| 16 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002616` | `t00002621` |
| 17 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002621` | `t00002622` |
| 18 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002481` | `t00002623` |
| 19 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002623` | `t00002624` |
| 20 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002624` | `t00002625` |
| 21 | `gt.Scalar` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002625` | `t00002626` |
| 22 | `is_nonzero.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002626` | `` |
| 23 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002625` | `` |
| 24 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002297` | `t00002627` |
| 25 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002627` | `t00002627` |
| 26 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002625` | `` |
| 27 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002299` | `t00002628` |
| 28 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002628` | `t00002628` |
| 29 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002627, t00002481` | `t00002629` |
| 30 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002629` | `t00002630` |
| 31 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002628, t00002481` | `t00002631` |
| 32 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002631` | `t00002632` |
| 33 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002618, t00002630` | `t00002633` |
| 34 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002618` | `t00002634` |
| 35 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002618` | `t00002635` |
| 36 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002635` | `t00002636` |
| 37 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002636, t00002634` | `t00002637` |
| 38 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002637, t00002632` | `t00002638` |
| 39 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope, attention` | `t00002633, t00002638` | `t00002639` |
| 40 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002620, t00002630` | `t00002640` |
| 41 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002620` | `t00002641` |
| 42 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002620` | `t00002642` |
| 43 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002642` | `t00002643` |
| 44 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002643, t00002641` | `t00002644` |
| 45 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002644, t00002632` | `t00002645` |
| 46 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002640, t00002645` | `t00002646` |
| 47 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002318, t00002646` | `t00002647` |
| 48 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002292, t00002622` | `t00002648` |
| 49 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002647` | `t00002649` |
| 50 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002639, t00002649` | `t00002650` |
| 51 | `div.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002650` | `t00002651` |
| 52 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002651, t00002652` | `t00002653` |
| 53 | `softmax.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002653` | `t00002654` |
| 54 | `to.dtype` | `model.layers.27.self_attn` | `True` | `True` | `mlp` | `t00002654` | `t00002655` |
| 55 | `dropout.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002655` | `t00002655` |
| 56 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention, attention_output` | `t00002655, t00002648` | `t00002656` |
| 57 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002656` | `t00002657` |
| 58 | `reshape.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` | `t00002657` | `t00002658` |
| 59 | `gt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00000057` | `t00002659` |
| 60 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002659` | `` |
| 61 | `linear.default` | `model.layers.27.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002658, t00002353` | `t00002660` |
| 62 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output, mlp` | `t00002605, t00002660` | `t00002661` |
| 63 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002661` | `t00002662` |
| 64 | `pow.Tensor_Scalar` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002662` | `t00002663` |
| 65 | `mean.dim` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002663` | `t00002664` |
| 66 | `add.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002664` | `t00002665` |
| 67 | `rsqrt.default` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002665` | `t00002666` |
| 68 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002662, t00002666` | `t00002667` |
| 69 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002667` | `t00002668` |
| 70 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002363, t00002668` | `t00002669` |
| 71 | `linear.default` | `model.layers.27.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002669, t00002365` | `t00002670` |
| 72 | `silu.default` | `model.layers.27.mlp.act_fn` | `True` | `True` | `mlp` | `t00002670` | `t00002671` |
| 73 | `linear.default` | `model.layers.27.mlp.up_proj` | `True` | `True` | `mlp` | `t00002669, t00002368` | `t00002672` |
| 74 | `mul.Tensor` | `model.layers.27.mlp` | `True` | `True` | `` | `t00002671, t00002672` | `t00002673` |
| 75 | `linear.default` | `model.layers.27.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002673, t00002371` | `t00002674` |
| 76 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output` | `t00002661, t00002674` | `t00002675` |
