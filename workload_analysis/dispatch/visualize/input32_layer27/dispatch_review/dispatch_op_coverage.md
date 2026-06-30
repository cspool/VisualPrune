# input32_layer27 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002976` | `t00002977` |
| 2 | `pow.Tensor_Scalar` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002977` | `t00002978` |
| 3 | `mean.dim` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002978` | `t00002979` |
| 4 | `add.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002979` | `t00002980` |
| 5 | `rsqrt.default` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002980` | `t00002981` |
| 6 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002977, t00002981` | `t00002982` |
| 7 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002982` | `t00002983` |
| 8 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002279, t00002983` | `t00002984` |
| 9 | `linear.default` | `model.layers.27.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002984, t00002281` | `t00002985` |
| 10 | `linear.default` | `model.layers.27.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002984, t00002283` | `t00002986` |
| 11 | `linear.default` | `model.layers.27.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002984, t00002285` | `t00002987` |
| 12 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002985` | `t00002988` |
| 13 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002988` | `t00002989` |
| 14 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002986` | `t00002990` |
| 15 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002990` | `t00002991` |
| 16 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002987` | `t00002992` |
| 17 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002992` | `t00002993` |
| 18 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002848` | `t00002994` |
| 19 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002994` | `t00002995` |
| 20 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002995` | `t00002996` |
| 21 | `gt.Scalar` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002996` | `t00002997` |
| 22 | `is_nonzero.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002997` | `` |
| 23 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002996` | `` |
| 24 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002297` | `t00002998` |
| 25 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002998` | `t00002998` |
| 26 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002996` | `` |
| 27 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002299` | `t00002999` |
| 28 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002999` | `t00002999` |
| 29 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002998, t00002848` | `t00003000` |
| 30 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00003000` | `t00003001` |
| 31 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002999, t00002848` | `t00003002` |
| 32 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00003002` | `t00003003` |
| 33 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002989, t00003001` | `t00003004` |
| 34 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002989` | `t00003005` |
| 35 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002989` | `t00003006` |
| 36 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00003006` | `t00003007` |
| 37 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00003007, t00003005` | `t00003008` |
| 38 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00003008, t00003003` | `t00003009` |
| 39 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope, attention` | `t00003004, t00003009` | `t00003010` |
| 40 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002991, t00003001` | `t00003011` |
| 41 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002991` | `t00003012` |
| 42 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002991` | `t00003013` |
| 43 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00003013` | `t00003014` |
| 44 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00003014, t00003012` | `t00003015` |
| 45 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00003015, t00003003` | `t00003016` |
| 46 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003011, t00003016` | `t00003017` |
| 47 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003018, t00003017` | `t00003019` |
| 48 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003020, t00002993` | `t00003021` |
| 49 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003019` | `t00003022` |
| 50 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003010, t00003022` | `t00003023` |
| 51 | `div.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003023` | `t00003024` |
| 52 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003024, t00003025` | `t00003026` |
| 53 | `softmax.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003026` | `t00003027` |
| 54 | `to.dtype` | `model.layers.27.self_attn` | `True` | `True` | `mlp` | `t00003027` | `t00003028` |
| 55 | `dropout.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00003028` | `t00003028` |
| 56 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention, attention_output` | `t00003028, t00003021` | `t00003029` |
| 57 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00003029` | `t00003030` |
| 58 | `reshape.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` | `t00003030` | `t00003031` |
| 59 | `gt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00000057` | `t00003032` |
| 60 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00003032` | `` |
| 61 | `linear.default` | `model.layers.27.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00003031, t00002353` | `t00003033` |
| 62 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output, mlp` | `t00002976, t00003033` | `t00003034` |
| 63 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003034` | `t00003035` |
| 64 | `pow.Tensor_Scalar` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003035` | `t00003036` |
| 65 | `mean.dim` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003036` | `t00003037` |
| 66 | `add.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003037` | `t00003038` |
| 67 | `rsqrt.default` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003038` | `t00003039` |
| 68 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003035, t00003039` | `t00003040` |
| 69 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003040` | `t00003041` |
| 70 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002363, t00003041` | `t00003042` |
| 71 | `linear.default` | `model.layers.27.mlp.gate_proj` | `True` | `True` | `mlp` | `t00003042, t00002365` | `t00003043` |
| 72 | `silu.default` | `model.layers.27.mlp.act_fn` | `True` | `True` | `mlp` | `t00003043` | `t00003044` |
| 73 | `linear.default` | `model.layers.27.mlp.up_proj` | `True` | `True` | `mlp` | `t00003042, t00002368` | `t00003045` |
| 74 | `mul.Tensor` | `model.layers.27.mlp` | `True` | `True` | `` | `t00003044, t00003045` | `t00003046` |
| 75 | `linear.default` | `model.layers.27.mlp.down_proj` | `True` | `True` | `attention_output` | `t00003046, t00002371` | `t00003047` |
| 76 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output` | `t00003034, t00003047` | `t00003048` |
