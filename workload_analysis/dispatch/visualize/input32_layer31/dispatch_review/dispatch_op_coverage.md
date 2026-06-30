# input32_layer31 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003121` | `t00003122` |
| 2 | `pow.Tensor_Scalar` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003122` | `t00003123` |
| 3 | `mean.dim` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003123` | `t00003124` |
| 4 | `add.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003124` | `t00003125` |
| 5 | `rsqrt.default` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003125` | `t00003126` |
| 6 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003122, t00003126` | `t00003127` |
| 7 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003127` | `t00003128` |
| 8 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002754, t00003128` | `t00003129` |
| 9 | `linear.default` | `model.layers.31.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00003129, t00002756` | `t00003130` |
| 10 | `linear.default` | `model.layers.31.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00003129, t00002758` | `t00003131` |
| 11 | `linear.default` | `model.layers.31.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00003129, t00002760` | `t00003132` |
| 12 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00003130` | `t00003133` |
| 13 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003133` | `t00003134` |
| 14 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00003131` | `t00003135` |
| 15 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003135` | `t00003136` |
| 16 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00003132` | `t00003137` |
| 17 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003137` | `t00003138` |
| 18 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002848` | `t00003139` |
| 19 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003139` | `t00003140` |
| 20 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003140` | `t00003141` |
| 21 | `gt.Scalar` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003141` | `t00003142` |
| 22 | `is_nonzero.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003142` | `` |
| 23 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003141` | `` |
| 24 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002772` | `t00003143` |
| 25 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003143` | `t00003143` |
| 26 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003141` | `` |
| 27 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002774` | `t00003144` |
| 28 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00003144` | `t00003144` |
| 29 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003143, t00002848` | `t00003145` |
| 30 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003145` | `t00003146` |
| 31 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003144, t00002848` | `t00003147` |
| 32 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003147` | `t00003148` |
| 33 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003134, t00003146` | `t00003149` |
| 34 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003134` | `t00003150` |
| 35 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003134` | `t00003151` |
| 36 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003151` | `t00003152` |
| 37 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003152, t00003150` | `t00003153` |
| 38 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00003153, t00003148` | `t00003154` |
| 39 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope, attention` | `t00003149, t00003154` | `t00003155` |
| 40 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003136, t00003146` | `t00003156` |
| 41 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003136` | `t00003157` |
| 42 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003136` | `t00003158` |
| 43 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003158` | `t00003159` |
| 44 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003159, t00003157` | `t00003160` |
| 45 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003160, t00003148` | `t00003161` |
| 46 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003156, t00003161` | `t00003162` |
| 47 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003163, t00003162` | `t00003164` |
| 48 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003165, t00003138` | `t00003166` |
| 49 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003164` | `t00003167` |
| 50 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003155, t00003167` | `t00003168` |
| 51 | `div.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003168` | `t00003169` |
| 52 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003169, t00003170` | `t00003171` |
| 53 | `softmax.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003171` | `t00003172` |
| 54 | `to.dtype` | `model.layers.31.self_attn` | `True` | `True` | `mlp` | `t00003172` | `t00003173` |
| 55 | `dropout.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00003173` | `t00003173` |
| 56 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention, attention_output` | `t00003173, t00003166` | `t00003174` |
| 57 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003174` | `t00003175` |
| 58 | `reshape.default` | `model.layers.31.self_attn` | `True` | `True` | `attention_output` | `t00003175` | `t00003176` |
| 59 | `gt.Scalar` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00000057` | `t00003177` |
| 60 | `is_nonzero.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00003177` | `` |
| 61 | `linear.default` | `model.layers.31.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00003176, t00002809` | `t00003178` |
| 62 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output, mlp` | `t00003121, t00003178` | `t00003179` |
| 63 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003179` | `t00003180` |
| 64 | `pow.Tensor_Scalar` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003180` | `t00003181` |
| 65 | `mean.dim` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003181` | `t00003182` |
| 66 | `add.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003182` | `t00003183` |
| 67 | `rsqrt.default` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003183` | `t00003184` |
| 68 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003180, t00003184` | `t00003185` |
| 69 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003185` | `t00003186` |
| 70 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002819, t00003186` | `t00003187` |
| 71 | `linear.default` | `model.layers.31.mlp.gate_proj` | `True` | `True` | `mlp` | `t00003187, t00002821` | `t00003188` |
| 72 | `silu.default` | `model.layers.31.mlp.act_fn` | `True` | `True` | `mlp` | `t00003188` | `t00003189` |
| 73 | `linear.default` | `model.layers.31.mlp.up_proj` | `True` | `True` | `mlp` | `t00003187, t00002824` | `t00003190` |
| 74 | `mul.Tensor` | `model.layers.31.mlp` | `True` | `True` | `` | `t00003189, t00003190` | `t00003191` |
| 75 | `linear.default` | `model.layers.31.mlp.down_proj` | `True` | `True` | `attention_output` | `t00003191, t00002827` | `t00003192` |
| 76 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output` | `t00003179, t00003192` | `t00003193` |
