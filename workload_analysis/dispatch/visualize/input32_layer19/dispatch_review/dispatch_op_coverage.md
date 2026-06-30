# input32_layer19 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002903` | `t00002904` |
| 2 | `pow.Tensor_Scalar` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002904` | `t00002905` |
| 3 | `mean.dim` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002905` | `t00002906` |
| 4 | `add.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002906` | `t00002907` |
| 5 | `rsqrt.default` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002907` | `t00002908` |
| 6 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002904, t00002908` | `t00002909` |
| 7 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002909` | `t00002910` |
| 8 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001461, t00002910` | `t00002911` |
| 9 | `linear.default` | `model.layers.19.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002911, t00001463` | `t00002912` |
| 10 | `linear.default` | `model.layers.19.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002911, t00001465` | `t00002913` |
| 11 | `linear.default` | `model.layers.19.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002911, t00001467` | `t00002914` |
| 12 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002912` | `t00002915` |
| 13 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002915` | `t00002916` |
| 14 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002913` | `t00002917` |
| 15 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002917` | `t00002918` |
| 16 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002914` | `t00002919` |
| 17 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002919` | `t00002920` |
| 18 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002848` | `t00002921` |
| 19 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002921` | `t00002922` |
| 20 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002922` | `t00002923` |
| 21 | `gt.Scalar` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002923` | `t00002924` |
| 22 | `is_nonzero.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002924` | `` |
| 23 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002923` | `` |
| 24 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001480` | `t00002925` |
| 25 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002925` | `t00002925` |
| 26 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002923` | `` |
| 27 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001482` | `t00002926` |
| 28 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002926` | `t00002926` |
| 29 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002925, t00002848` | `t00002927` |
| 30 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002927` | `t00002928` |
| 31 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002926, t00002848` | `t00002929` |
| 32 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002929` | `t00002930` |
| 33 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002916, t00002928` | `t00002931` |
| 34 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002916` | `t00002932` |
| 35 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002916` | `t00002933` |
| 36 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002933` | `t00002934` |
| 37 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002934, t00002932` | `t00002935` |
| 38 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002935, t00002930` | `t00002936` |
| 39 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope, attention` | `t00002931, t00002936` | `t00002937` |
| 40 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002918, t00002928` | `t00002938` |
| 41 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002918` | `t00002939` |
| 42 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002918` | `t00002940` |
| 43 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002940` | `t00002941` |
| 44 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002941, t00002939` | `t00002942` |
| 45 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002942, t00002930` | `t00002943` |
| 46 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002938, t00002943` | `t00002944` |
| 47 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002945, t00002944` | `t00002946` |
| 48 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002947, t00002920` | `t00002948` |
| 49 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002946` | `t00002949` |
| 50 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002937, t00002949` | `t00002950` |
| 51 | `div.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002950` | `t00002951` |
| 52 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002951, t00002952` | `t00002953` |
| 53 | `softmax.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002953` | `t00002954` |
| 54 | `to.dtype` | `model.layers.19.self_attn` | `True` | `True` | `mlp` | `t00002954` | `t00002955` |
| 55 | `dropout.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002955` | `t00002955` |
| 56 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention, attention_output` | `t00002955, t00002948` | `t00002956` |
| 57 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002956` | `t00002957` |
| 58 | `reshape.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` | `t00002957` | `t00002958` |
| 59 | `gt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00000057` | `t00002959` |
| 60 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002959` | `` |
| 61 | `linear.default` | `model.layers.19.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002958, t00001537` | `t00002960` |
| 62 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output, mlp` | `t00002903, t00002960` | `t00002961` |
| 63 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002961` | `t00002962` |
| 64 | `pow.Tensor_Scalar` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002962` | `t00002963` |
| 65 | `mean.dim` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002963` | `t00002964` |
| 66 | `add.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002964` | `t00002965` |
| 67 | `rsqrt.default` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002965` | `t00002966` |
| 68 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002962, t00002966` | `t00002967` |
| 69 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002967` | `t00002968` |
| 70 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001547, t00002968` | `t00002969` |
| 71 | `linear.default` | `model.layers.19.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002969, t00001549` | `t00002970` |
| 72 | `silu.default` | `model.layers.19.mlp.act_fn` | `True` | `True` | `mlp` | `t00002970` | `t00002971` |
| 73 | `linear.default` | `model.layers.19.mlp.up_proj` | `True` | `True` | `mlp` | `t00002969, t00001552` | `t00002972` |
| 74 | `mul.Tensor` | `model.layers.19.mlp` | `True` | `True` | `` | `t00002971, t00002972` | `t00002973` |
| 75 | `linear.default` | `model.layers.19.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002973, t00001555` | `t00002974` |
| 76 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output` | `t00002961, t00002974` | `t00002975` |
