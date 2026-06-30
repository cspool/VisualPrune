# input2_layer19 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002534` | `t00002535` |
| 2 | `pow.Tensor_Scalar` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002535` | `t00002536` |
| 3 | `mean.dim` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002536` | `t00002537` |
| 4 | `add.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002537` | `t00002538` |
| 5 | `rsqrt.default` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002538` | `t00002539` |
| 6 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002535, t00002539` | `t00002540` |
| 7 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002540` | `t00002541` |
| 8 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001461, t00002541` | `t00002542` |
| 9 | `linear.default` | `model.layers.19.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002542, t00001463` | `t00002543` |
| 10 | `linear.default` | `model.layers.19.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002542, t00001465` | `t00002544` |
| 11 | `linear.default` | `model.layers.19.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002542, t00001467` | `t00002545` |
| 12 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002543` | `t00002546` |
| 13 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002546` | `t00002547` |
| 14 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002544` | `t00002548` |
| 15 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002548` | `t00002549` |
| 16 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00002545` | `t00002550` |
| 17 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002550` | `t00002551` |
| 18 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002481` | `t00002552` |
| 19 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002552` | `t00002553` |
| 20 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002553` | `t00002554` |
| 21 | `gt.Scalar` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002554` | `t00002555` |
| 22 | `is_nonzero.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002555` | `` |
| 23 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002554` | `` |
| 24 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001480` | `t00002556` |
| 25 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002556` | `t00002556` |
| 26 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002554` | `` |
| 27 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001482` | `t00002557` |
| 28 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00002557` | `t00002557` |
| 29 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002556, t00002481` | `t00002558` |
| 30 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002558` | `t00002559` |
| 31 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002557, t00002481` | `t00002560` |
| 32 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002560` | `t00002561` |
| 33 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002547, t00002559` | `t00002562` |
| 34 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002547` | `t00002563` |
| 35 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002547` | `t00002564` |
| 36 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002564` | `t00002565` |
| 37 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002565, t00002563` | `t00002566` |
| 38 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00002566, t00002561` | `t00002567` |
| 39 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope, attention` | `t00002562, t00002567` | `t00002568` |
| 40 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002549, t00002559` | `t00002569` |
| 41 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002549` | `t00002570` |
| 42 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002549` | `t00002571` |
| 43 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002571` | `t00002572` |
| 44 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002572, t00002570` | `t00002573` |
| 45 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002573, t00002561` | `t00002574` |
| 46 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002569, t00002574` | `t00002575` |
| 47 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` | `t00001501, t00002575` | `t00002576` |
| 48 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` | `t00001474, t00002551` | `t00002577` |
| 49 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002576` | `t00002578` |
| 50 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002568, t00002578` | `t00002579` |
| 51 | `div.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002579` | `t00002580` |
| 52 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002580, t00002581` | `t00002582` |
| 53 | `softmax.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002582` | `t00002583` |
| 54 | `to.dtype` | `model.layers.19.self_attn` | `True` | `True` | `mlp` | `t00002583` | `t00002584` |
| 55 | `dropout.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00002584` | `t00002584` |
| 56 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention, attention_output` | `t00002584, t00002577` | `t00002585` |
| 57 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002585` | `t00002586` |
| 58 | `reshape.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` | `t00002586` | `t00002587` |
| 59 | `gt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00000057` | `t00002588` |
| 60 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00002588` | `` |
| 61 | `linear.default` | `model.layers.19.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002587, t00001537` | `t00002589` |
| 62 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output, mlp` | `t00002534, t00002589` | `t00002590` |
| 63 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002590` | `t00002591` |
| 64 | `pow.Tensor_Scalar` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002591` | `t00002592` |
| 65 | `mean.dim` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002592` | `t00002593` |
| 66 | `add.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002593` | `t00002594` |
| 67 | `rsqrt.default` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002594` | `t00002595` |
| 68 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002591, t00002595` | `t00002596` |
| 69 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002596` | `t00002597` |
| 70 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001547, t00002597` | `t00002598` |
| 71 | `linear.default` | `model.layers.19.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002598, t00001549` | `t00002599` |
| 72 | `silu.default` | `model.layers.19.mlp.act_fn` | `True` | `True` | `mlp` | `t00002599` | `t00002600` |
| 73 | `linear.default` | `model.layers.19.mlp.up_proj` | `True` | `True` | `mlp` | `t00002598, t00001552` | `t00002601` |
| 74 | `mul.Tensor` | `model.layers.19.mlp` | `True` | `True` | `` | `t00002600, t00002601` | `t00002602` |
| 75 | `linear.default` | `model.layers.19.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002602, t00001555` | `t00002603` |
| 76 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output` | `t00002590, t00002603` | `t00002604` |
