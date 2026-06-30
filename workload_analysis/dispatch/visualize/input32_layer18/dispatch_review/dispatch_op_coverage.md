# input32_layer18 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002830` | `t00002831` |
| 2 | `pow.Tensor_Scalar` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002831` | `t00002832` |
| 3 | `mean.dim` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002832` | `t00002833` |
| 4 | `add.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002833` | `t00002834` |
| 5 | `rsqrt.default` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002834` | `t00002835` |
| 6 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002831, t00002835` | `t00002836` |
| 7 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002836` | `t00002837` |
| 8 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001354, t00002837` | `t00002838` |
| 9 | `linear.default` | `model.layers.18.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002838, t00001356` | `t00002839` |
| 10 | `linear.default` | `model.layers.18.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002838, t00001358` | `t00002840` |
| 11 | `linear.default` | `model.layers.18.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002838, t00001360` | `t00002841` |
| 12 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002839` | `t00002842` |
| 13 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002842` | `t00002843` |
| 14 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002840` | `t00002844` |
| 15 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002844` | `t00002845` |
| 16 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002841` | `t00002846` |
| 17 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002846` | `t00002847` |
| 18 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002848` | `t00002849` |
| 19 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002849` | `t00002850` |
| 20 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002850` | `t00002851` |
| 21 | `gt.Scalar` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002851` | `t00002852` |
| 22 | `is_nonzero.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002852` | `` |
| 23 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002851` | `` |
| 24 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001372` | `t00002853` |
| 25 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002853` | `t00002853` |
| 26 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002851` | `` |
| 27 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001374` | `t00002854` |
| 28 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002854` | `t00002854` |
| 29 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002853, t00002848` | `t00002855` |
| 30 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002855` | `t00002856` |
| 31 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002854, t00002848` | `t00002857` |
| 32 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002857` | `t00002858` |
| 33 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002843, t00002856` | `t00002859` |
| 34 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002843` | `t00002860` |
| 35 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002843` | `t00002861` |
| 36 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002861` | `t00002862` |
| 37 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002862, t00002860` | `t00002863` |
| 38 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002863, t00002858` | `t00002864` |
| 39 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope, attention` | `t00002859, t00002864` | `t00002865` |
| 40 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002845, t00002856` | `t00002866` |
| 41 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002845` | `t00002867` |
| 42 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002845` | `t00002868` |
| 43 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002868` | `t00002869` |
| 44 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002869, t00002867` | `t00002870` |
| 45 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002870, t00002858` | `t00002871` |
| 46 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002866, t00002871` | `t00002872` |
| 47 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002873, t00002872` | `t00002874` |
| 48 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002875, t00002847` | `t00002876` |
| 49 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002874` | `t00002877` |
| 50 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002865, t00002877` | `t00002878` |
| 51 | `div.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002878` | `t00002879` |
| 52 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002879, t00002880` | `t00002881` |
| 53 | `softmax.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002881` | `t00002882` |
| 54 | `to.dtype` | `model.layers.18.self_attn` | `True` | `True` | `mlp` | `t00002882` | `t00002883` |
| 55 | `dropout.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002883` | `t00002883` |
| 56 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention, attention_output` | `t00002883, t00002876` | `t00002884` |
| 57 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002884` | `t00002885` |
| 58 | `reshape.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` | `t00002885` | `t00002886` |
| 59 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00000057` | `t00002887` |
| 60 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002887` | `` |
| 61 | `linear.default` | `model.layers.18.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002886, t00001432` | `t00002888` |
| 62 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output, mlp` | `t00002830, t00002888` | `t00002889` |
| 63 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002889` | `t00002890` |
| 64 | `pow.Tensor_Scalar` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002890` | `t00002891` |
| 65 | `mean.dim` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002891` | `t00002892` |
| 66 | `add.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002892` | `t00002893` |
| 67 | `rsqrt.default` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002893` | `t00002894` |
| 68 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002890, t00002894` | `t00002895` |
| 69 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002895` | `t00002896` |
| 70 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001442, t00002896` | `t00002897` |
| 71 | `linear.default` | `model.layers.18.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002897, t00001444` | `t00002898` |
| 72 | `silu.default` | `model.layers.18.mlp.act_fn` | `True` | `True` | `mlp` | `t00002898` | `t00002899` |
| 73 | `linear.default` | `model.layers.18.mlp.up_proj` | `True` | `True` | `mlp` | `t00002897, t00001447` | `t00002900` |
| 74 | `mul.Tensor` | `model.layers.18.mlp` | `True` | `True` | `` | `t00002899, t00002900` | `t00002901` |
| 75 | `linear.default` | `model.layers.18.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002901, t00001450` | `t00002902` |
| 76 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output` | `t00002889, t00002902` | `t00002903` |
