# input1_layer16 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001148` | `t00001149` |
| 2 | `pow.Tensor_Scalar` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001149` | `t00001150` |
| 3 | `mean.dim` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001150` | `t00001151` |
| 4 | `add.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001151` | `t00001152` |
| 5 | `rsqrt.default` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001152` | `t00001153` |
| 6 | `mul.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001149, t00001153` | `t00001154` |
| 7 | `to.dtype` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001154` | `t00001155` |
| 8 | `mul.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001156, t00001155` | `t00001157` |
| 9 | `linear.default` | `model.layers.16.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001157, t00001158` | `t00001159` |
| 10 | `linear.default` | `model.layers.16.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001157, t00001160` | `t00001161` |
| 11 | `linear.default` | `model.layers.16.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001157, t00001162` | `t00001163` |
| 12 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` | `t00001159` | `t00001164` |
| 13 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001164` | `t00001165` |
| 14 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` | `t00001161` | `t00001166` |
| 15 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001166` | `t00001167` |
| 16 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` | `t00001163` | `t00001168` |
| 17 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001168` | `t00001169` |
| 18 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00000023` | `t00001170` |
| 19 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001170` | `t00001171` |
| 20 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001171` | `t00001172` |
| 21 | `gt.Scalar` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001172` | `t00001173` |
| 22 | `is_nonzero.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001173` | `` |
| 23 | `item.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` | `t00001172` | `` |
| 24 | `slice.Tensor` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001174` | `t00001175` |
| 25 | `to.dtype` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` | `t00001175` | `t00001175` |
| 26 | `item.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` | `t00001172` | `` |
| 27 | `slice.Tensor` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001176` | `t00001177` |
| 28 | `to.dtype` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` | `t00001177` | `t00001177` |
| 29 | `index.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001175, t00000023` | `t00001178` |
| 30 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001178` | `t00001179` |
| 31 | `index.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001177, t00000023` | `t00001180` |
| 32 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001180` | `t00001181` |
| 33 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001165, t00001179` | `t00001182` |
| 34 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001165` | `t00001183` |
| 35 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001165` | `t00001184` |
| 36 | `neg.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001184` | `t00001185` |
| 37 | `cat.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001185, t00001183` | `t00001186` |
| 38 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` | `t00001186, t00001181` | `t00001187` |
| 39 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope, attention` | `t00001182, t00001187` | `t00001188` |
| 40 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001167, t00001179` | `t00001189` |
| 41 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001167` | `t00001190` |
| 42 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001167` | `t00001191` |
| 43 | `neg.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001191` | `t00001192` |
| 44 | `cat.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001192, t00001190` | `t00001193` |
| 45 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001193, t00001181` | `t00001194` |
| 46 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001189, t00001194` | `t00001195` |
| 47 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001195` | `t00001196` |
| 48 | `matmul.default` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001188, t00001196` | `t00001197` |
| 49 | `div.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001197` | `t00001198` |
| 50 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001198, t00000053` | `t00001199` |
| 51 | `softmax.int` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001199` | `t00001200` |
| 52 | `to.dtype` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001200` | `t00001201` |
| 53 | `dropout.default` | `model.layers.16.self_attn` | `True` | `True` | `attention` | `t00001201` | `t00001201` |
| 54 | `matmul.default` | `model.layers.16.self_attn` | `True` | `True` | `attention, attention_output` | `t00001201, t00001169` | `t00001202` |
| 55 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001202` | `t00001203` |
| 56 | `contiguous.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` | `t00001203` | `t00001204` |
| 57 | `reshape.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` | `t00001204` | `t00001205` |
| 58 | `gt.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001206` |
| 59 | `is_nonzero.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001206` | `` |
| 60 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00000023` | `t00001207` |
| 61 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001207` | `t00001208` |
| 62 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001208` | `t00001209` |
| 63 | `eq.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001209` | `t00001210` |
| 64 | `is_nonzero.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001210` | `` |
| 65 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001205` | `t00001211` |
| 66 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001201` | `t00001212` |
| 67 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001212` | `t00001213` |
| 68 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001213, t00001169` | `t00001214` |
| 69 | `permute.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001214` | `t00001215` |
| 70 | `contiguous.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` | `t00001215` | `t00001216` |
| 71 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001216` | `t00001217` |
| 72 | `item.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001218` | `` |
| 73 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001217` | `t00001219` |
| 74 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001211` | `t00001220` |
| 75 | `sub.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001220, t00001219` | `t00001221` |
| 76 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001211` | `t00001222` |
| 77 | `cosine_similarity.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001221, t00001222` | `t00001223` |
| 78 | `squeeze.dim` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001223` | `t00001224` |
| 79 | `lt.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001224` | `t00001225` |
| 80 | `any.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001225` | `t00001226` |
| 81 | `item.default` | `model.layers.16.self_attn` | `True` | `True` | `` | `t00001226` | `` |
| 82 | `linear.default` | `model.layers.16.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001205, t00001227` | `t00001228` |
| 83 | `add.Tensor` | `model.layers.16` | `True` | `True` | `attention_output, mlp` | `t00001148, t00001228` | `t00001229` |
| 84 | `to.dtype` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001229` | `t00001230` |
| 85 | `pow.Tensor_Scalar` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001230` | `t00001231` |
| 86 | `mean.dim` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001231` | `t00001232` |
| 87 | `add.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001232` | `t00001233` |
| 88 | `rsqrt.default` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001233` | `t00001234` |
| 89 | `mul.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001230, t00001234` | `t00001235` |
| 90 | `to.dtype` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001235` | `t00001236` |
| 91 | `mul.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001237, t00001236` | `t00001238` |
| 92 | `linear.default` | `model.layers.16.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001238, t00001239` | `t00001240` |
| 93 | `silu.default` | `model.layers.16.mlp.act_fn` | `True` | `True` | `mlp` | `t00001240` | `t00001241` |
| 94 | `linear.default` | `model.layers.16.mlp.up_proj` | `True` | `True` | `mlp` | `t00001238, t00001242` | `t00001243` |
| 95 | `mul.Tensor` | `model.layers.16.mlp` | `True` | `True` | `mlp` | `t00001241, t00001243` | `t00001244` |
| 96 | `linear.default` | `model.layers.16.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001244, t00001245` | `t00001246` |
| 97 | `add.Tensor` | `model.layers.16` | `True` | `True` | `attention_output` | `t00001229, t00001246` | `t00001247` |
