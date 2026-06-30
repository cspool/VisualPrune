# input1_layer10 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000554` | `t00000555` |
| 2 | `pow.Tensor_Scalar` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000555` | `t00000556` |
| 3 | `mean.dim` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000556` | `t00000557` |
| 4 | `add.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000557` | `t00000558` |
| 5 | `rsqrt.default` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000558` | `t00000559` |
| 6 | `mul.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000555, t00000559` | `t00000560` |
| 7 | `to.dtype` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000560` | `t00000561` |
| 8 | `mul.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000562, t00000561` | `t00000563` |
| 9 | `linear.default` | `model.layers.10.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000563, t00000564` | `t00000565` |
| 10 | `linear.default` | `model.layers.10.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000563, t00000566` | `t00000567` |
| 11 | `linear.default` | `model.layers.10.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000563, t00000568` | `t00000569` |
| 12 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` | `t00000565` | `t00000570` |
| 13 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000570` | `t00000571` |
| 14 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` | `t00000567` | `t00000572` |
| 15 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000572` | `t00000573` |
| 16 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` | `t00000569` | `t00000574` |
| 17 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000574` | `t00000575` |
| 18 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000023` | `t00000576` |
| 19 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000576` | `t00000577` |
| 20 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000577` | `t00000578` |
| 21 | `gt.Scalar` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000578` | `t00000579` |
| 22 | `is_nonzero.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000579` | `` |
| 23 | `item.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` | `t00000578` | `` |
| 24 | `slice.Tensor` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000580` | `t00000581` |
| 25 | `to.dtype` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` | `t00000581` | `t00000581` |
| 26 | `item.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` | `t00000578` | `` |
| 27 | `slice.Tensor` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000582` | `t00000583` |
| 28 | `to.dtype` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` | `t00000583` | `t00000583` |
| 29 | `index.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000581, t00000023` | `t00000584` |
| 30 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000584` | `t00000585` |
| 31 | `index.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000583, t00000023` | `t00000586` |
| 32 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000586` | `t00000587` |
| 33 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000571, t00000585` | `t00000588` |
| 34 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000571` | `t00000589` |
| 35 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000571` | `t00000590` |
| 36 | `neg.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000590` | `t00000591` |
| 37 | `cat.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000591, t00000589` | `t00000592` |
| 38 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` | `t00000592, t00000587` | `t00000593` |
| 39 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope, attention` | `t00000588, t00000593` | `t00000594` |
| 40 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000573, t00000585` | `t00000595` |
| 41 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000573` | `t00000596` |
| 42 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000573` | `t00000597` |
| 43 | `neg.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000597` | `t00000598` |
| 44 | `cat.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000598, t00000596` | `t00000599` |
| 45 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000599, t00000587` | `t00000600` |
| 46 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000595, t00000600` | `t00000601` |
| 47 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000601` | `t00000602` |
| 48 | `matmul.default` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000594, t00000602` | `t00000603` |
| 49 | `div.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000603` | `t00000604` |
| 50 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000604, t00000053` | `t00000605` |
| 51 | `softmax.int` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000605` | `t00000606` |
| 52 | `to.dtype` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000606` | `t00000607` |
| 53 | `dropout.default` | `model.layers.10.self_attn` | `True` | `True` | `attention` | `t00000607` | `t00000607` |
| 54 | `matmul.default` | `model.layers.10.self_attn` | `True` | `True` | `attention, attention_output` | `t00000607, t00000575` | `t00000608` |
| 55 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000608` | `t00000609` |
| 56 | `contiguous.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` | `t00000609` | `t00000610` |
| 57 | `reshape.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` | `t00000610` | `t00000611` |
| 58 | `gt.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000612` |
| 59 | `is_nonzero.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000612` | `` |
| 60 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000023` | `t00000613` |
| 61 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000613` | `t00000614` |
| 62 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000614` | `t00000615` |
| 63 | `eq.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000615` | `t00000616` |
| 64 | `is_nonzero.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000616` | `` |
| 65 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000611` | `t00000617` |
| 66 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000607` | `t00000618` |
| 67 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000618` | `t00000619` |
| 68 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000619, t00000575` | `t00000620` |
| 69 | `permute.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000620` | `t00000621` |
| 70 | `contiguous.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` | `t00000621` | `t00000622` |
| 71 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000622` | `t00000623` |
| 72 | `item.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000624` | `` |
| 73 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000623` | `t00000625` |
| 74 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000617` | `t00000626` |
| 75 | `sub.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000626, t00000625` | `t00000627` |
| 76 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000617` | `t00000628` |
| 77 | `cosine_similarity.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000627, t00000628` | `t00000629` |
| 78 | `squeeze.dim` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000629` | `t00000630` |
| 79 | `lt.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000630` | `t00000631` |
| 80 | `any.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000631` | `t00000632` |
| 81 | `item.default` | `model.layers.10.self_attn` | `True` | `True` | `` | `t00000632` | `` |
| 82 | `linear.default` | `model.layers.10.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000611, t00000633` | `t00000634` |
| 83 | `add.Tensor` | `model.layers.10` | `True` | `True` | `attention_output, mlp` | `t00000554, t00000634` | `t00000635` |
| 84 | `to.dtype` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000635` | `t00000636` |
| 85 | `pow.Tensor_Scalar` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000636` | `t00000637` |
| 86 | `mean.dim` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000637` | `t00000638` |
| 87 | `add.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000638` | `t00000639` |
| 88 | `rsqrt.default` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000639` | `t00000640` |
| 89 | `mul.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000636, t00000640` | `t00000641` |
| 90 | `to.dtype` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000641` | `t00000642` |
| 91 | `mul.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000643, t00000642` | `t00000644` |
| 92 | `linear.default` | `model.layers.10.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000644, t00000645` | `t00000646` |
| 93 | `silu.default` | `model.layers.10.mlp.act_fn` | `True` | `True` | `mlp` | `t00000646` | `t00000647` |
| 94 | `linear.default` | `model.layers.10.mlp.up_proj` | `True` | `True` | `mlp` | `t00000644, t00000648` | `t00000649` |
| 95 | `mul.Tensor` | `model.layers.10.mlp` | `True` | `True` | `mlp` | `t00000647, t00000649` | `t00000650` |
| 96 | `linear.default` | `model.layers.10.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000650, t00000651` | `t00000652` |
| 97 | `add.Tensor` | `model.layers.10` | `True` | `True` | `attention_output` | `t00000635, t00000652` | `t00000653` |
