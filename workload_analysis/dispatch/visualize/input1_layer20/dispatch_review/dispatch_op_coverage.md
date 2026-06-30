# input1_layer20 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001557` | `t00001558` |
| 2 | `pow.Tensor_Scalar` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001558` | `t00001559` |
| 3 | `mean.dim` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001559` | `t00001560` |
| 4 | `add.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001560` | `t00001561` |
| 5 | `rsqrt.default` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001561` | `t00001562` |
| 6 | `mul.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001558, t00001562` | `t00001563` |
| 7 | `to.dtype` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001563` | `t00001564` |
| 8 | `mul.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001565, t00001564` | `t00001566` |
| 9 | `linear.default` | `model.layers.20.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001566, t00001567` | `t00001568` |
| 10 | `linear.default` | `model.layers.20.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001566, t00001569` | `t00001570` |
| 11 | `linear.default` | `model.layers.20.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001566, t00001571` | `t00001572` |
| 12 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` | `t00001568` | `t00001573` |
| 13 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001573` | `t00001574` |
| 14 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` | `t00001570` | `t00001575` |
| 15 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001575` | `t00001576` |
| 16 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` | `t00001572` | `t00001577` |
| 17 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001577` | `t00001578` |
| 18 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001475` | `t00001579` |
| 19 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001579` | `t00001580` |
| 20 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001580` | `t00001581` |
| 21 | `gt.Scalar` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001581` | `t00001582` |
| 22 | `is_nonzero.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001582` | `` |
| 23 | `item.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` | `t00001581` | `` |
| 24 | `slice.Tensor` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001583` | `t00001584` |
| 25 | `to.dtype` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` | `t00001584` | `t00001584` |
| 26 | `item.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` | `t00001581` | `` |
| 27 | `slice.Tensor` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001585` | `t00001586` |
| 28 | `to.dtype` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` | `t00001586` | `t00001586` |
| 29 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001584, t00001475` | `t00001587` |
| 30 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001587` | `t00001588` |
| 31 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001586, t00001475` | `t00001589` |
| 32 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001589` | `t00001590` |
| 33 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001574, t00001588` | `t00001591` |
| 34 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001574` | `t00001592` |
| 35 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001574` | `t00001593` |
| 36 | `neg.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001593` | `t00001594` |
| 37 | `cat.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001594, t00001592` | `t00001595` |
| 38 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` | `t00001595, t00001590` | `t00001596` |
| 39 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope, attention` | `t00001591, t00001596` | `t00001597` |
| 40 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001576, t00001588` | `t00001598` |
| 41 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001576` | `t00001599` |
| 42 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001576` | `t00001600` |
| 43 | `neg.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001600` | `t00001601` |
| 44 | `cat.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001601, t00001599` | `t00001602` |
| 45 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001602, t00001590` | `t00001603` |
| 46 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001598, t00001603` | `t00001604` |
| 47 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001604` | `t00001605` |
| 48 | `matmul.default` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001597, t00001605` | `t00001606` |
| 49 | `div.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001606` | `t00001607` |
| 50 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001607, t00001505` | `t00001608` |
| 51 | `softmax.int` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001608` | `t00001609` |
| 52 | `to.dtype` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001609` | `t00001610` |
| 53 | `dropout.default` | `model.layers.20.self_attn` | `True` | `True` | `attention` | `t00001610` | `t00001610` |
| 54 | `matmul.default` | `model.layers.20.self_attn` | `True` | `True` | `attention, attention_output` | `t00001610, t00001578` | `t00001611` |
| 55 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001611` | `t00001612` |
| 56 | `contiguous.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` | `t00001612` | `t00001613` |
| 57 | `reshape.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` | `t00001613` | `t00001614` |
| 58 | `gt.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001615` |
| 59 | `is_nonzero.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001615` | `` |
| 60 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001475` | `t00001616` |
| 61 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001616` | `t00001617` |
| 62 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001617` | `t00001618` |
| 63 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00000057` | `t00001619` |
| 64 | `sub.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001618, t00001619` | `t00001620` |
| 65 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001620` | `t00001621` |
| 66 | `eq.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001621` | `t00001622` |
| 67 | `is_nonzero.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001622` | `` |
| 68 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001614` | `t00001623` |
| 69 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001610` | `t00001624` |
| 70 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001624` | `t00001625` |
| 71 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001625, t00001578` | `t00001626` |
| 72 | `permute.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001626` | `t00001627` |
| 73 | `contiguous.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` | `t00001627` | `t00001628` |
| 74 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001628` | `t00001629` |
| 75 | `arange.start` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00001630` |
| 76 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001629, t00001630` | `t00001631` |
| 77 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001623` | `t00001632` |
| 78 | `sub.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001632, t00001631` | `t00001633` |
| 79 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001623` | `t00001634` |
| 80 | `cosine_similarity.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001633, t00001634` | `t00001635` |
| 81 | `squeeze.dim` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001635` | `t00001636` |
| 82 | `lt.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001636` | `t00001637` |
| 83 | `any.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001637` | `t00001638` |
| 84 | `item.default` | `model.layers.20.self_attn` | `True` | `True` | `` | `t00001638` | `` |
| 85 | `linear.default` | `model.layers.20.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001614, t00001639` | `t00001640` |
| 86 | `add.Tensor` | `model.layers.20` | `True` | `True` | `attention_output, mlp` | `t00001557, t00001640` | `t00001641` |
| 87 | `to.dtype` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001641` | `t00001642` |
| 88 | `pow.Tensor_Scalar` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001642` | `t00001643` |
| 89 | `mean.dim` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001643` | `t00001644` |
| 90 | `add.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001644` | `t00001645` |
| 91 | `rsqrt.default` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001645` | `t00001646` |
| 92 | `mul.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001642, t00001646` | `t00001647` |
| 93 | `to.dtype` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001647` | `t00001648` |
| 94 | `mul.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001649, t00001648` | `t00001650` |
| 95 | `linear.default` | `model.layers.20.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001650, t00001651` | `t00001652` |
| 96 | `silu.default` | `model.layers.20.mlp.act_fn` | `True` | `True` | `mlp` | `t00001652` | `t00001653` |
| 97 | `linear.default` | `model.layers.20.mlp.up_proj` | `True` | `True` | `mlp` | `t00001650, t00001654` | `t00001655` |
| 98 | `mul.Tensor` | `model.layers.20.mlp` | `True` | `True` | `mlp` | `t00001653, t00001655` | `t00001656` |
| 99 | `linear.default` | `model.layers.20.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001656, t00001657` | `t00001658` |
| 100 | `add.Tensor` | `model.layers.20` | `True` | `True` | `attention_output` | `t00001641, t00001658` | `t00001659` |
