# input1_layer12 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000752` | `t00000753` |
| 2 | `pow.Tensor_Scalar` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000753` | `t00000754` |
| 3 | `mean.dim` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000754` | `t00000755` |
| 4 | `add.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000755` | `t00000756` |
| 5 | `rsqrt.default` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000756` | `t00000757` |
| 6 | `mul.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000753, t00000757` | `t00000758` |
| 7 | `to.dtype` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000758` | `t00000759` |
| 8 | `mul.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000760, t00000759` | `t00000761` |
| 9 | `linear.default` | `model.layers.12.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000761, t00000762` | `t00000763` |
| 10 | `linear.default` | `model.layers.12.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000761, t00000764` | `t00000765` |
| 11 | `linear.default` | `model.layers.12.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000761, t00000766` | `t00000767` |
| 12 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` | `t00000763` | `t00000768` |
| 13 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000768` | `t00000769` |
| 14 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` | `t00000765` | `t00000770` |
| 15 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000770` | `t00000771` |
| 16 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` | `t00000767` | `t00000772` |
| 17 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000772` | `t00000773` |
| 18 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000023` | `t00000774` |
| 19 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000774` | `t00000775` |
| 20 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000775` | `t00000776` |
| 21 | `gt.Scalar` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000776` | `t00000777` |
| 22 | `is_nonzero.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000777` | `` |
| 23 | `item.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` | `t00000776` | `` |
| 24 | `slice.Tensor` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000778` | `t00000779` |
| 25 | `to.dtype` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` | `t00000779` | `t00000779` |
| 26 | `item.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` | `t00000776` | `` |
| 27 | `slice.Tensor` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000780` | `t00000781` |
| 28 | `to.dtype` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` | `t00000781` | `t00000781` |
| 29 | `index.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000779, t00000023` | `t00000782` |
| 30 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000782` | `t00000783` |
| 31 | `index.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000781, t00000023` | `t00000784` |
| 32 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000784` | `t00000785` |
| 33 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000769, t00000783` | `t00000786` |
| 34 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000769` | `t00000787` |
| 35 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000769` | `t00000788` |
| 36 | `neg.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000788` | `t00000789` |
| 37 | `cat.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000789, t00000787` | `t00000790` |
| 38 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` | `t00000790, t00000785` | `t00000791` |
| 39 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope, attention` | `t00000786, t00000791` | `t00000792` |
| 40 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000771, t00000783` | `t00000793` |
| 41 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000771` | `t00000794` |
| 42 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000771` | `t00000795` |
| 43 | `neg.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000795` | `t00000796` |
| 44 | `cat.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000796, t00000794` | `t00000797` |
| 45 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000797, t00000785` | `t00000798` |
| 46 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000793, t00000798` | `t00000799` |
| 47 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000799` | `t00000800` |
| 48 | `matmul.default` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000792, t00000800` | `t00000801` |
| 49 | `div.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000801` | `t00000802` |
| 50 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000802, t00000053` | `t00000803` |
| 51 | `softmax.int` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000803` | `t00000804` |
| 52 | `to.dtype` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000804` | `t00000805` |
| 53 | `dropout.default` | `model.layers.12.self_attn` | `True` | `True` | `attention` | `t00000805` | `t00000805` |
| 54 | `matmul.default` | `model.layers.12.self_attn` | `True` | `True` | `attention, attention_output` | `t00000805, t00000773` | `t00000806` |
| 55 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000806` | `t00000807` |
| 56 | `contiguous.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` | `t00000807` | `t00000808` |
| 57 | `reshape.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` | `t00000808` | `t00000809` |
| 58 | `gt.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000810` |
| 59 | `is_nonzero.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000810` | `` |
| 60 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000023` | `t00000811` |
| 61 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000811` | `t00000812` |
| 62 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000812` | `t00000813` |
| 63 | `eq.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000813` | `t00000814` |
| 64 | `is_nonzero.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000814` | `` |
| 65 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000809` | `t00000815` |
| 66 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000805` | `t00000816` |
| 67 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000816` | `t00000817` |
| 68 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000817, t00000773` | `t00000818` |
| 69 | `permute.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000818` | `t00000819` |
| 70 | `contiguous.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` | `t00000819` | `t00000820` |
| 71 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000820` | `t00000821` |
| 72 | `item.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000822` | `` |
| 73 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000821` | `t00000823` |
| 74 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000815` | `t00000824` |
| 75 | `sub.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000824, t00000823` | `t00000825` |
| 76 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000815` | `t00000826` |
| 77 | `cosine_similarity.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000825, t00000826` | `t00000827` |
| 78 | `squeeze.dim` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000827` | `t00000828` |
| 79 | `lt.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000828` | `t00000829` |
| 80 | `any.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000829` | `t00000830` |
| 81 | `item.default` | `model.layers.12.self_attn` | `True` | `True` | `` | `t00000830` | `` |
| 82 | `linear.default` | `model.layers.12.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000809, t00000831` | `t00000832` |
| 83 | `add.Tensor` | `model.layers.12` | `True` | `True` | `attention_output, mlp` | `t00000752, t00000832` | `t00000833` |
| 84 | `to.dtype` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000833` | `t00000834` |
| 85 | `pow.Tensor_Scalar` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000834` | `t00000835` |
| 86 | `mean.dim` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000835` | `t00000836` |
| 87 | `add.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000836` | `t00000837` |
| 88 | `rsqrt.default` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000837` | `t00000838` |
| 89 | `mul.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000834, t00000838` | `t00000839` |
| 90 | `to.dtype` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000839` | `t00000840` |
| 91 | `mul.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000841, t00000840` | `t00000842` |
| 92 | `linear.default` | `model.layers.12.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000842, t00000843` | `t00000844` |
| 93 | `silu.default` | `model.layers.12.mlp.act_fn` | `True` | `True` | `mlp` | `t00000844` | `t00000845` |
| 94 | `linear.default` | `model.layers.12.mlp.up_proj` | `True` | `True` | `mlp` | `t00000842, t00000846` | `t00000847` |
| 95 | `mul.Tensor` | `model.layers.12.mlp` | `True` | `True` | `mlp` | `t00000845, t00000847` | `t00000848` |
| 96 | `linear.default` | `model.layers.12.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000848, t00000849` | `t00000850` |
| 97 | `add.Tensor` | `model.layers.12` | `True` | `True` | `attention_output` | `t00000833, t00000850` | `t00000851` |
