# input1_layer22 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001761` | `t00001762` |
| 2 | `pow.Tensor_Scalar` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001762` | `t00001763` |
| 3 | `mean.dim` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001763` | `t00001764` |
| 4 | `add.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001764` | `t00001765` |
| 5 | `rsqrt.default` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001765` | `t00001766` |
| 6 | `mul.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001762, t00001766` | `t00001767` |
| 7 | `to.dtype` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001767` | `t00001768` |
| 8 | `mul.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001769, t00001768` | `t00001770` |
| 9 | `linear.default` | `model.layers.22.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001770, t00001771` | `t00001772` |
| 10 | `linear.default` | `model.layers.22.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001770, t00001773` | `t00001774` |
| 11 | `linear.default` | `model.layers.22.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001770, t00001775` | `t00001776` |
| 12 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` | `t00001772` | `t00001777` |
| 13 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001777` | `t00001778` |
| 14 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` | `t00001774` | `t00001779` |
| 15 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001779` | `t00001780` |
| 16 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` | `t00001776` | `t00001781` |
| 17 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001781` | `t00001782` |
| 18 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001475` | `t00001783` |
| 19 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001783` | `t00001784` |
| 20 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001784` | `t00001785` |
| 21 | `gt.Scalar` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001785` | `t00001786` |
| 22 | `is_nonzero.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001786` | `` |
| 23 | `item.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` | `t00001785` | `` |
| 24 | `slice.Tensor` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001787` | `t00001788` |
| 25 | `to.dtype` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` | `t00001788` | `t00001788` |
| 26 | `item.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` | `t00001785` | `` |
| 27 | `slice.Tensor` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001789` | `t00001790` |
| 28 | `to.dtype` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` | `t00001790` | `t00001790` |
| 29 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001788, t00001475` | `t00001791` |
| 30 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001791` | `t00001792` |
| 31 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001790, t00001475` | `t00001793` |
| 32 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001793` | `t00001794` |
| 33 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001778, t00001792` | `t00001795` |
| 34 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001778` | `t00001796` |
| 35 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001778` | `t00001797` |
| 36 | `neg.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001797` | `t00001798` |
| 37 | `cat.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001798, t00001796` | `t00001799` |
| 38 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` | `t00001799, t00001794` | `t00001800` |
| 39 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope, attention` | `t00001795, t00001800` | `t00001801` |
| 40 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001780, t00001792` | `t00001802` |
| 41 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001780` | `t00001803` |
| 42 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001780` | `t00001804` |
| 43 | `neg.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001804` | `t00001805` |
| 44 | `cat.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001805, t00001803` | `t00001806` |
| 45 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001806, t00001794` | `t00001807` |
| 46 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001802, t00001807` | `t00001808` |
| 47 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001808` | `t00001809` |
| 48 | `matmul.default` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001801, t00001809` | `t00001810` |
| 49 | `div.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001810` | `t00001811` |
| 50 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001811, t00001505` | `t00001812` |
| 51 | `softmax.int` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001812` | `t00001813` |
| 52 | `to.dtype` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001813` | `t00001814` |
| 53 | `dropout.default` | `model.layers.22.self_attn` | `True` | `True` | `attention` | `t00001814` | `t00001814` |
| 54 | `matmul.default` | `model.layers.22.self_attn` | `True` | `True` | `attention, attention_output` | `t00001814, t00001782` | `t00001815` |
| 55 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001815` | `t00001816` |
| 56 | `contiguous.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` | `t00001816` | `t00001817` |
| 57 | `reshape.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` | `t00001817` | `t00001818` |
| 58 | `gt.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001819` |
| 59 | `is_nonzero.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001819` | `` |
| 60 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001475` | `t00001820` |
| 61 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001820` | `t00001821` |
| 62 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001821` | `t00001822` |
| 63 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00000057` | `t00001823` |
| 64 | `sub.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001822, t00001823` | `t00001824` |
| 65 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001824` | `t00001825` |
| 66 | `eq.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001825` | `t00001826` |
| 67 | `is_nonzero.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001826` | `` |
| 68 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001818` | `t00001827` |
| 69 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001814` | `t00001828` |
| 70 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001828` | `t00001829` |
| 71 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001829, t00001782` | `t00001830` |
| 72 | `permute.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001830` | `t00001831` |
| 73 | `contiguous.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` | `t00001831` | `t00001832` |
| 74 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001832` | `t00001833` |
| 75 | `arange.start` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00001834` |
| 76 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001833, t00001834` | `t00001835` |
| 77 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001827` | `t00001836` |
| 78 | `sub.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001836, t00001835` | `t00001837` |
| 79 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001827` | `t00001838` |
| 80 | `cosine_similarity.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001837, t00001838` | `t00001839` |
| 81 | `squeeze.dim` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001839` | `t00001840` |
| 82 | `lt.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001840` | `t00001841` |
| 83 | `any.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001841` | `t00001842` |
| 84 | `item.default` | `model.layers.22.self_attn` | `True` | `True` | `` | `t00001842` | `` |
| 85 | `linear.default` | `model.layers.22.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001818, t00001843` | `t00001844` |
| 86 | `add.Tensor` | `model.layers.22` | `True` | `True` | `attention_output, mlp` | `t00001761, t00001844` | `t00001845` |
| 87 | `to.dtype` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001845` | `t00001846` |
| 88 | `pow.Tensor_Scalar` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001846` | `t00001847` |
| 89 | `mean.dim` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001847` | `t00001848` |
| 90 | `add.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001848` | `t00001849` |
| 91 | `rsqrt.default` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001849` | `t00001850` |
| 92 | `mul.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001846, t00001850` | `t00001851` |
| 93 | `to.dtype` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001851` | `t00001852` |
| 94 | `mul.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001853, t00001852` | `t00001854` |
| 95 | `linear.default` | `model.layers.22.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001854, t00001855` | `t00001856` |
| 96 | `silu.default` | `model.layers.22.mlp.act_fn` | `True` | `True` | `mlp` | `t00001856` | `t00001857` |
| 97 | `linear.default` | `model.layers.22.mlp.up_proj` | `True` | `True` | `mlp` | `t00001854, t00001858` | `t00001859` |
| 98 | `mul.Tensor` | `model.layers.22.mlp` | `True` | `True` | `mlp` | `t00001857, t00001859` | `t00001860` |
| 99 | `linear.default` | `model.layers.22.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001860, t00001861` | `t00001862` |
| 100 | `add.Tensor` | `model.layers.22` | `True` | `True` | `attention_output` | `t00001845, t00001862` | `t00001863` |
