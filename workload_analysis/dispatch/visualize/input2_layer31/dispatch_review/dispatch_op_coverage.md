# input2_layer31 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002746` | `t00002747` |
| 2 | `pow.Tensor_Scalar` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002747` | `t00002748` |
| 3 | `mean.dim` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002748` | `t00002749` |
| 4 | `add.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002749` | `t00002750` |
| 5 | `rsqrt.default` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002750` | `t00002751` |
| 6 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002747, t00002751` | `t00002752` |
| 7 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002752` | `t00002753` |
| 8 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002754, t00002753` | `t00002755` |
| 9 | `linear.default` | `model.layers.31.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002755, t00002756` | `t00002757` |
| 10 | `linear.default` | `model.layers.31.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002755, t00002758` | `t00002759` |
| 11 | `linear.default` | `model.layers.31.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002755, t00002760` | `t00002761` |
| 12 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00002757` | `t00002762` |
| 13 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002762` | `t00002763` |
| 14 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00002759` | `t00002764` |
| 15 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002764` | `t00002765` |
| 16 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` | `t00002761` | `t00002766` |
| 17 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002766` | `t00002767` |
| 18 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002481` | `t00002768` |
| 19 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002768` | `t00002769` |
| 20 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002769` | `t00002770` |
| 21 | `gt.Scalar` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002770` | `t00002771` |
| 22 | `is_nonzero.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002771` | `` |
| 23 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002770` | `` |
| 24 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002772` | `t00002773` |
| 25 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002773` | `t00002773` |
| 26 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002770` | `` |
| 27 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002774` | `t00002775` |
| 28 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` | `t00002775` | `t00002775` |
| 29 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002773, t00002481` | `t00002776` |
| 30 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002776` | `t00002777` |
| 31 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002775, t00002481` | `t00002778` |
| 32 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002778` | `t00002779` |
| 33 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002763, t00002777` | `t00002780` |
| 34 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002763` | `t00002781` |
| 35 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002763` | `t00002782` |
| 36 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002782` | `t00002783` |
| 37 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002783, t00002781` | `t00002784` |
| 38 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` | `t00002784, t00002779` | `t00002785` |
| 39 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope, attention` | `t00002780, t00002785` | `t00002786` |
| 40 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002765, t00002777` | `t00002787` |
| 41 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002765` | `t00002788` |
| 42 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002765` | `t00002789` |
| 43 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002789` | `t00002790` |
| 44 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002790, t00002788` | `t00002791` |
| 45 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002791, t00002779` | `t00002792` |
| 46 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002787, t00002792` | `t00002793` |
| 47 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002794, t00002793` | `t00002795` |
| 48 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002796, t00002767` | `t00002797` |
| 49 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002795` | `t00002798` |
| 50 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002786, t00002798` | `t00002799` |
| 51 | `div.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002799` | `t00002800` |
| 52 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002800, t00002801` | `t00002802` |
| 53 | `softmax.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002802` | `t00002803` |
| 54 | `to.dtype` | `model.layers.31.self_attn` | `True` | `True` | `mlp` | `t00002803` | `t00002804` |
| 55 | `dropout.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` | `t00002804` | `t00002804` |
| 56 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention, attention_output` | `t00002804, t00002797` | `t00002805` |
| 57 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002805` | `t00002806` |
| 58 | `reshape.default` | `model.layers.31.self_attn` | `True` | `True` | `attention_output` | `t00002806` | `t00002807` |
| 59 | `gt.Scalar` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00000057` | `t00002808` |
| 60 | `is_nonzero.default` | `model.layers.31.self_attn` | `True` | `True` | `` | `t00002808` | `` |
| 61 | `linear.default` | `model.layers.31.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002807, t00002809` | `t00002810` |
| 62 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output, mlp` | `t00002746, t00002810` | `t00002811` |
| 63 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002811` | `t00002812` |
| 64 | `pow.Tensor_Scalar` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002812` | `t00002813` |
| 65 | `mean.dim` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002813` | `t00002814` |
| 66 | `add.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002814` | `t00002815` |
| 67 | `rsqrt.default` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002815` | `t00002816` |
| 68 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002812, t00002816` | `t00002817` |
| 69 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002817` | `t00002818` |
| 70 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002819, t00002818` | `t00002820` |
| 71 | `linear.default` | `model.layers.31.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002820, t00002821` | `t00002822` |
| 72 | `silu.default` | `model.layers.31.mlp.act_fn` | `True` | `True` | `mlp` | `t00002822` | `t00002823` |
| 73 | `linear.default` | `model.layers.31.mlp.up_proj` | `True` | `True` | `mlp` | `t00002820, t00002824` | `t00002825` |
| 74 | `mul.Tensor` | `model.layers.31.mlp` | `True` | `True` | `` | `t00002823, t00002825` | `t00002826` |
| 75 | `linear.default` | `model.layers.31.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002826, t00002827` | `t00002828` |
| 76 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output` | `t00002811, t00002828` | `t00002829` |
