# input1_layer8 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000356` | `t00000357` |
| 2 | `pow.Tensor_Scalar` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000357` | `t00000358` |
| 3 | `mean.dim` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000358` | `t00000359` |
| 4 | `add.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000359` | `t00000360` |
| 5 | `rsqrt.default` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000360` | `t00000361` |
| 6 | `mul.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000357, t00000361` | `t00000362` |
| 7 | `to.dtype` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000362` | `t00000363` |
| 8 | `mul.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000364, t00000363` | `t00000365` |
| 9 | `linear.default` | `model.layers.8.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000365, t00000366` | `t00000367` |
| 10 | `linear.default` | `model.layers.8.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000365, t00000368` | `t00000369` |
| 11 | `linear.default` | `model.layers.8.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000365, t00000370` | `t00000371` |
| 12 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` | `t00000367` | `t00000372` |
| 13 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000372` | `t00000373` |
| 14 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` | `t00000369` | `t00000374` |
| 15 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000374` | `t00000375` |
| 16 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` | `t00000371` | `t00000376` |
| 17 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000376` | `t00000377` |
| 18 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000023` | `t00000378` |
| 19 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000378` | `t00000379` |
| 20 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000379` | `t00000380` |
| 21 | `gt.Scalar` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000380` | `t00000381` |
| 22 | `is_nonzero.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000381` | `` |
| 23 | `item.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` | `t00000380` | `` |
| 24 | `slice.Tensor` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000382` | `t00000383` |
| 25 | `to.dtype` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` | `t00000383` | `t00000383` |
| 26 | `item.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` | `t00000380` | `` |
| 27 | `slice.Tensor` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000384` | `t00000385` |
| 28 | `to.dtype` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` | `t00000385` | `t00000385` |
| 29 | `index.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000383, t00000023` | `t00000386` |
| 30 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000386` | `t00000387` |
| 31 | `index.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000385, t00000023` | `t00000388` |
| 32 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000388` | `t00000389` |
| 33 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000373, t00000387` | `t00000390` |
| 34 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000373` | `t00000391` |
| 35 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000373` | `t00000392` |
| 36 | `neg.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000392` | `t00000393` |
| 37 | `cat.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000393, t00000391` | `t00000394` |
| 38 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` | `t00000394, t00000389` | `t00000395` |
| 39 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope, attention` | `t00000390, t00000395` | `t00000396` |
| 40 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000375, t00000387` | `t00000397` |
| 41 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000375` | `t00000398` |
| 42 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000375` | `t00000399` |
| 43 | `neg.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000399` | `t00000400` |
| 44 | `cat.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000400, t00000398` | `t00000401` |
| 45 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000401, t00000389` | `t00000402` |
| 46 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000397, t00000402` | `t00000403` |
| 47 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000403` | `t00000404` |
| 48 | `matmul.default` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000396, t00000404` | `t00000405` |
| 49 | `div.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000405` | `t00000406` |
| 50 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000406, t00000053` | `t00000407` |
| 51 | `softmax.int` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000407` | `t00000408` |
| 52 | `to.dtype` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000408` | `t00000409` |
| 53 | `dropout.default` | `model.layers.8.self_attn` | `True` | `True` | `attention` | `t00000409` | `t00000409` |
| 54 | `matmul.default` | `model.layers.8.self_attn` | `True` | `True` | `attention, attention_output` | `t00000409, t00000377` | `t00000410` |
| 55 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000410` | `t00000411` |
| 56 | `contiguous.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` | `t00000411` | `t00000412` |
| 57 | `reshape.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` | `t00000412` | `t00000413` |
| 58 | `gt.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000414` |
| 59 | `is_nonzero.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000414` | `` |
| 60 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000023` | `t00000415` |
| 61 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000415` | `t00000416` |
| 62 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000416` | `t00000417` |
| 63 | `eq.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000417` | `t00000418` |
| 64 | `is_nonzero.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000418` | `` |
| 65 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000413` | `t00000419` |
| 66 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000409` | `t00000420` |
| 67 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000420` | `t00000421` |
| 68 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000421, t00000377` | `t00000422` |
| 69 | `permute.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000422` | `t00000423` |
| 70 | `contiguous.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` | `t00000423` | `t00000424` |
| 71 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000424` | `t00000425` |
| 72 | `item.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000426` | `` |
| 73 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000425` | `t00000427` |
| 74 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000419` | `t00000428` |
| 75 | `sub.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000428, t00000427` | `t00000429` |
| 76 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000419` | `t00000430` |
| 77 | `cosine_similarity.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000429, t00000430` | `t00000431` |
| 78 | `squeeze.dim` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000431` | `t00000432` |
| 79 | `lt.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000432` | `t00000433` |
| 80 | `any.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000433` | `t00000434` |
| 81 | `item.default` | `model.layers.8.self_attn` | `True` | `True` | `` | `t00000434` | `` |
| 82 | `linear.default` | `model.layers.8.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000413, t00000435` | `t00000436` |
| 83 | `add.Tensor` | `model.layers.8` | `True` | `True` | `attention_output, mlp` | `t00000356, t00000436` | `t00000437` |
| 84 | `to.dtype` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000437` | `t00000438` |
| 85 | `pow.Tensor_Scalar` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000438` | `t00000439` |
| 86 | `mean.dim` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000439` | `t00000440` |
| 87 | `add.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000440` | `t00000441` |
| 88 | `rsqrt.default` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000441` | `t00000442` |
| 89 | `mul.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000438, t00000442` | `t00000443` |
| 90 | `to.dtype` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000443` | `t00000444` |
| 91 | `mul.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000445, t00000444` | `t00000446` |
| 92 | `linear.default` | `model.layers.8.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000446, t00000447` | `t00000448` |
| 93 | `silu.default` | `model.layers.8.mlp.act_fn` | `True` | `True` | `mlp` | `t00000448` | `t00000449` |
| 94 | `linear.default` | `model.layers.8.mlp.up_proj` | `True` | `True` | `mlp` | `t00000446, t00000450` | `t00000451` |
| 95 | `mul.Tensor` | `model.layers.8.mlp` | `True` | `True` | `mlp` | `t00000449, t00000451` | `t00000452` |
| 96 | `linear.default` | `model.layers.8.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000452, t00000453` | `t00000454` |
| 97 | `add.Tensor` | `model.layers.8` | `True` | `True` | `attention_output` | `t00000437, t00000454` | `t00000455` |
