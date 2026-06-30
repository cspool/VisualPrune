# input1_layer18 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `104`
- ops listed in coverage: `104`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001346` | `t00001347` |
| 2 | `pow.Tensor_Scalar` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001347` | `t00001348` |
| 3 | `mean.dim` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001348` | `t00001349` |
| 4 | `add.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001349` | `t00001350` |
| 5 | `rsqrt.default` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001350` | `t00001351` |
| 6 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001347, t00001351` | `t00001352` |
| 7 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001352` | `t00001353` |
| 8 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001354, t00001353` | `t00001355` |
| 9 | `linear.default` | `model.layers.18.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001355, t00001356` | `t00001357` |
| 10 | `linear.default` | `model.layers.18.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001355, t00001358` | `t00001359` |
| 11 | `linear.default` | `model.layers.18.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001355, t00001360` | `t00001361` |
| 12 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00001357` | `t00001362` |
| 13 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001362` | `t00001363` |
| 14 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00001359` | `t00001364` |
| 15 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001364` | `t00001365` |
| 16 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00001361` | `t00001366` |
| 17 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001366` | `t00001367` |
| 18 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00000023` | `t00001368` |
| 19 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001368` | `t00001369` |
| 20 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001369` | `t00001370` |
| 21 | `gt.Scalar` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001370` | `t00001371` |
| 22 | `is_nonzero.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001371` | `` |
| 23 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00001370` | `` |
| 24 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001372` | `t00001373` |
| 25 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00001373` | `t00001373` |
| 26 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00001370` | `` |
| 27 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001374` | `t00001375` |
| 28 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00001375` | `t00001375` |
| 29 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001373, t00000023` | `t00001376` |
| 30 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001376` | `t00001377` |
| 31 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001375, t00000023` | `t00001378` |
| 32 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001378` | `t00001379` |
| 33 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001363, t00001377` | `t00001380` |
| 34 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001363` | `t00001381` |
| 35 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001363` | `t00001382` |
| 36 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001382` | `t00001383` |
| 37 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001383, t00001381` | `t00001384` |
| 38 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00001384, t00001379` | `t00001385` |
| 39 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope, attention` | `t00001380, t00001385` | `t00001386` |
| 40 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001365, t00001377` | `t00001387` |
| 41 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001365` | `t00001388` |
| 42 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001365` | `t00001389` |
| 43 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001389` | `t00001390` |
| 44 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001390, t00001388` | `t00001391` |
| 45 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001391, t00001379` | `t00001392` |
| 46 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001387, t00001392` | `t00001393` |
| 47 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001393` | `t00001394` |
| 48 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001386, t00001394` | `t00001395` |
| 49 | `div.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001395` | `t00001396` |
| 50 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001396, t00000053` | `t00001397` |
| 51 | `softmax.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001397` | `t00001398` |
| 52 | `to.dtype` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001398` | `t00001399` |
| 53 | `dropout.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00001399` | `t00001399` |
| 54 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention, attention_output` | `t00001399, t00001367` | `t00001400` |
| 55 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001400` | `t00001401` |
| 56 | `contiguous.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` | `t00001401` | `t00001402` |
| 57 | `reshape.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` | `t00001402` | `t00001403` |
| 58 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001404` |
| 59 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001404` | `` |
| 60 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00000023` | `t00001405` |
| 61 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001405` | `t00001406` |
| 62 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001406` | `t00001407` |
| 63 | `eq.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001407` | `t00001408` |
| 64 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001408` | `` |
| 65 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001403` | `t00001409` |
| 66 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001399` | `t00001410` |
| 67 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001410` | `t00001411` |
| 68 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001411, t00001367` | `t00001412` |
| 69 | `permute.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001412` | `t00001413` |
| 70 | `contiguous.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` | `t00001413` | `t00001414` |
| 71 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001414` | `t00001415` |
| 72 | `item.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001416` | `` |
| 73 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001415` | `t00001417` |
| 74 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001409` | `t00001418` |
| 75 | `sub.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001418, t00001417` | `t00001419` |
| 76 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001409` | `t00001420` |
| 77 | `cosine_similarity.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001419, t00001420` | `t00001421` |
| 78 | `squeeze.dim` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001421` | `t00001422` |
| 79 | `lt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001422` | `t00001423` |
| 80 | `any.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001423` | `t00001424` |
| 81 | `item.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001424` | `` |
| 82 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001409` | `t00001425` |
| 83 | `sub.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001419, t00001425` | `t00001426` |
| 84 | `linalg_vector_norm.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001426` | `t00001427` |
| 85 | `squeeze.dim` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001427` | `t00001428` |
| 86 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001428` | `t00001429` |
| 87 | `where.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00001429` | `t00001430` |
| 88 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `mlp` | `t00001430` | `t00001431` |
| 89 | `linear.default` | `model.layers.18.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001403, t00001432` | `t00001433` |
| 90 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output, mlp` | `t00001346, t00001433` | `t00001434` |
| 91 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001434` | `t00001435` |
| 92 | `pow.Tensor_Scalar` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001435` | `t00001436` |
| 93 | `mean.dim` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001436` | `t00001437` |
| 94 | `add.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001437` | `t00001438` |
| 95 | `rsqrt.default` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001438` | `t00001439` |
| 96 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001435, t00001439` | `t00001440` |
| 97 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001440` | `t00001441` |
| 98 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001442, t00001441` | `t00001443` |
| 99 | `linear.default` | `model.layers.18.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001443, t00001444` | `t00001445` |
| 100 | `silu.default` | `model.layers.18.mlp.act_fn` | `True` | `True` | `mlp` | `t00001445` | `t00001446` |
| 101 | `linear.default` | `model.layers.18.mlp.up_proj` | `True` | `True` | `mlp` | `t00001443, t00001447` | `t00001448` |
| 102 | `mul.Tensor` | `model.layers.18.mlp` | `True` | `True` | `` | `t00001446, t00001448` | `t00001449` |
| 103 | `linear.default` | `model.layers.18.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001449, t00001450` | `t00001451` |
| 104 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output` | `t00001434, t00001451` | `t00001452` |
