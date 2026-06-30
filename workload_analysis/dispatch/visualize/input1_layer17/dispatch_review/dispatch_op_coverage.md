# input1_layer17 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001247` | `t00001248` |
| 2 | `pow.Tensor_Scalar` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001248` | `t00001249` |
| 3 | `mean.dim` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001249` | `t00001250` |
| 4 | `add.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001250` | `t00001251` |
| 5 | `rsqrt.default` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001251` | `t00001252` |
| 6 | `mul.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001248, t00001252` | `t00001253` |
| 7 | `to.dtype` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001253` | `t00001254` |
| 8 | `mul.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001255, t00001254` | `t00001256` |
| 9 | `linear.default` | `model.layers.17.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001256, t00001257` | `t00001258` |
| 10 | `linear.default` | `model.layers.17.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001256, t00001259` | `t00001260` |
| 11 | `linear.default` | `model.layers.17.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001256, t00001261` | `t00001262` |
| 12 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` | `t00001258` | `t00001263` |
| 13 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001263` | `t00001264` |
| 14 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` | `t00001260` | `t00001265` |
| 15 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001265` | `t00001266` |
| 16 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` | `t00001262` | `t00001267` |
| 17 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001267` | `t00001268` |
| 18 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00000023` | `t00001269` |
| 19 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001269` | `t00001270` |
| 20 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001270` | `t00001271` |
| 21 | `gt.Scalar` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001271` | `t00001272` |
| 22 | `is_nonzero.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001272` | `` |
| 23 | `item.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` | `t00001271` | `` |
| 24 | `slice.Tensor` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001273` | `t00001274` |
| 25 | `to.dtype` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` | `t00001274` | `t00001274` |
| 26 | `item.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` | `t00001271` | `` |
| 27 | `slice.Tensor` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001275` | `t00001276` |
| 28 | `to.dtype` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` | `t00001276` | `t00001276` |
| 29 | `index.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001274, t00000023` | `t00001277` |
| 30 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001277` | `t00001278` |
| 31 | `index.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001276, t00000023` | `t00001279` |
| 32 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001279` | `t00001280` |
| 33 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001264, t00001278` | `t00001281` |
| 34 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001264` | `t00001282` |
| 35 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001264` | `t00001283` |
| 36 | `neg.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001283` | `t00001284` |
| 37 | `cat.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001284, t00001282` | `t00001285` |
| 38 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` | `t00001285, t00001280` | `t00001286` |
| 39 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope, attention` | `t00001281, t00001286` | `t00001287` |
| 40 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001266, t00001278` | `t00001288` |
| 41 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001266` | `t00001289` |
| 42 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001266` | `t00001290` |
| 43 | `neg.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001290` | `t00001291` |
| 44 | `cat.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001291, t00001289` | `t00001292` |
| 45 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001292, t00001280` | `t00001293` |
| 46 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001288, t00001293` | `t00001294` |
| 47 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001294` | `t00001295` |
| 48 | `matmul.default` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001287, t00001295` | `t00001296` |
| 49 | `div.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001296` | `t00001297` |
| 50 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001297, t00000053` | `t00001298` |
| 51 | `softmax.int` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001298` | `t00001299` |
| 52 | `to.dtype` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001299` | `t00001300` |
| 53 | `dropout.default` | `model.layers.17.self_attn` | `True` | `True` | `attention` | `t00001300` | `t00001300` |
| 54 | `matmul.default` | `model.layers.17.self_attn` | `True` | `True` | `attention, attention_output` | `t00001300, t00001268` | `t00001301` |
| 55 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001301` | `t00001302` |
| 56 | `contiguous.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` | `t00001302` | `t00001303` |
| 57 | `reshape.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` | `t00001303` | `t00001304` |
| 58 | `gt.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001305` |
| 59 | `is_nonzero.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001305` | `` |
| 60 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00000023` | `t00001306` |
| 61 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001306` | `t00001307` |
| 62 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001307` | `t00001308` |
| 63 | `eq.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001308` | `t00001309` |
| 64 | `is_nonzero.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001309` | `` |
| 65 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001304` | `t00001310` |
| 66 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001300` | `t00001311` |
| 67 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001311` | `t00001312` |
| 68 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001312, t00001268` | `t00001313` |
| 69 | `permute.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001313` | `t00001314` |
| 70 | `contiguous.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` | `t00001314` | `t00001315` |
| 71 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001315` | `t00001316` |
| 72 | `item.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001317` | `` |
| 73 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001316` | `t00001318` |
| 74 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001310` | `t00001319` |
| 75 | `sub.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001319, t00001318` | `t00001320` |
| 76 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001310` | `t00001321` |
| 77 | `cosine_similarity.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001320, t00001321` | `t00001322` |
| 78 | `squeeze.dim` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001322` | `t00001323` |
| 79 | `lt.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001323` | `t00001324` |
| 80 | `any.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001324` | `t00001325` |
| 81 | `item.default` | `model.layers.17.self_attn` | `True` | `True` | `` | `t00001325` | `` |
| 82 | `linear.default` | `model.layers.17.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001304, t00001326` | `t00001327` |
| 83 | `add.Tensor` | `model.layers.17` | `True` | `True` | `attention_output, mlp` | `t00001247, t00001327` | `t00001328` |
| 84 | `to.dtype` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001328` | `t00001329` |
| 85 | `pow.Tensor_Scalar` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001329` | `t00001330` |
| 86 | `mean.dim` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001330` | `t00001331` |
| 87 | `add.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001331` | `t00001332` |
| 88 | `rsqrt.default` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001332` | `t00001333` |
| 89 | `mul.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001329, t00001333` | `t00001334` |
| 90 | `to.dtype` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001334` | `t00001335` |
| 91 | `mul.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001336, t00001335` | `t00001337` |
| 92 | `linear.default` | `model.layers.17.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001337, t00001338` | `t00001339` |
| 93 | `silu.default` | `model.layers.17.mlp.act_fn` | `True` | `True` | `mlp` | `t00001339` | `t00001340` |
| 94 | `linear.default` | `model.layers.17.mlp.up_proj` | `True` | `True` | `mlp` | `t00001337, t00001341` | `t00001342` |
| 95 | `mul.Tensor` | `model.layers.17.mlp` | `True` | `True` | `mlp` | `t00001340, t00001342` | `t00001343` |
| 96 | `linear.default` | `model.layers.17.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001343, t00001344` | `t00001345` |
| 97 | `add.Tensor` | `model.layers.17` | `True` | `True` | `attention_output` | `t00001328, t00001345` | `t00001346` |
