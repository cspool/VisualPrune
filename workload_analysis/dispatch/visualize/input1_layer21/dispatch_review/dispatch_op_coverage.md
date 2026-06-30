# input1_layer21 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001659` | `t00001660` |
| 2 | `pow.Tensor_Scalar` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001660` | `t00001661` |
| 3 | `mean.dim` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001661` | `t00001662` |
| 4 | `add.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001662` | `t00001663` |
| 5 | `rsqrt.default` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001663` | `t00001664` |
| 6 | `mul.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001660, t00001664` | `t00001665` |
| 7 | `to.dtype` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001665` | `t00001666` |
| 8 | `mul.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001667, t00001666` | `t00001668` |
| 9 | `linear.default` | `model.layers.21.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001668, t00001669` | `t00001670` |
| 10 | `linear.default` | `model.layers.21.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001668, t00001671` | `t00001672` |
| 11 | `linear.default` | `model.layers.21.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001668, t00001673` | `t00001674` |
| 12 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` | `t00001670` | `t00001675` |
| 13 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001675` | `t00001676` |
| 14 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` | `t00001672` | `t00001677` |
| 15 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001677` | `t00001678` |
| 16 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` | `t00001674` | `t00001679` |
| 17 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001679` | `t00001680` |
| 18 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001475` | `t00001681` |
| 19 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001681` | `t00001682` |
| 20 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001682` | `t00001683` |
| 21 | `gt.Scalar` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001683` | `t00001684` |
| 22 | `is_nonzero.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001684` | `` |
| 23 | `item.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` | `t00001683` | `` |
| 24 | `slice.Tensor` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001685` | `t00001686` |
| 25 | `to.dtype` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` | `t00001686` | `t00001686` |
| 26 | `item.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` | `t00001683` | `` |
| 27 | `slice.Tensor` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001687` | `t00001688` |
| 28 | `to.dtype` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` | `t00001688` | `t00001688` |
| 29 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001686, t00001475` | `t00001689` |
| 30 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001689` | `t00001690` |
| 31 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001688, t00001475` | `t00001691` |
| 32 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001691` | `t00001692` |
| 33 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001676, t00001690` | `t00001693` |
| 34 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001676` | `t00001694` |
| 35 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001676` | `t00001695` |
| 36 | `neg.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001695` | `t00001696` |
| 37 | `cat.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001696, t00001694` | `t00001697` |
| 38 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` | `t00001697, t00001692` | `t00001698` |
| 39 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope, attention` | `t00001693, t00001698` | `t00001699` |
| 40 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001678, t00001690` | `t00001700` |
| 41 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001678` | `t00001701` |
| 42 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001678` | `t00001702` |
| 43 | `neg.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001702` | `t00001703` |
| 44 | `cat.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001703, t00001701` | `t00001704` |
| 45 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001704, t00001692` | `t00001705` |
| 46 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001700, t00001705` | `t00001706` |
| 47 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001706` | `t00001707` |
| 48 | `matmul.default` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001699, t00001707` | `t00001708` |
| 49 | `div.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001708` | `t00001709` |
| 50 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001709, t00001505` | `t00001710` |
| 51 | `softmax.int` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001710` | `t00001711` |
| 52 | `to.dtype` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001711` | `t00001712` |
| 53 | `dropout.default` | `model.layers.21.self_attn` | `True` | `True` | `attention` | `t00001712` | `t00001712` |
| 54 | `matmul.default` | `model.layers.21.self_attn` | `True` | `True` | `attention, attention_output` | `t00001712, t00001680` | `t00001713` |
| 55 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001713` | `t00001714` |
| 56 | `contiguous.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` | `t00001714` | `t00001715` |
| 57 | `reshape.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` | `t00001715` | `t00001716` |
| 58 | `gt.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001717` |
| 59 | `is_nonzero.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001717` | `` |
| 60 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001475` | `t00001718` |
| 61 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001718` | `t00001719` |
| 62 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001719` | `t00001720` |
| 63 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00000057` | `t00001721` |
| 64 | `sub.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001720, t00001721` | `t00001722` |
| 65 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001722` | `t00001723` |
| 66 | `eq.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001723` | `t00001724` |
| 67 | `is_nonzero.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001724` | `` |
| 68 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001716` | `t00001725` |
| 69 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001712` | `t00001726` |
| 70 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001726` | `t00001727` |
| 71 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001727, t00001680` | `t00001728` |
| 72 | `permute.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001728` | `t00001729` |
| 73 | `contiguous.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` | `t00001729` | `t00001730` |
| 74 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001730` | `t00001731` |
| 75 | `arange.start` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00001732` |
| 76 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001731, t00001732` | `t00001733` |
| 77 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001725` | `t00001734` |
| 78 | `sub.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001734, t00001733` | `t00001735` |
| 79 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001725` | `t00001736` |
| 80 | `cosine_similarity.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001735, t00001736` | `t00001737` |
| 81 | `squeeze.dim` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001737` | `t00001738` |
| 82 | `lt.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001738` | `t00001739` |
| 83 | `any.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001739` | `t00001740` |
| 84 | `item.default` | `model.layers.21.self_attn` | `True` | `True` | `` | `t00001740` | `` |
| 85 | `linear.default` | `model.layers.21.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001716, t00001741` | `t00001742` |
| 86 | `add.Tensor` | `model.layers.21` | `True` | `True` | `attention_output, mlp` | `t00001659, t00001742` | `t00001743` |
| 87 | `to.dtype` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001743` | `t00001744` |
| 88 | `pow.Tensor_Scalar` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001744` | `t00001745` |
| 89 | `mean.dim` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001745` | `t00001746` |
| 90 | `add.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001746` | `t00001747` |
| 91 | `rsqrt.default` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001747` | `t00001748` |
| 92 | `mul.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001744, t00001748` | `t00001749` |
| 93 | `to.dtype` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001749` | `t00001750` |
| 94 | `mul.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001751, t00001750` | `t00001752` |
| 95 | `linear.default` | `model.layers.21.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001752, t00001753` | `t00001754` |
| 96 | `silu.default` | `model.layers.21.mlp.act_fn` | `True` | `True` | `mlp` | `t00001754` | `t00001755` |
| 97 | `linear.default` | `model.layers.21.mlp.up_proj` | `True` | `True` | `mlp` | `t00001752, t00001756` | `t00001757` |
| 98 | `mul.Tensor` | `model.layers.21.mlp` | `True` | `True` | `mlp` | `t00001755, t00001757` | `t00001758` |
| 99 | `linear.default` | `model.layers.21.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001758, t00001759` | `t00001760` |
| 100 | `add.Tensor` | `model.layers.21` | `True` | `True` | `attention_output` | `t00001743, t00001760` | `t00001761` |
