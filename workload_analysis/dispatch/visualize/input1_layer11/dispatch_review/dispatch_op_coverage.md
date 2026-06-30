# input1_layer11 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000653` | `t00000654` |
| 2 | `pow.Tensor_Scalar` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000654` | `t00000655` |
| 3 | `mean.dim` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000655` | `t00000656` |
| 4 | `add.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000656` | `t00000657` |
| 5 | `rsqrt.default` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000657` | `t00000658` |
| 6 | `mul.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000654, t00000658` | `t00000659` |
| 7 | `to.dtype` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000659` | `t00000660` |
| 8 | `mul.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000661, t00000660` | `t00000662` |
| 9 | `linear.default` | `model.layers.11.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000662, t00000663` | `t00000664` |
| 10 | `linear.default` | `model.layers.11.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000662, t00000665` | `t00000666` |
| 11 | `linear.default` | `model.layers.11.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000662, t00000667` | `t00000668` |
| 12 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` | `t00000664` | `t00000669` |
| 13 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000669` | `t00000670` |
| 14 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` | `t00000666` | `t00000671` |
| 15 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000671` | `t00000672` |
| 16 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` | `t00000668` | `t00000673` |
| 17 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000673` | `t00000674` |
| 18 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000023` | `t00000675` |
| 19 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000675` | `t00000676` |
| 20 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000676` | `t00000677` |
| 21 | `gt.Scalar` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000677` | `t00000678` |
| 22 | `is_nonzero.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000678` | `` |
| 23 | `item.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` | `t00000677` | `` |
| 24 | `slice.Tensor` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000679` | `t00000680` |
| 25 | `to.dtype` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` | `t00000680` | `t00000680` |
| 26 | `item.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` | `t00000677` | `` |
| 27 | `slice.Tensor` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000681` | `t00000682` |
| 28 | `to.dtype` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` | `t00000682` | `t00000682` |
| 29 | `index.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000680, t00000023` | `t00000683` |
| 30 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000683` | `t00000684` |
| 31 | `index.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000682, t00000023` | `t00000685` |
| 32 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000685` | `t00000686` |
| 33 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000670, t00000684` | `t00000687` |
| 34 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000670` | `t00000688` |
| 35 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000670` | `t00000689` |
| 36 | `neg.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000689` | `t00000690` |
| 37 | `cat.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000690, t00000688` | `t00000691` |
| 38 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` | `t00000691, t00000686` | `t00000692` |
| 39 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope, attention` | `t00000687, t00000692` | `t00000693` |
| 40 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000672, t00000684` | `t00000694` |
| 41 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000672` | `t00000695` |
| 42 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000672` | `t00000696` |
| 43 | `neg.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000696` | `t00000697` |
| 44 | `cat.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000697, t00000695` | `t00000698` |
| 45 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000698, t00000686` | `t00000699` |
| 46 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000694, t00000699` | `t00000700` |
| 47 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000700` | `t00000701` |
| 48 | `matmul.default` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000693, t00000701` | `t00000702` |
| 49 | `div.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000702` | `t00000703` |
| 50 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000703, t00000053` | `t00000704` |
| 51 | `softmax.int` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000704` | `t00000705` |
| 52 | `to.dtype` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000705` | `t00000706` |
| 53 | `dropout.default` | `model.layers.11.self_attn` | `True` | `True` | `attention` | `t00000706` | `t00000706` |
| 54 | `matmul.default` | `model.layers.11.self_attn` | `True` | `True` | `attention, attention_output` | `t00000706, t00000674` | `t00000707` |
| 55 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000707` | `t00000708` |
| 56 | `contiguous.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` | `t00000708` | `t00000709` |
| 57 | `reshape.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` | `t00000709` | `t00000710` |
| 58 | `gt.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000711` |
| 59 | `is_nonzero.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000711` | `` |
| 60 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000023` | `t00000712` |
| 61 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000712` | `t00000713` |
| 62 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000713` | `t00000714` |
| 63 | `eq.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000714` | `t00000715` |
| 64 | `is_nonzero.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000715` | `` |
| 65 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000710` | `t00000716` |
| 66 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000706` | `t00000717` |
| 67 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000717` | `t00000718` |
| 68 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000718, t00000674` | `t00000719` |
| 69 | `permute.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000719` | `t00000720` |
| 70 | `contiguous.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` | `t00000720` | `t00000721` |
| 71 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000721` | `t00000722` |
| 72 | `item.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000723` | `` |
| 73 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000722` | `t00000724` |
| 74 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000716` | `t00000725` |
| 75 | `sub.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000725, t00000724` | `t00000726` |
| 76 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000716` | `t00000727` |
| 77 | `cosine_similarity.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000726, t00000727` | `t00000728` |
| 78 | `squeeze.dim` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000728` | `t00000729` |
| 79 | `lt.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000729` | `t00000730` |
| 80 | `any.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000730` | `t00000731` |
| 81 | `item.default` | `model.layers.11.self_attn` | `True` | `True` | `` | `t00000731` | `` |
| 82 | `linear.default` | `model.layers.11.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000710, t00000732` | `t00000733` |
| 83 | `add.Tensor` | `model.layers.11` | `True` | `True` | `attention_output, mlp` | `t00000653, t00000733` | `t00000734` |
| 84 | `to.dtype` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000734` | `t00000735` |
| 85 | `pow.Tensor_Scalar` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000735` | `t00000736` |
| 86 | `mean.dim` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000736` | `t00000737` |
| 87 | `add.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000737` | `t00000738` |
| 88 | `rsqrt.default` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000738` | `t00000739` |
| 89 | `mul.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000735, t00000739` | `t00000740` |
| 90 | `to.dtype` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000740` | `t00000741` |
| 91 | `mul.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000742, t00000741` | `t00000743` |
| 92 | `linear.default` | `model.layers.11.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000743, t00000744` | `t00000745` |
| 93 | `silu.default` | `model.layers.11.mlp.act_fn` | `True` | `True` | `mlp` | `t00000745` | `t00000746` |
| 94 | `linear.default` | `model.layers.11.mlp.up_proj` | `True` | `True` | `mlp` | `t00000743, t00000747` | `t00000748` |
| 95 | `mul.Tensor` | `model.layers.11.mlp` | `True` | `True` | `mlp` | `t00000746, t00000748` | `t00000749` |
| 96 | `linear.default` | `model.layers.11.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000749, t00000750` | `t00000751` |
| 97 | `add.Tensor` | `model.layers.11` | `True` | `True` | `attention_output` | `t00000734, t00000751` | `t00000752` |
