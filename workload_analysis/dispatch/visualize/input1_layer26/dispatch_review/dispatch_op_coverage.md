# input1_layer26 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002169` | `t00002170` |
| 2 | `pow.Tensor_Scalar` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002170` | `t00002171` |
| 3 | `mean.dim` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002171` | `t00002172` |
| 4 | `add.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002172` | `t00002173` |
| 5 | `rsqrt.default` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002173` | `t00002174` |
| 6 | `mul.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002170, t00002174` | `t00002175` |
| 7 | `to.dtype` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002175` | `t00002176` |
| 8 | `mul.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002177, t00002176` | `t00002178` |
| 9 | `linear.default` | `model.layers.26.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002178, t00002179` | `t00002180` |
| 10 | `linear.default` | `model.layers.26.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002178, t00002181` | `t00002182` |
| 11 | `linear.default` | `model.layers.26.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002178, t00002183` | `t00002184` |
| 12 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` | `t00002180` | `t00002185` |
| 13 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002185` | `t00002186` |
| 14 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` | `t00002182` | `t00002187` |
| 15 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002187` | `t00002188` |
| 16 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` | `t00002184` | `t00002189` |
| 17 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002189` | `t00002190` |
| 18 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00001475` | `t00002191` |
| 19 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002191` | `t00002192` |
| 20 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002192` | `t00002193` |
| 21 | `gt.Scalar` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002193` | `t00002194` |
| 22 | `is_nonzero.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002194` | `` |
| 23 | `item.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` | `t00002193` | `` |
| 24 | `slice.Tensor` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002195` | `t00002196` |
| 25 | `to.dtype` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` | `t00002196` | `t00002196` |
| 26 | `item.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` | `t00002193` | `` |
| 27 | `slice.Tensor` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002197` | `t00002198` |
| 28 | `to.dtype` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` | `t00002198` | `t00002198` |
| 29 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002196, t00001475` | `t00002199` |
| 30 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002199` | `t00002200` |
| 31 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002198, t00001475` | `t00002201` |
| 32 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002201` | `t00002202` |
| 33 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002186, t00002200` | `t00002203` |
| 34 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002186` | `t00002204` |
| 35 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002186` | `t00002205` |
| 36 | `neg.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002205` | `t00002206` |
| 37 | `cat.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002206, t00002204` | `t00002207` |
| 38 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` | `t00002207, t00002202` | `t00002208` |
| 39 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope, attention` | `t00002203, t00002208` | `t00002209` |
| 40 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002188, t00002200` | `t00002210` |
| 41 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002188` | `t00002211` |
| 42 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002188` | `t00002212` |
| 43 | `neg.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002212` | `t00002213` |
| 44 | `cat.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002213, t00002211` | `t00002214` |
| 45 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002214, t00002202` | `t00002215` |
| 46 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002210, t00002215` | `t00002216` |
| 47 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002216` | `t00002217` |
| 48 | `matmul.default` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002209, t00002217` | `t00002218` |
| 49 | `div.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002218` | `t00002219` |
| 50 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002219, t00001505` | `t00002220` |
| 51 | `softmax.int` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002220` | `t00002221` |
| 52 | `to.dtype` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002221` | `t00002222` |
| 53 | `dropout.default` | `model.layers.26.self_attn` | `True` | `True` | `attention` | `t00002222` | `t00002222` |
| 54 | `matmul.default` | `model.layers.26.self_attn` | `True` | `True` | `attention, attention_output` | `t00002222, t00002190` | `t00002223` |
| 55 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002223` | `t00002224` |
| 56 | `contiguous.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` | `t00002224` | `t00002225` |
| 57 | `reshape.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` | `t00002225` | `t00002226` |
| 58 | `gt.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00002227` |
| 59 | `is_nonzero.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002227` | `` |
| 60 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00001475` | `t00002228` |
| 61 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002228` | `t00002229` |
| 62 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002229` | `t00002230` |
| 63 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00000057` | `t00002231` |
| 64 | `sub.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002230, t00002231` | `t00002232` |
| 65 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002232` | `t00002233` |
| 66 | `eq.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002233` | `t00002234` |
| 67 | `is_nonzero.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002234` | `` |
| 68 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002226` | `t00002235` |
| 69 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002222` | `t00002236` |
| 70 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002236` | `t00002237` |
| 71 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002237, t00002190` | `t00002238` |
| 72 | `permute.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002238` | `t00002239` |
| 73 | `contiguous.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` | `t00002239` | `t00002240` |
| 74 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002240` | `t00002241` |
| 75 | `arange.start` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00002242` |
| 76 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002241, t00002242` | `t00002243` |
| 77 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002235` | `t00002244` |
| 78 | `sub.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002244, t00002243` | `t00002245` |
| 79 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002235` | `t00002246` |
| 80 | `cosine_similarity.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002245, t00002246` | `t00002247` |
| 81 | `squeeze.dim` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002247` | `t00002248` |
| 82 | `lt.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002248` | `t00002249` |
| 83 | `any.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002249` | `t00002250` |
| 84 | `item.default` | `model.layers.26.self_attn` | `True` | `True` | `` | `t00002250` | `` |
| 85 | `linear.default` | `model.layers.26.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002226, t00002251` | `t00002252` |
| 86 | `add.Tensor` | `model.layers.26` | `True` | `True` | `attention_output, mlp` | `t00002169, t00002252` | `t00002253` |
| 87 | `to.dtype` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002253` | `t00002254` |
| 88 | `pow.Tensor_Scalar` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002254` | `t00002255` |
| 89 | `mean.dim` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002255` | `t00002256` |
| 90 | `add.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002256` | `t00002257` |
| 91 | `rsqrt.default` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002257` | `t00002258` |
| 92 | `mul.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002254, t00002258` | `t00002259` |
| 93 | `to.dtype` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002259` | `t00002260` |
| 94 | `mul.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002261, t00002260` | `t00002262` |
| 95 | `linear.default` | `model.layers.26.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002262, t00002263` | `t00002264` |
| 96 | `silu.default` | `model.layers.26.mlp.act_fn` | `True` | `True` | `mlp` | `t00002264` | `t00002265` |
| 97 | `linear.default` | `model.layers.26.mlp.up_proj` | `True` | `True` | `mlp` | `t00002262, t00002266` | `t00002267` |
| 98 | `mul.Tensor` | `model.layers.26.mlp` | `True` | `True` | `mlp` | `t00002265, t00002267` | `t00002268` |
| 99 | `linear.default` | `model.layers.26.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002268, t00002269` | `t00002270` |
| 100 | `add.Tensor` | `model.layers.26` | `True` | `True` | `attention_output` | `t00002253, t00002270` | `t00002271` |
