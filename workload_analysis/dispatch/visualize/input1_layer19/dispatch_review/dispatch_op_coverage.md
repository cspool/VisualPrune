# input1_layer19 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001453` | `t00001454` |
| 2 | `pow.Tensor_Scalar` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001454` | `t00001455` |
| 3 | `mean.dim` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001455` | `t00001456` |
| 4 | `add.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001456` | `t00001457` |
| 5 | `rsqrt.default` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001457` | `t00001458` |
| 6 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001454, t00001458` | `t00001459` |
| 7 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001459` | `t00001460` |
| 8 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001461, t00001460` | `t00001462` |
| 9 | `linear.default` | `model.layers.19.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001462, t00001463` | `t00001464` |
| 10 | `linear.default` | `model.layers.19.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001462, t00001465` | `t00001466` |
| 11 | `linear.default` | `model.layers.19.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001462, t00001467` | `t00001468` |
| 12 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00001464` | `t00001469` |
| 13 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001469` | `t00001470` |
| 14 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00001466` | `t00001471` |
| 15 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001471` | `t00001472` |
| 16 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` | `t00001468` | `t00001473` |
| 17 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001473` | `t00001474` |
| 18 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001475` | `t00001476` |
| 19 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001476` | `t00001477` |
| 20 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001477` | `t00001478` |
| 21 | `gt.Scalar` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001478` | `t00001479` |
| 22 | `is_nonzero.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001479` | `` |
| 23 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00001478` | `` |
| 24 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001480` | `t00001481` |
| 25 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00001481` | `t00001481` |
| 26 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00001478` | `` |
| 27 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001482` | `t00001483` |
| 28 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` | `t00001483` | `t00001483` |
| 29 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001481, t00001475` | `t00001484` |
| 30 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001484` | `t00001485` |
| 31 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001483, t00001475` | `t00001486` |
| 32 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001486` | `t00001487` |
| 33 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001470, t00001485` | `t00001488` |
| 34 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001470` | `t00001489` |
| 35 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001470` | `t00001490` |
| 36 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001490` | `t00001491` |
| 37 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001491, t00001489` | `t00001492` |
| 38 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` | `t00001492, t00001487` | `t00001493` |
| 39 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope, attention` | `t00001488, t00001493` | `t00001494` |
| 40 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001472, t00001485` | `t00001495` |
| 41 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001472` | `t00001496` |
| 42 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001472` | `t00001497` |
| 43 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001497` | `t00001498` |
| 44 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001498, t00001496` | `t00001499` |
| 45 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001499, t00001487` | `t00001500` |
| 46 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001495, t00001500` | `t00001501` |
| 47 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001501` | `t00001502` |
| 48 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001494, t00001502` | `t00001503` |
| 49 | `div.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001503` | `t00001504` |
| 50 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001504, t00001505` | `t00001506` |
| 51 | `softmax.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001506` | `t00001507` |
| 52 | `to.dtype` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001507` | `t00001508` |
| 53 | `dropout.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` | `t00001508` | `t00001508` |
| 54 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention, attention_output` | `t00001508, t00001474` | `t00001509` |
| 55 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001509` | `t00001510` |
| 56 | `contiguous.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` | `t00001510` | `t00001511` |
| 57 | `reshape.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` | `t00001511` | `t00001512` |
| 58 | `gt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001513` |
| 59 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001513` | `` |
| 60 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001475` | `t00001514` |
| 61 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001514` | `t00001515` |
| 62 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001515` | `t00001516` |
| 63 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00000057` | `t00001517` |
| 64 | `sub.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001516, t00001517` | `t00001518` |
| 65 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001518` | `t00001519` |
| 66 | `eq.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001519` | `t00001520` |
| 67 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001520` | `` |
| 68 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001512` | `t00001521` |
| 69 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001508` | `t00001522` |
| 70 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001522` | `t00001523` |
| 71 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001523, t00001474` | `t00001524` |
| 72 | `permute.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001524` | `t00001525` |
| 73 | `contiguous.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` | `t00001525` | `t00001526` |
| 74 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001526` | `t00001527` |
| 75 | `arange.start` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00001528` |
| 76 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001527, t00001528` | `t00001529` |
| 77 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001521` | `t00001530` |
| 78 | `sub.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001530, t00001529` | `t00001531` |
| 79 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001521` | `t00001532` |
| 80 | `cosine_similarity.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001531, t00001532` | `t00001533` |
| 81 | `squeeze.dim` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001533` | `t00001534` |
| 82 | `lt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001534` | `t00001535` |
| 83 | `any.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001535` | `t00001536` |
| 84 | `item.default` | `model.layers.19.self_attn` | `True` | `True` | `` | `t00001536` | `` |
| 85 | `linear.default` | `model.layers.19.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001512, t00001537` | `t00001538` |
| 86 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output, mlp` | `t00001453, t00001538` | `t00001539` |
| 87 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001539` | `t00001540` |
| 88 | `pow.Tensor_Scalar` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001540` | `t00001541` |
| 89 | `mean.dim` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001541` | `t00001542` |
| 90 | `add.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001542` | `t00001543` |
| 91 | `rsqrt.default` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001543` | `t00001544` |
| 92 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001540, t00001544` | `t00001545` |
| 93 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001545` | `t00001546` |
| 94 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001547, t00001546` | `t00001548` |
| 95 | `linear.default` | `model.layers.19.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001548, t00001549` | `t00001550` |
| 96 | `silu.default` | `model.layers.19.mlp.act_fn` | `True` | `True` | `mlp` | `t00001550` | `t00001551` |
| 97 | `linear.default` | `model.layers.19.mlp.up_proj` | `True` | `True` | `mlp` | `t00001548, t00001552` | `t00001553` |
| 98 | `mul.Tensor` | `model.layers.19.mlp` | `True` | `True` | `mlp` | `t00001551, t00001553` | `t00001554` |
| 99 | `linear.default` | `model.layers.19.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001554, t00001555` | `t00001556` |
| 100 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output` | `t00001539, t00001556` | `t00001557` |
