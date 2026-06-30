# input1_layer9 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000455` | `t00000456` |
| 2 | `pow.Tensor_Scalar` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000456` | `t00000457` |
| 3 | `mean.dim` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000457` | `t00000458` |
| 4 | `add.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000458` | `t00000459` |
| 5 | `rsqrt.default` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000459` | `t00000460` |
| 6 | `mul.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000456, t00000460` | `t00000461` |
| 7 | `to.dtype` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000461` | `t00000462` |
| 8 | `mul.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000463, t00000462` | `t00000464` |
| 9 | `linear.default` | `model.layers.9.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000464, t00000465` | `t00000466` |
| 10 | `linear.default` | `model.layers.9.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000464, t00000467` | `t00000468` |
| 11 | `linear.default` | `model.layers.9.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000464, t00000469` | `t00000470` |
| 12 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` | `t00000466` | `t00000471` |
| 13 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000471` | `t00000472` |
| 14 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` | `t00000468` | `t00000473` |
| 15 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000473` | `t00000474` |
| 16 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` | `t00000470` | `t00000475` |
| 17 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000475` | `t00000476` |
| 18 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000023` | `t00000477` |
| 19 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000477` | `t00000478` |
| 20 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000478` | `t00000479` |
| 21 | `gt.Scalar` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000479` | `t00000480` |
| 22 | `is_nonzero.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000480` | `` |
| 23 | `item.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` | `t00000479` | `` |
| 24 | `slice.Tensor` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000481` | `t00000482` |
| 25 | `to.dtype` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` | `t00000482` | `t00000482` |
| 26 | `item.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` | `t00000479` | `` |
| 27 | `slice.Tensor` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000483` | `t00000484` |
| 28 | `to.dtype` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` | `t00000484` | `t00000484` |
| 29 | `index.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000482, t00000023` | `t00000485` |
| 30 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000485` | `t00000486` |
| 31 | `index.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000484, t00000023` | `t00000487` |
| 32 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000487` | `t00000488` |
| 33 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000472, t00000486` | `t00000489` |
| 34 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000472` | `t00000490` |
| 35 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000472` | `t00000491` |
| 36 | `neg.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000491` | `t00000492` |
| 37 | `cat.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000492, t00000490` | `t00000493` |
| 38 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` | `t00000493, t00000488` | `t00000494` |
| 39 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope, attention` | `t00000489, t00000494` | `t00000495` |
| 40 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000474, t00000486` | `t00000496` |
| 41 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000474` | `t00000497` |
| 42 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000474` | `t00000498` |
| 43 | `neg.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000498` | `t00000499` |
| 44 | `cat.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000499, t00000497` | `t00000500` |
| 45 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000500, t00000488` | `t00000501` |
| 46 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000496, t00000501` | `t00000502` |
| 47 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000502` | `t00000503` |
| 48 | `matmul.default` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000495, t00000503` | `t00000504` |
| 49 | `div.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000504` | `t00000505` |
| 50 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000505, t00000053` | `t00000506` |
| 51 | `softmax.int` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000506` | `t00000507` |
| 52 | `to.dtype` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000507` | `t00000508` |
| 53 | `dropout.default` | `model.layers.9.self_attn` | `True` | `True` | `attention` | `t00000508` | `t00000508` |
| 54 | `matmul.default` | `model.layers.9.self_attn` | `True` | `True` | `attention, attention_output` | `t00000508, t00000476` | `t00000509` |
| 55 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000509` | `t00000510` |
| 56 | `contiguous.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` | `t00000510` | `t00000511` |
| 57 | `reshape.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` | `t00000511` | `t00000512` |
| 58 | `gt.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000513` |
| 59 | `is_nonzero.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000513` | `` |
| 60 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000023` | `t00000514` |
| 61 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000514` | `t00000515` |
| 62 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000515` | `t00000516` |
| 63 | `eq.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000516` | `t00000517` |
| 64 | `is_nonzero.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000517` | `` |
| 65 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000512` | `t00000518` |
| 66 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000508` | `t00000519` |
| 67 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000519` | `t00000520` |
| 68 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000520, t00000476` | `t00000521` |
| 69 | `permute.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000521` | `t00000522` |
| 70 | `contiguous.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` | `t00000522` | `t00000523` |
| 71 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000523` | `t00000524` |
| 72 | `item.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000525` | `` |
| 73 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000524` | `t00000526` |
| 74 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000518` | `t00000527` |
| 75 | `sub.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000527, t00000526` | `t00000528` |
| 76 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000518` | `t00000529` |
| 77 | `cosine_similarity.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000528, t00000529` | `t00000530` |
| 78 | `squeeze.dim` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000530` | `t00000531` |
| 79 | `lt.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000531` | `t00000532` |
| 80 | `any.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000532` | `t00000533` |
| 81 | `item.default` | `model.layers.9.self_attn` | `True` | `True` | `` | `t00000533` | `` |
| 82 | `linear.default` | `model.layers.9.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000512, t00000534` | `t00000535` |
| 83 | `add.Tensor` | `model.layers.9` | `True` | `True` | `attention_output, mlp` | `t00000455, t00000535` | `t00000536` |
| 84 | `to.dtype` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000536` | `t00000537` |
| 85 | `pow.Tensor_Scalar` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000537` | `t00000538` |
| 86 | `mean.dim` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000538` | `t00000539` |
| 87 | `add.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000539` | `t00000540` |
| 88 | `rsqrt.default` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000540` | `t00000541` |
| 89 | `mul.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000537, t00000541` | `t00000542` |
| 90 | `to.dtype` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000542` | `t00000543` |
| 91 | `mul.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000544, t00000543` | `t00000545` |
| 92 | `linear.default` | `model.layers.9.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000545, t00000546` | `t00000547` |
| 93 | `silu.default` | `model.layers.9.mlp.act_fn` | `True` | `True` | `mlp` | `t00000547` | `t00000548` |
| 94 | `linear.default` | `model.layers.9.mlp.up_proj` | `True` | `True` | `mlp` | `t00000545, t00000549` | `t00000550` |
| 95 | `mul.Tensor` | `model.layers.9.mlp` | `True` | `True` | `mlp` | `t00000548, t00000550` | `t00000551` |
| 96 | `linear.default` | `model.layers.9.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000551, t00000552` | `t00000553` |
| 97 | `add.Tensor` | `model.layers.9` | `True` | `True` | `attention_output` | `t00000536, t00000553` | `t00000554` |
