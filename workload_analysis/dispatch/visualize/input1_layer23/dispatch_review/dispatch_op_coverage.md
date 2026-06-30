# input1_layer23 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001863` | `t00001864` |
| 2 | `pow.Tensor_Scalar` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001864` | `t00001865` |
| 3 | `mean.dim` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001865` | `t00001866` |
| 4 | `add.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001866` | `t00001867` |
| 5 | `rsqrt.default` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001867` | `t00001868` |
| 6 | `mul.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001864, t00001868` | `t00001869` |
| 7 | `to.dtype` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001869` | `t00001870` |
| 8 | `mul.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001871, t00001870` | `t00001872` |
| 9 | `linear.default` | `model.layers.23.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001872, t00001873` | `t00001874` |
| 10 | `linear.default` | `model.layers.23.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001872, t00001875` | `t00001876` |
| 11 | `linear.default` | `model.layers.23.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001872, t00001877` | `t00001878` |
| 12 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` | `t00001874` | `t00001879` |
| 13 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001879` | `t00001880` |
| 14 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` | `t00001876` | `t00001881` |
| 15 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001881` | `t00001882` |
| 16 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` | `t00001878` | `t00001883` |
| 17 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001883` | `t00001884` |
| 18 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001475` | `t00001885` |
| 19 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001885` | `t00001886` |
| 20 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001886` | `t00001887` |
| 21 | `gt.Scalar` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001887` | `t00001888` |
| 22 | `is_nonzero.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001888` | `` |
| 23 | `item.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` | `t00001887` | `` |
| 24 | `slice.Tensor` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001889` | `t00001890` |
| 25 | `to.dtype` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` | `t00001890` | `t00001890` |
| 26 | `item.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` | `t00001887` | `` |
| 27 | `slice.Tensor` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001891` | `t00001892` |
| 28 | `to.dtype` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` | `t00001892` | `t00001892` |
| 29 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001890, t00001475` | `t00001893` |
| 30 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001893` | `t00001894` |
| 31 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001892, t00001475` | `t00001895` |
| 32 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001895` | `t00001896` |
| 33 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001880, t00001894` | `t00001897` |
| 34 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001880` | `t00001898` |
| 35 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001880` | `t00001899` |
| 36 | `neg.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001899` | `t00001900` |
| 37 | `cat.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001900, t00001898` | `t00001901` |
| 38 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` | `t00001901, t00001896` | `t00001902` |
| 39 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope, attention` | `t00001897, t00001902` | `t00001903` |
| 40 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001882, t00001894` | `t00001904` |
| 41 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001882` | `t00001905` |
| 42 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001882` | `t00001906` |
| 43 | `neg.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001906` | `t00001907` |
| 44 | `cat.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001907, t00001905` | `t00001908` |
| 45 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001908, t00001896` | `t00001909` |
| 46 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001904, t00001909` | `t00001910` |
| 47 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001910` | `t00001911` |
| 48 | `matmul.default` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001903, t00001911` | `t00001912` |
| 49 | `div.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001912` | `t00001913` |
| 50 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001913, t00001505` | `t00001914` |
| 51 | `softmax.int` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001914` | `t00001915` |
| 52 | `to.dtype` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001915` | `t00001916` |
| 53 | `dropout.default` | `model.layers.23.self_attn` | `True` | `True` | `attention` | `t00001916` | `t00001916` |
| 54 | `matmul.default` | `model.layers.23.self_attn` | `True` | `True` | `attention, attention_output` | `t00001916, t00001884` | `t00001917` |
| 55 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001917` | `t00001918` |
| 56 | `contiguous.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` | `t00001918` | `t00001919` |
| 57 | `reshape.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` | `t00001919` | `t00001920` |
| 58 | `gt.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001921` |
| 59 | `is_nonzero.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001921` | `` |
| 60 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001475` | `t00001922` |
| 61 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001922` | `t00001923` |
| 62 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001923` | `t00001924` |
| 63 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00000057` | `t00001925` |
| 64 | `sub.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001924, t00001925` | `t00001926` |
| 65 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001926` | `t00001927` |
| 66 | `eq.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001927` | `t00001928` |
| 67 | `is_nonzero.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001928` | `` |
| 68 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001920` | `t00001929` |
| 69 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001916` | `t00001930` |
| 70 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001930` | `t00001931` |
| 71 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001931, t00001884` | `t00001932` |
| 72 | `permute.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001932` | `t00001933` |
| 73 | `contiguous.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` | `t00001933` | `t00001934` |
| 74 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001934` | `t00001935` |
| 75 | `arange.start` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00001936` |
| 76 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001935, t00001936` | `t00001937` |
| 77 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001929` | `t00001938` |
| 78 | `sub.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001938, t00001937` | `t00001939` |
| 79 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001929` | `t00001940` |
| 80 | `cosine_similarity.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001939, t00001940` | `t00001941` |
| 81 | `squeeze.dim` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001941` | `t00001942` |
| 82 | `lt.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001942` | `t00001943` |
| 83 | `any.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001943` | `t00001944` |
| 84 | `item.default` | `model.layers.23.self_attn` | `True` | `True` | `` | `t00001944` | `` |
| 85 | `linear.default` | `model.layers.23.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001920, t00001945` | `t00001946` |
| 86 | `add.Tensor` | `model.layers.23` | `True` | `True` | `attention_output, mlp` | `t00001863, t00001946` | `t00001947` |
| 87 | `to.dtype` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001947` | `t00001948` |
| 88 | `pow.Tensor_Scalar` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001948` | `t00001949` |
| 89 | `mean.dim` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001949` | `t00001950` |
| 90 | `add.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001950` | `t00001951` |
| 91 | `rsqrt.default` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001951` | `t00001952` |
| 92 | `mul.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001948, t00001952` | `t00001953` |
| 93 | `to.dtype` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001953` | `t00001954` |
| 94 | `mul.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001955, t00001954` | `t00001956` |
| 95 | `linear.default` | `model.layers.23.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001956, t00001957` | `t00001958` |
| 96 | `silu.default` | `model.layers.23.mlp.act_fn` | `True` | `True` | `mlp` | `t00001958` | `t00001959` |
| 97 | `linear.default` | `model.layers.23.mlp.up_proj` | `True` | `True` | `mlp` | `t00001956, t00001960` | `t00001961` |
| 98 | `mul.Tensor` | `model.layers.23.mlp` | `True` | `True` | `mlp` | `t00001959, t00001961` | `t00001962` |
| 99 | `linear.default` | `model.layers.23.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001962, t00001963` | `t00001964` |
| 100 | `add.Tensor` | `model.layers.23` | `True` | `True` | `attention_output` | `t00001947, t00001964` | `t00001965` |
