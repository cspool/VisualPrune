# input1_layer14 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000950` | `t00000951` |
| 2 | `pow.Tensor_Scalar` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000951` | `t00000952` |
| 3 | `mean.dim` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000952` | `t00000953` |
| 4 | `add.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000953` | `t00000954` |
| 5 | `rsqrt.default` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000954` | `t00000955` |
| 6 | `mul.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000951, t00000955` | `t00000956` |
| 7 | `to.dtype` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000956` | `t00000957` |
| 8 | `mul.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000958, t00000957` | `t00000959` |
| 9 | `linear.default` | `model.layers.14.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000959, t00000960` | `t00000961` |
| 10 | `linear.default` | `model.layers.14.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000959, t00000962` | `t00000963` |
| 11 | `linear.default` | `model.layers.14.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000959, t00000964` | `t00000965` |
| 12 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` | `t00000961` | `t00000966` |
| 13 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000966` | `t00000967` |
| 14 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` | `t00000963` | `t00000968` |
| 15 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000968` | `t00000969` |
| 16 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` | `t00000965` | `t00000970` |
| 17 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000970` | `t00000971` |
| 18 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000023` | `t00000972` |
| 19 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000972` | `t00000973` |
| 20 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000973` | `t00000974` |
| 21 | `gt.Scalar` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000974` | `t00000975` |
| 22 | `is_nonzero.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000975` | `` |
| 23 | `item.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` | `t00000974` | `` |
| 24 | `slice.Tensor` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000976` | `t00000977` |
| 25 | `to.dtype` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` | `t00000977` | `t00000977` |
| 26 | `item.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` | `t00000974` | `` |
| 27 | `slice.Tensor` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000978` | `t00000979` |
| 28 | `to.dtype` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` | `t00000979` | `t00000979` |
| 29 | `index.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000977, t00000023` | `t00000980` |
| 30 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000980` | `t00000981` |
| 31 | `index.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000979, t00000023` | `t00000982` |
| 32 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000982` | `t00000983` |
| 33 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000967, t00000981` | `t00000984` |
| 34 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000967` | `t00000985` |
| 35 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000967` | `t00000986` |
| 36 | `neg.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000986` | `t00000987` |
| 37 | `cat.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000987, t00000985` | `t00000988` |
| 38 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` | `t00000988, t00000983` | `t00000989` |
| 39 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope, attention` | `t00000984, t00000989` | `t00000990` |
| 40 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000969, t00000981` | `t00000991` |
| 41 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000969` | `t00000992` |
| 42 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000969` | `t00000993` |
| 43 | `neg.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000993` | `t00000994` |
| 44 | `cat.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000994, t00000992` | `t00000995` |
| 45 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000995, t00000983` | `t00000996` |
| 46 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00000991, t00000996` | `t00000997` |
| 47 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00000997` | `t00000998` |
| 48 | `matmul.default` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00000990, t00000998` | `t00000999` |
| 49 | `div.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00000999` | `t00001000` |
| 50 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00001000, t00000053` | `t00001001` |
| 51 | `softmax.int` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00001001` | `t00001002` |
| 52 | `to.dtype` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001002` | `t00001003` |
| 53 | `dropout.default` | `model.layers.14.self_attn` | `True` | `True` | `attention` | `t00001003` | `t00001003` |
| 54 | `matmul.default` | `model.layers.14.self_attn` | `True` | `True` | `attention, attention_output` | `t00001003, t00000971` | `t00001004` |
| 55 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001004` | `t00001005` |
| 56 | `contiguous.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` | `t00001005` | `t00001006` |
| 57 | `reshape.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` | `t00001006` | `t00001007` |
| 58 | `gt.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001008` |
| 59 | `is_nonzero.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001008` | `` |
| 60 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00000023` | `t00001009` |
| 61 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001009` | `t00001010` |
| 62 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001010` | `t00001011` |
| 63 | `eq.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001011` | `t00001012` |
| 64 | `is_nonzero.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001012` | `` |
| 65 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001007` | `t00001013` |
| 66 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001003` | `t00001014` |
| 67 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001014` | `t00001015` |
| 68 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001015, t00000971` | `t00001016` |
| 69 | `permute.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001016` | `t00001017` |
| 70 | `contiguous.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` | `t00001017` | `t00001018` |
| 71 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001018` | `t00001019` |
| 72 | `item.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001020` | `` |
| 73 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001019` | `t00001021` |
| 74 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001013` | `t00001022` |
| 75 | `sub.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001022, t00001021` | `t00001023` |
| 76 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001013` | `t00001024` |
| 77 | `cosine_similarity.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001023, t00001024` | `t00001025` |
| 78 | `squeeze.dim` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001025` | `t00001026` |
| 79 | `lt.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001026` | `t00001027` |
| 80 | `any.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001027` | `t00001028` |
| 81 | `item.default` | `model.layers.14.self_attn` | `True` | `True` | `` | `t00001028` | `` |
| 82 | `linear.default` | `model.layers.14.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001007, t00001029` | `t00001030` |
| 83 | `add.Tensor` | `model.layers.14` | `True` | `True` | `attention_output, mlp` | `t00000950, t00001030` | `t00001031` |
| 84 | `to.dtype` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001031` | `t00001032` |
| 85 | `pow.Tensor_Scalar` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001032` | `t00001033` |
| 86 | `mean.dim` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001033` | `t00001034` |
| 87 | `add.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001034` | `t00001035` |
| 88 | `rsqrt.default` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001035` | `t00001036` |
| 89 | `mul.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001032, t00001036` | `t00001037` |
| 90 | `to.dtype` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001037` | `t00001038` |
| 91 | `mul.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001039, t00001038` | `t00001040` |
| 92 | `linear.default` | `model.layers.14.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001040, t00001041` | `t00001042` |
| 93 | `silu.default` | `model.layers.14.mlp.act_fn` | `True` | `True` | `mlp` | `t00001042` | `t00001043` |
| 94 | `linear.default` | `model.layers.14.mlp.up_proj` | `True` | `True` | `mlp` | `t00001040, t00001044` | `t00001045` |
| 95 | `mul.Tensor` | `model.layers.14.mlp` | `True` | `True` | `mlp` | `t00001043, t00001045` | `t00001046` |
| 96 | `linear.default` | `model.layers.14.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001046, t00001047` | `t00001048` |
| 97 | `add.Tensor` | `model.layers.14` | `True` | `True` | `attention_output` | `t00001031, t00001048` | `t00001049` |
