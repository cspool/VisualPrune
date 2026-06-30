# input1_layer24 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001965` | `t00001966` |
| 2 | `pow.Tensor_Scalar` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001966` | `t00001967` |
| 3 | `mean.dim` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001967` | `t00001968` |
| 4 | `add.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001968` | `t00001969` |
| 5 | `rsqrt.default` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001969` | `t00001970` |
| 6 | `mul.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001966, t00001970` | `t00001971` |
| 7 | `to.dtype` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001971` | `t00001972` |
| 8 | `mul.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001973, t00001972` | `t00001974` |
| 9 | `linear.default` | `model.layers.24.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001974, t00001975` | `t00001976` |
| 10 | `linear.default` | `model.layers.24.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001974, t00001977` | `t00001978` |
| 11 | `linear.default` | `model.layers.24.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001974, t00001979` | `t00001980` |
| 12 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` | `t00001976` | `t00001981` |
| 13 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001981` | `t00001982` |
| 14 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` | `t00001978` | `t00001983` |
| 15 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001983` | `t00001984` |
| 16 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` | `t00001980` | `t00001985` |
| 17 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001985` | `t00001986` |
| 18 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001475` | `t00001987` |
| 19 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001987` | `t00001988` |
| 20 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001988` | `t00001989` |
| 21 | `gt.Scalar` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001989` | `t00001990` |
| 22 | `is_nonzero.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001990` | `` |
| 23 | `item.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` | `t00001989` | `` |
| 24 | `slice.Tensor` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001991` | `t00001992` |
| 25 | `to.dtype` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` | `t00001992` | `t00001992` |
| 26 | `item.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` | `t00001989` | `` |
| 27 | `slice.Tensor` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001993` | `t00001994` |
| 28 | `to.dtype` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` | `t00001994` | `t00001994` |
| 29 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001992, t00001475` | `t00001995` |
| 30 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001995` | `t00001996` |
| 31 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001994, t00001475` | `t00001997` |
| 32 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001997` | `t00001998` |
| 33 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001982, t00001996` | `t00001999` |
| 34 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001982` | `t00002000` |
| 35 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00001982` | `t00002001` |
| 36 | `neg.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00002001` | `t00002002` |
| 37 | `cat.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00002002, t00002000` | `t00002003` |
| 38 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` | `t00002003, t00001998` | `t00002004` |
| 39 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope, attention` | `t00001999, t00002004` | `t00002005` |
| 40 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001984, t00001996` | `t00002006` |
| 41 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001984` | `t00002007` |
| 42 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001984` | `t00002008` |
| 43 | `neg.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002008` | `t00002009` |
| 44 | `cat.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002009, t00002007` | `t00002010` |
| 45 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002010, t00001998` | `t00002011` |
| 46 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002006, t00002011` | `t00002012` |
| 47 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002012` | `t00002013` |
| 48 | `matmul.default` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002005, t00002013` | `t00002014` |
| 49 | `div.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002014` | `t00002015` |
| 50 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002015, t00001505` | `t00002016` |
| 51 | `softmax.int` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002016` | `t00002017` |
| 52 | `to.dtype` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002017` | `t00002018` |
| 53 | `dropout.default` | `model.layers.24.self_attn` | `True` | `True` | `attention` | `t00002018` | `t00002018` |
| 54 | `matmul.default` | `model.layers.24.self_attn` | `True` | `True` | `attention, attention_output` | `t00002018, t00001986` | `t00002019` |
| 55 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002019` | `t00002020` |
| 56 | `contiguous.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` | `t00002020` | `t00002021` |
| 57 | `reshape.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` | `t00002021` | `t00002022` |
| 58 | `gt.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00002023` |
| 59 | `is_nonzero.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002023` | `` |
| 60 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00001475` | `t00002024` |
| 61 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002024` | `t00002025` |
| 62 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002025` | `t00002026` |
| 63 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00000057` | `t00002027` |
| 64 | `sub.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002026, t00002027` | `t00002028` |
| 65 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002028` | `t00002029` |
| 66 | `eq.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002029` | `t00002030` |
| 67 | `is_nonzero.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002030` | `` |
| 68 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002022` | `t00002031` |
| 69 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002018` | `t00002032` |
| 70 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002032` | `t00002033` |
| 71 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002033, t00001986` | `t00002034` |
| 72 | `permute.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002034` | `t00002035` |
| 73 | `contiguous.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` | `t00002035` | `t00002036` |
| 74 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002036` | `t00002037` |
| 75 | `arange.start` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00002038` |
| 76 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002037, t00002038` | `t00002039` |
| 77 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002031` | `t00002040` |
| 78 | `sub.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002040, t00002039` | `t00002041` |
| 79 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002031` | `t00002042` |
| 80 | `cosine_similarity.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002041, t00002042` | `t00002043` |
| 81 | `squeeze.dim` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002043` | `t00002044` |
| 82 | `lt.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002044` | `t00002045` |
| 83 | `any.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002045` | `t00002046` |
| 84 | `item.default` | `model.layers.24.self_attn` | `True` | `True` | `` | `t00002046` | `` |
| 85 | `linear.default` | `model.layers.24.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002022, t00002047` | `t00002048` |
| 86 | `add.Tensor` | `model.layers.24` | `True` | `True` | `attention_output, mlp` | `t00001965, t00002048` | `t00002049` |
| 87 | `to.dtype` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002049` | `t00002050` |
| 88 | `pow.Tensor_Scalar` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002050` | `t00002051` |
| 89 | `mean.dim` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002051` | `t00002052` |
| 90 | `add.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002052` | `t00002053` |
| 91 | `rsqrt.default` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002053` | `t00002054` |
| 92 | `mul.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002050, t00002054` | `t00002055` |
| 93 | `to.dtype` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002055` | `t00002056` |
| 94 | `mul.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002057, t00002056` | `t00002058` |
| 95 | `linear.default` | `model.layers.24.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002058, t00002059` | `t00002060` |
| 96 | `silu.default` | `model.layers.24.mlp.act_fn` | `True` | `True` | `mlp` | `t00002060` | `t00002061` |
| 97 | `linear.default` | `model.layers.24.mlp.up_proj` | `True` | `True` | `mlp` | `t00002058, t00002062` | `t00002063` |
| 98 | `mul.Tensor` | `model.layers.24.mlp` | `True` | `True` | `mlp` | `t00002061, t00002063` | `t00002064` |
| 99 | `linear.default` | `model.layers.24.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002064, t00002065` | `t00002066` |
| 100 | `add.Tensor` | `model.layers.24` | `True` | `True` | `attention_output` | `t00002049, t00002066` | `t00002067` |
