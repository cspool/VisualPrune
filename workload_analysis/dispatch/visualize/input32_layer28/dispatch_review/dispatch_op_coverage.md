# input32_layer28 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003048` | `t00003049` |
| 2 | `pow.Tensor_Scalar` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003049` | `t00003050` |
| 3 | `mean.dim` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003050` | `t00003051` |
| 4 | `add.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003051` | `t00003052` |
| 5 | `rsqrt.default` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003052` | `t00003053` |
| 6 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003049, t00003053` | `t00003054` |
| 7 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00003054` | `t00003055` |
| 8 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002382, t00003055` | `t00003056` |
| 9 | `linear.default` | `model.layers.28.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00003056, t00002384` | `t00003057` |
| 10 | `linear.default` | `model.layers.28.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00003056, t00002386` | `t00003058` |
| 11 | `linear.default` | `model.layers.28.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00003056, t00002388` | `t00003059` |
| 12 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00003057` | `t00003060` |
| 13 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003060` | `t00003061` |
| 14 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00003058` | `t00003062` |
| 15 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003062` | `t00003063` |
| 16 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00003059` | `t00003064` |
| 17 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00003064` | `t00003065` |
| 18 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002848` | `t00003066` |
| 19 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003066` | `t00003067` |
| 20 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003067` | `t00003068` |
| 21 | `gt.Scalar` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003068` | `t00003069` |
| 22 | `is_nonzero.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003069` | `` |
| 23 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003068` | `` |
| 24 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002401` | `t00003070` |
| 25 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003070` | `t00003070` |
| 26 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003068` | `` |
| 27 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002403` | `t00003071` |
| 28 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00003071` | `t00003071` |
| 29 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003070, t00002848` | `t00003072` |
| 30 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003072` | `t00003073` |
| 31 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003071, t00002848` | `t00003074` |
| 32 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003074` | `t00003075` |
| 33 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003061, t00003073` | `t00003076` |
| 34 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003061` | `t00003077` |
| 35 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003061` | `t00003078` |
| 36 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003078` | `t00003079` |
| 37 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003079, t00003077` | `t00003080` |
| 38 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00003080, t00003075` | `t00003081` |
| 39 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope, attention` | `t00003076, t00003081` | `t00003082` |
| 40 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003063, t00003073` | `t00003083` |
| 41 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003063` | `t00003084` |
| 42 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003063` | `t00003085` |
| 43 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003085` | `t00003086` |
| 44 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003086, t00003084` | `t00003087` |
| 45 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003087, t00003075` | `t00003088` |
| 46 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003083, t00003088` | `t00003089` |
| 47 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003090, t00003089` | `t00003091` |
| 48 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` | `t00003092, t00003065` | `t00003093` |
| 49 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003091` | `t00003094` |
| 50 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003082, t00003094` | `t00003095` |
| 51 | `div.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003095` | `t00003096` |
| 52 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003096, t00003097` | `t00003098` |
| 53 | `softmax.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003098` | `t00003099` |
| 54 | `to.dtype` | `model.layers.28.self_attn` | `True` | `True` | `mlp` | `t00003099` | `t00003100` |
| 55 | `dropout.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00003100` | `t00003100` |
| 56 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention, attention_output` | `t00003100, t00003093` | `t00003101` |
| 57 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003101` | `t00003102` |
| 58 | `reshape.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` | `t00003102` | `t00003103` |
| 59 | `gt.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00000057` | `t00003104` |
| 60 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00003104` | `` |
| 61 | `linear.default` | `model.layers.28.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00003103, t00002442` | `t00003105` |
| 62 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output, mlp` | `t00003048, t00003105` | `t00003106` |
| 63 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003106` | `t00003107` |
| 64 | `pow.Tensor_Scalar` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003107` | `t00003108` |
| 65 | `mean.dim` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003108` | `t00003109` |
| 66 | `add.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003109` | `t00003110` |
| 67 | `rsqrt.default` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003110` | `t00003111` |
| 68 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003107, t00003111` | `t00003112` |
| 69 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00003112` | `t00003113` |
| 70 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002452, t00003113` | `t00003114` |
| 71 | `linear.default` | `model.layers.28.mlp.gate_proj` | `True` | `True` | `mlp` | `t00003114, t00002454` | `t00003115` |
| 72 | `silu.default` | `model.layers.28.mlp.act_fn` | `True` | `True` | `mlp` | `t00003115` | `t00003116` |
| 73 | `linear.default` | `model.layers.28.mlp.up_proj` | `True` | `True` | `mlp` | `t00003114, t00002457` | `t00003117` |
| 74 | `mul.Tensor` | `model.layers.28.mlp` | `True` | `True` | `` | `t00003116, t00003117` | `t00003118` |
| 75 | `linear.default` | `model.layers.28.mlp.down_proj` | `True` | `True` | `attention_output` | `t00003118, t00002460` | `t00003119` |
| 76 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output` | `t00003106, t00003119` | `t00003120` |
