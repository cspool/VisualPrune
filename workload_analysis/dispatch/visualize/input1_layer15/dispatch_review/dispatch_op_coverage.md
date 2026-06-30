# input1_layer15 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001049` | `t00001050` |
| 2 | `pow.Tensor_Scalar` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001050` | `t00001051` |
| 3 | `mean.dim` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001051` | `t00001052` |
| 4 | `add.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001052` | `t00001053` |
| 5 | `rsqrt.default` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001053` | `t00001054` |
| 6 | `mul.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001050, t00001054` | `t00001055` |
| 7 | `to.dtype` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001055` | `t00001056` |
| 8 | `mul.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001057, t00001056` | `t00001058` |
| 9 | `linear.default` | `model.layers.15.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00001058, t00001059` | `t00001060` |
| 10 | `linear.default` | `model.layers.15.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00001058, t00001061` | `t00001062` |
| 11 | `linear.default` | `model.layers.15.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00001058, t00001063` | `t00001064` |
| 12 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` | `t00001060` | `t00001065` |
| 13 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001065` | `t00001066` |
| 14 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` | `t00001062` | `t00001067` |
| 15 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001067` | `t00001068` |
| 16 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` | `t00001064` | `t00001069` |
| 17 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00001069` | `t00001070` |
| 18 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00000023` | `t00001071` |
| 19 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001071` | `t00001072` |
| 20 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001072` | `t00001073` |
| 21 | `gt.Scalar` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001073` | `t00001074` |
| 22 | `is_nonzero.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00001074` | `` |
| 23 | `item.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` | `t00001073` | `` |
| 24 | `slice.Tensor` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001075` | `t00001076` |
| 25 | `to.dtype` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` | `t00001076` | `t00001076` |
| 26 | `item.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` | `t00001073` | `` |
| 27 | `slice.Tensor` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001077` | `t00001078` |
| 28 | `to.dtype` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` | `t00001078` | `t00001078` |
| 29 | `index.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001076, t00000023` | `t00001079` |
| 30 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001079` | `t00001080` |
| 31 | `index.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001078, t00000023` | `t00001081` |
| 32 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001081` | `t00001082` |
| 33 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001066, t00001080` | `t00001083` |
| 34 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001066` | `t00001084` |
| 35 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001066` | `t00001085` |
| 36 | `neg.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001085` | `t00001086` |
| 37 | `cat.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001086, t00001084` | `t00001087` |
| 38 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` | `t00001087, t00001082` | `t00001088` |
| 39 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope, attention` | `t00001083, t00001088` | `t00001089` |
| 40 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001068, t00001080` | `t00001090` |
| 41 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001068` | `t00001091` |
| 42 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001068` | `t00001092` |
| 43 | `neg.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001092` | `t00001093` |
| 44 | `cat.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001093, t00001091` | `t00001094` |
| 45 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001094, t00001082` | `t00001095` |
| 46 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001090, t00001095` | `t00001096` |
| 47 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001096` | `t00001097` |
| 48 | `matmul.default` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001089, t00001097` | `t00001098` |
| 49 | `div.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001098` | `t00001099` |
| 50 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001099, t00000053` | `t00001100` |
| 51 | `softmax.int` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001100` | `t00001101` |
| 52 | `to.dtype` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001101` | `t00001102` |
| 53 | `dropout.default` | `model.layers.15.self_attn` | `True` | `True` | `attention` | `t00001102` | `t00001102` |
| 54 | `matmul.default` | `model.layers.15.self_attn` | `True` | `True` | `attention, attention_output` | `t00001102, t00001070` | `t00001103` |
| 55 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001103` | `t00001104` |
| 56 | `contiguous.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` | `t00001104` | `t00001105` |
| 57 | `reshape.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` | `t00001105` | `t00001106` |
| 58 | `gt.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00001107` |
| 59 | `is_nonzero.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001107` | `` |
| 60 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00000023` | `t00001108` |
| 61 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001108` | `t00001109` |
| 62 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001109` | `t00001110` |
| 63 | `eq.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001110` | `t00001111` |
| 64 | `is_nonzero.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001111` | `` |
| 65 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001106` | `t00001112` |
| 66 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001102` | `t00001113` |
| 67 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001113` | `t00001114` |
| 68 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001114, t00001070` | `t00001115` |
| 69 | `permute.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001115` | `t00001116` |
| 70 | `contiguous.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` | `t00001116` | `t00001117` |
| 71 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001117` | `t00001118` |
| 72 | `item.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001119` | `` |
| 73 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001118` | `t00001120` |
| 74 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001112` | `t00001121` |
| 75 | `sub.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001121, t00001120` | `t00001122` |
| 76 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001112` | `t00001123` |
| 77 | `cosine_similarity.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001122, t00001123` | `t00001124` |
| 78 | `squeeze.dim` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001124` | `t00001125` |
| 79 | `lt.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001125` | `t00001126` |
| 80 | `any.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00001126` | `t00001127` |
| 81 | `item.default` | `model.layers.15.self_attn` | `True` | `True` | `` | `t00001127` | `` |
| 82 | `linear.default` | `model.layers.15.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00001106, t00001128` | `t00001129` |
| 83 | `add.Tensor` | `model.layers.15` | `True` | `True` | `attention_output, mlp` | `t00001049, t00001129` | `t00001130` |
| 84 | `to.dtype` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001130` | `t00001131` |
| 85 | `pow.Tensor_Scalar` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001131` | `t00001132` |
| 86 | `mean.dim` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001132` | `t00001133` |
| 87 | `add.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001133` | `t00001134` |
| 88 | `rsqrt.default` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001134` | `t00001135` |
| 89 | `mul.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001131, t00001135` | `t00001136` |
| 90 | `to.dtype` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001136` | `t00001137` |
| 91 | `mul.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001138, t00001137` | `t00001139` |
| 92 | `linear.default` | `model.layers.15.mlp.gate_proj` | `True` | `True` | `mlp` | `t00001139, t00001140` | `t00001141` |
| 93 | `silu.default` | `model.layers.15.mlp.act_fn` | `True` | `True` | `mlp` | `t00001141` | `t00001142` |
| 94 | `linear.default` | `model.layers.15.mlp.up_proj` | `True` | `True` | `mlp` | `t00001139, t00001143` | `t00001144` |
| 95 | `mul.Tensor` | `model.layers.15.mlp` | `True` | `True` | `mlp` | `t00001142, t00001144` | `t00001145` |
| 96 | `linear.default` | `model.layers.15.mlp.down_proj` | `True` | `True` | `attention_output` | `t00001145, t00001146` | `t00001147` |
| 97 | `add.Tensor` | `model.layers.15` | `True` | `True` | `attention_output` | `t00001130, t00001147` | `t00001148` |
