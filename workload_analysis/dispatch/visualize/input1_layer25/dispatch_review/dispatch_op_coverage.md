# input1_layer25 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002067` | `t00002068` |
| 2 | `pow.Tensor_Scalar` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002068` | `t00002069` |
| 3 | `mean.dim` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002069` | `t00002070` |
| 4 | `add.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002070` | `t00002071` |
| 5 | `rsqrt.default` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002071` | `t00002072` |
| 6 | `mul.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002068, t00002072` | `t00002073` |
| 7 | `to.dtype` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002073` | `t00002074` |
| 8 | `mul.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002075, t00002074` | `t00002076` |
| 9 | `linear.default` | `model.layers.25.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002076, t00002077` | `t00002078` |
| 10 | `linear.default` | `model.layers.25.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002076, t00002079` | `t00002080` |
| 11 | `linear.default` | `model.layers.25.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002076, t00002081` | `t00002082` |
| 12 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` | `t00002078` | `t00002083` |
| 13 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002083` | `t00002084` |
| 14 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` | `t00002080` | `t00002085` |
| 15 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002085` | `t00002086` |
| 16 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` | `t00002082` | `t00002087` |
| 17 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002087` | `t00002088` |
| 18 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00001475` | `t00002089` |
| 19 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002089` | `t00002090` |
| 20 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002090` | `t00002091` |
| 21 | `gt.Scalar` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002091` | `t00002092` |
| 22 | `is_nonzero.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002092` | `` |
| 23 | `item.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` | `t00002091` | `` |
| 24 | `slice.Tensor` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002093` | `t00002094` |
| 25 | `to.dtype` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` | `t00002094` | `t00002094` |
| 26 | `item.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` | `t00002091` | `` |
| 27 | `slice.Tensor` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002095` | `t00002096` |
| 28 | `to.dtype` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` | `t00002096` | `t00002096` |
| 29 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002094, t00001475` | `t00002097` |
| 30 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002097` | `t00002098` |
| 31 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002096, t00001475` | `t00002099` |
| 32 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002099` | `t00002100` |
| 33 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002084, t00002098` | `t00002101` |
| 34 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002084` | `t00002102` |
| 35 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002084` | `t00002103` |
| 36 | `neg.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002103` | `t00002104` |
| 37 | `cat.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002104, t00002102` | `t00002105` |
| 38 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` | `t00002105, t00002100` | `t00002106` |
| 39 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope, attention` | `t00002101, t00002106` | `t00002107` |
| 40 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002086, t00002098` | `t00002108` |
| 41 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002086` | `t00002109` |
| 42 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002086` | `t00002110` |
| 43 | `neg.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002110` | `t00002111` |
| 44 | `cat.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002111, t00002109` | `t00002112` |
| 45 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002112, t00002100` | `t00002113` |
| 46 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002108, t00002113` | `t00002114` |
| 47 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002114` | `t00002115` |
| 48 | `matmul.default` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002107, t00002115` | `t00002116` |
| 49 | `div.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002116` | `t00002117` |
| 50 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002117, t00001505` | `t00002118` |
| 51 | `softmax.int` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002118` | `t00002119` |
| 52 | `to.dtype` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002119` | `t00002120` |
| 53 | `dropout.default` | `model.layers.25.self_attn` | `True` | `True` | `attention` | `t00002120` | `t00002120` |
| 54 | `matmul.default` | `model.layers.25.self_attn` | `True` | `True` | `attention, attention_output` | `t00002120, t00002088` | `t00002121` |
| 55 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002121` | `t00002122` |
| 56 | `contiguous.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` | `t00002122` | `t00002123` |
| 57 | `reshape.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` | `t00002123` | `t00002124` |
| 58 | `gt.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00002125` |
| 59 | `is_nonzero.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002125` | `` |
| 60 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00001475` | `t00002126` |
| 61 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002126` | `t00002127` |
| 62 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002127` | `t00002128` |
| 63 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00000057` | `t00002129` |
| 64 | `sub.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002128, t00002129` | `t00002130` |
| 65 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002130` | `t00002131` |
| 66 | `eq.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002131` | `t00002132` |
| 67 | `is_nonzero.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002132` | `` |
| 68 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002124` | `t00002133` |
| 69 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002120` | `t00002134` |
| 70 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002134` | `t00002135` |
| 71 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002135, t00002088` | `t00002136` |
| 72 | `permute.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002136` | `t00002137` |
| 73 | `contiguous.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` | `t00002137` | `t00002138` |
| 74 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002138` | `t00002139` |
| 75 | `arange.start` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00002140` |
| 76 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002139, t00002140` | `t00002141` |
| 77 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002133` | `t00002142` |
| 78 | `sub.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002142, t00002141` | `t00002143` |
| 79 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002133` | `t00002144` |
| 80 | `cosine_similarity.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002143, t00002144` | `t00002145` |
| 81 | `squeeze.dim` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002145` | `t00002146` |
| 82 | `lt.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002146` | `t00002147` |
| 83 | `any.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002147` | `t00002148` |
| 84 | `item.default` | `model.layers.25.self_attn` | `True` | `True` | `` | `t00002148` | `` |
| 85 | `linear.default` | `model.layers.25.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002124, t00002149` | `t00002150` |
| 86 | `add.Tensor` | `model.layers.25` | `True` | `True` | `attention_output, mlp` | `t00002067, t00002150` | `t00002151` |
| 87 | `to.dtype` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002151` | `t00002152` |
| 88 | `pow.Tensor_Scalar` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002152` | `t00002153` |
| 89 | `mean.dim` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002153` | `t00002154` |
| 90 | `add.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002154` | `t00002155` |
| 91 | `rsqrt.default` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002155` | `t00002156` |
| 92 | `mul.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002152, t00002156` | `t00002157` |
| 93 | `to.dtype` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002157` | `t00002158` |
| 94 | `mul.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002159, t00002158` | `t00002160` |
| 95 | `linear.default` | `model.layers.25.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002160, t00002161` | `t00002162` |
| 96 | `silu.default` | `model.layers.25.mlp.act_fn` | `True` | `True` | `mlp` | `t00002162` | `t00002163` |
| 97 | `linear.default` | `model.layers.25.mlp.up_proj` | `True` | `True` | `mlp` | `t00002160, t00002164` | `t00002165` |
| 98 | `mul.Tensor` | `model.layers.25.mlp` | `True` | `True` | `mlp` | `t00002163, t00002165` | `t00002166` |
| 99 | `linear.default` | `model.layers.25.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002166, t00002167` | `t00002168` |
| 100 | `add.Tensor` | `model.layers.25` | `True` | `True` | `attention_output` | `t00002151, t00002168` | `t00002169` |
