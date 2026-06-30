# input1_layer27 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002271` | `t00002272` |
| 2 | `pow.Tensor_Scalar` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002272` | `t00002273` |
| 3 | `mean.dim` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002273` | `t00002274` |
| 4 | `add.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002274` | `t00002275` |
| 5 | `rsqrt.default` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002275` | `t00002276` |
| 6 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002272, t00002276` | `t00002277` |
| 7 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002277` | `t00002278` |
| 8 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002279, t00002278` | `t00002280` |
| 9 | `linear.default` | `model.layers.27.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002280, t00002281` | `t00002282` |
| 10 | `linear.default` | `model.layers.27.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002280, t00002283` | `t00002284` |
| 11 | `linear.default` | `model.layers.27.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002280, t00002285` | `t00002286` |
| 12 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002282` | `t00002287` |
| 13 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002287` | `t00002288` |
| 14 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002284` | `t00002289` |
| 15 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002289` | `t00002290` |
| 16 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` | `t00002286` | `t00002291` |
| 17 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002291` | `t00002292` |
| 18 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00001475` | `t00002293` |
| 19 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002293` | `t00002294` |
| 20 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002294` | `t00002295` |
| 21 | `gt.Scalar` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002295` | `t00002296` |
| 22 | `is_nonzero.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00002296` | `` |
| 23 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002295` | `` |
| 24 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002297` | `t00002298` |
| 25 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002298` | `t00002298` |
| 26 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002295` | `` |
| 27 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002299` | `t00002300` |
| 28 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` | `t00002300` | `t00002300` |
| 29 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002298, t00001475` | `t00002301` |
| 30 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002301` | `t00002302` |
| 31 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002300, t00001475` | `t00002303` |
| 32 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002303` | `t00002304` |
| 33 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002288, t00002302` | `t00002305` |
| 34 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002288` | `t00002306` |
| 35 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002288` | `t00002307` |
| 36 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002307` | `t00002308` |
| 37 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002308, t00002306` | `t00002309` |
| 38 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` | `t00002309, t00002304` | `t00002310` |
| 39 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope, attention` | `t00002305, t00002310` | `t00002311` |
| 40 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002290, t00002302` | `t00002312` |
| 41 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002290` | `t00002313` |
| 42 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002290` | `t00002314` |
| 43 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002314` | `t00002315` |
| 44 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002315, t00002313` | `t00002316` |
| 45 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002316, t00002304` | `t00002317` |
| 46 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002312, t00002317` | `t00002318` |
| 47 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002318` | `t00002319` |
| 48 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002311, t00002319` | `t00002320` |
| 49 | `div.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002320` | `t00002321` |
| 50 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002321, t00001505` | `t00002322` |
| 51 | `softmax.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002322` | `t00002323` |
| 52 | `to.dtype` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002323` | `t00002324` |
| 53 | `dropout.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` | `t00002324` | `t00002324` |
| 54 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention, attention_output` | `t00002324, t00002292` | `t00002325` |
| 55 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002325` | `t00002326` |
| 56 | `contiguous.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` | `t00002326` | `t00002327` |
| 57 | `reshape.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` | `t00002327` | `t00002328` |
| 58 | `gt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00002329` |
| 59 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002329` | `` |
| 60 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00001475` | `t00002330` |
| 61 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002330` | `t00002331` |
| 62 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002331` | `t00002332` |
| 63 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00000057` | `t00002333` |
| 64 | `sub.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002332, t00002333` | `t00002334` |
| 65 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002334` | `t00002335` |
| 66 | `eq.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002335` | `t00002336` |
| 67 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002336` | `` |
| 68 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002328` | `t00002337` |
| 69 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002324` | `t00002338` |
| 70 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002338` | `t00002339` |
| 71 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002339, t00002292` | `t00002340` |
| 72 | `permute.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002340` | `t00002341` |
| 73 | `contiguous.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` | `t00002341` | `t00002342` |
| 74 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002342` | `t00002343` |
| 75 | `arange.start` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `` | `t00002344` |
| 76 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002343, t00002344` | `t00002345` |
| 77 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002337` | `t00002346` |
| 78 | `sub.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002346, t00002345` | `t00002347` |
| 79 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002337` | `t00002348` |
| 80 | `cosine_similarity.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002347, t00002348` | `t00002349` |
| 81 | `squeeze.dim` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002349` | `t00002350` |
| 82 | `lt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002350` | `t00002351` |
| 83 | `any.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00002351` | `t00002352` |
| 84 | `item.default` | `model.layers.27.self_attn` | `True` | `True` | `` | `t00002352` | `` |
| 85 | `linear.default` | `model.layers.27.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002328, t00002353` | `t00002354` |
| 86 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output, mlp` | `t00002271, t00002354` | `t00002355` |
| 87 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002355` | `t00002356` |
| 88 | `pow.Tensor_Scalar` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002356` | `t00002357` |
| 89 | `mean.dim` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002357` | `t00002358` |
| 90 | `add.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002358` | `t00002359` |
| 91 | `rsqrt.default` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002359` | `t00002360` |
| 92 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002356, t00002360` | `t00002361` |
| 93 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002361` | `t00002362` |
| 94 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002363, t00002362` | `t00002364` |
| 95 | `linear.default` | `model.layers.27.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002364, t00002365` | `t00002366` |
| 96 | `silu.default` | `model.layers.27.mlp.act_fn` | `True` | `True` | `mlp` | `t00002366` | `t00002367` |
| 97 | `linear.default` | `model.layers.27.mlp.up_proj` | `True` | `True` | `mlp` | `t00002364, t00002368` | `t00002369` |
| 98 | `mul.Tensor` | `model.layers.27.mlp` | `True` | `True` | `mlp` | `t00002367, t00002369` | `t00002370` |
| 99 | `linear.default` | `model.layers.27.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002370, t00002371` | `t00002372` |
| 100 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output` | `t00002355, t00002372` | `t00002373` |
