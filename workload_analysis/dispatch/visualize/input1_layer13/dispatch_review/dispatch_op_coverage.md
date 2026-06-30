# input1_layer13 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000851` | `t00000852` |
| 2 | `pow.Tensor_Scalar` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000852` | `t00000853` |
| 3 | `mean.dim` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000853` | `t00000854` |
| 4 | `add.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000854` | `t00000855` |
| 5 | `rsqrt.default` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000855` | `t00000856` |
| 6 | `mul.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000852, t00000856` | `t00000857` |
| 7 | `to.dtype` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000857` | `t00000858` |
| 8 | `mul.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00000859, t00000858` | `t00000860` |
| 9 | `linear.default` | `model.layers.13.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00000860, t00000861` | `t00000862` |
| 10 | `linear.default` | `model.layers.13.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00000860, t00000863` | `t00000864` |
| 11 | `linear.default` | `model.layers.13.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00000860, t00000865` | `t00000866` |
| 12 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` | `t00000862` | `t00000867` |
| 13 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000867` | `t00000868` |
| 14 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` | `t00000864` | `t00000869` |
| 15 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000869` | `t00000870` |
| 16 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` | `t00000866` | `t00000871` |
| 17 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00000871` | `t00000872` |
| 18 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000023` | `t00000873` |
| 19 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000873` | `t00000874` |
| 20 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000874` | `t00000875` |
| 21 | `gt.Scalar` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000875` | `t00000876` |
| 22 | `is_nonzero.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` | `t00000876` | `` |
| 23 | `item.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` | `t00000875` | `` |
| 24 | `slice.Tensor` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000877` | `t00000878` |
| 25 | `to.dtype` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` | `t00000878` | `t00000878` |
| 26 | `item.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` | `t00000875` | `` |
| 27 | `slice.Tensor` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00000879` | `t00000880` |
| 28 | `to.dtype` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` | `t00000880` | `t00000880` |
| 29 | `index.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000878, t00000023` | `t00000881` |
| 30 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000881` | `t00000882` |
| 31 | `index.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000880, t00000023` | `t00000883` |
| 32 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000883` | `t00000884` |
| 33 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000868, t00000882` | `t00000885` |
| 34 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000868` | `t00000886` |
| 35 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000868` | `t00000887` |
| 36 | `neg.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000887` | `t00000888` |
| 37 | `cat.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000888, t00000886` | `t00000889` |
| 38 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` | `t00000889, t00000884` | `t00000890` |
| 39 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope, attention` | `t00000885, t00000890` | `t00000891` |
| 40 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000870, t00000882` | `t00000892` |
| 41 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000870` | `t00000893` |
| 42 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000870` | `t00000894` |
| 43 | `neg.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000894` | `t00000895` |
| 44 | `cat.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000895, t00000893` | `t00000896` |
| 45 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000896, t00000884` | `t00000897` |
| 46 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000892, t00000897` | `t00000898` |
| 47 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000898` | `t00000899` |
| 48 | `matmul.default` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000891, t00000899` | `t00000900` |
| 49 | `div.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000900` | `t00000901` |
| 50 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000901, t00000053` | `t00000902` |
| 51 | `softmax.int` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000902` | `t00000903` |
| 52 | `to.dtype` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000903` | `t00000904` |
| 53 | `dropout.default` | `model.layers.13.self_attn` | `True` | `True` | `attention` | `t00000904` | `t00000904` |
| 54 | `matmul.default` | `model.layers.13.self_attn` | `True` | `True` | `attention, attention_output` | `t00000904, t00000872` | `t00000905` |
| 55 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000905` | `t00000906` |
| 56 | `contiguous.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` | `t00000906` | `t00000907` |
| 57 | `reshape.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` | `t00000907` | `t00000908` |
| 58 | `gt.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000057` | `t00000909` |
| 59 | `is_nonzero.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000909` | `` |
| 60 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000023` | `t00000910` |
| 61 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000910` | `t00000911` |
| 62 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000911` | `t00000912` |
| 63 | `eq.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000912` | `t00000913` |
| 64 | `is_nonzero.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000913` | `` |
| 65 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000908` | `t00000914` |
| 66 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000904` | `t00000915` |
| 67 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000915` | `t00000916` |
| 68 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000916, t00000872` | `t00000917` |
| 69 | `permute.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000917` | `t00000918` |
| 70 | `contiguous.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` | `t00000918` | `t00000919` |
| 71 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000919` | `t00000920` |
| 72 | `item.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000921` | `` |
| 73 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000920` | `t00000922` |
| 74 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000914` | `t00000923` |
| 75 | `sub.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000923, t00000922` | `t00000924` |
| 76 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000914` | `t00000925` |
| 77 | `cosine_similarity.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000924, t00000925` | `t00000926` |
| 78 | `squeeze.dim` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000926` | `t00000927` |
| 79 | `lt.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000927` | `t00000928` |
| 80 | `any.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` | `t00000928` | `t00000929` |
| 81 | `item.default` | `model.layers.13.self_attn` | `True` | `True` | `` | `t00000929` | `` |
| 82 | `linear.default` | `model.layers.13.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00000908, t00000930` | `t00000931` |
| 83 | `add.Tensor` | `model.layers.13` | `True` | `True` | `attention_output, mlp` | `t00000851, t00000931` | `t00000932` |
| 84 | `to.dtype` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000932` | `t00000933` |
| 85 | `pow.Tensor_Scalar` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000933` | `t00000934` |
| 86 | `mean.dim` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000934` | `t00000935` |
| 87 | `add.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000935` | `t00000936` |
| 88 | `rsqrt.default` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000936` | `t00000937` |
| 89 | `mul.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000933, t00000937` | `t00000938` |
| 90 | `to.dtype` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000938` | `t00000939` |
| 91 | `mul.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` | `t00000940, t00000939` | `t00000941` |
| 92 | `linear.default` | `model.layers.13.mlp.gate_proj` | `True` | `True` | `mlp` | `t00000941, t00000942` | `t00000943` |
| 93 | `silu.default` | `model.layers.13.mlp.act_fn` | `True` | `True` | `mlp` | `t00000943` | `t00000944` |
| 94 | `linear.default` | `model.layers.13.mlp.up_proj` | `True` | `True` | `mlp` | `t00000941, t00000945` | `t00000946` |
| 95 | `mul.Tensor` | `model.layers.13.mlp` | `True` | `True` | `mlp` | `t00000944, t00000946` | `t00000947` |
| 96 | `linear.default` | `model.layers.13.mlp.down_proj` | `True` | `True` | `attention_output` | `t00000947, t00000948` | `t00000949` |
| 97 | `add.Tensor` | `model.layers.13` | `True` | `True` | `attention_output` | `t00000932, t00000949` | `t00000950` |
