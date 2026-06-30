# input2_layer28 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002675` | `t00002676` |
| 2 | `pow.Tensor_Scalar` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002676` | `t00002677` |
| 3 | `mean.dim` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002677` | `t00002678` |
| 4 | `add.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002678` | `t00002679` |
| 5 | `rsqrt.default` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002679` | `t00002680` |
| 6 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002676, t00002680` | `t00002681` |
| 7 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002681` | `t00002682` |
| 8 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002382, t00002682` | `t00002683` |
| 9 | `linear.default` | `model.layers.28.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002683, t00002384` | `t00002684` |
| 10 | `linear.default` | `model.layers.28.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002683, t00002386` | `t00002685` |
| 11 | `linear.default` | `model.layers.28.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002683, t00002388` | `t00002686` |
| 12 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002684` | `t00002687` |
| 13 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002687` | `t00002688` |
| 14 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002685` | `t00002689` |
| 15 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002689` | `t00002690` |
| 16 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002686` | `t00002691` |
| 17 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002691` | `t00002692` |
| 18 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002481` | `t00002693` |
| 19 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002693` | `t00002694` |
| 20 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002694` | `t00002695` |
| 21 | `gt.Scalar` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002695` | `t00002696` |
| 22 | `is_nonzero.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002696` | `` |
| 23 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002695` | `` |
| 24 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002401` | `t00002697` |
| 25 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002697` | `t00002697` |
| 26 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002695` | `` |
| 27 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002403` | `t00002698` |
| 28 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002698` | `t00002698` |
| 29 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002697, t00002481` | `t00002699` |
| 30 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002699` | `t00002700` |
| 31 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002698, t00002481` | `t00002701` |
| 32 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002701` | `t00002702` |
| 33 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002688, t00002700` | `t00002703` |
| 34 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002688` | `t00002704` |
| 35 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002688` | `t00002705` |
| 36 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002705` | `t00002706` |
| 37 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002706, t00002704` | `t00002707` |
| 38 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002707, t00002702` | `t00002708` |
| 39 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope, attention` | `t00002703, t00002708` | `t00002709` |
| 40 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002690, t00002700` | `t00002710` |
| 41 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002690` | `t00002711` |
| 42 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002690` | `t00002712` |
| 43 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002712` | `t00002713` |
| 44 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002713, t00002711` | `t00002714` |
| 45 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002714, t00002702` | `t00002715` |
| 46 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002710, t00002715` | `t00002716` |
| 47 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002422, t00002716` | `t00002717` |
| 48 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` | `t00002395, t00002692` | `t00002718` |
| 49 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002717` | `t00002719` |
| 50 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002709, t00002719` | `t00002720` |
| 51 | `div.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002720` | `t00002721` |
| 52 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002721, t00002722` | `t00002723` |
| 53 | `softmax.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002723` | `t00002724` |
| 54 | `to.dtype` | `model.layers.28.self_attn` | `True` | `True` | `mlp` | `t00002724` | `t00002725` |
| 55 | `dropout.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002725` | `t00002725` |
| 56 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention, attention_output` | `t00002725, t00002718` | `t00002726` |
| 57 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002726` | `t00002727` |
| 58 | `reshape.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` | `t00002727` | `t00002728` |
| 59 | `gt.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00000057` | `t00002729` |
| 60 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002729` | `` |
| 61 | `linear.default` | `model.layers.28.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002728, t00002442` | `t00002730` |
| 62 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output, mlp` | `t00002675, t00002730` | `t00002731` |
| 63 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002731` | `t00002732` |
| 64 | `pow.Tensor_Scalar` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002732` | `t00002733` |
| 65 | `mean.dim` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002733` | `t00002734` |
| 66 | `add.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002734` | `t00002735` |
| 67 | `rsqrt.default` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002735` | `t00002736` |
| 68 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002732, t00002736` | `t00002737` |
| 69 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002737` | `t00002738` |
| 70 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002452, t00002738` | `t00002739` |
| 71 | `linear.default` | `model.layers.28.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002739, t00002454` | `t00002740` |
| 72 | `silu.default` | `model.layers.28.mlp.act_fn` | `True` | `True` | `mlp` | `t00002740` | `t00002741` |
| 73 | `linear.default` | `model.layers.28.mlp.up_proj` | `True` | `True` | `mlp` | `t00002739, t00002457` | `t00002742` |
| 74 | `mul.Tensor` | `model.layers.28.mlp` | `True` | `True` | `` | `t00002741, t00002742` | `t00002743` |
| 75 | `linear.default` | `model.layers.28.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002743, t00002460` | `t00002744` |
| 76 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output` | `t00002731, t00002744` | `t00002745` |
