# input2_layer18 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002463` | `t00002464` |
| 2 | `pow.Tensor_Scalar` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002464` | `t00002465` |
| 3 | `mean.dim` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002465` | `t00002466` |
| 4 | `add.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002466` | `t00002467` |
| 5 | `rsqrt.default` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002467` | `t00002468` |
| 6 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002464, t00002468` | `t00002469` |
| 7 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002469` | `t00002470` |
| 8 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00001354, t00002470` | `t00002471` |
| 9 | `linear.default` | `model.layers.18.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002471, t00001356` | `t00002472` |
| 10 | `linear.default` | `model.layers.18.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002471, t00001358` | `t00002473` |
| 11 | `linear.default` | `model.layers.18.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002471, t00001360` | `t00002474` |
| 12 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002472` | `t00002475` |
| 13 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002475` | `t00002476` |
| 14 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002473` | `t00002477` |
| 15 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002477` | `t00002478` |
| 16 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` | `t00002474` | `t00002479` |
| 17 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002479` | `t00002480` |
| 18 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002481` | `t00002482` |
| 19 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002482` | `t00002483` |
| 20 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002483` | `t00002484` |
| 21 | `gt.Scalar` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002484` | `t00002485` |
| 22 | `is_nonzero.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002485` | `` |
| 23 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002484` | `` |
| 24 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001372` | `t00002486` |
| 25 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002486` | `t00002486` |
| 26 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002484` | `` |
| 27 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00001374` | `t00002487` |
| 28 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` | `t00002487` | `t00002487` |
| 29 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002486, t00002481` | `t00002488` |
| 30 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002488` | `t00002489` |
| 31 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002487, t00002481` | `t00002490` |
| 32 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002490` | `t00002491` |
| 33 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002476, t00002489` | `t00002492` |
| 34 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002476` | `t00002493` |
| 35 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002476` | `t00002494` |
| 36 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002494` | `t00002495` |
| 37 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002495, t00002493` | `t00002496` |
| 38 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` | `t00002496, t00002491` | `t00002497` |
| 39 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope, attention` | `t00002492, t00002497` | `t00002498` |
| 40 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002478, t00002489` | `t00002499` |
| 41 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002478` | `t00002500` |
| 42 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002478` | `t00002501` |
| 43 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002501` | `t00002502` |
| 44 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002502, t00002500` | `t00002503` |
| 45 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002503, t00002491` | `t00002504` |
| 46 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002499, t00002504` | `t00002505` |
| 47 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` | `t00001393, t00002505` | `t00002506` |
| 48 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` | `t00001367, t00002480` | `t00002507` |
| 49 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002506` | `t00002508` |
| 50 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002498, t00002508` | `t00002509` |
| 51 | `div.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002509` | `t00002510` |
| 52 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002510, t00002511` | `t00002512` |
| 53 | `softmax.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002512` | `t00002513` |
| 54 | `to.dtype` | `model.layers.18.self_attn` | `True` | `True` | `mlp` | `t00002513` | `t00002514` |
| 55 | `dropout.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` | `t00002514` | `t00002514` |
| 56 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention, attention_output` | `t00002514, t00002507` | `t00002515` |
| 57 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002515` | `t00002516` |
| 58 | `reshape.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` | `t00002516` | `t00002517` |
| 59 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00000057` | `t00002518` |
| 60 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `` | `t00002518` | `` |
| 61 | `linear.default` | `model.layers.18.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002517, t00001432` | `t00002519` |
| 62 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output, mlp` | `t00002463, t00002519` | `t00002520` |
| 63 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002520` | `t00002521` |
| 64 | `pow.Tensor_Scalar` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002521` | `t00002522` |
| 65 | `mean.dim` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002522` | `t00002523` |
| 66 | `add.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002523` | `t00002524` |
| 67 | `rsqrt.default` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002524` | `t00002525` |
| 68 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002521, t00002525` | `t00002526` |
| 69 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002526` | `t00002527` |
| 70 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` | `t00001442, t00002527` | `t00002528` |
| 71 | `linear.default` | `model.layers.18.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002528, t00001444` | `t00002529` |
| 72 | `silu.default` | `model.layers.18.mlp.act_fn` | `True` | `True` | `mlp` | `t00002529` | `t00002530` |
| 73 | `linear.default` | `model.layers.18.mlp.up_proj` | `True` | `True` | `mlp` | `t00002528, t00001447` | `t00002531` |
| 74 | `mul.Tensor` | `model.layers.18.mlp` | `True` | `True` | `` | `t00002530, t00002531` | `t00002532` |
| 75 | `linear.default` | `model.layers.18.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002532, t00001450` | `t00002533` |
| 76 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output` | `t00002520, t00002533` | `t00002534` |
