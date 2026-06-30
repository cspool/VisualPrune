# input1_layer28 Dispatch Op Coverage

This file lists every dispatch op row exactly once and connects it to runtime module split and tensor-id evidence.

- ops in dispatch rows: `83`
- ops listed in coverage: `83`
- missing event_op_index values: `[]`
- duplicate event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence | Tensor ID inputs | Tensor ID outputs |
|---:|---|---|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002374` | `t00002375` |
| 2 | `pow.Tensor_Scalar` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002375` | `t00002376` |
| 3 | `mean.dim` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002376` | `t00002377` |
| 4 | `add.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002377` | `t00002378` |
| 5 | `rsqrt.default` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002378` | `t00002379` |
| 6 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002375, t00002379` | `t00002380` |
| 7 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002380` | `t00002381` |
| 8 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` | `t00002382, t00002381` | `t00002383` |
| 9 | `linear.default` | `model.layers.28.self_attn.q_proj` | `True` | `True` | `qkv_projection` | `t00002383, t00002384` | `t00002385` |
| 10 | `linear.default` | `model.layers.28.self_attn.k_proj` | `True` | `True` | `qkv_projection` | `t00002383, t00002386` | `t00002387` |
| 11 | `linear.default` | `model.layers.28.self_attn.v_proj` | `True` | `True` | `qkv_projection` | `t00002383, t00002388` | `t00002389` |
| 12 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002385` | `t00002390` |
| 13 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002390` | `t00002391` |
| 14 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002387` | `t00002392` |
| 15 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002392` | `t00002393` |
| 16 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` | `t00002389` | `t00002394` |
| 17 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` | `t00002394` | `t00002395` |
| 18 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002396` | `t00002397` |
| 19 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002397` | `t00002398` |
| 20 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002398` | `t00002399` |
| 21 | `gt.Scalar` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002399` | `t00002400` |
| 22 | `is_nonzero.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002400` | `` |
| 23 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002399` | `` |
| 24 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002401` | `t00002402` |
| 25 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002402` | `t00002402` |
| 26 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002399` | `` |
| 27 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` | `t00002403` | `t00002404` |
| 28 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` | `t00002404` | `t00002404` |
| 29 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002402, t00002396` | `t00002405` |
| 30 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002405` | `t00002406` |
| 31 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002404, t00002396` | `t00002407` |
| 32 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002407` | `t00002408` |
| 33 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002391, t00002406` | `t00002409` |
| 34 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002391` | `t00002410` |
| 35 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002391` | `t00002411` |
| 36 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002411` | `t00002412` |
| 37 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002412, t00002410` | `t00002413` |
| 38 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` | `t00002413, t00002408` | `t00002414` |
| 39 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope, attention` | `t00002409, t00002414` | `t00002415` |
| 40 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002393, t00002406` | `t00002416` |
| 41 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002393` | `t00002417` |
| 42 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002393` | `t00002418` |
| 43 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002418` | `t00002419` |
| 44 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002419, t00002417` | `t00002420` |
| 45 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002420, t00002408` | `t00002421` |
| 46 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002416, t00002421` | `t00002422` |
| 47 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002422` | `t00002423` |
| 48 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002415, t00002423` | `t00002424` |
| 49 | `div.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002424` | `t00002425` |
| 50 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002425, t00002426` | `t00002427` |
| 51 | `softmax.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002427` | `t00002428` |
| 52 | `to.dtype` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002428` | `t00002429` |
| 53 | `dropout.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` | `t00002429` | `t00002429` |
| 54 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention, attention_output` | `t00002429, t00002395` | `t00002430` |
| 55 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002430` | `t00002431` |
| 56 | `contiguous.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` | `t00002431` | `t00002432` |
| 57 | `reshape.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` | `t00002432` | `t00002433` |
| 58 | `gt.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00000057` | `t00002434` |
| 59 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002434` | `` |
| 60 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002396` | `t00002435` |
| 61 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002435` | `t00002436` |
| 62 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` | `t00002436` | `t00002437` |
| 63 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` | `t00000057` | `t00002438` |
| 64 | `sub.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002437, t00002438` | `t00002439` |
| 65 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` | `t00002439` | `t00002440` |
| 66 | `eq.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002440` | `t00002441` |
| 67 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` | `t00002441` | `` |
| 68 | `linear.default` | `model.layers.28.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` | `t00002433, t00002442` | `t00002443` |
| 69 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output, mlp` | `t00002374, t00002443` | `t00002444` |
| 70 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002444` | `t00002445` |
| 71 | `pow.Tensor_Scalar` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002445` | `t00002446` |
| 72 | `mean.dim` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002446` | `t00002447` |
| 73 | `add.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002447` | `t00002448` |
| 74 | `rsqrt.default` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002448` | `t00002449` |
| 75 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002445, t00002449` | `t00002450` |
| 76 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002450` | `t00002451` |
| 77 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` | `t00002452, t00002451` | `t00002453` |
| 78 | `linear.default` | `model.layers.28.mlp.gate_proj` | `True` | `True` | `mlp` | `t00002453, t00002454` | `t00002455` |
| 79 | `silu.default` | `model.layers.28.mlp.act_fn` | `True` | `True` | `` | `t00002455` | `t00002456` |
| 80 | `linear.default` | `model.layers.28.mlp.up_proj` | `True` | `True` | `` | `t00002453, t00002457` | `t00002458` |
| 81 | `mul.Tensor` | `model.layers.28.mlp` | `True` | `True` | `` | `t00002456, t00002458` | `t00002459` |
| 82 | `linear.default` | `model.layers.28.mlp.down_proj` | `True` | `True` | `attention_output` | `t00002459, t00002460` | `t00002461` |
| 83 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output` | `t00002444, t00002461` | `t00002462` |
