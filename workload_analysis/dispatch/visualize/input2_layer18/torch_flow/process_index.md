# input2_layer18 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input2_layer18/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input2_layer18/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input2_layer18/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002463, t00001354`
- `input_rmsnorm` outputs: `t00002471`
- `qkv_projection` inputs: `t00002471, t00001356, t00001358, t00001360`
- `qkv_projection` outputs: `t00002476, t00002478, t00002480`
- `rope` inputs: `t00002483, t00001372, t00001374, t00002481, t00002476`
- `rope` outputs: `t00002484, t00002498`
- `kv_cache_concat` inputs: `t00001393, t00002505, t00001367, t00002480`
- `kv_cache_concat` outputs: `t00002506, t00002507`
- `attention` inputs: `t00002475, t00002477, t00002479, t00002492, t00002497, t00002499, t00002504, t00002506, t00002511, t00002507`
- `attention` outputs: `t00002476, t00002478, t00002480, t00002505, t00002513, t00002515`
- `attention_output` inputs: `t00002514, t00002507, t00002516, t00001432, t00002463, t00002532, t00001450`
- `attention_output` outputs: `t00002515, t00002534`
- `mlp` inputs: `t00002513, t00002517, t00001432, t00002463, t00001442, t00001444, t00001447`
- `mlp` outputs: `t00002514, t00002530, t00002531`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input2_layer18/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input2_layer18/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input2_layer18/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.18.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.18.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.18.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.18.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.18.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.18.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.18.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.18.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.18.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.18.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002463']` outputs=`['t00002464']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002464']` outputs=`['t00002465']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002465']` outputs=`['t00002466']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002466']` outputs=`['t00002467']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002467']` outputs=`['t00002468']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002464', 't00002468']` outputs=`['t00002469']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002469']` outputs=`['t00002470']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001354', 't00002470']` outputs=`['t00002471']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002471', 't00001356']` outputs=`['t00002472']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002471', 't00001358']` outputs=`['t00002473']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002471', 't00001360']` outputs=`['t00002474']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002472']` outputs=`['t00002475']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002475']` outputs=`['t00002476']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002473']` outputs=`['t00002477']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002477']` outputs=`['t00002478']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002474']` outputs=`['t00002479']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002479']` outputs=`['t00002480']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002483']` outputs=`['t00002484']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001372']` outputs=`['t00002486']` -> shape=[625, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001374']` outputs=`['t00002487']` -> shape=[625, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002486', 't00002481']` outputs=`['t00002488']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002488']` outputs=`['t00002489']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002487', 't00002481']` outputs=`['t00002490']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002490']` outputs=`['t00002491']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002476', 't00002489']` outputs=`['t00002492']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002476']` outputs=`['t00002493']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002476']` outputs=`['t00002494']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002494']` outputs=`['t00002495']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002495', 't00002493']` outputs=`['t00002496']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002496', 't00002491']` outputs=`['t00002497']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002492', 't00002497']` outputs=`['t00002498']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00001393', 't00002505']` outputs=`['t00002506']` -> shape=[1, 32, 625, 128], dtype=float16
- `#48 cat.default` inputs=`['t00001367', 't00002480']` outputs=`['t00002507']` -> shape=[1, 32, 625, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002475']` outputs=`['t00002476']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002477']` outputs=`['t00002478']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002479']` outputs=`['t00002480']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002492', 't00002497']` outputs=`['t00002498']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002499', 't00002504']` outputs=`['t00002505']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002506']` outputs=`['t00002508']` -> shape=[1, 32, 128, 625], dtype=float16
- `#50 matmul.default` inputs=`['t00002498', 't00002508']` outputs=`['t00002509']` -> shape=[1, 32, 1, 625], dtype=float16
- `#51 div.Tensor` inputs=`['t00002509']` outputs=`['t00002510']` -> shape=[1, 32, 1, 625], dtype=float16
- `#52 add.Tensor` inputs=`['t00002510', 't00002511']` outputs=`['t00002512']` -> shape=[1, 32, 1, 625], dtype=float16
- `#53 softmax.int` inputs=`['t00002512']` outputs=`['t00002513']` -> shape=[1, 32, 1, 625], dtype=float32
- `#55 dropout.default` inputs=`['t00002514']` outputs=`['t00002514']` -> shape=[1, 32, 1, 625], dtype=float16
- `#56 matmul.default` inputs=`['t00002514', 't00002507']` outputs=`['t00002515']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002514', 't00002507']` outputs=`['t00002515']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002516']` outputs=`['t00002517']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002517', 't00001432']` outputs=`['t00002519']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002463', 't00002519']` outputs=`['t00002520']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002532', 't00001450']` outputs=`['t00002533']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002520', 't00002533']` outputs=`['t00002534']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002513']` outputs=`['t00002514']` -> shape=[1, 32, 1, 625], dtype=float16
- `#61 linear.default` inputs=`['t00002517', 't00001432']` outputs=`['t00002519']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002463', 't00002519']` outputs=`['t00002520']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002520']` outputs=`['t00002521']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002521']` outputs=`['t00002522']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002522']` outputs=`['t00002523']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002523']` outputs=`['t00002524']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002524']` outputs=`['t00002525']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002521', 't00002525']` outputs=`['t00002526']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002526']` outputs=`['t00002527']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00001442', 't00002527']` outputs=`['t00002528']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002528', 't00001444']` outputs=`['t00002529']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002529']` outputs=`['t00002530']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002528', 't00001447']` outputs=`['t00002531']` -> shape=[1, 1, 11008], dtype=float16
