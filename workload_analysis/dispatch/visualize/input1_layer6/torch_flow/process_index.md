# input1_layer6 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer6/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `attention_output`
6. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000178, t00000186`
- `input_rmsnorm` outputs: `t00000187`
- `qkv_projection` inputs: `t00000187, t00000188, t00000190, t00000192`
- `qkv_projection` outputs: `t00000195, t00000197, t00000199`
- `rope` inputs: `t00000201, t00000204, t00000206, t00000023, t00000195`
- `rope` outputs: `t00000202, t00000218`
- `attention` inputs: `t00000194, t00000196, t00000198, t00000212, t00000217, t00000219, t00000224, t00000053`
- `attention` outputs: `t00000195, t00000197, t00000230, t00000232`
- `attention_output` inputs: `t00000231, t00000199, t00000233, t00000237, t00000178, t00000254, t00000255`
- `attention_output` outputs: `t00000232, t00000257`
- `mlp` inputs: `t00000230, t00000235, t00000237, t00000178, t00000247, t00000249, t00000252`
- `mlp` outputs: `t00000231, t00000251, t00000253`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer6/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer6/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer6/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `75`
- ops listed in coverage: `75`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.6.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.6.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.6.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.6.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.6.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.6.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.6.self_attn` | `True` | `True` | `mlp` |
| 53 | `dropout.default` | `model.layers.6.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.6.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.6.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.6.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 59 | `is_nonzero.default` | `model.layers.6.self_attn` | `True` | `True` | `` |
| 60 | `linear.default` | `model.layers.6.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 61 | `add.Tensor` | `model.layers.6` | `True` | `True` | `attention_output, mlp` |
| 62 | `to.dtype` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 63 | `pow.Tensor_Scalar` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `mean.dim` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `add.Tensor` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `rsqrt.default` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `mul.Tensor` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `to.dtype` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `mul.Tensor` | `model.layers.6.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `linear.default` | `model.layers.6.mlp.gate_proj` | `True` | `True` | `mlp` |
| 71 | `silu.default` | `model.layers.6.mlp.act_fn` | `True` | `True` | `mlp` |
| 72 | `linear.default` | `model.layers.6.mlp.up_proj` | `True` | `True` | `mlp` |
| 73 | `mul.Tensor` | `model.layers.6.mlp` | `True` | `True` | `` |
| 74 | `linear.default` | `model.layers.6.mlp.down_proj` | `True` | `True` | `attention_output` |
| 75 | `add.Tensor` | `model.layers.6` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000178']` outputs=`['t00000179']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000179']` outputs=`['t00000180']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000180']` outputs=`['t00000181']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000181']` outputs=`['t00000182']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000182']` outputs=`['t00000183']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000179', 't00000183']` outputs=`['t00000184']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000184']` outputs=`['t00000185']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000186', 't00000185']` outputs=`['t00000187']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000187', 't00000188']` outputs=`['t00000189']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000187', 't00000190']` outputs=`['t00000191']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000187', 't00000192']` outputs=`['t00000193']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000189']` outputs=`['t00000194']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000194']` outputs=`['t00000195']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000191']` outputs=`['t00000196']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000196']` outputs=`['t00000197']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000193']` outputs=`['t00000198']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000198']` outputs=`['t00000199']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000201']` outputs=`['t00000202']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000204']` outputs=`['t00000205']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000206']` outputs=`['t00000207']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000205', 't00000023']` outputs=`['t00000208']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000208']` outputs=`['t00000209']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000207', 't00000023']` outputs=`['t00000210']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000210']` outputs=`['t00000211']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000195', 't00000209']` outputs=`['t00000212']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000195']` outputs=`['t00000213']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000195']` outputs=`['t00000214']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000214']` outputs=`['t00000215']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000215', 't00000213']` outputs=`['t00000216']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000216', 't00000211']` outputs=`['t00000217']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000212', 't00000217']` outputs=`['t00000218']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000194']` outputs=`['t00000195']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000196']` outputs=`['t00000197']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000198']` outputs=`['t00000199']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000212', 't00000217']` outputs=`['t00000218']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000219', 't00000224']` outputs=`['t00000225']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000225']` outputs=`['t00000226']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000218', 't00000226']` outputs=`['t00000227']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000227']` outputs=`['t00000228']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000228', 't00000053']` outputs=`['t00000229']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000229']` outputs=`['t00000230']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000231']` outputs=`['t00000231']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000231', 't00000199']` outputs=`['t00000232']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention_output`
- `#54 matmul.default` inputs=`['t00000231', 't00000199']` outputs=`['t00000232']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000233']` outputs=`['t00000234']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000234']` outputs=`['t00000235']` -> shape=[1, 624, 4096], dtype=float16
- `#60 linear.default` inputs=`['t00000235', 't00000237']` outputs=`['t00000238']` -> shape=[1, 624, 4096], dtype=float16
- `#61 add.Tensor` inputs=`['t00000178', 't00000238']` outputs=`['t00000239']` -> shape=[1, 624, 4096], dtype=float16
- `#74 linear.default` inputs=`['t00000254', 't00000255']` outputs=`['t00000256']` -> shape=[1, 624, 4096], dtype=float16
- `#75 add.Tensor` inputs=`['t00000239', 't00000256']` outputs=`['t00000257']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#52 to.dtype` inputs=`['t00000230']` outputs=`['t00000231']` -> shape=[1, 32, 624, 624], dtype=float16
- `#60 linear.default` inputs=`['t00000235', 't00000237']` outputs=`['t00000238']` -> shape=[1, 624, 4096], dtype=float16
- `#61 add.Tensor` inputs=`['t00000178', 't00000238']` outputs=`['t00000239']` -> shape=[1, 624, 4096], dtype=float16
- `#62 to.dtype` inputs=`['t00000239']` outputs=`['t00000240']` -> shape=[1, 624, 4096], dtype=float32
- `#63 pow.Tensor_Scalar` inputs=`['t00000240']` outputs=`['t00000241']` -> shape=[1, 624, 4096], dtype=float32
- `#64 mean.dim` inputs=`['t00000241']` outputs=`['t00000242']` -> shape=[1, 624, 1], dtype=float32
- `#65 add.Tensor` inputs=`['t00000242']` outputs=`['t00000243']` -> shape=[1, 624, 1], dtype=float32
- `#66 rsqrt.default` inputs=`['t00000243']` outputs=`['t00000244']` -> shape=[1, 624, 1], dtype=float32
- `#67 mul.Tensor` inputs=`['t00000240', 't00000244']` outputs=`['t00000245']` -> shape=[1, 624, 4096], dtype=float32
- `#68 to.dtype` inputs=`['t00000245']` outputs=`['t00000246']` -> shape=[1, 624, 4096], dtype=float16
- `#69 mul.Tensor` inputs=`['t00000247', 't00000246']` outputs=`['t00000248']` -> shape=[1, 624, 4096], dtype=float16
- `#70 linear.default` inputs=`['t00000248', 't00000249']` outputs=`['t00000250']` -> shape=[1, 624, 11008], dtype=float16
- `#71 silu.default` inputs=`['t00000250']` outputs=`['t00000251']` -> shape=[1, 624, 11008], dtype=float16
- `#72 linear.default` inputs=`['t00000248', 't00000252']` outputs=`['t00000253']` -> shape=[1, 624, 11008], dtype=float16
