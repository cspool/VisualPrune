# input1_layer24 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer24/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001965, t00001973`
- `input_rmsnorm` outputs: `t00001974`
- `qkv_projection` inputs: `t00001974, t00001975, t00001977, t00001979`
- `qkv_projection` outputs: `t00001982, t00001984, t00001986`
- `rope` inputs: `t00001988, t00001991, t00001993, t00001475, t00001982`
- `rope` outputs: `t00001989, t00002005`
- `attention` inputs: `t00001981, t00001983, t00001985, t00001999, t00002004, t00002006, t00002011, t00001505`
- `attention` outputs: `t00001982, t00001984, t00002017, t00002019`
- `visipruner_similarity_check` inputs: `t00001989, t00000057, t00002026, t00002027, t00002030, t00002040, t00002039, t00002042, t00002045`
- `visipruner_similarity_check` outputs: `t00002028, t00002038, t00002043, t00002046`
- `attention_output` inputs: `t00002018, t00001986, t00002020, t00002035, t00002047, t00001965, t00002064, t00002065`
- `attention_output` outputs: `t00002019, t00002036, t00002067`
- `mlp` inputs: `t00002022, t00002047, t00001965, t00002057, t00002059, t00002062`
- `mlp` outputs: `t00002064`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer24/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer24/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer24/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.24.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.24.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.24.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.24.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.24.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.24.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.24.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.24.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.24.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.24.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.24.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.24` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.24.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.24.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.24.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.24.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.24.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.24.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.24` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001965']` outputs=`['t00001966']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001966']` outputs=`['t00001967']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001967']` outputs=`['t00001968']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001968']` outputs=`['t00001969']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001969']` outputs=`['t00001970']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001966', 't00001970']` outputs=`['t00001971']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001971']` outputs=`['t00001972']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001973', 't00001972']` outputs=`['t00001974']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001974', 't00001975']` outputs=`['t00001976']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001974', 't00001977']` outputs=`['t00001978']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001974', 't00001979']` outputs=`['t00001980']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001976']` outputs=`['t00001981']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001981']` outputs=`['t00001982']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001978']` outputs=`['t00001983']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001983']` outputs=`['t00001984']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001980']` outputs=`['t00001985']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001985']` outputs=`['t00001986']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001988']` outputs=`['t00001989']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001991']` outputs=`['t00001992']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001993']` outputs=`['t00001994']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001992', 't00001475']` outputs=`['t00001995']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001995']` outputs=`['t00001996']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001994', 't00001475']` outputs=`['t00001997']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001997']` outputs=`['t00001998']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001982', 't00001996']` outputs=`['t00001999']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001982']` outputs=`['t00002000']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001982']` outputs=`['t00002001']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002001']` outputs=`['t00002002']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002002', 't00002000']` outputs=`['t00002003']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002003', 't00001998']` outputs=`['t00002004']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001999', 't00002004']` outputs=`['t00002005']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001981']` outputs=`['t00001982']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001983']` outputs=`['t00001984']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001985']` outputs=`['t00001986']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001999', 't00002004']` outputs=`['t00002005']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002006', 't00002011']` outputs=`['t00002012']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002012']` outputs=`['t00002013']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00002005', 't00002013']` outputs=`['t00002014']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00002014']` outputs=`['t00002015']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00002015', 't00001505']` outputs=`['t00002016']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00002016']` outputs=`['t00002017']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00002018']` outputs=`['t00002018']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00002018', 't00001986']` outputs=`['t00002019']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001989']` outputs=`['t00001990']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001990']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00002023']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00002023']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00002026', 't00002027']` outputs=`['t00002028']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00002030']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00002038']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00002040', 't00002039']` outputs=`['t00002041']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00002041', 't00002042']` outputs=`['t00002043']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00002045']` outputs=`['t00002046']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00002018', 't00001986']` outputs=`['t00002019']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002020']` outputs=`['t00002021']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002021']` outputs=`['t00002022']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00002035']` outputs=`['t00002036']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00002022', 't00002047']` outputs=`['t00002048']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001965', 't00002048']` outputs=`['t00002049']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00002064', 't00002065']` outputs=`['t00002066']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00002049', 't00002066']` outputs=`['t00002067']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00002022', 't00002047']` outputs=`['t00002048']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001965', 't00002048']` outputs=`['t00002049']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00002049']` outputs=`['t00002050']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00002050']` outputs=`['t00002051']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00002051']` outputs=`['t00002052']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00002052']` outputs=`['t00002053']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00002053']` outputs=`['t00002054']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00002050', 't00002054']` outputs=`['t00002055']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00002055']` outputs=`['t00002056']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00002057', 't00002056']` outputs=`['t00002058']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00002058', 't00002059']` outputs=`['t00002060']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00002060']` outputs=`['t00002061']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00002058', 't00002062']` outputs=`['t00002063']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00002061', 't00002063']` outputs=`['t00002064']` -> shape=[1, 58, 11008], dtype=float16
