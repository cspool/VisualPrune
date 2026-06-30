# input1_layer8 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer8/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer8/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer8/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000356, t00000364`
- `input_rmsnorm` outputs: `t00000365`
- `qkv_projection` inputs: `t00000365, t00000366, t00000368, t00000370`
- `qkv_projection` outputs: `t00000373, t00000375, t00000377`
- `rope` inputs: `t00000379, t00000382, t00000384, t00000023, t00000373`
- `rope` outputs: `t00000380, t00000396`
- `attention` inputs: `t00000372, t00000374, t00000376, t00000390, t00000395, t00000397, t00000402, t00000053`
- `attention` outputs: `t00000373, t00000375, t00000408, t00000410`
- `visipruner_similarity_check` inputs: `t00000380, t00000057, t00000418, t00000428, t00000427, t00000430, t00000433`
- `visipruner_similarity_check` outputs: `t00000431, t00000434`
- `attention_output` inputs: `t00000409, t00000377, t00000411, t00000423, t00000435, t00000356, t00000452, t00000453`
- `attention_output` outputs: `t00000410, t00000424, t00000455`
- `mlp` inputs: `t00000413, t00000435, t00000356, t00000445, t00000447, t00000450`
- `mlp` outputs: `t00000452`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer8/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer8/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer8/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.8.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.8.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.8.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.8.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.8.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.8.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.8.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.8.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.8.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.8.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.8.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.8` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.8.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.8.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.8.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.8.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.8.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.8.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.8` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000356']` outputs=`['t00000357']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000357']` outputs=`['t00000358']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000358']` outputs=`['t00000359']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000359']` outputs=`['t00000360']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000360']` outputs=`['t00000361']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000357', 't00000361']` outputs=`['t00000362']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000362']` outputs=`['t00000363']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000364', 't00000363']` outputs=`['t00000365']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000365', 't00000366']` outputs=`['t00000367']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000365', 't00000368']` outputs=`['t00000369']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000365', 't00000370']` outputs=`['t00000371']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000367']` outputs=`['t00000372']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000372']` outputs=`['t00000373']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000369']` outputs=`['t00000374']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000374']` outputs=`['t00000375']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000371']` outputs=`['t00000376']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000376']` outputs=`['t00000377']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000379']` outputs=`['t00000380']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000382']` outputs=`['t00000383']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000384']` outputs=`['t00000385']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000383', 't00000023']` outputs=`['t00000386']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000386']` outputs=`['t00000387']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000385', 't00000023']` outputs=`['t00000388']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000388']` outputs=`['t00000389']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000373', 't00000387']` outputs=`['t00000390']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000373']` outputs=`['t00000391']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000373']` outputs=`['t00000392']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000392']` outputs=`['t00000393']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000393', 't00000391']` outputs=`['t00000394']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000394', 't00000389']` outputs=`['t00000395']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000390', 't00000395']` outputs=`['t00000396']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000372']` outputs=`['t00000373']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000374']` outputs=`['t00000375']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000376']` outputs=`['t00000377']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000390', 't00000395']` outputs=`['t00000396']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000397', 't00000402']` outputs=`['t00000403']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000403']` outputs=`['t00000404']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000396', 't00000404']` outputs=`['t00000405']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000405']` outputs=`['t00000406']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000406', 't00000053']` outputs=`['t00000407']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000407']` outputs=`['t00000408']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000409']` outputs=`['t00000409']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000409', 't00000377']` outputs=`['t00000410']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000380']` outputs=`['t00000381']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000381']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000414']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000414']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000418']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000428', 't00000427']` outputs=`['t00000429']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000429', 't00000430']` outputs=`['t00000431']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000433']` outputs=`['t00000434']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00000409', 't00000377']` outputs=`['t00000410']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000411']` outputs=`['t00000412']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000412']` outputs=`['t00000413']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000423']` outputs=`['t00000424']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000413', 't00000435']` outputs=`['t00000436']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000356', 't00000436']` outputs=`['t00000437']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000452', 't00000453']` outputs=`['t00000454']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000437', 't00000454']` outputs=`['t00000455']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000413', 't00000435']` outputs=`['t00000436']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000356', 't00000436']` outputs=`['t00000437']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000437']` outputs=`['t00000438']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000438']` outputs=`['t00000439']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000439']` outputs=`['t00000440']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000440']` outputs=`['t00000441']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000441']` outputs=`['t00000442']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000438', 't00000442']` outputs=`['t00000443']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000443']` outputs=`['t00000444']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000445', 't00000444']` outputs=`['t00000446']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000446', 't00000447']` outputs=`['t00000448']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000448']` outputs=`['t00000449']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000446', 't00000450']` outputs=`['t00000451']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000449', 't00000451']` outputs=`['t00000452']` -> shape=[1, 624, 11008], dtype=float16
