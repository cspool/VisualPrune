# input1_layer23 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer23/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer23/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer23/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001863, t00001871`
- `input_rmsnorm` outputs: `t00001872`
- `qkv_projection` inputs: `t00001872, t00001873, t00001875, t00001877`
- `qkv_projection` outputs: `t00001880, t00001882, t00001884`
- `rope` inputs: `t00001886, t00001889, t00001891, t00001475, t00001880`
- `rope` outputs: `t00001887, t00001903`
- `attention` inputs: `t00001879, t00001881, t00001883, t00001897, t00001902, t00001904, t00001909, t00001505`
- `attention` outputs: `t00001880, t00001882, t00001915, t00001917`
- `visipruner_similarity_check` inputs: `t00001887, t00000057, t00001924, t00001925, t00001928, t00001938, t00001937, t00001940, t00001943`
- `visipruner_similarity_check` outputs: `t00001926, t00001936, t00001941, t00001944`
- `attention_output` inputs: `t00001916, t00001884, t00001918, t00001933, t00001945, t00001863, t00001962, t00001963`
- `attention_output` outputs: `t00001917, t00001934, t00001965`
- `mlp` inputs: `t00001920, t00001945, t00001863, t00001955, t00001957, t00001960`
- `mlp` outputs: `t00001962`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer23/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer23/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer23/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.23.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.23.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.23.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.23.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.23.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.23.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.23.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.23.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.23.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.23.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.23.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.23` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.23.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.23.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.23.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.23.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.23.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.23.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.23` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001863']` outputs=`['t00001864']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001864']` outputs=`['t00001865']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001865']` outputs=`['t00001866']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001866']` outputs=`['t00001867']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001867']` outputs=`['t00001868']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001864', 't00001868']` outputs=`['t00001869']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001869']` outputs=`['t00001870']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001871', 't00001870']` outputs=`['t00001872']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001872', 't00001873']` outputs=`['t00001874']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001872', 't00001875']` outputs=`['t00001876']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001872', 't00001877']` outputs=`['t00001878']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001874']` outputs=`['t00001879']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001879']` outputs=`['t00001880']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001876']` outputs=`['t00001881']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001881']` outputs=`['t00001882']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001878']` outputs=`['t00001883']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001883']` outputs=`['t00001884']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001886']` outputs=`['t00001887']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001889']` outputs=`['t00001890']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001891']` outputs=`['t00001892']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001890', 't00001475']` outputs=`['t00001893']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001893']` outputs=`['t00001894']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001892', 't00001475']` outputs=`['t00001895']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001895']` outputs=`['t00001896']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001880', 't00001894']` outputs=`['t00001897']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001880']` outputs=`['t00001898']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001880']` outputs=`['t00001899']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001899']` outputs=`['t00001900']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001900', 't00001898']` outputs=`['t00001901']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001901', 't00001896']` outputs=`['t00001902']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001897', 't00001902']` outputs=`['t00001903']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001879']` outputs=`['t00001880']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001881']` outputs=`['t00001882']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001883']` outputs=`['t00001884']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001897', 't00001902']` outputs=`['t00001903']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001904', 't00001909']` outputs=`['t00001910']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001910']` outputs=`['t00001911']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00001903', 't00001911']` outputs=`['t00001912']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00001912']` outputs=`['t00001913']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00001913', 't00001505']` outputs=`['t00001914']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00001914']` outputs=`['t00001915']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00001916']` outputs=`['t00001916']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00001916', 't00001884']` outputs=`['t00001917']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001887']` outputs=`['t00001888']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001888']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001921']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001921']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00001924', 't00001925']` outputs=`['t00001926']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00001928']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00001936']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00001938', 't00001937']` outputs=`['t00001939']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00001939', 't00001940']` outputs=`['t00001941']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00001943']` outputs=`['t00001944']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00001916', 't00001884']` outputs=`['t00001917']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001918']` outputs=`['t00001919']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001919']` outputs=`['t00001920']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00001933']` outputs=`['t00001934']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00001920', 't00001945']` outputs=`['t00001946']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001863', 't00001946']` outputs=`['t00001947']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001962', 't00001963']` outputs=`['t00001964']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00001947', 't00001964']` outputs=`['t00001965']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00001920', 't00001945']` outputs=`['t00001946']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001863', 't00001946']` outputs=`['t00001947']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00001947']` outputs=`['t00001948']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00001948']` outputs=`['t00001949']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00001949']` outputs=`['t00001950']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00001950']` outputs=`['t00001951']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00001951']` outputs=`['t00001952']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00001948', 't00001952']` outputs=`['t00001953']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00001953']` outputs=`['t00001954']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00001955', 't00001954']` outputs=`['t00001956']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00001956', 't00001957']` outputs=`['t00001958']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00001958']` outputs=`['t00001959']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00001956', 't00001960']` outputs=`['t00001961']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001959', 't00001961']` outputs=`['t00001962']` -> shape=[1, 58, 11008], dtype=float16
