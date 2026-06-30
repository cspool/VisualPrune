# input32_layer31 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input32_layer31/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input32_layer31/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input32_layer31/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00003121, t00002754`
- `input_rmsnorm` outputs: `t00003129`
- `qkv_projection` inputs: `t00003129, t00002756, t00002758, t00002760`
- `qkv_projection` outputs: `t00003134, t00003136, t00003138`
- `rope` inputs: `t00003140, t00002772, t00002774, t00002848, t00003134`
- `rope` outputs: `t00003141, t00003155`
- `kv_cache_concat` inputs: `t00003163, t00003162, t00003165, t00003138`
- `kv_cache_concat` outputs: `t00003164, t00003166`
- `attention` inputs: `t00003133, t00003135, t00003137, t00003149, t00003154, t00003156, t00003161, t00003164, t00003170, t00003166`
- `attention` outputs: `t00003134, t00003136, t00003138, t00003162, t00003172, t00003174`
- `attention_output` inputs: `t00003173, t00003166, t00003175, t00002809, t00003121, t00003191, t00002827`
- `attention_output` outputs: `t00003174, t00003193`
- `mlp` inputs: `t00003172, t00003176, t00002809, t00003121, t00002819, t00002821, t00002824`
- `mlp` outputs: `t00003173, t00003189, t00003190`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input32_layer31/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input32_layer31/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input32_layer31/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.31.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.31.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.31.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.31.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.31.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.31.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.31.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.31.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.31.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.31.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.31.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00003121']` outputs=`['t00003122']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00003122']` outputs=`['t00003123']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00003123']` outputs=`['t00003124']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00003124']` outputs=`['t00003125']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00003125']` outputs=`['t00003126']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00003122', 't00003126']` outputs=`['t00003127']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00003127']` outputs=`['t00003128']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002754', 't00003128']` outputs=`['t00003129']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00003129', 't00002756']` outputs=`['t00003130']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00003129', 't00002758']` outputs=`['t00003131']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00003129', 't00002760']` outputs=`['t00003132']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00003130']` outputs=`['t00003133']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00003133']` outputs=`['t00003134']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00003131']` outputs=`['t00003135']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00003135']` outputs=`['t00003136']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00003132']` outputs=`['t00003137']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00003137']` outputs=`['t00003138']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00003140']` outputs=`['t00003141']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002772']` outputs=`['t00003143']` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002774']` outputs=`['t00003144']` -> shape=[655, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00003143', 't00002848']` outputs=`['t00003145']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00003145']` outputs=`['t00003146']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00003144', 't00002848']` outputs=`['t00003147']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00003147']` outputs=`['t00003148']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00003134', 't00003146']` outputs=`['t00003149']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00003134']` outputs=`['t00003150']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00003134']` outputs=`['t00003151']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00003151']` outputs=`['t00003152']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00003152', 't00003150']` outputs=`['t00003153']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00003153', 't00003148']` outputs=`['t00003154']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003149', 't00003154']` outputs=`['t00003155']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00003163', 't00003162']` outputs=`['t00003164']` -> shape=[1, 32, 79, 128], dtype=float16
- `#48 cat.default` inputs=`['t00003165', 't00003138']` outputs=`['t00003166']` -> shape=[1, 32, 79, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00003133']` outputs=`['t00003134']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00003135']` outputs=`['t00003136']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00003137']` outputs=`['t00003138']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003149', 't00003154']` outputs=`['t00003155']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00003156', 't00003161']` outputs=`['t00003162']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00003164']` outputs=`['t00003167']` -> shape=[1, 32, 128, 79], dtype=float16
- `#50 matmul.default` inputs=`['t00003155', 't00003167']` outputs=`['t00003168']` -> shape=[1, 32, 1, 79], dtype=float16
- `#51 div.Tensor` inputs=`['t00003168']` outputs=`['t00003169']` -> shape=[1, 32, 1, 79], dtype=float16
- `#52 add.Tensor` inputs=`['t00003169', 't00003170']` outputs=`['t00003171']` -> shape=[1, 32, 1, 79], dtype=float16
- `#53 softmax.int` inputs=`['t00003171']` outputs=`['t00003172']` -> shape=[1, 32, 1, 79], dtype=float32
- `#55 dropout.default` inputs=`['t00003173']` outputs=`['t00003173']` -> shape=[1, 32, 1, 79], dtype=float16
- `#56 matmul.default` inputs=`['t00003173', 't00003166']` outputs=`['t00003174']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00003173', 't00003166']` outputs=`['t00003174']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00003175']` outputs=`['t00003176']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00003176', 't00002809']` outputs=`['t00003178']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00003121', 't00003178']` outputs=`['t00003179']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00003191', 't00002827']` outputs=`['t00003192']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00003179', 't00003192']` outputs=`['t00003193']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00003172']` outputs=`['t00003173']` -> shape=[1, 32, 1, 79], dtype=float16
- `#61 linear.default` inputs=`['t00003176', 't00002809']` outputs=`['t00003178']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00003121', 't00003178']` outputs=`['t00003179']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00003179']` outputs=`['t00003180']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00003180']` outputs=`['t00003181']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00003181']` outputs=`['t00003182']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00003182']` outputs=`['t00003183']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00003183']` outputs=`['t00003184']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00003180', 't00003184']` outputs=`['t00003185']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00003185']` outputs=`['t00003186']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002819', 't00003186']` outputs=`['t00003187']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00003187', 't00002821']` outputs=`['t00003188']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00003188']` outputs=`['t00003189']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00003187', 't00002824']` outputs=`['t00003190']` -> shape=[1, 1, 11008], dtype=float16
