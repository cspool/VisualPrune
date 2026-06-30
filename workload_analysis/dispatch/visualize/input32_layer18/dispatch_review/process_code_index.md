# input32_layer18 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input32_layer18/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input32_layer18/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input32_layer18/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002830, t00001354`
- `input_rmsnorm` outputs: `t00002838`
- `qkv_projection` inputs: `t00002838, t00001356, t00001358, t00001360`
- `qkv_projection` outputs: `t00002843, t00002845, t00002847`
- `rope` inputs: `t00002850, t00001372, t00001374, t00002848, t00002843`
- `rope` outputs: `t00002851, t00002865`
- `kv_cache_concat` inputs: `t00002873, t00002872, t00002875, t00002847`
- `kv_cache_concat` outputs: `t00002874, t00002876`
- `attention` inputs: `t00002842, t00002844, t00002846, t00002859, t00002864, t00002866, t00002871, t00002874, t00002880, t00002876`
- `attention` outputs: `t00002843, t00002845, t00002847, t00002872, t00002882, t00002884`
- `attention_output` inputs: `t00002883, t00002876, t00002885, t00001432, t00002830, t00002901, t00001450`
- `attention_output` outputs: `t00002884, t00002903`
- `mlp` inputs: `t00002882, t00002886, t00001432, t00002830, t00001442, t00001444, t00001447`
- `mlp` outputs: `t00002883, t00002899, t00002900`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input32_layer18/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input32_layer18/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input32_layer18/dispatch_review/dispatch_op_coverage.md`
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
- `#1 to.dtype` inputs=`['t00002830']` outputs=`['t00002831']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002831']` outputs=`['t00002832']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002832']` outputs=`['t00002833']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002833']` outputs=`['t00002834']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002834']` outputs=`['t00002835']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002831', 't00002835']` outputs=`['t00002836']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002836']` outputs=`['t00002837']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001354', 't00002837']` outputs=`['t00002838']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002838', 't00001356']` outputs=`['t00002839']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002838', 't00001358']` outputs=`['t00002840']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002838', 't00001360']` outputs=`['t00002841']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002839']` outputs=`['t00002842']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002842']` outputs=`['t00002843']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002840']` outputs=`['t00002844']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002844']` outputs=`['t00002845']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002841']` outputs=`['t00002846']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002846']` outputs=`['t00002847']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002850']` outputs=`['t00002851']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001372']` outputs=`['t00002853']` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001374']` outputs=`['t00002854']` -> shape=[655, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002853', 't00002848']` outputs=`['t00002855']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002855']` outputs=`['t00002856']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002854', 't00002848']` outputs=`['t00002857']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002857']` outputs=`['t00002858']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002843', 't00002856']` outputs=`['t00002859']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002843']` outputs=`['t00002860']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002843']` outputs=`['t00002861']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002861']` outputs=`['t00002862']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002862', 't00002860']` outputs=`['t00002863']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002863', 't00002858']` outputs=`['t00002864']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002859', 't00002864']` outputs=`['t00002865']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00002873', 't00002872']` outputs=`['t00002874']` -> shape=[1, 32, 655, 128], dtype=float16
- `#48 cat.default` inputs=`['t00002875', 't00002847']` outputs=`['t00002876']` -> shape=[1, 32, 655, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002842']` outputs=`['t00002843']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002844']` outputs=`['t00002845']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002846']` outputs=`['t00002847']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002859', 't00002864']` outputs=`['t00002865']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002866', 't00002871']` outputs=`['t00002872']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002874']` outputs=`['t00002877']` -> shape=[1, 32, 128, 655], dtype=float16
- `#50 matmul.default` inputs=`['t00002865', 't00002877']` outputs=`['t00002878']` -> shape=[1, 32, 1, 655], dtype=float16
- `#51 div.Tensor` inputs=`['t00002878']` outputs=`['t00002879']` -> shape=[1, 32, 1, 655], dtype=float16
- `#52 add.Tensor` inputs=`['t00002879', 't00002880']` outputs=`['t00002881']` -> shape=[1, 32, 1, 655], dtype=float16
- `#53 softmax.int` inputs=`['t00002881']` outputs=`['t00002882']` -> shape=[1, 32, 1, 655], dtype=float32
- `#55 dropout.default` inputs=`['t00002883']` outputs=`['t00002883']` -> shape=[1, 32, 1, 655], dtype=float16
- `#56 matmul.default` inputs=`['t00002883', 't00002876']` outputs=`['t00002884']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002883', 't00002876']` outputs=`['t00002884']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002885']` outputs=`['t00002886']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002886', 't00001432']` outputs=`['t00002888']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002830', 't00002888']` outputs=`['t00002889']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002901', 't00001450']` outputs=`['t00002902']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002889', 't00002902']` outputs=`['t00002903']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002882']` outputs=`['t00002883']` -> shape=[1, 32, 1, 655], dtype=float16
- `#61 linear.default` inputs=`['t00002886', 't00001432']` outputs=`['t00002888']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002830', 't00002888']` outputs=`['t00002889']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002889']` outputs=`['t00002890']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002890']` outputs=`['t00002891']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002891']` outputs=`['t00002892']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002892']` outputs=`['t00002893']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002893']` outputs=`['t00002894']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002890', 't00002894']` outputs=`['t00002895']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002895']` outputs=`['t00002896']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00001442', 't00002896']` outputs=`['t00002897']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002897', 't00001444']` outputs=`['t00002898']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002898']` outputs=`['t00002899']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002897', 't00001447']` outputs=`['t00002900']` -> shape=[1, 1, 11008], dtype=float16
