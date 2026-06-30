# input32_layer27 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input32_layer27/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input32_layer27/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input32_layer27/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002976, t00002279`
- `input_rmsnorm` outputs: `t00002984`
- `qkv_projection` inputs: `t00002984, t00002281, t00002283, t00002285`
- `qkv_projection` outputs: `t00002989, t00002991, t00002993`
- `rope` inputs: `t00002995, t00002297, t00002299, t00002848, t00002989`
- `rope` outputs: `t00002996, t00003010`
- `kv_cache_concat` inputs: `t00003018, t00003017, t00003020, t00002993`
- `kv_cache_concat` outputs: `t00003019, t00003021`
- `attention` inputs: `t00002988, t00002990, t00002992, t00003004, t00003009, t00003011, t00003016, t00003019, t00003025, t00003021`
- `attention` outputs: `t00002989, t00002991, t00002993, t00003017, t00003027, t00003029`
- `attention_output` inputs: `t00003028, t00003021, t00003030, t00002353, t00002976, t00003046, t00002371`
- `attention_output` outputs: `t00003029, t00003048`
- `mlp` inputs: `t00003027, t00003031, t00002353, t00002976, t00002363, t00002365, t00002368`
- `mlp` outputs: `t00003028, t00003044, t00003045`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input32_layer27/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input32_layer27/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input32_layer27/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.27.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.27.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.27.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.27.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.27.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.27.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.27.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.27.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.27.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.27.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002976']` outputs=`['t00002977']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002977']` outputs=`['t00002978']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002978']` outputs=`['t00002979']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002979']` outputs=`['t00002980']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002980']` outputs=`['t00002981']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002977', 't00002981']` outputs=`['t00002982']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002982']` outputs=`['t00002983']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002279', 't00002983']` outputs=`['t00002984']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002984', 't00002281']` outputs=`['t00002985']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002984', 't00002283']` outputs=`['t00002986']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002984', 't00002285']` outputs=`['t00002987']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002985']` outputs=`['t00002988']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002988']` outputs=`['t00002989']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002986']` outputs=`['t00002990']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002990']` outputs=`['t00002991']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002987']` outputs=`['t00002992']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002992']` outputs=`['t00002993']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002995']` outputs=`['t00002996']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002297']` outputs=`['t00002998']` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002299']` outputs=`['t00002999']` -> shape=[655, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002998', 't00002848']` outputs=`['t00003000']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00003000']` outputs=`['t00003001']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002999', 't00002848']` outputs=`['t00003002']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00003002']` outputs=`['t00003003']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002989', 't00003001']` outputs=`['t00003004']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002989']` outputs=`['t00003005']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002989']` outputs=`['t00003006']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00003006']` outputs=`['t00003007']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00003007', 't00003005']` outputs=`['t00003008']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00003008', 't00003003']` outputs=`['t00003009']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003004', 't00003009']` outputs=`['t00003010']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00003018', 't00003017']` outputs=`['t00003019']` -> shape=[1, 32, 89, 128], dtype=float16
- `#48 cat.default` inputs=`['t00003020', 't00002993']` outputs=`['t00003021']` -> shape=[1, 32, 89, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002988']` outputs=`['t00002989']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002990']` outputs=`['t00002991']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002992']` outputs=`['t00002993']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003004', 't00003009']` outputs=`['t00003010']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00003011', 't00003016']` outputs=`['t00003017']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00003019']` outputs=`['t00003022']` -> shape=[1, 32, 128, 89], dtype=float16
- `#50 matmul.default` inputs=`['t00003010', 't00003022']` outputs=`['t00003023']` -> shape=[1, 32, 1, 89], dtype=float16
- `#51 div.Tensor` inputs=`['t00003023']` outputs=`['t00003024']` -> shape=[1, 32, 1, 89], dtype=float16
- `#52 add.Tensor` inputs=`['t00003024', 't00003025']` outputs=`['t00003026']` -> shape=[1, 32, 1, 89], dtype=float16
- `#53 softmax.int` inputs=`['t00003026']` outputs=`['t00003027']` -> shape=[1, 32, 1, 89], dtype=float32
- `#55 dropout.default` inputs=`['t00003028']` outputs=`['t00003028']` -> shape=[1, 32, 1, 89], dtype=float16
- `#56 matmul.default` inputs=`['t00003028', 't00003021']` outputs=`['t00003029']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00003028', 't00003021']` outputs=`['t00003029']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00003030']` outputs=`['t00003031']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00003031', 't00002353']` outputs=`['t00003033']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002976', 't00003033']` outputs=`['t00003034']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00003046', 't00002371']` outputs=`['t00003047']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00003034', 't00003047']` outputs=`['t00003048']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00003027']` outputs=`['t00003028']` -> shape=[1, 32, 1, 89], dtype=float16
- `#61 linear.default` inputs=`['t00003031', 't00002353']` outputs=`['t00003033']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002976', 't00003033']` outputs=`['t00003034']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00003034']` outputs=`['t00003035']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00003035']` outputs=`['t00003036']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00003036']` outputs=`['t00003037']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00003037']` outputs=`['t00003038']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00003038']` outputs=`['t00003039']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00003035', 't00003039']` outputs=`['t00003040']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00003040']` outputs=`['t00003041']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002363', 't00003041']` outputs=`['t00003042']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00003042', 't00002365']` outputs=`['t00003043']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00003043']` outputs=`['t00003044']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00003042', 't00002368']` outputs=`['t00003045']` -> shape=[1, 1, 11008], dtype=float16
