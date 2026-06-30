# input2_layer27 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input2_layer27/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002605, t00002279`
- `input_rmsnorm` outputs: `t00002613`
- `qkv_projection` inputs: `t00002613, t00002281, t00002283, t00002285`
- `qkv_projection` outputs: `t00002618, t00002620, t00002622`
- `rope` inputs: `t00002624, t00002297, t00002299, t00002481, t00002618`
- `rope` outputs: `t00002625, t00002639`
- `kv_cache_concat` inputs: `t00002318, t00002646, t00002292, t00002622`
- `kv_cache_concat` outputs: `t00002647, t00002648`
- `attention` inputs: `t00002617, t00002619, t00002621, t00002633, t00002638, t00002640, t00002645, t00002647, t00002652, t00002648`
- `attention` outputs: `t00002618, t00002620, t00002622, t00002646, t00002654, t00002656`
- `attention_output` inputs: `t00002655, t00002648, t00002657, t00002353, t00002605, t00002673, t00002371`
- `attention_output` outputs: `t00002656, t00002675`
- `mlp` inputs: `t00002654, t00002658, t00002353, t00002605, t00002363, t00002365, t00002368`
- `mlp` outputs: `t00002655, t00002671, t00002672`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input2_layer27/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input2_layer27/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input2_layer27/dispatch_review/dispatch_op_coverage.md`
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
- `#1 to.dtype` inputs=`['t00002605']` outputs=`['t00002606']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002606']` outputs=`['t00002607']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002607']` outputs=`['t00002608']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002608']` outputs=`['t00002609']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002609']` outputs=`['t00002610']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002606', 't00002610']` outputs=`['t00002611']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002611']` outputs=`['t00002612']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002279', 't00002612']` outputs=`['t00002613']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002613', 't00002281']` outputs=`['t00002614']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002613', 't00002283']` outputs=`['t00002615']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002613', 't00002285']` outputs=`['t00002616']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002614']` outputs=`['t00002617']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002617']` outputs=`['t00002618']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002615']` outputs=`['t00002619']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002619']` outputs=`['t00002620']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002616']` outputs=`['t00002621']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002621']` outputs=`['t00002622']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002624']` outputs=`['t00002625']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002297']` outputs=`['t00002627']` -> shape=[625, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002299']` outputs=`['t00002628']` -> shape=[625, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002627', 't00002481']` outputs=`['t00002629']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002629']` outputs=`['t00002630']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002628', 't00002481']` outputs=`['t00002631']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002631']` outputs=`['t00002632']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002618', 't00002630']` outputs=`['t00002633']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002618']` outputs=`['t00002634']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002618']` outputs=`['t00002635']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002635']` outputs=`['t00002636']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002636', 't00002634']` outputs=`['t00002637']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002637', 't00002632']` outputs=`['t00002638']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002633', 't00002638']` outputs=`['t00002639']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00002318', 't00002646']` outputs=`['t00002647']` -> shape=[1, 32, 59, 128], dtype=float16
- `#48 cat.default` inputs=`['t00002292', 't00002622']` outputs=`['t00002648']` -> shape=[1, 32, 59, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002617']` outputs=`['t00002618']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002619']` outputs=`['t00002620']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002621']` outputs=`['t00002622']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002633', 't00002638']` outputs=`['t00002639']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002640', 't00002645']` outputs=`['t00002646']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002647']` outputs=`['t00002649']` -> shape=[1, 32, 128, 59], dtype=float16
- `#50 matmul.default` inputs=`['t00002639', 't00002649']` outputs=`['t00002650']` -> shape=[1, 32, 1, 59], dtype=float16
- `#51 div.Tensor` inputs=`['t00002650']` outputs=`['t00002651']` -> shape=[1, 32, 1, 59], dtype=float16
- `#52 add.Tensor` inputs=`['t00002651', 't00002652']` outputs=`['t00002653']` -> shape=[1, 32, 1, 59], dtype=float16
- `#53 softmax.int` inputs=`['t00002653']` outputs=`['t00002654']` -> shape=[1, 32, 1, 59], dtype=float32
- `#55 dropout.default` inputs=`['t00002655']` outputs=`['t00002655']` -> shape=[1, 32, 1, 59], dtype=float16
- `#56 matmul.default` inputs=`['t00002655', 't00002648']` outputs=`['t00002656']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002655', 't00002648']` outputs=`['t00002656']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002657']` outputs=`['t00002658']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002658', 't00002353']` outputs=`['t00002660']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002605', 't00002660']` outputs=`['t00002661']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002673', 't00002371']` outputs=`['t00002674']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002661', 't00002674']` outputs=`['t00002675']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002654']` outputs=`['t00002655']` -> shape=[1, 32, 1, 59], dtype=float16
- `#61 linear.default` inputs=`['t00002658', 't00002353']` outputs=`['t00002660']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002605', 't00002660']` outputs=`['t00002661']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002661']` outputs=`['t00002662']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002662']` outputs=`['t00002663']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002663']` outputs=`['t00002664']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002664']` outputs=`['t00002665']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002665']` outputs=`['t00002666']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002662', 't00002666']` outputs=`['t00002667']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002667']` outputs=`['t00002668']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002363', 't00002668']` outputs=`['t00002669']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002669', 't00002365']` outputs=`['t00002670']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002670']` outputs=`['t00002671']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002669', 't00002368']` outputs=`['t00002672']` -> shape=[1, 1, 11008], dtype=float16
