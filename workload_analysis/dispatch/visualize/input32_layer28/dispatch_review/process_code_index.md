# input32_layer28 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input32_layer28/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00003048, t00002382`
- `input_rmsnorm` outputs: `t00003056`
- `qkv_projection` inputs: `t00003056, t00002384, t00002386, t00002388`
- `qkv_projection` outputs: `t00003061, t00003063, t00003065`
- `rope` inputs: `t00003067, t00002401, t00002403, t00002848, t00003061`
- `rope` outputs: `t00003068, t00003082`
- `kv_cache_concat` inputs: `t00003090, t00003089, t00003092, t00003065`
- `kv_cache_concat` outputs: `t00003091, t00003093`
- `attention` inputs: `t00003060, t00003062, t00003064, t00003076, t00003081, t00003083, t00003088, t00003091, t00003097, t00003093`
- `attention` outputs: `t00003061, t00003063, t00003065, t00003089, t00003099, t00003101`
- `attention_output` inputs: `t00003100, t00003093, t00003102, t00002442, t00003048, t00003118, t00002460`
- `attention_output` outputs: `t00003101, t00003120`
- `mlp` inputs: `t00003099, t00003103, t00002442, t00003048, t00002452, t00002454, t00002457`
- `mlp` outputs: `t00003100, t00003116, t00003117`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input32_layer28/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input32_layer28/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input32_layer28/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.28.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.28.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.28.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.28.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.28.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.28.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.28.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.28.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.28.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.28.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.28.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.28.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.28.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00003048']` outputs=`['t00003049']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00003049']` outputs=`['t00003050']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00003050']` outputs=`['t00003051']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00003051']` outputs=`['t00003052']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00003052']` outputs=`['t00003053']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00003049', 't00003053']` outputs=`['t00003054']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00003054']` outputs=`['t00003055']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002382', 't00003055']` outputs=`['t00003056']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00003056', 't00002384']` outputs=`['t00003057']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00003056', 't00002386']` outputs=`['t00003058']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00003056', 't00002388']` outputs=`['t00003059']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00003057']` outputs=`['t00003060']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00003060']` outputs=`['t00003061']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00003058']` outputs=`['t00003062']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00003062']` outputs=`['t00003063']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00003059']` outputs=`['t00003064']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00003064']` outputs=`['t00003065']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00003067']` outputs=`['t00003068']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002401']` outputs=`['t00003070']` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002403']` outputs=`['t00003071']` -> shape=[655, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00003070', 't00002848']` outputs=`['t00003072']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00003072']` outputs=`['t00003073']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00003071', 't00002848']` outputs=`['t00003074']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00003074']` outputs=`['t00003075']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00003061', 't00003073']` outputs=`['t00003076']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00003061']` outputs=`['t00003077']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00003061']` outputs=`['t00003078']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00003078']` outputs=`['t00003079']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00003079', 't00003077']` outputs=`['t00003080']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00003080', 't00003075']` outputs=`['t00003081']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003076', 't00003081']` outputs=`['t00003082']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00003090', 't00003089']` outputs=`['t00003091']` -> shape=[1, 32, 79, 128], dtype=float16
- `#48 cat.default` inputs=`['t00003092', 't00003065']` outputs=`['t00003093']` -> shape=[1, 32, 79, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00003060']` outputs=`['t00003061']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00003062']` outputs=`['t00003063']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00003064']` outputs=`['t00003065']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00003076', 't00003081']` outputs=`['t00003082']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00003083', 't00003088']` outputs=`['t00003089']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00003091']` outputs=`['t00003094']` -> shape=[1, 32, 128, 79], dtype=float16
- `#50 matmul.default` inputs=`['t00003082', 't00003094']` outputs=`['t00003095']` -> shape=[1, 32, 1, 79], dtype=float16
- `#51 div.Tensor` inputs=`['t00003095']` outputs=`['t00003096']` -> shape=[1, 32, 1, 79], dtype=float16
- `#52 add.Tensor` inputs=`['t00003096', 't00003097']` outputs=`['t00003098']` -> shape=[1, 32, 1, 79], dtype=float16
- `#53 softmax.int` inputs=`['t00003098']` outputs=`['t00003099']` -> shape=[1, 32, 1, 79], dtype=float32
- `#55 dropout.default` inputs=`['t00003100']` outputs=`['t00003100']` -> shape=[1, 32, 1, 79], dtype=float16
- `#56 matmul.default` inputs=`['t00003100', 't00003093']` outputs=`['t00003101']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00003100', 't00003093']` outputs=`['t00003101']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00003102']` outputs=`['t00003103']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00003103', 't00002442']` outputs=`['t00003105']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00003048', 't00003105']` outputs=`['t00003106']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00003118', 't00002460']` outputs=`['t00003119']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00003106', 't00003119']` outputs=`['t00003120']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00003099']` outputs=`['t00003100']` -> shape=[1, 32, 1, 79], dtype=float16
- `#61 linear.default` inputs=`['t00003103', 't00002442']` outputs=`['t00003105']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00003048', 't00003105']` outputs=`['t00003106']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00003106']` outputs=`['t00003107']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00003107']` outputs=`['t00003108']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00003108']` outputs=`['t00003109']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00003109']` outputs=`['t00003110']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00003110']` outputs=`['t00003111']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00003107', 't00003111']` outputs=`['t00003112']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00003112']` outputs=`['t00003113']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002452', 't00003113']` outputs=`['t00003114']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00003114', 't00002454']` outputs=`['t00003115']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00003115']` outputs=`['t00003116']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00003114', 't00002457']` outputs=`['t00003117']` -> shape=[1, 1, 11008], dtype=float16
