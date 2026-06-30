# input32_layer19 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input32_layer19/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input32_layer19/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input32_layer19/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002903, t00001461`
- `input_rmsnorm` outputs: `t00002911`
- `qkv_projection` inputs: `t00002911, t00001463, t00001465, t00001467`
- `qkv_projection` outputs: `t00002916, t00002918, t00002920`
- `rope` inputs: `t00002922, t00001480, t00001482, t00002848, t00002916`
- `rope` outputs: `t00002923, t00002937`
- `kv_cache_concat` inputs: `t00002945, t00002944, t00002947, t00002920`
- `kv_cache_concat` outputs: `t00002946, t00002948`
- `attention` inputs: `t00002915, t00002917, t00002919, t00002931, t00002936, t00002938, t00002943, t00002946, t00002952, t00002948`
- `attention` outputs: `t00002916, t00002918, t00002920, t00002944, t00002954, t00002956`
- `attention_output` inputs: `t00002955, t00002948, t00002957, t00001537, t00002903, t00002973, t00001555`
- `attention_output` outputs: `t00002956, t00002975`
- `mlp` inputs: `t00002954, t00002958, t00001537, t00002903, t00001547, t00001549, t00001552`
- `mlp` outputs: `t00002955, t00002971, t00002972`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input32_layer19/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input32_layer19/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input32_layer19/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.19.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.19.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.19.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.19.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.19.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.19.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.19.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.19.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.19.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.19.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002903']` outputs=`['t00002904']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002904']` outputs=`['t00002905']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002905']` outputs=`['t00002906']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002906']` outputs=`['t00002907']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002907']` outputs=`['t00002908']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002904', 't00002908']` outputs=`['t00002909']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002909']` outputs=`['t00002910']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001461', 't00002910']` outputs=`['t00002911']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002911', 't00001463']` outputs=`['t00002912']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002911', 't00001465']` outputs=`['t00002913']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002911', 't00001467']` outputs=`['t00002914']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002912']` outputs=`['t00002915']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002915']` outputs=`['t00002916']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002913']` outputs=`['t00002917']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002917']` outputs=`['t00002918']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002914']` outputs=`['t00002919']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002919']` outputs=`['t00002920']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002922']` outputs=`['t00002923']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001480']` outputs=`['t00002925']` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001482']` outputs=`['t00002926']` -> shape=[655, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002925', 't00002848']` outputs=`['t00002927']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002927']` outputs=`['t00002928']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002926', 't00002848']` outputs=`['t00002929']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002929']` outputs=`['t00002930']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002916', 't00002928']` outputs=`['t00002931']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002916']` outputs=`['t00002932']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002916']` outputs=`['t00002933']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002933']` outputs=`['t00002934']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002934', 't00002932']` outputs=`['t00002935']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002935', 't00002930']` outputs=`['t00002936']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002931', 't00002936']` outputs=`['t00002937']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00002945', 't00002944']` outputs=`['t00002946']` -> shape=[1, 32, 89, 128], dtype=float16
- `#48 cat.default` inputs=`['t00002947', 't00002920']` outputs=`['t00002948']` -> shape=[1, 32, 89, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002915']` outputs=`['t00002916']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002917']` outputs=`['t00002918']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002919']` outputs=`['t00002920']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002931', 't00002936']` outputs=`['t00002937']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002938', 't00002943']` outputs=`['t00002944']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002946']` outputs=`['t00002949']` -> shape=[1, 32, 128, 89], dtype=float16
- `#50 matmul.default` inputs=`['t00002937', 't00002949']` outputs=`['t00002950']` -> shape=[1, 32, 1, 89], dtype=float16
- `#51 div.Tensor` inputs=`['t00002950']` outputs=`['t00002951']` -> shape=[1, 32, 1, 89], dtype=float16
- `#52 add.Tensor` inputs=`['t00002951', 't00002952']` outputs=`['t00002953']` -> shape=[1, 32, 1, 89], dtype=float16
- `#53 softmax.int` inputs=`['t00002953']` outputs=`['t00002954']` -> shape=[1, 32, 1, 89], dtype=float32
- `#55 dropout.default` inputs=`['t00002955']` outputs=`['t00002955']` -> shape=[1, 32, 1, 89], dtype=float16
- `#56 matmul.default` inputs=`['t00002955', 't00002948']` outputs=`['t00002956']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002955', 't00002948']` outputs=`['t00002956']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002957']` outputs=`['t00002958']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002958', 't00001537']` outputs=`['t00002960']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002903', 't00002960']` outputs=`['t00002961']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002973', 't00001555']` outputs=`['t00002974']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002961', 't00002974']` outputs=`['t00002975']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002954']` outputs=`['t00002955']` -> shape=[1, 32, 1, 89], dtype=float16
- `#61 linear.default` inputs=`['t00002958', 't00001537']` outputs=`['t00002960']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002903', 't00002960']` outputs=`['t00002961']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002961']` outputs=`['t00002962']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002962']` outputs=`['t00002963']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002963']` outputs=`['t00002964']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002964']` outputs=`['t00002965']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002965']` outputs=`['t00002966']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002962', 't00002966']` outputs=`['t00002967']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002967']` outputs=`['t00002968']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00001547', 't00002968']` outputs=`['t00002969']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002969', 't00001549']` outputs=`['t00002970']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002970']` outputs=`['t00002971']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002969', 't00001552']` outputs=`['t00002972']` -> shape=[1, 1, 11008], dtype=float16
