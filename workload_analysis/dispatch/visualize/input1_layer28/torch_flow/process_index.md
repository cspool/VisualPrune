# input1_layer28 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer28/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `attention_output`
6. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002374, t00002382`
- `input_rmsnorm` outputs: `t00002383`
- `qkv_projection` inputs: `t00002383, t00002384, t00002386, t00002388`
- `qkv_projection` outputs: `t00002391, t00002393, t00002395`
- `rope` inputs: `t00002398, t00002401, t00002403, t00002396, t00002391`
- `rope` outputs: `t00002399, t00002415`
- `attention` inputs: `t00002390, t00002392, t00002394, t00002409, t00002414, t00002416, t00002421, t00002426`
- `attention` outputs: `t00002391, t00002393, t00002428, t00002430`
- `attention_output` inputs: `t00002429, t00002395, t00002431, t00002442, t00002374, t00002459, t00002460`
- `attention_output` outputs: `t00002430, t00002462`
- `mlp` inputs: `t00002436, t00000057, t00002439, t00002433, t00002442, t00002374, t00002452, t00002454`
- `mlp` outputs: `t00002437, t00002438, t00002440, t00002455`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer28/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer28/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer28/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `83`
- ops listed in coverage: `83`
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
| 47 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.28.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.28.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.28.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 59 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 60 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` |
| 63 | `mul.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` |
| 64 | `sub.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 65 | `add.Tensor` | `model.layers.28.self_attn` | `True` | `True` | `mlp` |
| 66 | `eq.Scalar` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.28.self_attn` | `True` | `True` | `` |
| 68 | `linear.default` | `model.layers.28.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 69 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output, mlp` |
| 70 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `pow.Tensor_Scalar` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 72 | `mean.dim` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 73 | `add.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 74 | `rsqrt.default` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 75 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 76 | `to.dtype` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 77 | `mul.Tensor` | `model.layers.28.post_attention_layernorm` | `True` | `True` | `mlp` |
| 78 | `linear.default` | `model.layers.28.mlp.gate_proj` | `True` | `True` | `mlp` |
| 79 | `silu.default` | `model.layers.28.mlp.act_fn` | `True` | `True` | `` |
| 80 | `linear.default` | `model.layers.28.mlp.up_proj` | `True` | `True` | `` |
| 81 | `mul.Tensor` | `model.layers.28.mlp` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.28.mlp.down_proj` | `True` | `True` | `attention_output` |
| 83 | `add.Tensor` | `model.layers.28` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002374']` outputs=`['t00002375']` -> shape=[1, 48, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002375']` outputs=`['t00002376']` -> shape=[1, 48, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002376']` outputs=`['t00002377']` -> shape=[1, 48, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002377']` outputs=`['t00002378']` -> shape=[1, 48, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002378']` outputs=`['t00002379']` -> shape=[1, 48, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002375', 't00002379']` outputs=`['t00002380']` -> shape=[1, 48, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002380']` outputs=`['t00002381']` -> shape=[1, 48, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002382', 't00002381']` outputs=`['t00002383']` -> shape=[1, 48, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002383', 't00002384']` outputs=`['t00002385']` -> shape=[1, 48, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002383', 't00002386']` outputs=`['t00002387']` -> shape=[1, 48, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002383', 't00002388']` outputs=`['t00002389']` -> shape=[1, 48, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002385']` outputs=`['t00002390']` -> shape=[1, 48, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002390']` outputs=`['t00002391']` -> shape=[1, 32, 48, 128], dtype=float16
- `#14 view.default` inputs=`['t00002387']` outputs=`['t00002392']` -> shape=[1, 48, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002392']` outputs=`['t00002393']` -> shape=[1, 32, 48, 128], dtype=float16
- `#16 view.default` inputs=`['t00002389']` outputs=`['t00002394']` -> shape=[1, 48, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002394']` outputs=`['t00002395']` -> shape=[1, 32, 48, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002398']` outputs=`['t00002399']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002401']` outputs=`['t00002402']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002403']` outputs=`['t00002404']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002402', 't00002396']` outputs=`['t00002405']` -> shape=[1, 48, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002405']` outputs=`['t00002406']` -> shape=[1, 1, 48, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002404', 't00002396']` outputs=`['t00002407']` -> shape=[1, 48, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002407']` outputs=`['t00002408']` -> shape=[1, 1, 48, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002391', 't00002406']` outputs=`['t00002409']` -> shape=[1, 32, 48, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002391']` outputs=`['t00002410']` -> shape=[1, 32, 48, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002391']` outputs=`['t00002411']` -> shape=[1, 32, 48, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002411']` outputs=`['t00002412']` -> shape=[1, 32, 48, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002412', 't00002410']` outputs=`['t00002413']` -> shape=[1, 32, 48, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002413', 't00002408']` outputs=`['t00002414']` -> shape=[1, 32, 48, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002409', 't00002414']` outputs=`['t00002415']` -> shape=[1, 32, 48, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002390']` outputs=`['t00002391']` -> shape=[1, 32, 48, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002392']` outputs=`['t00002393']` -> shape=[1, 32, 48, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002394']` outputs=`['t00002395']` -> shape=[1, 32, 48, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002409', 't00002414']` outputs=`['t00002415']` -> shape=[1, 32, 48, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002416', 't00002421']` outputs=`['t00002422']` -> shape=[1, 32, 48, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002422']` outputs=`['t00002423']` -> shape=[1, 32, 128, 48], dtype=float16
- `#48 matmul.default` inputs=`['t00002415', 't00002423']` outputs=`['t00002424']` -> shape=[1, 32, 48, 48], dtype=float16
- `#49 div.Tensor` inputs=`['t00002424']` outputs=`['t00002425']` -> shape=[1, 32, 48, 48], dtype=float16
- `#50 add.Tensor` inputs=`['t00002425', 't00002426']` outputs=`['t00002427']` -> shape=[1, 32, 48, 48], dtype=float16
- `#51 softmax.int` inputs=`['t00002427']` outputs=`['t00002428']` -> shape=[1, 32, 48, 48], dtype=float32
- `#53 dropout.default` inputs=`['t00002429']` outputs=`['t00002429']` -> shape=[1, 32, 48, 48], dtype=float16
- `#54 matmul.default` inputs=`['t00002429', 't00002395']` outputs=`['t00002430']` -> shape=[1, 32, 48, 128], dtype=float16

### `attention_output`
- `#54 matmul.default` inputs=`['t00002429', 't00002395']` outputs=`['t00002430']` -> shape=[1, 32, 48, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002431']` outputs=`['t00002432']` -> shape=[1, 48, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002432']` outputs=`['t00002433']` -> shape=[1, 48, 4096], dtype=float16
- `#68 linear.default` inputs=`['t00002433', 't00002442']` outputs=`['t00002443']` -> shape=[1, 48, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00002374', 't00002443']` outputs=`['t00002444']` -> shape=[1, 48, 4096], dtype=float16
- `#82 linear.default` inputs=`['t00002459', 't00002460']` outputs=`['t00002461']` -> shape=[1, 48, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00002444', 't00002461']` outputs=`['t00002462']` -> shape=[1, 48, 4096], dtype=float16

### `mlp`
- `#62 add.Tensor` inputs=`['t00002436']` outputs=`['t00002437']` -> shape=[], dtype=int64
- `#63 mul.Tensor` inputs=`['t00000057']` outputs=`['t00002438']` -> shape=[], dtype=int64
- `#65 add.Tensor` inputs=`['t00002439']` outputs=`['t00002440']` -> shape=[], dtype=int64
- `#68 linear.default` inputs=`['t00002433', 't00002442']` outputs=`['t00002443']` -> shape=[1, 48, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00002374', 't00002443']` outputs=`['t00002444']` -> shape=[1, 48, 4096], dtype=float16
- `#70 to.dtype` inputs=`['t00002444']` outputs=`['t00002445']` -> shape=[1, 48, 4096], dtype=float32
- `#71 pow.Tensor_Scalar` inputs=`['t00002445']` outputs=`['t00002446']` -> shape=[1, 48, 4096], dtype=float32
- `#72 mean.dim` inputs=`['t00002446']` outputs=`['t00002447']` -> shape=[1, 48, 1], dtype=float32
- `#73 add.Tensor` inputs=`['t00002447']` outputs=`['t00002448']` -> shape=[1, 48, 1], dtype=float32
- `#74 rsqrt.default` inputs=`['t00002448']` outputs=`['t00002449']` -> shape=[1, 48, 1], dtype=float32
- `#75 mul.Tensor` inputs=`['t00002445', 't00002449']` outputs=`['t00002450']` -> shape=[1, 48, 4096], dtype=float32
- `#76 to.dtype` inputs=`['t00002450']` outputs=`['t00002451']` -> shape=[1, 48, 4096], dtype=float16
- `#77 mul.Tensor` inputs=`['t00002452', 't00002451']` outputs=`['t00002453']` -> shape=[1, 48, 4096], dtype=float16
- `#78 linear.default` inputs=`['t00002453', 't00002454']` outputs=`['t00002455']` -> shape=[1, 48, 11008], dtype=float16
