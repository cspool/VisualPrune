# input2_layer28 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input2_layer28/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002675, t00002382`
- `input_rmsnorm` outputs: `t00002683`
- `qkv_projection` inputs: `t00002683, t00002384, t00002386, t00002388`
- `qkv_projection` outputs: `t00002688, t00002690, t00002692`
- `rope` inputs: `t00002694, t00002401, t00002403, t00002481, t00002688`
- `rope` outputs: `t00002695, t00002709`
- `kv_cache_concat` inputs: `t00002422, t00002716, t00002395, t00002692`
- `kv_cache_concat` outputs: `t00002717, t00002718`
- `attention` inputs: `t00002687, t00002689, t00002691, t00002703, t00002708, t00002710, t00002715, t00002717, t00002722, t00002718`
- `attention` outputs: `t00002688, t00002690, t00002692, t00002716, t00002724, t00002726`
- `attention_output` inputs: `t00002725, t00002718, t00002727, t00002442, t00002675, t00002743, t00002460`
- `attention_output` outputs: `t00002726, t00002745`
- `mlp` inputs: `t00002724, t00002728, t00002442, t00002675, t00002452, t00002454, t00002457`
- `mlp` outputs: `t00002725, t00002741, t00002742`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input2_layer28/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input2_layer28/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input2_layer28/dispatch_review/dispatch_op_coverage.md`
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
- `#1 to.dtype` inputs=`['t00002675']` outputs=`['t00002676']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002676']` outputs=`['t00002677']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002677']` outputs=`['t00002678']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002678']` outputs=`['t00002679']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002679']` outputs=`['t00002680']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002676', 't00002680']` outputs=`['t00002681']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002681']` outputs=`['t00002682']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002382', 't00002682']` outputs=`['t00002683']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002683', 't00002384']` outputs=`['t00002684']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002683', 't00002386']` outputs=`['t00002685']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002683', 't00002388']` outputs=`['t00002686']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002684']` outputs=`['t00002687']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002687']` outputs=`['t00002688']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002685']` outputs=`['t00002689']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002689']` outputs=`['t00002690']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002686']` outputs=`['t00002691']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002691']` outputs=`['t00002692']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002694']` outputs=`['t00002695']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002401']` outputs=`['t00002697']` -> shape=[625, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002403']` outputs=`['t00002698']` -> shape=[625, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002697', 't00002481']` outputs=`['t00002699']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002699']` outputs=`['t00002700']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002698', 't00002481']` outputs=`['t00002701']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002701']` outputs=`['t00002702']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002688', 't00002700']` outputs=`['t00002703']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002688']` outputs=`['t00002704']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002688']` outputs=`['t00002705']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002705']` outputs=`['t00002706']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002706', 't00002704']` outputs=`['t00002707']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002707', 't00002702']` outputs=`['t00002708']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002703', 't00002708']` outputs=`['t00002709']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00002422', 't00002716']` outputs=`['t00002717']` -> shape=[1, 32, 49, 128], dtype=float16
- `#48 cat.default` inputs=`['t00002395', 't00002692']` outputs=`['t00002718']` -> shape=[1, 32, 49, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002687']` outputs=`['t00002688']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002689']` outputs=`['t00002690']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002691']` outputs=`['t00002692']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002703', 't00002708']` outputs=`['t00002709']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002710', 't00002715']` outputs=`['t00002716']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002717']` outputs=`['t00002719']` -> shape=[1, 32, 128, 49], dtype=float16
- `#50 matmul.default` inputs=`['t00002709', 't00002719']` outputs=`['t00002720']` -> shape=[1, 32, 1, 49], dtype=float16
- `#51 div.Tensor` inputs=`['t00002720']` outputs=`['t00002721']` -> shape=[1, 32, 1, 49], dtype=float16
- `#52 add.Tensor` inputs=`['t00002721', 't00002722']` outputs=`['t00002723']` -> shape=[1, 32, 1, 49], dtype=float16
- `#53 softmax.int` inputs=`['t00002723']` outputs=`['t00002724']` -> shape=[1, 32, 1, 49], dtype=float32
- `#55 dropout.default` inputs=`['t00002725']` outputs=`['t00002725']` -> shape=[1, 32, 1, 49], dtype=float16
- `#56 matmul.default` inputs=`['t00002725', 't00002718']` outputs=`['t00002726']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002725', 't00002718']` outputs=`['t00002726']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002727']` outputs=`['t00002728']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002728', 't00002442']` outputs=`['t00002730']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002675', 't00002730']` outputs=`['t00002731']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002743', 't00002460']` outputs=`['t00002744']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002731', 't00002744']` outputs=`['t00002745']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002724']` outputs=`['t00002725']` -> shape=[1, 32, 1, 49], dtype=float16
- `#61 linear.default` inputs=`['t00002728', 't00002442']` outputs=`['t00002730']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002675', 't00002730']` outputs=`['t00002731']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002731']` outputs=`['t00002732']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002732']` outputs=`['t00002733']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002733']` outputs=`['t00002734']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002734']` outputs=`['t00002735']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002735']` outputs=`['t00002736']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002732', 't00002736']` outputs=`['t00002737']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002737']` outputs=`['t00002738']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002452', 't00002738']` outputs=`['t00002739']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002739', 't00002454']` outputs=`['t00002740']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002740']` outputs=`['t00002741']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002739', 't00002457']` outputs=`['t00002742']` -> shape=[1, 1, 11008], dtype=float16
