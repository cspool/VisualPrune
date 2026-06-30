# input1_layer16 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer16/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer16/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer16/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001148, t00001156`
- `input_rmsnorm` outputs: `t00001157`
- `qkv_projection` inputs: `t00001157, t00001158, t00001160, t00001162`
- `qkv_projection` outputs: `t00001165, t00001167, t00001169`
- `rope` inputs: `t00001171, t00001174, t00001176, t00000023, t00001165`
- `rope` outputs: `t00001172, t00001188`
- `attention` inputs: `t00001164, t00001166, t00001168, t00001182, t00001187, t00001189, t00001194, t00000053`
- `attention` outputs: `t00001165, t00001167, t00001200, t00001202`
- `visipruner_similarity_check` inputs: `t00001172, t00000057, t00001210, t00001220, t00001219, t00001222, t00001225`
- `visipruner_similarity_check` outputs: `t00001223, t00001226`
- `attention_output` inputs: `t00001201, t00001169, t00001203, t00001215, t00001227, t00001148, t00001244, t00001245`
- `attention_output` outputs: `t00001202, t00001216, t00001247`
- `mlp` inputs: `t00001205, t00001227, t00001148, t00001237, t00001239, t00001242`
- `mlp` outputs: `t00001244`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer16/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer16/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer16/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.16.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.16.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.16.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.16.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.16.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.16.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.16.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.16.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.16.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.16.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.16.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.16` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.16.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.16.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.16.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.16.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.16.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.16.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.16` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001148']` outputs=`['t00001149']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001149']` outputs=`['t00001150']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001150']` outputs=`['t00001151']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001151']` outputs=`['t00001152']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001152']` outputs=`['t00001153']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001149', 't00001153']` outputs=`['t00001154']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001154']` outputs=`['t00001155']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001156', 't00001155']` outputs=`['t00001157']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001157', 't00001158']` outputs=`['t00001159']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001157', 't00001160']` outputs=`['t00001161']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001157', 't00001162']` outputs=`['t00001163']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001159']` outputs=`['t00001164']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001164']` outputs=`['t00001165']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00001161']` outputs=`['t00001166']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001166']` outputs=`['t00001167']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00001163']` outputs=`['t00001168']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001168']` outputs=`['t00001169']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001171']` outputs=`['t00001172']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001174']` outputs=`['t00001175']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001176']` outputs=`['t00001177']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001175', 't00000023']` outputs=`['t00001178']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001178']` outputs=`['t00001179']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001177', 't00000023']` outputs=`['t00001180']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001180']` outputs=`['t00001181']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001165', 't00001179']` outputs=`['t00001182']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001165']` outputs=`['t00001183']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001165']` outputs=`['t00001184']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001184']` outputs=`['t00001185']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001185', 't00001183']` outputs=`['t00001186']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001186', 't00001181']` outputs=`['t00001187']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001182', 't00001187']` outputs=`['t00001188']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001164']` outputs=`['t00001165']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001166']` outputs=`['t00001167']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001168']` outputs=`['t00001169']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001182', 't00001187']` outputs=`['t00001188']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001189', 't00001194']` outputs=`['t00001195']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001195']` outputs=`['t00001196']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00001188', 't00001196']` outputs=`['t00001197']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00001197']` outputs=`['t00001198']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00001198', 't00000053']` outputs=`['t00001199']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00001199']` outputs=`['t00001200']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00001201']` outputs=`['t00001201']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00001201', 't00001169']` outputs=`['t00001202']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001172']` outputs=`['t00001173']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001173']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001206']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001206']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001210']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001220', 't00001219']` outputs=`['t00001221']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001221', 't00001222']` outputs=`['t00001223']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001225']` outputs=`['t00001226']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00001201', 't00001169']` outputs=`['t00001202']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001203']` outputs=`['t00001204']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001204']` outputs=`['t00001205']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00001215']` outputs=`['t00001216']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00001205', 't00001227']` outputs=`['t00001228']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001148', 't00001228']` outputs=`['t00001229']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00001244', 't00001245']` outputs=`['t00001246']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00001229', 't00001246']` outputs=`['t00001247']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00001205', 't00001227']` outputs=`['t00001228']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001148', 't00001228']` outputs=`['t00001229']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00001229']` outputs=`['t00001230']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00001230']` outputs=`['t00001231']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00001231']` outputs=`['t00001232']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00001232']` outputs=`['t00001233']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00001233']` outputs=`['t00001234']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00001230', 't00001234']` outputs=`['t00001235']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00001235']` outputs=`['t00001236']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00001237', 't00001236']` outputs=`['t00001238']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00001238', 't00001239']` outputs=`['t00001240']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00001240']` outputs=`['t00001241']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00001238', 't00001242']` outputs=`['t00001243']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00001241', 't00001243']` outputs=`['t00001244']` -> shape=[1, 624, 11008], dtype=float16
