# input1_layer13 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer13/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer13/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer13/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000851, t00000859`
- `input_rmsnorm` outputs: `t00000860`
- `qkv_projection` inputs: `t00000860, t00000861, t00000863, t00000865`
- `qkv_projection` outputs: `t00000868, t00000870, t00000872`
- `rope` inputs: `t00000874, t00000877, t00000879, t00000023, t00000868`
- `rope` outputs: `t00000875, t00000891`
- `attention` inputs: `t00000867, t00000869, t00000871, t00000885, t00000890, t00000892, t00000897, t00000053`
- `attention` outputs: `t00000868, t00000870, t00000903, t00000905`
- `visipruner_similarity_check` inputs: `t00000875, t00000057, t00000913, t00000923, t00000922, t00000925, t00000928`
- `visipruner_similarity_check` outputs: `t00000926, t00000929`
- `attention_output` inputs: `t00000904, t00000872, t00000906, t00000918, t00000930, t00000851, t00000947, t00000948`
- `attention_output` outputs: `t00000905, t00000919, t00000950`
- `mlp` inputs: `t00000908, t00000930, t00000851, t00000940, t00000942, t00000945`
- `mlp` outputs: `t00000947`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer13/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer13/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer13/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.13.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.13.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.13.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.13.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.13.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.13.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.13.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.13.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.13.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.13.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.13.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.13` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.13.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.13.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.13.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.13.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.13.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.13.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.13` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000851']` outputs=`['t00000852']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000852']` outputs=`['t00000853']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000853']` outputs=`['t00000854']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000854']` outputs=`['t00000855']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000855']` outputs=`['t00000856']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000852', 't00000856']` outputs=`['t00000857']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000857']` outputs=`['t00000858']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000859', 't00000858']` outputs=`['t00000860']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000860', 't00000861']` outputs=`['t00000862']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000860', 't00000863']` outputs=`['t00000864']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000860', 't00000865']` outputs=`['t00000866']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000862']` outputs=`['t00000867']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000867']` outputs=`['t00000868']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000864']` outputs=`['t00000869']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000869']` outputs=`['t00000870']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000866']` outputs=`['t00000871']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000871']` outputs=`['t00000872']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000874']` outputs=`['t00000875']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000877']` outputs=`['t00000878']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000879']` outputs=`['t00000880']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000878', 't00000023']` outputs=`['t00000881']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000881']` outputs=`['t00000882']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000880', 't00000023']` outputs=`['t00000883']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000883']` outputs=`['t00000884']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000868', 't00000882']` outputs=`['t00000885']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000868']` outputs=`['t00000886']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000868']` outputs=`['t00000887']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000887']` outputs=`['t00000888']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000888', 't00000886']` outputs=`['t00000889']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000889', 't00000884']` outputs=`['t00000890']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000885', 't00000890']` outputs=`['t00000891']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000867']` outputs=`['t00000868']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000869']` outputs=`['t00000870']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000871']` outputs=`['t00000872']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000885', 't00000890']` outputs=`['t00000891']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000892', 't00000897']` outputs=`['t00000898']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000898']` outputs=`['t00000899']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000891', 't00000899']` outputs=`['t00000900']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000900']` outputs=`['t00000901']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000901', 't00000053']` outputs=`['t00000902']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000902']` outputs=`['t00000903']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000904']` outputs=`['t00000904']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000904', 't00000872']` outputs=`['t00000905']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000875']` outputs=`['t00000876']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000876']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000909']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000909']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000913']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000923', 't00000922']` outputs=`['t00000924']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000924', 't00000925']` outputs=`['t00000926']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000928']` outputs=`['t00000929']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00000904', 't00000872']` outputs=`['t00000905']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000906']` outputs=`['t00000907']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000907']` outputs=`['t00000908']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000918']` outputs=`['t00000919']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000908', 't00000930']` outputs=`['t00000931']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000851', 't00000931']` outputs=`['t00000932']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000947', 't00000948']` outputs=`['t00000949']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000932', 't00000949']` outputs=`['t00000950']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000908', 't00000930']` outputs=`['t00000931']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000851', 't00000931']` outputs=`['t00000932']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000932']` outputs=`['t00000933']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000933']` outputs=`['t00000934']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000934']` outputs=`['t00000935']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000935']` outputs=`['t00000936']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000936']` outputs=`['t00000937']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000933', 't00000937']` outputs=`['t00000938']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000938']` outputs=`['t00000939']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000940', 't00000939']` outputs=`['t00000941']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000941', 't00000942']` outputs=`['t00000943']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000943']` outputs=`['t00000944']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000941', 't00000945']` outputs=`['t00000946']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000944', 't00000946']` outputs=`['t00000947']` -> shape=[1, 624, 11008], dtype=float16
