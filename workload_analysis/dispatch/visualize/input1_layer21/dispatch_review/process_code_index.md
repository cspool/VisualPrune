# input1_layer21 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer21/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer21/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer21/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001659, t00001667`
- `input_rmsnorm` outputs: `t00001668`
- `qkv_projection` inputs: `t00001668, t00001669, t00001671, t00001673`
- `qkv_projection` outputs: `t00001676, t00001678, t00001680`
- `rope` inputs: `t00001682, t00001685, t00001687, t00001475, t00001676`
- `rope` outputs: `t00001683, t00001699`
- `attention` inputs: `t00001675, t00001677, t00001679, t00001693, t00001698, t00001700, t00001705, t00001505`
- `attention` outputs: `t00001676, t00001678, t00001711, t00001713`
- `visipruner_similarity_check` inputs: `t00001683, t00000057, t00001720, t00001721, t00001724, t00001734, t00001733, t00001736, t00001739`
- `visipruner_similarity_check` outputs: `t00001722, t00001732, t00001737, t00001740`
- `attention_output` inputs: `t00001712, t00001680, t00001714, t00001729, t00001741, t00001659, t00001758, t00001759`
- `attention_output` outputs: `t00001713, t00001730, t00001761`
- `mlp` inputs: `t00001716, t00001741, t00001659, t00001751, t00001753, t00001756`
- `mlp` outputs: `t00001758`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer21/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer21/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer21/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.21.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.21.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.21.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.21.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.21.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.21.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.21.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.21.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.21.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.21.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.21.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.21` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.21.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.21.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.21.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.21.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.21.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.21.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.21` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001659']` outputs=`['t00001660']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001660']` outputs=`['t00001661']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001661']` outputs=`['t00001662']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001662']` outputs=`['t00001663']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001663']` outputs=`['t00001664']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001660', 't00001664']` outputs=`['t00001665']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001665']` outputs=`['t00001666']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001667', 't00001666']` outputs=`['t00001668']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001668', 't00001669']` outputs=`['t00001670']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001668', 't00001671']` outputs=`['t00001672']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001668', 't00001673']` outputs=`['t00001674']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001670']` outputs=`['t00001675']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001675']` outputs=`['t00001676']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001672']` outputs=`['t00001677']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001677']` outputs=`['t00001678']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001674']` outputs=`['t00001679']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001679']` outputs=`['t00001680']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001682']` outputs=`['t00001683']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001685']` outputs=`['t00001686']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001687']` outputs=`['t00001688']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001686', 't00001475']` outputs=`['t00001689']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001689']` outputs=`['t00001690']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001688', 't00001475']` outputs=`['t00001691']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001691']` outputs=`['t00001692']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001676', 't00001690']` outputs=`['t00001693']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001676']` outputs=`['t00001694']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001676']` outputs=`['t00001695']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001695']` outputs=`['t00001696']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001696', 't00001694']` outputs=`['t00001697']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001697', 't00001692']` outputs=`['t00001698']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001693', 't00001698']` outputs=`['t00001699']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001675']` outputs=`['t00001676']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001677']` outputs=`['t00001678']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001679']` outputs=`['t00001680']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001693', 't00001698']` outputs=`['t00001699']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001700', 't00001705']` outputs=`['t00001706']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001706']` outputs=`['t00001707']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00001699', 't00001707']` outputs=`['t00001708']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00001708']` outputs=`['t00001709']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00001709', 't00001505']` outputs=`['t00001710']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00001710']` outputs=`['t00001711']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00001712']` outputs=`['t00001712']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00001712', 't00001680']` outputs=`['t00001713']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001683']` outputs=`['t00001684']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001684']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001717']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001717']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00001720', 't00001721']` outputs=`['t00001722']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00001724']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00001732']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00001734', 't00001733']` outputs=`['t00001735']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00001735', 't00001736']` outputs=`['t00001737']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00001739']` outputs=`['t00001740']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00001712', 't00001680']` outputs=`['t00001713']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001714']` outputs=`['t00001715']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001715']` outputs=`['t00001716']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00001729']` outputs=`['t00001730']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00001716', 't00001741']` outputs=`['t00001742']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001659', 't00001742']` outputs=`['t00001743']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001758', 't00001759']` outputs=`['t00001760']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00001743', 't00001760']` outputs=`['t00001761']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00001716', 't00001741']` outputs=`['t00001742']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001659', 't00001742']` outputs=`['t00001743']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00001743']` outputs=`['t00001744']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00001744']` outputs=`['t00001745']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00001745']` outputs=`['t00001746']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00001746']` outputs=`['t00001747']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00001747']` outputs=`['t00001748']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00001744', 't00001748']` outputs=`['t00001749']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00001749']` outputs=`['t00001750']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00001751', 't00001750']` outputs=`['t00001752']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00001752', 't00001753']` outputs=`['t00001754']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00001754']` outputs=`['t00001755']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00001752', 't00001756']` outputs=`['t00001757']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001755', 't00001757']` outputs=`['t00001758']` -> shape=[1, 58, 11008], dtype=float16
