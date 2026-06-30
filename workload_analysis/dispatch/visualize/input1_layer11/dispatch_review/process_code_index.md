# input1_layer11 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer11/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000653, t00000661`
- `input_rmsnorm` outputs: `t00000662`
- `qkv_projection` inputs: `t00000662, t00000663, t00000665, t00000667`
- `qkv_projection` outputs: `t00000670, t00000672, t00000674`
- `rope` inputs: `t00000676, t00000679, t00000681, t00000023, t00000670`
- `rope` outputs: `t00000677, t00000693`
- `attention` inputs: `t00000669, t00000671, t00000673, t00000687, t00000692, t00000694, t00000699, t00000053`
- `attention` outputs: `t00000670, t00000672, t00000705, t00000707`
- `visipruner_similarity_check` inputs: `t00000677, t00000057, t00000715, t00000725, t00000724, t00000727, t00000730`
- `visipruner_similarity_check` outputs: `t00000728, t00000731`
- `attention_output` inputs: `t00000706, t00000674, t00000708, t00000720, t00000732, t00000653, t00000749, t00000750`
- `attention_output` outputs: `t00000707, t00000721, t00000752`
- `mlp` inputs: `t00000710, t00000732, t00000653, t00000742, t00000744, t00000747`
- `mlp` outputs: `t00000749`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer11/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer11/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer11/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.11.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.11.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.11.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.11.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.11.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.11.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.11.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.11.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.11.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.11.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.11.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.11` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.11.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.11.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.11.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.11.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.11.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.11.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.11` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000653']` outputs=`['t00000654']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000654']` outputs=`['t00000655']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000655']` outputs=`['t00000656']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000656']` outputs=`['t00000657']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000657']` outputs=`['t00000658']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000654', 't00000658']` outputs=`['t00000659']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000659']` outputs=`['t00000660']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000661', 't00000660']` outputs=`['t00000662']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000662', 't00000663']` outputs=`['t00000664']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000662', 't00000665']` outputs=`['t00000666']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000662', 't00000667']` outputs=`['t00000668']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000664']` outputs=`['t00000669']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000669']` outputs=`['t00000670']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000666']` outputs=`['t00000671']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000671']` outputs=`['t00000672']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000668']` outputs=`['t00000673']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000673']` outputs=`['t00000674']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000676']` outputs=`['t00000677']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000679']` outputs=`['t00000680']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000681']` outputs=`['t00000682']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000680', 't00000023']` outputs=`['t00000683']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000683']` outputs=`['t00000684']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000682', 't00000023']` outputs=`['t00000685']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000685']` outputs=`['t00000686']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000670', 't00000684']` outputs=`['t00000687']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000670']` outputs=`['t00000688']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000670']` outputs=`['t00000689']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000689']` outputs=`['t00000690']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000690', 't00000688']` outputs=`['t00000691']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000691', 't00000686']` outputs=`['t00000692']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000687', 't00000692']` outputs=`['t00000693']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000669']` outputs=`['t00000670']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000671']` outputs=`['t00000672']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000673']` outputs=`['t00000674']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000687', 't00000692']` outputs=`['t00000693']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000694', 't00000699']` outputs=`['t00000700']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000700']` outputs=`['t00000701']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000693', 't00000701']` outputs=`['t00000702']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000702']` outputs=`['t00000703']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000703', 't00000053']` outputs=`['t00000704']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000704']` outputs=`['t00000705']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000706']` outputs=`['t00000706']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000706', 't00000674']` outputs=`['t00000707']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000677']` outputs=`['t00000678']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000678']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000711']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000711']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000715']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000725', 't00000724']` outputs=`['t00000726']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000726', 't00000727']` outputs=`['t00000728']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000730']` outputs=`['t00000731']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00000706', 't00000674']` outputs=`['t00000707']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000708']` outputs=`['t00000709']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000709']` outputs=`['t00000710']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000720']` outputs=`['t00000721']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000710', 't00000732']` outputs=`['t00000733']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000653', 't00000733']` outputs=`['t00000734']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000749', 't00000750']` outputs=`['t00000751']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000734', 't00000751']` outputs=`['t00000752']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000710', 't00000732']` outputs=`['t00000733']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000653', 't00000733']` outputs=`['t00000734']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000734']` outputs=`['t00000735']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000735']` outputs=`['t00000736']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000736']` outputs=`['t00000737']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000737']` outputs=`['t00000738']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000738']` outputs=`['t00000739']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000735', 't00000739']` outputs=`['t00000740']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000740']` outputs=`['t00000741']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000742', 't00000741']` outputs=`['t00000743']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000743', 't00000744']` outputs=`['t00000745']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000745']` outputs=`['t00000746']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000743', 't00000747']` outputs=`['t00000748']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000746', 't00000748']` outputs=`['t00000749']` -> shape=[1, 624, 11008], dtype=float16
