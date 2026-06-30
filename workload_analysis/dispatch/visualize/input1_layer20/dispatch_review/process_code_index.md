# input1_layer20 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer20/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001557, t00001565`
- `input_rmsnorm` outputs: `t00001566`
- `qkv_projection` inputs: `t00001566, t00001567, t00001569, t00001571`
- `qkv_projection` outputs: `t00001574, t00001576, t00001578`
- `rope` inputs: `t00001580, t00001583, t00001585, t00001475, t00001574`
- `rope` outputs: `t00001581, t00001597`
- `attention` inputs: `t00001573, t00001575, t00001577, t00001591, t00001596, t00001598, t00001603, t00001505`
- `attention` outputs: `t00001574, t00001576, t00001609, t00001611`
- `visipruner_similarity_check` inputs: `t00001581, t00000057, t00001618, t00001619, t00001622, t00001632, t00001631, t00001634, t00001637`
- `visipruner_similarity_check` outputs: `t00001620, t00001630, t00001635, t00001638`
- `attention_output` inputs: `t00001610, t00001578, t00001612, t00001627, t00001639, t00001557, t00001656, t00001657`
- `attention_output` outputs: `t00001611, t00001628, t00001659`
- `mlp` inputs: `t00001614, t00001639, t00001557, t00001649, t00001651, t00001654`
- `mlp` outputs: `t00001656`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer20/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer20/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer20/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.20.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.20.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.20.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.20.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.20.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.20.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.20.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.20.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.20.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.20.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.20.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.20` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.20.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.20.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.20.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.20.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.20.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.20.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.20` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001557']` outputs=`['t00001558']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001558']` outputs=`['t00001559']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001559']` outputs=`['t00001560']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001560']` outputs=`['t00001561']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001561']` outputs=`['t00001562']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001558', 't00001562']` outputs=`['t00001563']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001563']` outputs=`['t00001564']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001565', 't00001564']` outputs=`['t00001566']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001566', 't00001567']` outputs=`['t00001568']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001566', 't00001569']` outputs=`['t00001570']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001566', 't00001571']` outputs=`['t00001572']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001568']` outputs=`['t00001573']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001573']` outputs=`['t00001574']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001570']` outputs=`['t00001575']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001575']` outputs=`['t00001576']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001572']` outputs=`['t00001577']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001577']` outputs=`['t00001578']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001580']` outputs=`['t00001581']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001583']` outputs=`['t00001584']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001585']` outputs=`['t00001586']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001584', 't00001475']` outputs=`['t00001587']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001587']` outputs=`['t00001588']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001586', 't00001475']` outputs=`['t00001589']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001589']` outputs=`['t00001590']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001574', 't00001588']` outputs=`['t00001591']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001574']` outputs=`['t00001592']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001574']` outputs=`['t00001593']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001593']` outputs=`['t00001594']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001594', 't00001592']` outputs=`['t00001595']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001595', 't00001590']` outputs=`['t00001596']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001591', 't00001596']` outputs=`['t00001597']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001573']` outputs=`['t00001574']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001575']` outputs=`['t00001576']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001577']` outputs=`['t00001578']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001591', 't00001596']` outputs=`['t00001597']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001598', 't00001603']` outputs=`['t00001604']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001604']` outputs=`['t00001605']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00001597', 't00001605']` outputs=`['t00001606']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00001606']` outputs=`['t00001607']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00001607', 't00001505']` outputs=`['t00001608']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00001608']` outputs=`['t00001609']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00001610']` outputs=`['t00001610']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00001610', 't00001578']` outputs=`['t00001611']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001581']` outputs=`['t00001582']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001582']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001615']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001615']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00001618', 't00001619']` outputs=`['t00001620']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00001622']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00001630']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00001632', 't00001631']` outputs=`['t00001633']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00001633', 't00001634']` outputs=`['t00001635']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00001637']` outputs=`['t00001638']` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` inputs=`['t00001610', 't00001578']` outputs=`['t00001611']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001612']` outputs=`['t00001613']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001613']` outputs=`['t00001614']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00001627']` outputs=`['t00001628']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00001614', 't00001639']` outputs=`['t00001640']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001557', 't00001640']` outputs=`['t00001641']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001656', 't00001657']` outputs=`['t00001658']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00001641', 't00001658']` outputs=`['t00001659']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00001614', 't00001639']` outputs=`['t00001640']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001557', 't00001640']` outputs=`['t00001641']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00001641']` outputs=`['t00001642']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00001642']` outputs=`['t00001643']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00001643']` outputs=`['t00001644']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00001644']` outputs=`['t00001645']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00001645']` outputs=`['t00001646']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00001642', 't00001646']` outputs=`['t00001647']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00001647']` outputs=`['t00001648']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00001649', 't00001648']` outputs=`['t00001650']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00001650', 't00001651']` outputs=`['t00001652']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00001652']` outputs=`['t00001653']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00001650', 't00001654']` outputs=`['t00001655']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001653', 't00001655']` outputs=`['t00001656']` -> shape=[1, 58, 11008], dtype=float16
