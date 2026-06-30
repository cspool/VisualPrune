# input2_layer31 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input2_layer31/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002746, t00002754`
- `input_rmsnorm` outputs: `t00002755`
- `qkv_projection` inputs: `t00002755, t00002756, t00002758, t00002760`
- `qkv_projection` outputs: `t00002763, t00002765, t00002767`
- `rope` inputs: `t00002769, t00002772, t00002774, t00002481, t00002763`
- `rope` outputs: `t00002770, t00002786`
- `kv_cache_concat` inputs: `t00002794, t00002793, t00002796, t00002767`
- `kv_cache_concat` outputs: `t00002795, t00002797`
- `attention` inputs: `t00002762, t00002764, t00002766, t00002780, t00002785, t00002787, t00002792, t00002795, t00002801, t00002797`
- `attention` outputs: `t00002763, t00002765, t00002767, t00002793, t00002803, t00002805`
- `attention_output` inputs: `t00002804, t00002797, t00002806, t00002809, t00002746, t00002826, t00002827`
- `attention_output` outputs: `t00002805, t00002829`
- `mlp` inputs: `t00002803, t00002807, t00002809, t00002746, t00002819, t00002821, t00002824`
- `mlp` outputs: `t00002804, t00002823, t00002825`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input2_layer31/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input2_layer31/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input2_layer31/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `76`
- ops listed in coverage: `76`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.31.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.31.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.31.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.31.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.31.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 47 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` |
| 48 | `cat.default` | `model.layers.31.self_attn` | `True` | `True` | `kv_cache_concat` |
| 49 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 50 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 51 | `div.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 52 | `add.Tensor` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 53 | `softmax.int` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 54 | `to.dtype` | `model.layers.31.self_attn` | `True` | `True` | `mlp` |
| 55 | `dropout.default` | `model.layers.31.self_attn` | `True` | `True` | `attention` |
| 56 | `matmul.default` | `model.layers.31.self_attn` | `True` | `True` | `attention, attention_output` |
| 57 | `transpose.int` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 58 | `reshape.default` | `model.layers.31.self_attn` | `True` | `True` | `attention_output` |
| 59 | `gt.Scalar` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 60 | `is_nonzero.default` | `model.layers.31.self_attn` | `True` | `True` | `` |
| 61 | `linear.default` | `model.layers.31.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 62 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output, mlp` |
| 63 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 64 | `pow.Tensor_Scalar` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 65 | `mean.dim` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 66 | `add.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 67 | `rsqrt.default` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 68 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 69 | `to.dtype` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 70 | `mul.Tensor` | `model.layers.31.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `linear.default` | `model.layers.31.mlp.gate_proj` | `True` | `True` | `mlp` |
| 72 | `silu.default` | `model.layers.31.mlp.act_fn` | `True` | `True` | `mlp` |
| 73 | `linear.default` | `model.layers.31.mlp.up_proj` | `True` | `True` | `mlp` |
| 74 | `mul.Tensor` | `model.layers.31.mlp` | `True` | `True` | `` |
| 75 | `linear.default` | `model.layers.31.mlp.down_proj` | `True` | `True` | `attention_output` |
| 76 | `add.Tensor` | `model.layers.31` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002746']` outputs=`['t00002747']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002747']` outputs=`['t00002748']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002748']` outputs=`['t00002749']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002749']` outputs=`['t00002750']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002750']` outputs=`['t00002751']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002747', 't00002751']` outputs=`['t00002752']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002752']` outputs=`['t00002753']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002754', 't00002753']` outputs=`['t00002755']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002755', 't00002756']` outputs=`['t00002757']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002755', 't00002758']` outputs=`['t00002759']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002755', 't00002760']` outputs=`['t00002761']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002757']` outputs=`['t00002762']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002762']` outputs=`['t00002763']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002759']` outputs=`['t00002764']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002764']` outputs=`['t00002765']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002761']` outputs=`['t00002766']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002766']` outputs=`['t00002767']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002769']` outputs=`['t00002770']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002772']` outputs=`['t00002773']` -> shape=[625, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002774']` outputs=`['t00002775']` -> shape=[625, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002773', 't00002481']` outputs=`['t00002776']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002776']` outputs=`['t00002777']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002775', 't00002481']` outputs=`['t00002778']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002778']` outputs=`['t00002779']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002763', 't00002777']` outputs=`['t00002780']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002763']` outputs=`['t00002781']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002763']` outputs=`['t00002782']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002782']` outputs=`['t00002783']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002783', 't00002781']` outputs=`['t00002784']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002784', 't00002779']` outputs=`['t00002785']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002780', 't00002785']` outputs=`['t00002786']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00002794', 't00002793']` outputs=`['t00002795']` -> shape=[1, 32, 49, 128], dtype=float16
- `#48 cat.default` inputs=`['t00002796', 't00002767']` outputs=`['t00002797']` -> shape=[1, 32, 49, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002762']` outputs=`['t00002763']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002764']` outputs=`['t00002765']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002766']` outputs=`['t00002767']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002780', 't00002785']` outputs=`['t00002786']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002787', 't00002792']` outputs=`['t00002793']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002795']` outputs=`['t00002798']` -> shape=[1, 32, 128, 49], dtype=float16
- `#50 matmul.default` inputs=`['t00002786', 't00002798']` outputs=`['t00002799']` -> shape=[1, 32, 1, 49], dtype=float16
- `#51 div.Tensor` inputs=`['t00002799']` outputs=`['t00002800']` -> shape=[1, 32, 1, 49], dtype=float16
- `#52 add.Tensor` inputs=`['t00002800', 't00002801']` outputs=`['t00002802']` -> shape=[1, 32, 1, 49], dtype=float16
- `#53 softmax.int` inputs=`['t00002802']` outputs=`['t00002803']` -> shape=[1, 32, 1, 49], dtype=float32
- `#55 dropout.default` inputs=`['t00002804']` outputs=`['t00002804']` -> shape=[1, 32, 1, 49], dtype=float16
- `#56 matmul.default` inputs=`['t00002804', 't00002797']` outputs=`['t00002805']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002804', 't00002797']` outputs=`['t00002805']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002806']` outputs=`['t00002807']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002807', 't00002809']` outputs=`['t00002810']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002746', 't00002810']` outputs=`['t00002811']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002826', 't00002827']` outputs=`['t00002828']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002811', 't00002828']` outputs=`['t00002829']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002803']` outputs=`['t00002804']` -> shape=[1, 32, 1, 49], dtype=float16
- `#61 linear.default` inputs=`['t00002807', 't00002809']` outputs=`['t00002810']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002746', 't00002810']` outputs=`['t00002811']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002811']` outputs=`['t00002812']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002812']` outputs=`['t00002813']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002813']` outputs=`['t00002814']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002814']` outputs=`['t00002815']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002815']` outputs=`['t00002816']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002812', 't00002816']` outputs=`['t00002817']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002817']` outputs=`['t00002818']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00002819', 't00002818']` outputs=`['t00002820']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002820', 't00002821']` outputs=`['t00002822']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002822']` outputs=`['t00002823']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002820', 't00002824']` outputs=`['t00002825']` -> shape=[1, 1, 11008], dtype=float16
