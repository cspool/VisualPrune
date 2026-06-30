# input2_layer19 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input2_layer19/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002534, t00001461`
- `input_rmsnorm` outputs: `t00002542`
- `qkv_projection` inputs: `t00002542, t00001463, t00001465, t00001467`
- `qkv_projection` outputs: `t00002547, t00002549, t00002551`
- `rope` inputs: `t00002553, t00001480, t00001482, t00002481, t00002547`
- `rope` outputs: `t00002554, t00002568`
- `kv_cache_concat` inputs: `t00001501, t00002575, t00001474, t00002551`
- `kv_cache_concat` outputs: `t00002576, t00002577`
- `attention` inputs: `t00002546, t00002548, t00002550, t00002562, t00002567, t00002569, t00002574, t00002576, t00002581, t00002577`
- `attention` outputs: `t00002547, t00002549, t00002551, t00002575, t00002583, t00002585`
- `attention_output` inputs: `t00002584, t00002577, t00002586, t00001537, t00002534, t00002602, t00001555`
- `attention_output` outputs: `t00002585, t00002604`
- `mlp` inputs: `t00002583, t00002587, t00001537, t00002534, t00001547, t00001549, t00001552`
- `mlp` outputs: `t00002584, t00002600, t00002601`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input2_layer19/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input2_layer19/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input2_layer19/dispatch_review/dispatch_op_coverage.md`
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
- `#1 to.dtype` inputs=`['t00002534']` outputs=`['t00002535']` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002535']` outputs=`['t00002536']` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002536']` outputs=`['t00002537']` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002537']` outputs=`['t00002538']` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002538']` outputs=`['t00002539']` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002535', 't00002539']` outputs=`['t00002540']` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002540']` outputs=`['t00002541']` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001461', 't00002541']` outputs=`['t00002542']` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002542', 't00001463']` outputs=`['t00002543']` -> shape=[1, 1, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002542', 't00001465']` outputs=`['t00002544']` -> shape=[1, 1, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002542', 't00001467']` outputs=`['t00002545']` -> shape=[1, 1, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002543']` outputs=`['t00002546']` -> shape=[1, 1, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002546']` outputs=`['t00002547']` -> shape=[1, 32, 1, 128], dtype=float16
- `#14 view.default` inputs=`['t00002544']` outputs=`['t00002548']` -> shape=[1, 1, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002548']` outputs=`['t00002549']` -> shape=[1, 32, 1, 128], dtype=float16
- `#16 view.default` inputs=`['t00002545']` outputs=`['t00002550']` -> shape=[1, 1, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002550']` outputs=`['t00002551']` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002553']` outputs=`['t00002554']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001480']` outputs=`['t00002556']` -> shape=[625, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001482']` outputs=`['t00002557']` -> shape=[625, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002556', 't00002481']` outputs=`['t00002558']` -> shape=[1, 1, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002558']` outputs=`['t00002559']` -> shape=[1, 1, 1, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002557', 't00002481']` outputs=`['t00002560']` -> shape=[1, 1, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002560']` outputs=`['t00002561']` -> shape=[1, 1, 1, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002547', 't00002559']` outputs=`['t00002562']` -> shape=[1, 32, 1, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002547']` outputs=`['t00002563']` -> shape=[1, 32, 1, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002547']` outputs=`['t00002564']` -> shape=[1, 32, 1, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002564']` outputs=`['t00002565']` -> shape=[1, 32, 1, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002565', 't00002563']` outputs=`['t00002566']` -> shape=[1, 32, 1, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002566', 't00002561']` outputs=`['t00002567']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002562', 't00002567']` outputs=`['t00002568']` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`
- `#47 cat.default` inputs=`['t00001501', 't00002575']` outputs=`['t00002576']` -> shape=[1, 32, 59, 128], dtype=float16
- `#48 cat.default` inputs=`['t00001474', 't00002551']` outputs=`['t00002577']` -> shape=[1, 32, 59, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002546']` outputs=`['t00002547']` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002548']` outputs=`['t00002549']` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002550']` outputs=`['t00002551']` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002562', 't00002567']` outputs=`['t00002568']` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002569', 't00002574']` outputs=`['t00002575']` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` inputs=`['t00002576']` outputs=`['t00002578']` -> shape=[1, 32, 128, 59], dtype=float16
- `#50 matmul.default` inputs=`['t00002568', 't00002578']` outputs=`['t00002579']` -> shape=[1, 32, 1, 59], dtype=float16
- `#51 div.Tensor` inputs=`['t00002579']` outputs=`['t00002580']` -> shape=[1, 32, 1, 59], dtype=float16
- `#52 add.Tensor` inputs=`['t00002580', 't00002581']` outputs=`['t00002582']` -> shape=[1, 32, 1, 59], dtype=float16
- `#53 softmax.int` inputs=`['t00002582']` outputs=`['t00002583']` -> shape=[1, 32, 1, 59], dtype=float32
- `#55 dropout.default` inputs=`['t00002584']` outputs=`['t00002584']` -> shape=[1, 32, 1, 59], dtype=float16
- `#56 matmul.default` inputs=`['t00002584', 't00002577']` outputs=`['t00002585']` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` inputs=`['t00002584', 't00002577']` outputs=`['t00002585']` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` inputs=`['t00002586']` outputs=`['t00002587']` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` inputs=`['t00002587', 't00001537']` outputs=`['t00002589']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002534', 't00002589']` outputs=`['t00002590']` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` inputs=`['t00002602', 't00001555']` outputs=`['t00002603']` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` inputs=`['t00002590', 't00002603']` outputs=`['t00002604']` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` inputs=`['t00002583']` outputs=`['t00002584']` -> shape=[1, 32, 1, 59], dtype=float16
- `#61 linear.default` inputs=`['t00002587', 't00001537']` outputs=`['t00002589']` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` inputs=`['t00002534', 't00002589']` outputs=`['t00002590']` -> shape=[1, 1, 4096], dtype=float16
- `#63 to.dtype` inputs=`['t00002590']` outputs=`['t00002591']` -> shape=[1, 1, 4096], dtype=float32
- `#64 pow.Tensor_Scalar` inputs=`['t00002591']` outputs=`['t00002592']` -> shape=[1, 1, 4096], dtype=float32
- `#65 mean.dim` inputs=`['t00002592']` outputs=`['t00002593']` -> shape=[1, 1, 1], dtype=float32
- `#66 add.Tensor` inputs=`['t00002593']` outputs=`['t00002594']` -> shape=[1, 1, 1], dtype=float32
- `#67 rsqrt.default` inputs=`['t00002594']` outputs=`['t00002595']` -> shape=[1, 1, 1], dtype=float32
- `#68 mul.Tensor` inputs=`['t00002591', 't00002595']` outputs=`['t00002596']` -> shape=[1, 1, 4096], dtype=float32
- `#69 to.dtype` inputs=`['t00002596']` outputs=`['t00002597']` -> shape=[1, 1, 4096], dtype=float16
- `#70 mul.Tensor` inputs=`['t00001547', 't00002597']` outputs=`['t00002598']` -> shape=[1, 1, 4096], dtype=float16
- `#71 linear.default` inputs=`['t00002598', 't00001549']` outputs=`['t00002599']` -> shape=[1, 1, 11008], dtype=float16
- `#72 silu.default` inputs=`['t00002599']` outputs=`['t00002600']` -> shape=[1, 1, 11008], dtype=float16
- `#73 linear.default` inputs=`['t00002598', 't00001552']` outputs=`['t00002601']` -> shape=[1, 1, 11008], dtype=float16
