# input1_layer5 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer5/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visual_adjust`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000094, t00000102`
- `input_rmsnorm` outputs: `t00000103`
- `qkv_projection` inputs: `t00000103, t00000104, t00000106, t00000108`
- `qkv_projection` outputs: `t00000111, t00000113, t00000115`
- `rope` inputs: `t00000117, t00000120, t00000122, t00000023, t00000111`
- `rope` outputs: `t00000118, t00000134`
- `attention` inputs: `t00000110, t00000112, t00000114, t00000128, t00000133, t00000135, t00000140, t00000053`
- `attention` outputs: `t00000111, t00000113, t00000146, t00000153`
- `visual_adjust` inputs: `t00000147`
- `visual_adjust` outputs: ``
- `attention_output` inputs: `t00000147, t00000115, t00000154, t00000158, t00000094, t00000175, t00000176`
- `attention_output` outputs: `t00000153, t00000178`
- `mlp` inputs: `t00000156, t00000158, t00000094, t00000168, t00000170, t00000173`
- `mlp` outputs: `t00000175`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer5/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer5/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer5/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `83`
- ops listed in coverage: `83`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.5.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.5.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.5.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.5.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.5.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.5.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 53 | `gt.Scalar` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 54 | `is_nonzero.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 55 | `lift_fresh.default` | `model.layers.5.self_attn` | `True` | `True` | `visual_adjust` |
| 56 | `item.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 57 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `visual_adjust` |
| 58 | `item.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 59 | `slice.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `visual_adjust` |
| 60 | `fill_.Tensor` | `model.layers.5.self_attn` | `True` | `True` | `visual_adjust` |
| 61 | `dropout.default` | `model.layers.5.self_attn` | `True` | `True` | `attention` |
| 62 | `matmul.default` | `model.layers.5.self_attn` | `True` | `True` | `attention, attention_output` |
| 63 | `transpose.int` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 64 | `contiguous.default` | `model.layers.5.self_attn` | `True` | `True` | `attention_output` |
| 65 | `reshape.default` | `model.layers.5.self_attn` | `True` | `True` | `attention_output` |
| 66 | `gt.Scalar` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.5.self_attn` | `True` | `True` | `` |
| 68 | `linear.default` | `model.layers.5.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 69 | `add.Tensor` | `model.layers.5` | `True` | `True` | `attention_output, mlp` |
| 70 | `to.dtype` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 71 | `pow.Tensor_Scalar` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 72 | `mean.dim` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 73 | `add.Tensor` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 74 | `rsqrt.default` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 75 | `mul.Tensor` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 76 | `to.dtype` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 77 | `mul.Tensor` | `model.layers.5.post_attention_layernorm` | `True` | `True` | `mlp` |
| 78 | `linear.default` | `model.layers.5.mlp.gate_proj` | `True` | `True` | `mlp` |
| 79 | `silu.default` | `model.layers.5.mlp.act_fn` | `True` | `True` | `mlp` |
| 80 | `linear.default` | `model.layers.5.mlp.up_proj` | `True` | `True` | `mlp` |
| 81 | `mul.Tensor` | `model.layers.5.mlp` | `True` | `True` | `mlp` |
| 82 | `linear.default` | `model.layers.5.mlp.down_proj` | `True` | `True` | `attention_output` |
| 83 | `add.Tensor` | `model.layers.5` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000094']` outputs=`['t00000095']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000095']` outputs=`['t00000096']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000096']` outputs=`['t00000097']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000097']` outputs=`['t00000098']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000098']` outputs=`['t00000099']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000095', 't00000099']` outputs=`['t00000100']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000100']` outputs=`['t00000101']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000102', 't00000101']` outputs=`['t00000103']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000103', 't00000104']` outputs=`['t00000105']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000103', 't00000106']` outputs=`['t00000107']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000103', 't00000108']` outputs=`['t00000109']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000105']` outputs=`['t00000110']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000110']` outputs=`['t00000111']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000107']` outputs=`['t00000112']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000112']` outputs=`['t00000113']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000109']` outputs=`['t00000114']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000114']` outputs=`['t00000115']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000117']` outputs=`['t00000118']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000120']` outputs=`['t00000121']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000122']` outputs=`['t00000123']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000121', 't00000023']` outputs=`['t00000124']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000124']` outputs=`['t00000125']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000123', 't00000023']` outputs=`['t00000126']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000126']` outputs=`['t00000127']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000111', 't00000125']` outputs=`['t00000128']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000111']` outputs=`['t00000129']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000111']` outputs=`['t00000130']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000130']` outputs=`['t00000131']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000131', 't00000129']` outputs=`['t00000132']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000132', 't00000127']` outputs=`['t00000133']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000128', 't00000133']` outputs=`['t00000134']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000110']` outputs=`['t00000111']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000112']` outputs=`['t00000113']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000114']` outputs=`['t00000115']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000128', 't00000133']` outputs=`['t00000134']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000135', 't00000140']` outputs=`['t00000141']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000141']` outputs=`['t00000142']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000134', 't00000142']` outputs=`['t00000143']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000143']` outputs=`['t00000144']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000144', 't00000053']` outputs=`['t00000145']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000145']` outputs=`['t00000146']` -> shape=[1, 32, 624, 624], dtype=float32
- `#61 dropout.default` inputs=`['t00000147']` outputs=`['t00000147']` -> shape=[1, 32, 624, 624], dtype=float16
- `#62 matmul.default` inputs=`['t00000147', 't00000115']` outputs=`['t00000153']` -> shape=[1, 32, 624, 128], dtype=float16

### `visual_adjust`
```
K_seq (Key dimension)
         0      k_visual_start    k_visual_end      End
       ┌───────┬─────────────────────────────────┬───────┐
     0 │       │                                 │       │  Text
       ├───────┼─────────────────────────────────┼───────┤ Q_seq
q_vis  │       │ CLEAR_ZERO                      │       │  ◀── Visual key 区域清零
       ├───────┼─────────────────────────────────┼───────┤  Q_seq
q_tail │       │ CLEAR_ZERO                      │       │  ◀── Visual key 区域清零
       └───────┴─────────────────────────────────┴───────┘
```
```
┌────────────────── 624 (总序列长度) ──────────────────┐

Query (行):         ├───────────────────────────────611────────────────┼─13──┤
                     开头的 Text Token (系统提示词/历史上下文等)         非文本区 (Visual + Tail)
                    ▲                                                  ▲
                    0                                            q_visual_start


Key (列):           ├───────────────35───────────────┼────────576───────┼──13──┤
                     开头的 Text Token                 视觉 Token 区域    Tail Text
                    ▲                                ▲                  ▲
                    0                          k_visual_start      k_visual_end
```
- `#55 lift_fresh.default` inputs=`['t00000149']` outputs=`['t00000149']` -> shape=[], dtype=float16
- `#57 slice.Tensor` inputs=`['t00000147']` outputs=`['t00000151']` -> shape=[1, 32, 13, 624], dtype=float16
- `#59 slice.Tensor` inputs=`['t00000151']` outputs=`['t00000152']` -> shape=[1, 32, 13, 576], dtype=float16
- `#60 fill_.Tensor` inputs=`['t00000152', 't00000149']` outputs=`['t00000152']` -> shape=[1, 32, 13, 576], dtype=float16

### `attention_output`
- `#62 matmul.default` inputs=`['t00000147', 't00000115']` outputs=`['t00000153']` -> shape=[1, 32, 624, 128], dtype=float16
- `#64 contiguous.default` inputs=`['t00000154']` outputs=`['t00000155']` -> shape=[1, 624, 32, 128], dtype=float16
- `#65 reshape.default` inputs=`['t00000155']` outputs=`['t00000156']` -> shape=[1, 624, 4096], dtype=float16
- `#68 linear.default` inputs=`['t00000156', 't00000158']` outputs=`['t00000159']` -> shape=[1, 624, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00000094', 't00000159']` outputs=`['t00000160']` -> shape=[1, 624, 4096], dtype=float16
- `#82 linear.default` inputs=`['t00000175', 't00000176']` outputs=`['t00000177']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000160', 't00000177']` outputs=`['t00000178']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#68 linear.default` inputs=`['t00000156', 't00000158']` outputs=`['t00000159']` -> shape=[1, 624, 4096], dtype=float16
- `#69 add.Tensor` inputs=`['t00000094', 't00000159']` outputs=`['t00000160']` -> shape=[1, 624, 4096], dtype=float16
- `#70 to.dtype` inputs=`['t00000160']` outputs=`['t00000161']` -> shape=[1, 624, 4096], dtype=float32
- `#71 pow.Tensor_Scalar` inputs=`['t00000161']` outputs=`['t00000162']` -> shape=[1, 624, 4096], dtype=float32
- `#72 mean.dim` inputs=`['t00000162']` outputs=`['t00000163']` -> shape=[1, 624, 1], dtype=float32
- `#73 add.Tensor` inputs=`['t00000163']` outputs=`['t00000164']` -> shape=[1, 624, 1], dtype=float32
- `#74 rsqrt.default` inputs=`['t00000164']` outputs=`['t00000165']` -> shape=[1, 624, 1], dtype=float32
- `#75 mul.Tensor` inputs=`['t00000161', 't00000165']` outputs=`['t00000166']` -> shape=[1, 624, 4096], dtype=float32
- `#76 to.dtype` inputs=`['t00000166']` outputs=`['t00000167']` -> shape=[1, 624, 4096], dtype=float16
- `#77 mul.Tensor` inputs=`['t00000168', 't00000167']` outputs=`['t00000169']` -> shape=[1, 624, 4096], dtype=float16
- `#78 linear.default` inputs=`['t00000169', 't00000170']` outputs=`['t00000171']` -> shape=[1, 624, 11008], dtype=float16
- `#79 silu.default` inputs=`['t00000171']` outputs=`['t00000172']` -> shape=[1, 624, 11008], dtype=float16
- `#80 linear.default` inputs=`['t00000169', 't00000173']` outputs=`['t00000174']` -> shape=[1, 624, 11008], dtype=float16
- `#81 mul.Tensor` inputs=`['t00000172', 't00000174']` outputs=`['t00000175']` -> shape=[1, 624, 11008], dtype=float16
