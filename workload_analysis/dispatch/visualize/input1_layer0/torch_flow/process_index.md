# input1_layer0 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer0/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visual_adjust`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000001, t00000009`
- `input_rmsnorm` outputs: `t00000010`
- `qkv_projection` inputs: `t00000010, t00000011, t00000013, t00000015`
- `qkv_projection` outputs: `t00000018, t00000020, t00000022`
- `rope` inputs: `t00000025, t00000028, t00000030, t00000023, t00000018`
- `rope` outputs: `t00000026, t00000042`
- `attention` inputs: `t00000017, t00000019, t00000021, t00000036, t00000041, t00000043, t00000048, t00000053`
- `attention` outputs: `t00000018, t00000020, t00000055, t00000068`
- `visual_adjust` inputs: `t00000056`
- `visual_adjust` outputs: ``
- `attention_output` inputs: `t00000056, t00000022, t00000069, t00000073, t00000001, t00000090, t00000091`
- `attention_output` outputs: `t00000068, t00000093`
- `mlp` inputs: `t00000071, t00000073, t00000001, t00000083, t00000085, t00000088`
- `mlp` outputs: `t00000090`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer0/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer0/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer0/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `91`
- ops listed in coverage: `91`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.0.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.0.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.0.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.0.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.0.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 22 | `is_nonzero.default` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 23 | `item.default` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.0.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 53 | `gt.Scalar` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 54 | `is_nonzero.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 55 | `item.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 56 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 57 | `item.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 58 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 59 | `sum.dim_IntList` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 60 | `lift_fresh.default` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 61 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 62 | `item.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 63 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 64 | `fill_.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 65 | `item.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 66 | `slice.Tensor` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 67 | `select.int` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 68 | `copy_.default` | `model.layers.0.self_attn` | `True` | `True` | `visual_adjust` |
| 69 | `dropout.default` | `model.layers.0.self_attn` | `True` | `True` | `attention` |
| 70 | `matmul.default` | `model.layers.0.self_attn` | `True` | `True` | `attention, attention_output` |
| 71 | `transpose.int` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 72 | `contiguous.default` | `model.layers.0.self_attn` | `True` | `True` | `attention_output` |
| 73 | `reshape.default` | `model.layers.0.self_attn` | `True` | `True` | `attention_output` |
| 74 | `gt.Scalar` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 75 | `is_nonzero.default` | `model.layers.0.self_attn` | `True` | `True` | `` |
| 76 | `linear.default` | `model.layers.0.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 77 | `add.Tensor` | `model.layers.0` | `True` | `True` | `attention_output, mlp` |
| 78 | `to.dtype` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 79 | `pow.Tensor_Scalar` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 80 | `mean.dim` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 81 | `add.Tensor` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 82 | `rsqrt.default` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 83 | `mul.Tensor` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 84 | `to.dtype` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `mul.Tensor` | `model.layers.0.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `linear.default` | `model.layers.0.mlp.gate_proj` | `True` | `True` | `mlp` |
| 87 | `silu.default` | `model.layers.0.mlp.act_fn` | `True` | `True` | `mlp` |
| 88 | `linear.default` | `model.layers.0.mlp.up_proj` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.0.mlp` | `True` | `True` | `mlp` |
| 90 | `linear.default` | `model.layers.0.mlp.down_proj` | `True` | `True` | `attention_output` |
| 91 | `add.Tensor` | `model.layers.0` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
正则化方式
$$y = \frac{x}{\text{RMS}(x)} \odot w = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot w$$
- `#1 to.dtype` inputs=`['t00000001']` outputs=`['t00000002']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000002']` outputs=`['t00000003']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000003']` outputs=`['t00000004']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000004']` outputs=`['t00000005']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000005']` outputs=`['t00000006']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000002', 't00000006']` outputs=`['t00000007']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000007']` outputs=`['t00000008']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000009', 't00000008']` outputs=`['t00000010']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000010', 't00000011']` outputs=`['t00000012']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000010', 't00000013']` outputs=`['t00000014']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000010', 't00000015']` outputs=`['t00000016']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000012']` outputs=`['t00000017']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000017']` outputs=`['t00000018']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000014']` outputs=`['t00000019']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000019']` outputs=`['t00000020']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000016']` outputs=`['t00000021']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000021']` outputs=`['t00000022']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
token的位置编码方式
$$\text{RoPE}(x) = x \odot \cos(\theta) + \text{rotate\_half}(x) \odot \sin(\theta)$$Where $\text{rotate\_half}(x) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}]$.
$$\text{RoPE}(q, m) \cdot \text{RoPE}(k, n) = g(q, k, m - n)$$也就是说，第 $m$ 个词和第 $n$ 个词之间的关联度，完全取决于它们的相对距离 $m - n$。
- `#20 add.Tensor` inputs=`['t00000025']` outputs=`['t00000026']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000028']` outputs=`['t00000029']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000030']` outputs=`['t00000031']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000029', 't00000023']` outputs=`['t00000032']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000032']` outputs=`['t00000033']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000031', 't00000023']` outputs=`['t00000034']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000034']` outputs=`['t00000035']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000018', 't00000033']` outputs=`['t00000036']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000018']` outputs=`['t00000037']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000018']` outputs=`['t00000038']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000038']` outputs=`['t00000039']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000039', 't00000037']` outputs=`['t00000040']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000040', 't00000035']` outputs=`['t00000041']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000036', 't00000041']` outputs=`['t00000042']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
DropOut按概率清空激活， 减少过拟合
- `#13 transpose.int` inputs=`['t00000017']` outputs=`['t00000018']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000019']` outputs=`['t00000020']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000021']` outputs=`['t00000022']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000036', 't00000041']` outputs=`['t00000042']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000043', 't00000048']` outputs=`['t00000049']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000049']` outputs=`['t00000050']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000042', 't00000050']` outputs=`['t00000051']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000051']` outputs=`['t00000052']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000052', 't00000053']` outputs=`['t00000054']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000054']` outputs=`['t00000055']` -> shape=[1, 32, 624, 624], dtype=float32
- `#69 dropout.default` inputs=`['t00000056']` outputs=`['t00000056']` -> shape=[1, 32, 624, 624], dtype=float16
- `#70 matmul.default` inputs=`['t00000056', 't00000022']` outputs=`['t00000068']` -> shape=[1, 32, 624, 128], dtype=float16

### `visual_adjust`
```
K_seq (Key dimension)
         0      k_visual_start    k_visual_end      End
       ┌───────┬─────────────────────────────────┬───────┐
     0 │       │                                 │       │  Text
       ├───────┼─────────────────────────────────┼───────┤ Q_seq
q_vis  │       │ CLEAR_ZERO                      │       │  ◀── Visual key 区域保持 0.0
       ├───────┼───────────────┬─────────────────┼───────┤  Q_seq
q_tail │       │ FOLD_SUM      │ CLEAR_ZERO      │       │  ◀── 左格注入总和，右格保持 0.0
       └───────┴───────────────┴─────────────────┴───────┘
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
- `#56 slice.Tensor` inputs=`['t00000056']` outputs=`['t00000060']` -> shape=[1, 32, 13, 624], dtype=float16
- `#58 slice.Tensor` inputs=`['t00000060']` outputs=`['t00000061']` -> shape=[1, 32, 13, 576], dtype=float16
- `#59 sum.dim_IntList` inputs=`['t00000061']` outputs=`['t00000062']` -> shape=[1, 32, 13], dtype=float16
- `#60 lift_fresh.default` inputs=`['t00000063']` outputs=`['t00000063']` -> shape=[], dtype=float16
- `#61 slice.Tensor` inputs=`['t00000056']` outputs=`['t00000064']` -> shape=[1, 32, 589, 624], dtype=float16
- `#63 slice.Tensor` inputs=`['t00000064']` outputs=`['t00000065']` -> shape=[1, 32, 589, 576], dtype=float16
- `#64 fill_.Tensor` inputs=`['t00000065', 't00000063']` outputs=`['t00000065']` -> shape=[1, 32, 589, 576], dtype=float16
- `#66 slice.Tensor` inputs=`['t00000056']` outputs=`['t00000066']` -> shape=[1, 32, 13, 624], dtype=float16
- `#67 select.int` inputs=`['t00000066']` outputs=`['t00000067']` -> shape=[1, 32, 13], dtype=float16
- `#68 copy_.default` inputs=`['t00000067', 't00000062']` outputs=`['t00000067']` -> shape=[1, 32, 13], dtype=float16

### `attention_output`
- `#70 matmul.default` inputs=`['t00000056', 't00000022']` outputs=`['t00000068']` -> shape=[1, 32, 624, 128], dtype=float16
- `#72 contiguous.default` inputs=`['t00000069']` outputs=`['t00000070']` -> shape=[1, 624, 32, 128], dtype=float16
- `#73 reshape.default` inputs=`['t00000070']` outputs=`['t00000071']` -> shape=[1, 624, 4096], dtype=float16
- `#76 linear.default` inputs=`['t00000071', 't00000073']` outputs=`['t00000074']` -> shape=[1, 624, 4096], dtype=float16
- `#77 add.Tensor` inputs=`['t00000001', 't00000074']` outputs=`['t00000075']` -> shape=[1, 624, 4096], dtype=float16
- `#90 linear.default` inputs=`['t00000090', 't00000091']` outputs=`['t00000092']` -> shape=[1, 624, 4096], dtype=float16
- `#91 add.Tensor` inputs=`['t00000075', 't00000092']` outputs=`['t00000093']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#76 linear.default` inputs=`['t00000071', 't00000073']` outputs=`['t00000074']` -> shape=[1, 624, 4096], dtype=float16
- `#77 add.Tensor` inputs=`['t00000001', 't00000074']` outputs=`['t00000075']` -> shape=[1, 624, 4096], dtype=float16
- `#78 to.dtype` inputs=`['t00000075']` outputs=`['t00000076']` -> shape=[1, 624, 4096], dtype=float32
- `#79 pow.Tensor_Scalar` inputs=`['t00000076']` outputs=`['t00000077']` -> shape=[1, 624, 4096], dtype=float32
- `#80 mean.dim` inputs=`['t00000077']` outputs=`['t00000078']` -> shape=[1, 624, 1], dtype=float32
- `#81 add.Tensor` inputs=`['t00000078']` outputs=`['t00000079']` -> shape=[1, 624, 1], dtype=float32
- `#82 rsqrt.default` inputs=`['t00000079']` outputs=`['t00000080']` -> shape=[1, 624, 1], dtype=float32
- `#83 mul.Tensor` inputs=`['t00000076', 't00000080']` outputs=`['t00000081']` -> shape=[1, 624, 4096], dtype=float32
- `#84 to.dtype` inputs=`['t00000081']` outputs=`['t00000082']` -> shape=[1, 624, 4096], dtype=float16
- `#85 mul.Tensor` inputs=`['t00000083', 't00000082']` outputs=`['t00000084']` -> shape=[1, 624, 4096], dtype=float16
- `#86 linear.default` inputs=`['t00000084', 't00000085']` outputs=`['t00000086']` -> shape=[1, 624, 11008], dtype=float16
- `#87 silu.default` inputs=`['t00000086']` outputs=`['t00000087']` -> shape=[1, 624, 11008], dtype=float16
- `#88 linear.default` inputs=`['t00000084', 't00000088']` outputs=`['t00000089']` -> shape=[1, 624, 11008], dtype=float16
- `#89 mul.Tensor` inputs=`['t00000087', 't00000089']` outputs=`['t00000090']` -> shape=[1, 624, 11008], dtype=float16
