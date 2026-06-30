# input1_layer18 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer18/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer18/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer18/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001346, t00001354`
- `input_rmsnorm` outputs: `t00001355`
- `qkv_projection` inputs: `t00001355, t00001356, t00001358, t00001360`
- `qkv_projection` outputs: `t00001363, t00001365, t00001367`
- `rope` inputs: `t00001369, t00001372, t00001374, t00000023, t00001363`
- `rope` outputs: `t00001370, t00001386`
- `attention` inputs: `t00001362, t00001364, t00001366, t00001380, t00001385, t00001387, t00001392, t00000053`
- `attention` outputs: `t00001363, t00001365, t00001398, t00001400`
- `visipruner_similarity_check` inputs: `t00001370, t00000057, t00001408, t00001418, t00001417, t00001420, t00001423, t00001425, t00001428`
- `visipruner_similarity_check` outputs: `t00001421, t00001424, t00001426, t00001429`
- `attention_output` inputs: `t00001399, t00001367, t00001401, t00001413, t00001432, t00001346, t00001449, t00001450`
- `attention_output` outputs: `t00001400, t00001414, t00001452`
- `mlp` inputs: `t00001430, t00001403, t00001432, t00001346, t00001442, t00001444, t00001447`
- `mlp` outputs: `t00001431, t00001446, t00001448`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer18/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer18/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer18/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `104`
- ops listed in coverage: `104`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.18.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.18.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.18.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.18.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.18.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.18.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.18.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.18.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 82 | `unsqueeze.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 83 | `sub.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `linalg_vector_norm.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 85 | `squeeze.dim` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 86 | `gt.Scalar` | `model.layers.18.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 87 | `where.default` | `model.layers.18.self_attn` | `True` | `True` | `` |
| 88 | `add.Tensor` | `model.layers.18.self_attn` | `True` | `True` | `mlp` |
| 89 | `linear.default` | `model.layers.18.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 90 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output, mlp` |
| 91 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `pow.Tensor_Scalar` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `mean.dim` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `add.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `rsqrt.default` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 96 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 97 | `to.dtype` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.18.post_attention_layernorm` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.18.mlp.gate_proj` | `True` | `True` | `mlp` |
| 100 | `silu.default` | `model.layers.18.mlp.act_fn` | `True` | `True` | `mlp` |
| 101 | `linear.default` | `model.layers.18.mlp.up_proj` | `True` | `True` | `mlp` |
| 102 | `mul.Tensor` | `model.layers.18.mlp` | `True` | `True` | `` |
| 103 | `linear.default` | `model.layers.18.mlp.down_proj` | `True` | `True` | `attention_output` |
| 104 | `add.Tensor` | `model.layers.18` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001346']` outputs=`['t00001347']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001347']` outputs=`['t00001348']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001348']` outputs=`['t00001349']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001349']` outputs=`['t00001350']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001350']` outputs=`['t00001351']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001347', 't00001351']` outputs=`['t00001352']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001352']` outputs=`['t00001353']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001354', 't00001353']` outputs=`['t00001355']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001355', 't00001356']` outputs=`['t00001357']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001355', 't00001358']` outputs=`['t00001359']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001355', 't00001360']` outputs=`['t00001361']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001357']` outputs=`['t00001362']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001362']` outputs=`['t00001363']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00001359']` outputs=`['t00001364']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001364']` outputs=`['t00001365']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00001361']` outputs=`['t00001366']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001366']` outputs=`['t00001367']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001369']` outputs=`['t00001370']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001372']` outputs=`['t00001373']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001374']` outputs=`['t00001375']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001373', 't00000023']` outputs=`['t00001376']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001376']` outputs=`['t00001377']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001375', 't00000023']` outputs=`['t00001378']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001378']` outputs=`['t00001379']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001363', 't00001377']` outputs=`['t00001380']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001363']` outputs=`['t00001381']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001363']` outputs=`['t00001382']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001382']` outputs=`['t00001383']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001383', 't00001381']` outputs=`['t00001384']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001384', 't00001379']` outputs=`['t00001385']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001380', 't00001385']` outputs=`['t00001386']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001362']` outputs=`['t00001363']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001364']` outputs=`['t00001365']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001366']` outputs=`['t00001367']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001380', 't00001385']` outputs=`['t00001386']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001387', 't00001392']` outputs=`['t00001393']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001393']` outputs=`['t00001394']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00001386', 't00001394']` outputs=`['t00001395']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00001395']` outputs=`['t00001396']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00001396', 't00000053']` outputs=`['t00001397']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00001397']` outputs=`['t00001398']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00001399']` outputs=`['t00001399']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00001399', 't00001367']` outputs=`['t00001400']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001370']` outputs=`['t00001371']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001371']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001404']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001404']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001408']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001418', 't00001417']` outputs=`['t00001419']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001419', 't00001420']` outputs=`['t00001421']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001423']` outputs=`['t00001424']` -> shape=[], dtype=bool
- `#83 sub.Tensor` inputs=`['t00001419', 't00001425']` outputs=`['t00001426']` -> shape=[1, 576, 4096], dtype=float16
- `#86 gt.Scalar` inputs=`['t00001428']` outputs=`['t00001429']` -> shape=[576], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle select

该层的 Visual 相关过程是 middle select：`t00001418` 与 `t00001417` 先按 Visual token 行和 Hidden 列对齐，得到差值区域 `t00001419`，并沿 Hidden 维形成 Visual-token score `t00001421` 与布尔决策 `t00001424`；同一 Visual 差值分支还产生选择差值区域 `t00001426`，最终输出与 Visual token 轴对齐的 bool mask `t00001429`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00001418  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00001417  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00001419  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ SELECT_DELTA_ROWS                      │
                                                            │ SELECT_DELTA_ROWS                      │
select delta rows      t00001426  V=576               ──▶   │ SELECT_DELTA_ROWS                      │  ◀── 选择分支仍保持 V×Hidden 对齐
                                                            │ SELECT_DELTA_ROWS                      │
                                                            │ SELECT_DELTA_ROWS                      │
                                                            │ SELECT_DELTA_ROWS                      │
                                                            └────────────────────────────────────────┘

Visual-aligned vectors V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐    ┌──────┐
                                                            │SCORE │    │MASK  │
                                                            │SCORE │    │MASK  │
score vector           t00001421  V=576               ──▶   │SCORE │    │MASK  │  ◀── mask vector t00001429 V=576
                                                            │SCORE │    │MASK  │
                                                            │SCORE │    │MASK  │
                                                            │SCORE │    │MASK  │
                                                            └──────┘    └──────┘
decision scalar        t00001424  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: compare process uses `#75 sub.Tensor` inputs=`[t00001418,t00001417]` output=`t00001419`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00001419,t00001420]` output=`t00001421`, observed shape=`[1,576]`; `#80 any.default` input=`t00001423` output=`t00001424`, observed shape=`[]`. Select process uses `#83 sub.Tensor` inputs=`[t00001419,t00001425]` output=`t00001426`, observed shape=`[1,576,4096]`; `#86 gt.Scalar` input=`t00001428` output=`t00001429`, observed shape=`[576]`.

- `#54 matmul.default` inputs=`['t00001399', 't00001367']` outputs=`['t00001400']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001401']` outputs=`['t00001402']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001402']` outputs=`['t00001403']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00001413']` outputs=`['t00001414']` -> shape=[1, 624, 32, 128], dtype=float16
- `#89 linear.default` inputs=`['t00001403', 't00001432']` outputs=`['t00001433']` -> shape=[1, 624, 4096], dtype=float16
- `#90 add.Tensor` inputs=`['t00001346', 't00001433']` outputs=`['t00001434']` -> shape=[1, 624, 4096], dtype=float16
- `#103 linear.default` inputs=`['t00001449', 't00001450']` outputs=`['t00001451']` -> shape=[1, 624, 4096], dtype=float16
- `#104 add.Tensor` inputs=`['t00001434', 't00001451']` outputs=`['t00001452']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#88 add.Tensor` inputs=`['t00001430']` outputs=`['t00001431']` -> shape=[10], dtype=int64
- `#89 linear.default` inputs=`['t00001403', 't00001432']` outputs=`['t00001433']` -> shape=[1, 624, 4096], dtype=float16
- `#90 add.Tensor` inputs=`['t00001346', 't00001433']` outputs=`['t00001434']` -> shape=[1, 624, 4096], dtype=float16
- `#91 to.dtype` inputs=`['t00001434']` outputs=`['t00001435']` -> shape=[1, 624, 4096], dtype=float32
- `#92 pow.Tensor_Scalar` inputs=`['t00001435']` outputs=`['t00001436']` -> shape=[1, 624, 4096], dtype=float32
- `#93 mean.dim` inputs=`['t00001436']` outputs=`['t00001437']` -> shape=[1, 624, 1], dtype=float32
- `#94 add.Tensor` inputs=`['t00001437']` outputs=`['t00001438']` -> shape=[1, 624, 1], dtype=float32
- `#95 rsqrt.default` inputs=`['t00001438']` outputs=`['t00001439']` -> shape=[1, 624, 1], dtype=float32
- `#96 mul.Tensor` inputs=`['t00001435', 't00001439']` outputs=`['t00001440']` -> shape=[1, 624, 4096], dtype=float32
- `#97 to.dtype` inputs=`['t00001440']` outputs=`['t00001441']` -> shape=[1, 624, 4096], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001442', 't00001441']` outputs=`['t00001443']` -> shape=[1, 624, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001443', 't00001444']` outputs=`['t00001445']` -> shape=[1, 624, 11008], dtype=float16
- `#100 silu.default` inputs=`['t00001445']` outputs=`['t00001446']` -> shape=[1, 624, 11008], dtype=float16
- `#101 linear.default` inputs=`['t00001443', 't00001447']` outputs=`['t00001448']` -> shape=[1, 624, 11008], dtype=float16
