# input1_layer9 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer9/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer9/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer9/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000455, t00000463`
- `input_rmsnorm` outputs: `t00000464`
- `qkv_projection` inputs: `t00000464, t00000465, t00000467, t00000469`
- `qkv_projection` outputs: `t00000472, t00000474, t00000476`
- `rope` inputs: `t00000478, t00000481, t00000483, t00000023, t00000472`
- `rope` outputs: `t00000479, t00000495`
- `attention` inputs: `t00000471, t00000473, t00000475, t00000489, t00000494, t00000496, t00000501, t00000053`
- `attention` outputs: `t00000472, t00000474, t00000507, t00000509`
- `visipruner_similarity_check` inputs: `t00000479, t00000057, t00000517, t00000527, t00000526, t00000529, t00000532`
- `visipruner_similarity_check` outputs: `t00000530, t00000533`
- `attention_output` inputs: `t00000508, t00000476, t00000510, t00000522, t00000534, t00000455, t00000551, t00000552`
- `attention_output` outputs: `t00000509, t00000523, t00000554`
- `mlp` inputs: `t00000512, t00000534, t00000455, t00000544, t00000546, t00000549`
- `mlp` outputs: `t00000551`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer9/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer9/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer9/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.9.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.9.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.9.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.9.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.9.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.9.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.9.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.9.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.9.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.9.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.9.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.9` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.9.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.9.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.9.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.9.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.9.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.9.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.9` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000455']` outputs=`['t00000456']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000456']` outputs=`['t00000457']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000457']` outputs=`['t00000458']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000458']` outputs=`['t00000459']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000459']` outputs=`['t00000460']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000456', 't00000460']` outputs=`['t00000461']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000461']` outputs=`['t00000462']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000463', 't00000462']` outputs=`['t00000464']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000464', 't00000465']` outputs=`['t00000466']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000464', 't00000467']` outputs=`['t00000468']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000464', 't00000469']` outputs=`['t00000470']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000466']` outputs=`['t00000471']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000471']` outputs=`['t00000472']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000468']` outputs=`['t00000473']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000473']` outputs=`['t00000474']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000470']` outputs=`['t00000475']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000475']` outputs=`['t00000476']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000478']` outputs=`['t00000479']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000481']` outputs=`['t00000482']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000483']` outputs=`['t00000484']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000482', 't00000023']` outputs=`['t00000485']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000485']` outputs=`['t00000486']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000484', 't00000023']` outputs=`['t00000487']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000487']` outputs=`['t00000488']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000472', 't00000486']` outputs=`['t00000489']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000472']` outputs=`['t00000490']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000472']` outputs=`['t00000491']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000491']` outputs=`['t00000492']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000492', 't00000490']` outputs=`['t00000493']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000493', 't00000488']` outputs=`['t00000494']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000489', 't00000494']` outputs=`['t00000495']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000471']` outputs=`['t00000472']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000473']` outputs=`['t00000474']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000475']` outputs=`['t00000476']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000489', 't00000494']` outputs=`['t00000495']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000496', 't00000501']` outputs=`['t00000502']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000502']` outputs=`['t00000503']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000495', 't00000503']` outputs=`['t00000504']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000504']` outputs=`['t00000505']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000505', 't00000053']` outputs=`['t00000506']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000506']` outputs=`['t00000507']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000508']` outputs=`['t00000508']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000508', 't00000476']` outputs=`['t00000509']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000479']` outputs=`['t00000480']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000480']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000513']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000513']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000517']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000527', 't00000526']` outputs=`['t00000528']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000528', 't00000529']` outputs=`['t00000530']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000532']` outputs=`['t00000533']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00000527` 与 `t00000526` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00000528`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00000530`，最后规约为布尔决策 `t00000533`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00000527  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00000526  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00000528  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00000530  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00000533  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00000527,t00000526]` output=`t00000528`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00000528,t00000529]` output=`t00000530`, observed shape=`[1,576]`; `#80 any.default` input=`t00000532` output=`t00000533`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00000508', 't00000476']` outputs=`['t00000509']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000510']` outputs=`['t00000511']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000511']` outputs=`['t00000512']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000522']` outputs=`['t00000523']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000512', 't00000534']` outputs=`['t00000535']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000455', 't00000535']` outputs=`['t00000536']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000551', 't00000552']` outputs=`['t00000553']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000536', 't00000553']` outputs=`['t00000554']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000512', 't00000534']` outputs=`['t00000535']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000455', 't00000535']` outputs=`['t00000536']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000536']` outputs=`['t00000537']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000537']` outputs=`['t00000538']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000538']` outputs=`['t00000539']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000539']` outputs=`['t00000540']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000540']` outputs=`['t00000541']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000537', 't00000541']` outputs=`['t00000542']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000542']` outputs=`['t00000543']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000544', 't00000543']` outputs=`['t00000545']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000545', 't00000546']` outputs=`['t00000547']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000547']` outputs=`['t00000548']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000545', 't00000549']` outputs=`['t00000550']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000548', 't00000550']` outputs=`['t00000551']` -> shape=[1, 624, 11008], dtype=float16
