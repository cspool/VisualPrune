# input1_layer7 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer7/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer7/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer7/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000257, t00000265`
- `input_rmsnorm` outputs: `t00000266`
- `qkv_projection` inputs: `t00000266, t00000267, t00000269, t00000271`
- `qkv_projection` outputs: `t00000274, t00000276, t00000278`
- `rope` inputs: `t00000280, t00000283, t00000285, t00000023, t00000274`
- `rope` outputs: `t00000281, t00000297`
- `attention` inputs: `t00000273, t00000275, t00000277, t00000291, t00000296, t00000298, t00000303, t00000053`
- `attention` outputs: `t00000274, t00000276, t00000309, t00000311`
- `visipruner_similarity_check` inputs: `t00000281, t00000057, t00000319, t00000329, t00000328, t00000331, t00000334`
- `visipruner_similarity_check` outputs: `t00000332, t00000335`
- `attention_output` inputs: `t00000310, t00000278, t00000312, t00000324, t00000336, t00000257, t00000353, t00000354`
- `attention_output` outputs: `t00000311, t00000325, t00000356`
- `mlp` inputs: `t00000314, t00000336, t00000257, t00000346, t00000348, t00000351`
- `mlp` outputs: `t00000353`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer7/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer7/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer7/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.7.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.7.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.7.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.7.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.7.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.7.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.7.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.7.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.7.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.7.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.7.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.7.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.7.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.7.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.7` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.7.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.7.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.7.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.7.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.7.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.7.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.7` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000257']` outputs=`['t00000258']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000258']` outputs=`['t00000259']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000259']` outputs=`['t00000260']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000260']` outputs=`['t00000261']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000261']` outputs=`['t00000262']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000258', 't00000262']` outputs=`['t00000263']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000263']` outputs=`['t00000264']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000265', 't00000264']` outputs=`['t00000266']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000266', 't00000267']` outputs=`['t00000268']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000266', 't00000269']` outputs=`['t00000270']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000266', 't00000271']` outputs=`['t00000272']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000268']` outputs=`['t00000273']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000273']` outputs=`['t00000274']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000270']` outputs=`['t00000275']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000275']` outputs=`['t00000276']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000272']` outputs=`['t00000277']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000277']` outputs=`['t00000278']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000280']` outputs=`['t00000281']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000283']` outputs=`['t00000284']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000285']` outputs=`['t00000286']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000284', 't00000023']` outputs=`['t00000287']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000287']` outputs=`['t00000288']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000286', 't00000023']` outputs=`['t00000289']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000289']` outputs=`['t00000290']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000274', 't00000288']` outputs=`['t00000291']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000274']` outputs=`['t00000292']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000274']` outputs=`['t00000293']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000293']` outputs=`['t00000294']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000294', 't00000292']` outputs=`['t00000295']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000295', 't00000290']` outputs=`['t00000296']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000291', 't00000296']` outputs=`['t00000297']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000273']` outputs=`['t00000274']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000275']` outputs=`['t00000276']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000277']` outputs=`['t00000278']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000291', 't00000296']` outputs=`['t00000297']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000298', 't00000303']` outputs=`['t00000304']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000304']` outputs=`['t00000305']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000297', 't00000305']` outputs=`['t00000306']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000306']` outputs=`['t00000307']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000307', 't00000053']` outputs=`['t00000308']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000308']` outputs=`['t00000309']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000310']` outputs=`['t00000310']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000310', 't00000278']` outputs=`['t00000311']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000281']` outputs=`['t00000282']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000282']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000315']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000315']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000319']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000329', 't00000328']` outputs=`['t00000330']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000330', 't00000331']` outputs=`['t00000332']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000334']` outputs=`['t00000335']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00000329` 与 `t00000328` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00000330`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00000332`，最后规约为布尔决策 `t00000335`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00000329  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00000328  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00000330  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00000332  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00000335  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00000329,t00000328]` output=`t00000330`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00000330,t00000331]` output=`t00000332`, observed shape=`[1,576]`; `#80 any.default` input=`t00000334` output=`t00000335`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00000310', 't00000278']` outputs=`['t00000311']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000312']` outputs=`['t00000313']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000313']` outputs=`['t00000314']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000324']` outputs=`['t00000325']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000314', 't00000336']` outputs=`['t00000337']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000257', 't00000337']` outputs=`['t00000338']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000353', 't00000354']` outputs=`['t00000355']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000338', 't00000355']` outputs=`['t00000356']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000314', 't00000336']` outputs=`['t00000337']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000257', 't00000337']` outputs=`['t00000338']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000338']` outputs=`['t00000339']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000339']` outputs=`['t00000340']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000340']` outputs=`['t00000341']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000341']` outputs=`['t00000342']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000342']` outputs=`['t00000343']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000339', 't00000343']` outputs=`['t00000344']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000344']` outputs=`['t00000345']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000346', 't00000345']` outputs=`['t00000347']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000347', 't00000348']` outputs=`['t00000349']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000349']` outputs=`['t00000350']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000347', 't00000351']` outputs=`['t00000352']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000350', 't00000352']` outputs=`['t00000353']` -> shape=[1, 624, 11008], dtype=float16
