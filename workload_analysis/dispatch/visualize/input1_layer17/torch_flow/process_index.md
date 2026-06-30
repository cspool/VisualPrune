# input1_layer17 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer17/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer17/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer17/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001247, t00001255`
- `input_rmsnorm` outputs: `t00001256`
- `qkv_projection` inputs: `t00001256, t00001257, t00001259, t00001261`
- `qkv_projection` outputs: `t00001264, t00001266, t00001268`
- `rope` inputs: `t00001270, t00001273, t00001275, t00000023, t00001264`
- `rope` outputs: `t00001271, t00001287`
- `attention` inputs: `t00001263, t00001265, t00001267, t00001281, t00001286, t00001288, t00001293, t00000053`
- `attention` outputs: `t00001264, t00001266, t00001299, t00001301`
- `visipruner_similarity_check` inputs: `t00001271, t00000057, t00001309, t00001319, t00001318, t00001321, t00001324`
- `visipruner_similarity_check` outputs: `t00001322, t00001325`
- `attention_output` inputs: `t00001300, t00001268, t00001302, t00001314, t00001326, t00001247, t00001343, t00001344`
- `attention_output` outputs: `t00001301, t00001315, t00001346`
- `mlp` inputs: `t00001304, t00001326, t00001247, t00001336, t00001338, t00001341`
- `mlp` outputs: `t00001343`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer17/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer17/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer17/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.17.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.17.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.17.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.17.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.17.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.17.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.17.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.17.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.17.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.17.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.17.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.17` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.17.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.17.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.17.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.17.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.17.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.17.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.17` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001247']` outputs=`['t00001248']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001248']` outputs=`['t00001249']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001249']` outputs=`['t00001250']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001250']` outputs=`['t00001251']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001251']` outputs=`['t00001252']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001248', 't00001252']` outputs=`['t00001253']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001253']` outputs=`['t00001254']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001255', 't00001254']` outputs=`['t00001256']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001256', 't00001257']` outputs=`['t00001258']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001256', 't00001259']` outputs=`['t00001260']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001256', 't00001261']` outputs=`['t00001262']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001258']` outputs=`['t00001263']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001263']` outputs=`['t00001264']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00001260']` outputs=`['t00001265']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001265']` outputs=`['t00001266']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00001262']` outputs=`['t00001267']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001267']` outputs=`['t00001268']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001270']` outputs=`['t00001271']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001273']` outputs=`['t00001274']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001275']` outputs=`['t00001276']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001274', 't00000023']` outputs=`['t00001277']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001277']` outputs=`['t00001278']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001276', 't00000023']` outputs=`['t00001279']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001279']` outputs=`['t00001280']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001264', 't00001278']` outputs=`['t00001281']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001264']` outputs=`['t00001282']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001264']` outputs=`['t00001283']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001283']` outputs=`['t00001284']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001284', 't00001282']` outputs=`['t00001285']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001285', 't00001280']` outputs=`['t00001286']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001281', 't00001286']` outputs=`['t00001287']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001263']` outputs=`['t00001264']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001265']` outputs=`['t00001266']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001267']` outputs=`['t00001268']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001281', 't00001286']` outputs=`['t00001287']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001288', 't00001293']` outputs=`['t00001294']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001294']` outputs=`['t00001295']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00001287', 't00001295']` outputs=`['t00001296']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00001296']` outputs=`['t00001297']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00001297', 't00000053']` outputs=`['t00001298']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00001298']` outputs=`['t00001299']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00001300']` outputs=`['t00001300']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00001300', 't00001268']` outputs=`['t00001301']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001271']` outputs=`['t00001272']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001272']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001305']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001305']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001309']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001319', 't00001318']` outputs=`['t00001320']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001320', 't00001321']` outputs=`['t00001322']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001324']` outputs=`['t00001325']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00001319` 与 `t00001318` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00001320`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00001322`，最后规约为布尔决策 `t00001325`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00001319  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00001318  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00001320  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00001322  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00001325  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00001319,t00001318]` output=`t00001320`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00001320,t00001321]` output=`t00001322`, observed shape=`[1,576]`; `#80 any.default` input=`t00001324` output=`t00001325`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00001300', 't00001268']` outputs=`['t00001301']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001302']` outputs=`['t00001303']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001303']` outputs=`['t00001304']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00001314']` outputs=`['t00001315']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00001304', 't00001326']` outputs=`['t00001327']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001247', 't00001327']` outputs=`['t00001328']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00001343', 't00001344']` outputs=`['t00001345']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00001328', 't00001345']` outputs=`['t00001346']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00001304', 't00001326']` outputs=`['t00001327']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001247', 't00001327']` outputs=`['t00001328']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00001328']` outputs=`['t00001329']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00001329']` outputs=`['t00001330']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00001330']` outputs=`['t00001331']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00001331']` outputs=`['t00001332']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00001332']` outputs=`['t00001333']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00001329', 't00001333']` outputs=`['t00001334']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00001334']` outputs=`['t00001335']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00001336', 't00001335']` outputs=`['t00001337']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00001337', 't00001338']` outputs=`['t00001339']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00001339']` outputs=`['t00001340']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00001337', 't00001341']` outputs=`['t00001342']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00001340', 't00001342']` outputs=`['t00001343']` -> shape=[1, 624, 11008], dtype=float16
