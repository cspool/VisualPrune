# input1_layer14 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer14/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000950, t00000958`
- `input_rmsnorm` outputs: `t00000959`
- `qkv_projection` inputs: `t00000959, t00000960, t00000962, t00000964`
- `qkv_projection` outputs: `t00000967, t00000969, t00000971`
- `rope` inputs: `t00000973, t00000976, t00000978, t00000023, t00000967`
- `rope` outputs: `t00000974, t00000990`
- `attention` inputs: `t00000966, t00000968, t00000970, t00000984, t00000989, t00000991, t00000996, t00000053`
- `attention` outputs: `t00000967, t00000969, t00001002, t00001004`
- `visipruner_similarity_check` inputs: `t00000974, t00000057, t00001012, t00001022, t00001021, t00001024, t00001027`
- `visipruner_similarity_check` outputs: `t00001025, t00001028`
- `attention_output` inputs: `t00001003, t00000971, t00001005, t00001017, t00001029, t00000950, t00001046, t00001047`
- `attention_output` outputs: `t00001004, t00001018, t00001049`
- `mlp` inputs: `t00001007, t00001029, t00000950, t00001039, t00001041, t00001044`
- `mlp` outputs: `t00001046`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer14/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer14/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer14/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.14.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.14.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.14.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.14.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.14.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.14.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.14.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.14.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.14.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.14.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.14.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.14` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.14.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.14.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.14.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.14.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.14.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.14.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.14` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000950']` outputs=`['t00000951']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000951']` outputs=`['t00000952']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000952']` outputs=`['t00000953']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000953']` outputs=`['t00000954']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000954']` outputs=`['t00000955']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000951', 't00000955']` outputs=`['t00000956']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000956']` outputs=`['t00000957']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000958', 't00000957']` outputs=`['t00000959']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000959', 't00000960']` outputs=`['t00000961']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000959', 't00000962']` outputs=`['t00000963']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000959', 't00000964']` outputs=`['t00000965']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000961']` outputs=`['t00000966']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000966']` outputs=`['t00000967']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000963']` outputs=`['t00000968']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000968']` outputs=`['t00000969']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000965']` outputs=`['t00000970']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000970']` outputs=`['t00000971']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000973']` outputs=`['t00000974']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000976']` outputs=`['t00000977']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000978']` outputs=`['t00000979']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000977', 't00000023']` outputs=`['t00000980']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000980']` outputs=`['t00000981']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000979', 't00000023']` outputs=`['t00000982']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000982']` outputs=`['t00000983']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000967', 't00000981']` outputs=`['t00000984']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000967']` outputs=`['t00000985']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000967']` outputs=`['t00000986']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000986']` outputs=`['t00000987']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000987', 't00000985']` outputs=`['t00000988']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000988', 't00000983']` outputs=`['t00000989']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000984', 't00000989']` outputs=`['t00000990']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000966']` outputs=`['t00000967']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000968']` outputs=`['t00000969']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000970']` outputs=`['t00000971']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000984', 't00000989']` outputs=`['t00000990']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000991', 't00000996']` outputs=`['t00000997']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000997']` outputs=`['t00000998']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000990', 't00000998']` outputs=`['t00000999']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000999']` outputs=`['t00001000']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00001000', 't00000053']` outputs=`['t00001001']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00001001']` outputs=`['t00001002']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00001003']` outputs=`['t00001003']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00001003', 't00000971']` outputs=`['t00001004']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000974']` outputs=`['t00000975']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000975']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001008']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001008']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001012']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001022', 't00001021']` outputs=`['t00001023']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001023', 't00001024']` outputs=`['t00001025']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001027']` outputs=`['t00001028']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00001022` 与 `t00001021` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00001023`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00001025`，最后规约为布尔决策 `t00001028`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00001022  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00001021  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00001023  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00001025  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00001028  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00001022,t00001021]` output=`t00001023`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00001023,t00001024]` output=`t00001025`, observed shape=`[1,576]`; `#80 any.default` input=`t00001027` output=`t00001028`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00001003', 't00000971']` outputs=`['t00001004']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001005']` outputs=`['t00001006']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001006']` outputs=`['t00001007']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00001017']` outputs=`['t00001018']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00001007', 't00001029']` outputs=`['t00001030']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000950', 't00001030']` outputs=`['t00001031']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00001046', 't00001047']` outputs=`['t00001048']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00001031', 't00001048']` outputs=`['t00001049']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00001007', 't00001029']` outputs=`['t00001030']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000950', 't00001030']` outputs=`['t00001031']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00001031']` outputs=`['t00001032']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00001032']` outputs=`['t00001033']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00001033']` outputs=`['t00001034']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00001034']` outputs=`['t00001035']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00001035']` outputs=`['t00001036']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00001032', 't00001036']` outputs=`['t00001037']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00001037']` outputs=`['t00001038']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00001039', 't00001038']` outputs=`['t00001040']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00001040', 't00001041']` outputs=`['t00001042']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00001042']` outputs=`['t00001043']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00001040', 't00001044']` outputs=`['t00001045']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00001043', 't00001045']` outputs=`['t00001046']` -> shape=[1, 624, 11008], dtype=float16
