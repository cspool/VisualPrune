# input1_layer15 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer15/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer15/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer15/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001049, t00001057`
- `input_rmsnorm` outputs: `t00001058`
- `qkv_projection` inputs: `t00001058, t00001059, t00001061, t00001063`
- `qkv_projection` outputs: `t00001066, t00001068, t00001070`
- `rope` inputs: `t00001072, t00001075, t00001077, t00000023, t00001066`
- `rope` outputs: `t00001073, t00001089`
- `attention` inputs: `t00001065, t00001067, t00001069, t00001083, t00001088, t00001090, t00001095, t00000053`
- `attention` outputs: `t00001066, t00001068, t00001101, t00001103`
- `visipruner_similarity_check` inputs: `t00001073, t00000057, t00001111, t00001121, t00001120, t00001123, t00001126`
- `visipruner_similarity_check` outputs: `t00001124, t00001127`
- `attention_output` inputs: `t00001102, t00001070, t00001104, t00001116, t00001128, t00001049, t00001145, t00001146`
- `attention_output` outputs: `t00001103, t00001117, t00001148`
- `mlp` inputs: `t00001106, t00001128, t00001049, t00001138, t00001140, t00001143`
- `mlp` outputs: `t00001145`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer15/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer15/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer15/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.15.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.15.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.15.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.15.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.15.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.15.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.15.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.15.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.15.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.15.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.15.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.15` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.15.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.15.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.15.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.15.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.15.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.15.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.15` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001049']` outputs=`['t00001050']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001050']` outputs=`['t00001051']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001051']` outputs=`['t00001052']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001052']` outputs=`['t00001053']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001053']` outputs=`['t00001054']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001050', 't00001054']` outputs=`['t00001055']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001055']` outputs=`['t00001056']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001057', 't00001056']` outputs=`['t00001058']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001058', 't00001059']` outputs=`['t00001060']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001058', 't00001061']` outputs=`['t00001062']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001058', 't00001063']` outputs=`['t00001064']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001060']` outputs=`['t00001065']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001065']` outputs=`['t00001066']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00001062']` outputs=`['t00001067']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001067']` outputs=`['t00001068']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00001064']` outputs=`['t00001069']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001069']` outputs=`['t00001070']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001072']` outputs=`['t00001073']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001075']` outputs=`['t00001076']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001077']` outputs=`['t00001078']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001076', 't00000023']` outputs=`['t00001079']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001079']` outputs=`['t00001080']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001078', 't00000023']` outputs=`['t00001081']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001081']` outputs=`['t00001082']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001066', 't00001080']` outputs=`['t00001083']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001066']` outputs=`['t00001084']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001066']` outputs=`['t00001085']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001085']` outputs=`['t00001086']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001086', 't00001084']` outputs=`['t00001087']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001087', 't00001082']` outputs=`['t00001088']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001083', 't00001088']` outputs=`['t00001089']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001065']` outputs=`['t00001066']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001067']` outputs=`['t00001068']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001069']` outputs=`['t00001070']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001083', 't00001088']` outputs=`['t00001089']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001090', 't00001095']` outputs=`['t00001096']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001096']` outputs=`['t00001097']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00001089', 't00001097']` outputs=`['t00001098']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00001098']` outputs=`['t00001099']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00001099', 't00000053']` outputs=`['t00001100']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00001100']` outputs=`['t00001101']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00001102']` outputs=`['t00001102']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00001102', 't00001070']` outputs=`['t00001103']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001073']` outputs=`['t00001074']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001074']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001107']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001107']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00001111']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00001121', 't00001120']` outputs=`['t00001122']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00001122', 't00001123']` outputs=`['t00001124']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00001126']` outputs=`['t00001127']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00001121` 与 `t00001120` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00001122`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00001124`，最后规约为布尔决策 `t00001127`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00001121  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00001120  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00001122  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00001124  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00001127  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00001121,t00001120]` output=`t00001122`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00001122,t00001123]` output=`t00001124`, observed shape=`[1,576]`; `#80 any.default` input=`t00001126` output=`t00001127`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00001102', 't00001070']` outputs=`['t00001103']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001104']` outputs=`['t00001105']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001105']` outputs=`['t00001106']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00001116']` outputs=`['t00001117']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00001106', 't00001128']` outputs=`['t00001129']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001049', 't00001129']` outputs=`['t00001130']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00001145', 't00001146']` outputs=`['t00001147']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00001130', 't00001147']` outputs=`['t00001148']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00001106', 't00001128']` outputs=`['t00001129']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00001049', 't00001129']` outputs=`['t00001130']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00001130']` outputs=`['t00001131']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00001131']` outputs=`['t00001132']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00001132']` outputs=`['t00001133']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00001133']` outputs=`['t00001134']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00001134']` outputs=`['t00001135']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00001131', 't00001135']` outputs=`['t00001136']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00001136']` outputs=`['t00001137']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00001138', 't00001137']` outputs=`['t00001139']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00001139', 't00001140']` outputs=`['t00001141']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00001141']` outputs=`['t00001142']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00001139', 't00001143']` outputs=`['t00001144']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00001142', 't00001144']` outputs=`['t00001145']` -> shape=[1, 624, 11008], dtype=float16
