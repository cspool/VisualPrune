# input1_layer12 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer12/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer12/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer12/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000752, t00000760`
- `input_rmsnorm` outputs: `t00000761`
- `qkv_projection` inputs: `t00000761, t00000762, t00000764, t00000766`
- `qkv_projection` outputs: `t00000769, t00000771, t00000773`
- `rope` inputs: `t00000775, t00000778, t00000780, t00000023, t00000769`
- `rope` outputs: `t00000776, t00000792`
- `attention` inputs: `t00000768, t00000770, t00000772, t00000786, t00000791, t00000793, t00000798, t00000053`
- `attention` outputs: `t00000769, t00000771, t00000804, t00000806`
- `visipruner_similarity_check` inputs: `t00000776, t00000057, t00000814, t00000824, t00000823, t00000826, t00000829`
- `visipruner_similarity_check` outputs: `t00000827, t00000830`
- `attention_output` inputs: `t00000805, t00000773, t00000807, t00000819, t00000831, t00000752, t00000848, t00000849`
- `attention_output` outputs: `t00000806, t00000820, t00000851`
- `mlp` inputs: `t00000809, t00000831, t00000752, t00000841, t00000843, t00000846`
- `mlp` outputs: `t00000848`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer12/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer12/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer12/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.12.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.12.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.12.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.12.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.12.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.12.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.12.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.12.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.12.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.12.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.12.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.12` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.12.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.12.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.12.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.12.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.12.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.12.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.12` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000752']` outputs=`['t00000753']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000753']` outputs=`['t00000754']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000754']` outputs=`['t00000755']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000755']` outputs=`['t00000756']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000756']` outputs=`['t00000757']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000753', 't00000757']` outputs=`['t00000758']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000758']` outputs=`['t00000759']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000760', 't00000759']` outputs=`['t00000761']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000761', 't00000762']` outputs=`['t00000763']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000761', 't00000764']` outputs=`['t00000765']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000761', 't00000766']` outputs=`['t00000767']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000763']` outputs=`['t00000768']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000768']` outputs=`['t00000769']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000765']` outputs=`['t00000770']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000770']` outputs=`['t00000771']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000767']` outputs=`['t00000772']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000772']` outputs=`['t00000773']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000775']` outputs=`['t00000776']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000778']` outputs=`['t00000779']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000780']` outputs=`['t00000781']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000779', 't00000023']` outputs=`['t00000782']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000782']` outputs=`['t00000783']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000781', 't00000023']` outputs=`['t00000784']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000784']` outputs=`['t00000785']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000769', 't00000783']` outputs=`['t00000786']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000769']` outputs=`['t00000787']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000769']` outputs=`['t00000788']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000788']` outputs=`['t00000789']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000789', 't00000787']` outputs=`['t00000790']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000790', 't00000785']` outputs=`['t00000791']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000786', 't00000791']` outputs=`['t00000792']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000768']` outputs=`['t00000769']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000770']` outputs=`['t00000771']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000772']` outputs=`['t00000773']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000786', 't00000791']` outputs=`['t00000792']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000793', 't00000798']` outputs=`['t00000799']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000799']` outputs=`['t00000800']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000792', 't00000800']` outputs=`['t00000801']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000801']` outputs=`['t00000802']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000802', 't00000053']` outputs=`['t00000803']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000803']` outputs=`['t00000804']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000805']` outputs=`['t00000805']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000805', 't00000773']` outputs=`['t00000806']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000776']` outputs=`['t00000777']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000777']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000810']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000810']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000814']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000824', 't00000823']` outputs=`['t00000825']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000825', 't00000826']` outputs=`['t00000827']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000829']` outputs=`['t00000830']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00000824` 与 `t00000823` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00000825`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00000827`，最后规约为布尔决策 `t00000830`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00000824  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00000823  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00000825  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00000827  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00000830  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00000824,t00000823]` output=`t00000825`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00000825,t00000826]` output=`t00000827`, observed shape=`[1,576]`; `#80 any.default` input=`t00000829` output=`t00000830`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00000805', 't00000773']` outputs=`['t00000806']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000807']` outputs=`['t00000808']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000808']` outputs=`['t00000809']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000819']` outputs=`['t00000820']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000809', 't00000831']` outputs=`['t00000832']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000752', 't00000832']` outputs=`['t00000833']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000848', 't00000849']` outputs=`['t00000850']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000833', 't00000850']` outputs=`['t00000851']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000809', 't00000831']` outputs=`['t00000832']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000752', 't00000832']` outputs=`['t00000833']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000833']` outputs=`['t00000834']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000834']` outputs=`['t00000835']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000835']` outputs=`['t00000836']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000836']` outputs=`['t00000837']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000837']` outputs=`['t00000838']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000834', 't00000838']` outputs=`['t00000839']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000839']` outputs=`['t00000840']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000841', 't00000840']` outputs=`['t00000842']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000842', 't00000843']` outputs=`['t00000844']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000844']` outputs=`['t00000845']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000842', 't00000846']` outputs=`['t00000847']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000845', 't00000847']` outputs=`['t00000848']` -> shape=[1, 624, 11008], dtype=float16
