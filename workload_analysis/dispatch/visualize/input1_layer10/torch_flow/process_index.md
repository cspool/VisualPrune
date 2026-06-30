# input1_layer10 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer10/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00000554, t00000562`
- `input_rmsnorm` outputs: `t00000563`
- `qkv_projection` inputs: `t00000563, t00000564, t00000566, t00000568`
- `qkv_projection` outputs: `t00000571, t00000573, t00000575`
- `rope` inputs: `t00000577, t00000580, t00000582, t00000023, t00000571`
- `rope` outputs: `t00000578, t00000594`
- `attention` inputs: `t00000570, t00000572, t00000574, t00000588, t00000593, t00000595, t00000600, t00000053`
- `attention` outputs: `t00000571, t00000573, t00000606, t00000608`
- `visipruner_similarity_check` inputs: `t00000578, t00000057, t00000616, t00000626, t00000625, t00000628, t00000631`
- `visipruner_similarity_check` outputs: `t00000629, t00000632`
- `attention_output` inputs: `t00000607, t00000575, t00000609, t00000621, t00000633, t00000554, t00000650, t00000651`
- `attention_output` outputs: `t00000608, t00000622, t00000653`
- `mlp` inputs: `t00000611, t00000633, t00000554, t00000643, t00000645, t00000648`
- `mlp` outputs: `t00000650`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer10/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer10/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer10/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.10.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.10.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.10.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.10.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.10.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.10.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.10.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 63 | `eq.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 64 | `is_nonzero.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 66 | `select.int` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 67 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 68 | `mul.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 69 | `permute.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 70 | `contiguous.default` | `model.layers.10.self_attn` | `True` | `True` | `attention_output` |
| 71 | `view.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 72 | `item.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 73 | `slice.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 74 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 75 | `sub.Tensor` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `unsqueeze.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 77 | `cosine_similarity.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 78 | `squeeze.dim` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 79 | `lt.Scalar` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 80 | `any.default` | `model.layers.10.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `item.default` | `model.layers.10.self_attn` | `True` | `True` | `` |
| 82 | `linear.default` | `model.layers.10.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 83 | `add.Tensor` | `model.layers.10` | `True` | `True` | `attention_output, mlp` |
| 84 | `to.dtype` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 85 | `pow.Tensor_Scalar` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 86 | `mean.dim` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 87 | `add.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `rsqrt.default` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mul.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `to.dtype` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `mul.Tensor` | `model.layers.10.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `linear.default` | `model.layers.10.mlp.gate_proj` | `True` | `True` | `mlp` |
| 93 | `silu.default` | `model.layers.10.mlp.act_fn` | `True` | `True` | `mlp` |
| 94 | `linear.default` | `model.layers.10.mlp.up_proj` | `True` | `True` | `mlp` |
| 95 | `mul.Tensor` | `model.layers.10.mlp` | `True` | `True` | `mlp` |
| 96 | `linear.default` | `model.layers.10.mlp.down_proj` | `True` | `True` | `attention_output` |
| 97 | `add.Tensor` | `model.layers.10` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00000554']` outputs=`['t00000555']` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00000555']` outputs=`['t00000556']` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00000556']` outputs=`['t00000557']` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00000557']` outputs=`['t00000558']` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00000558']` outputs=`['t00000559']` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00000555', 't00000559']` outputs=`['t00000560']` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00000560']` outputs=`['t00000561']` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00000562', 't00000561']` outputs=`['t00000563']` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00000563', 't00000564']` outputs=`['t00000565']` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00000563', 't00000566']` outputs=`['t00000567']` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00000563', 't00000568']` outputs=`['t00000569']` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` inputs=`['t00000565']` outputs=`['t00000570']` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00000570']` outputs=`['t00000571']` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` inputs=`['t00000567']` outputs=`['t00000572']` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000572']` outputs=`['t00000573']` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` inputs=`['t00000569']` outputs=`['t00000574']` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000574']` outputs=`['t00000575']` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00000577']` outputs=`['t00000578']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00000580']` outputs=`['t00000581']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00000582']` outputs=`['t00000583']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00000581', 't00000023']` outputs=`['t00000584']` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00000584']` outputs=`['t00000585']` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00000583', 't00000023']` outputs=`['t00000586']` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00000586']` outputs=`['t00000587']` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00000571', 't00000585']` outputs=`['t00000588']` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00000571']` outputs=`['t00000589']` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00000571']` outputs=`['t00000590']` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` inputs=`['t00000590']` outputs=`['t00000591']` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` inputs=`['t00000591', 't00000589']` outputs=`['t00000592']` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00000592', 't00000587']` outputs=`['t00000593']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000588', 't00000593']` outputs=`['t00000594']` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00000570']` outputs=`['t00000571']` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00000572']` outputs=`['t00000573']` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00000574']` outputs=`['t00000575']` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00000588', 't00000593']` outputs=`['t00000594']` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00000595', 't00000600']` outputs=`['t00000601']` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00000601']` outputs=`['t00000602']` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` inputs=`['t00000594', 't00000602']` outputs=`['t00000603']` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` inputs=`['t00000603']` outputs=`['t00000604']` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` inputs=`['t00000604', 't00000053']` outputs=`['t00000605']` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` inputs=`['t00000605']` outputs=`['t00000606']` -> shape=[1, 32, 624, 624], dtype=float32
- `#53 dropout.default` inputs=`['t00000607']` outputs=`['t00000607']` -> shape=[1, 32, 624, 624], dtype=float16
- `#54 matmul.default` inputs=`['t00000607', 't00000575']` outputs=`['t00000608']` -> shape=[1, 32, 624, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00000578']` outputs=`['t00000579']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00000579']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00000612']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00000612']` outputs=`[]` -> True
- `#64 is_nonzero.default` inputs=`['t00000616']` outputs=`[]` -> True
- `#75 sub.Tensor` inputs=`['t00000626', 't00000625']` outputs=`['t00000627']` -> shape=[1, 576, 4096], dtype=float16
- `#77 cosine_similarity.default` inputs=`['t00000627', 't00000628']` outputs=`['t00000629']` -> shape=[1, 576], dtype=float16
- `#80 any.default` inputs=`['t00000631']` outputs=`['t00000632']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune middle probe

该层的 Visual 相关过程是 middle probe：`t00000626` 与 `t00000625` 是按 Visual token 行和 Hidden 列对齐的两组表示；运行时先得到差值区域 `t00000627`，再沿 Hidden 维为每个 Visual token 生成相似度分数 `t00000629`，最后规约为布尔决策 `t00000632`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Visual token axis V=576 (compressed to 6 rows)           Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
                                                            ┌────────────────────────────────────────┐
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
current visual rows    t00000626  V=576               ──▶   │ CURRENT_VISUAL_ROWS                    │  ◀── 40:6 接近 4096:576
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            │ CURRENT_VISUAL_ROWS                    │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
reference visual rows  t00000625  V=576               ──▶   │ REFERENCE_VISUAL_ROWS                  │  ◀── 与 current 同一 V×Hidden 坐标对齐
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            │ REFERENCE_VISUAL_ROWS                  │
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
delta / compare rows   t00000627  V=576               ──▶   │ DELTA_COMPARE_ROWS                     │  ◀── current-reference 后沿 Hidden 维比较
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            │ DELTA_COMPARE_ROWS                     │
                                                            └────────────────────────────────────────┘

Visual score axis V=576 (same 6-row compression; 1-col expanded)
                                                            ┌──────┐
                                                            │SCORE │
                                                            │SCORE │
score vector           t00000629  V=576               ──▶   │SCORE │  ◀── 每个 Visual token 行保留一个 score
                                                            │SCORE │
                                                            │SCORE │
                                                            │SCORE │
                                                            └──────┘
decision scalar        t00000632  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 score vector 做 any 归约
```

Evidence: `#75 sub.Tensor` inputs=`[t00000626,t00000625]` output=`t00000627`, observed shape=`[1,576,4096]`; `#77 cosine_similarity.default` inputs=`[t00000627,t00000628]` output=`t00000629`, observed shape=`[1,576]`; `#80 any.default` input=`t00000631` output=`t00000632`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00000607', 't00000575']` outputs=`['t00000608']` -> shape=[1, 32, 624, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00000609']` outputs=`['t00000610']` -> shape=[1, 624, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00000610']` outputs=`['t00000611']` -> shape=[1, 624, 4096], dtype=float16
- `#70 contiguous.default` inputs=`['t00000621']` outputs=`['t00000622']` -> shape=[1, 624, 32, 128], dtype=float16
- `#82 linear.default` inputs=`['t00000611', 't00000633']` outputs=`['t00000634']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000554', 't00000634']` outputs=`['t00000635']` -> shape=[1, 624, 4096], dtype=float16
- `#96 linear.default` inputs=`['t00000650', 't00000651']` outputs=`['t00000652']` -> shape=[1, 624, 4096], dtype=float16
- `#97 add.Tensor` inputs=`['t00000635', 't00000652']` outputs=`['t00000653']` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#82 linear.default` inputs=`['t00000611', 't00000633']` outputs=`['t00000634']` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` inputs=`['t00000554', 't00000634']` outputs=`['t00000635']` -> shape=[1, 624, 4096], dtype=float16
- `#84 to.dtype` inputs=`['t00000635']` outputs=`['t00000636']` -> shape=[1, 624, 4096], dtype=float32
- `#85 pow.Tensor_Scalar` inputs=`['t00000636']` outputs=`['t00000637']` -> shape=[1, 624, 4096], dtype=float32
- `#86 mean.dim` inputs=`['t00000637']` outputs=`['t00000638']` -> shape=[1, 624, 1], dtype=float32
- `#87 add.Tensor` inputs=`['t00000638']` outputs=`['t00000639']` -> shape=[1, 624, 1], dtype=float32
- `#88 rsqrt.default` inputs=`['t00000639']` outputs=`['t00000640']` -> shape=[1, 624, 1], dtype=float32
- `#89 mul.Tensor` inputs=`['t00000636', 't00000640']` outputs=`['t00000641']` -> shape=[1, 624, 4096], dtype=float32
- `#90 to.dtype` inputs=`['t00000641']` outputs=`['t00000642']` -> shape=[1, 624, 4096], dtype=float16
- `#91 mul.Tensor` inputs=`['t00000643', 't00000642']` outputs=`['t00000644']` -> shape=[1, 624, 4096], dtype=float16
- `#92 linear.default` inputs=`['t00000644', 't00000645']` outputs=`['t00000646']` -> shape=[1, 624, 11008], dtype=float16
- `#93 silu.default` inputs=`['t00000646']` outputs=`['t00000647']` -> shape=[1, 624, 11008], dtype=float16
- `#94 linear.default` inputs=`['t00000644', 't00000648']` outputs=`['t00000649']` -> shape=[1, 624, 11008], dtype=float16
- `#95 mul.Tensor` inputs=`['t00000647', 't00000649']` outputs=`['t00000650']` -> shape=[1, 624, 11008], dtype=float16
