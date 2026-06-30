# input1_layer22 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer22/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer22/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer22/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001761, t00001769`
- `input_rmsnorm` outputs: `t00001770`
- `qkv_projection` inputs: `t00001770, t00001771, t00001773, t00001775`
- `qkv_projection` outputs: `t00001778, t00001780, t00001782`
- `rope` inputs: `t00001784, t00001787, t00001789, t00001475, t00001778`
- `rope` outputs: `t00001785, t00001801`
- `attention` inputs: `t00001777, t00001779, t00001781, t00001795, t00001800, t00001802, t00001807, t00001505`
- `attention` outputs: `t00001778, t00001780, t00001813, t00001815`
- `visipruner_similarity_check` inputs: `t00001785, t00000057, t00001822, t00001823, t00001826, t00001836, t00001835, t00001838, t00001841`
- `visipruner_similarity_check` outputs: `t00001824, t00001834, t00001839, t00001842`
- `attention_output` inputs: `t00001814, t00001782, t00001816, t00001831, t00001843, t00001761, t00001860, t00001861`
- `attention_output` outputs: `t00001815, t00001832, t00001863`
- `mlp` inputs: `t00001818, t00001843, t00001761, t00001853, t00001855, t00001858`
- `mlp` outputs: `t00001860`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer22/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer22/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer22/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.22.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.22.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.22.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.22.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.22.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.22.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.22.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.22.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.22.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.22.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.22.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.22` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.22.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.22.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.22.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.22.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.22.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.22.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.22` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001761']` outputs=`['t00001762']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001762']` outputs=`['t00001763']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001763']` outputs=`['t00001764']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001764']` outputs=`['t00001765']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001765']` outputs=`['t00001766']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001762', 't00001766']` outputs=`['t00001767']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001767']` outputs=`['t00001768']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001769', 't00001768']` outputs=`['t00001770']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001770', 't00001771']` outputs=`['t00001772']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001770', 't00001773']` outputs=`['t00001774']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001770', 't00001775']` outputs=`['t00001776']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001772']` outputs=`['t00001777']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001777']` outputs=`['t00001778']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001774']` outputs=`['t00001779']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001779']` outputs=`['t00001780']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001776']` outputs=`['t00001781']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001781']` outputs=`['t00001782']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001784']` outputs=`['t00001785']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001787']` outputs=`['t00001788']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001789']` outputs=`['t00001790']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001788', 't00001475']` outputs=`['t00001791']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001791']` outputs=`['t00001792']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001790', 't00001475']` outputs=`['t00001793']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001793']` outputs=`['t00001794']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001778', 't00001792']` outputs=`['t00001795']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001778']` outputs=`['t00001796']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001778']` outputs=`['t00001797']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001797']` outputs=`['t00001798']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001798', 't00001796']` outputs=`['t00001799']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001799', 't00001794']` outputs=`['t00001800']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001795', 't00001800']` outputs=`['t00001801']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001777']` outputs=`['t00001778']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001779']` outputs=`['t00001780']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001781']` outputs=`['t00001782']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001795', 't00001800']` outputs=`['t00001801']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001802', 't00001807']` outputs=`['t00001808']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001808']` outputs=`['t00001809']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00001801', 't00001809']` outputs=`['t00001810']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00001810']` outputs=`['t00001811']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00001811', 't00001505']` outputs=`['t00001812']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00001812']` outputs=`['t00001813']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00001814']` outputs=`['t00001814']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00001814', 't00001782']` outputs=`['t00001815']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001785']` outputs=`['t00001786']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001786']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001819']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001819']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00001822', 't00001823']` outputs=`['t00001824']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00001826']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00001834']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00001836', 't00001835']` outputs=`['t00001837']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00001837', 't00001838']` outputs=`['t00001839']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00001841']` outputs=`['t00001842']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune deep exit check

该层的 Visual 相关过程是 deep exit check：运行时先生成 `P=10` 的 probe index 向量 `t00001834`，再对 probe/reference 表示做 `P×Hidden` 对齐差值 `t00001837`；随后沿 Hidden 维得到每个 probe index 的 score `t00001839`，最后规约为布尔决策 `t00001842`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Probe index axis P=10 (highly compressed to 1 row)      Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
probe index vector     t00001834  P=10                ──▶   [PROBE_INDEX_VECTOR]                    ◀── arange.start 观测到的 P=10 index

                                                            ┌────────────────────────────────────────┐
selected probe rows    t00001836  P=10                ──▶   │ SELECTED_PROBE_ROWS                    │  ◀── 40:1 压缩表示 4096:10
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
reference probe rows   t00001835  P=10                ──▶   │ REFERENCE_PROBE_ROWS                   │  ◀── 与 selected 同一 P×Hidden 坐标对齐
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
delta / compare rows   t00001837  P=10                ──▶   │ PROBE_DELTA_COMPARE                    │  ◀── selected-reference 后沿 Hidden 维比较
                                                            └────────────────────────────────────────┘

Probe score axis P=10 (same 1-row compression; 1-col expanded)
                                                            ┌──────┐
score vector           t00001839  P=10                ──▶   │SCORE │  ◀── 每个 probe index 保留一个 score
                                                            └──────┘
decision scalar        t00001842  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 probe score 做 any 归约
```

Evidence: `#75 arange.start` output=`t00001834`, observed shape=`[10]`; `#78 sub.Tensor` inputs=`[t00001836,t00001835]` output=`t00001837`, observed shape=`[1,10,4096]`; `#80 cosine_similarity.default` inputs=`[t00001837,t00001838]` output=`t00001839`, observed shape=`[1,10]`; `#83 any.default` input=`t00001841` output=`t00001842`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00001814', 't00001782']` outputs=`['t00001815']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001816']` outputs=`['t00001817']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001817']` outputs=`['t00001818']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00001831']` outputs=`['t00001832']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00001818', 't00001843']` outputs=`['t00001844']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001761', 't00001844']` outputs=`['t00001845']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001860', 't00001861']` outputs=`['t00001862']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00001845', 't00001862']` outputs=`['t00001863']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00001818', 't00001843']` outputs=`['t00001844']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001761', 't00001844']` outputs=`['t00001845']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00001845']` outputs=`['t00001846']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00001846']` outputs=`['t00001847']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00001847']` outputs=`['t00001848']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00001848']` outputs=`['t00001849']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00001849']` outputs=`['t00001850']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00001846', 't00001850']` outputs=`['t00001851']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00001851']` outputs=`['t00001852']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00001853', 't00001852']` outputs=`['t00001854']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00001854', 't00001855']` outputs=`['t00001856']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00001856']` outputs=`['t00001857']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00001854', 't00001858']` outputs=`['t00001859']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001857', 't00001859']` outputs=`['t00001860']` -> shape=[1, 58, 11008], dtype=float16
