# input1_layer25 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer25/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002067, t00002075`
- `input_rmsnorm` outputs: `t00002076`
- `qkv_projection` inputs: `t00002076, t00002077, t00002079, t00002081`
- `qkv_projection` outputs: `t00002084, t00002086, t00002088`
- `rope` inputs: `t00002090, t00002093, t00002095, t00001475, t00002084`
- `rope` outputs: `t00002091, t00002107`
- `attention` inputs: `t00002083, t00002085, t00002087, t00002101, t00002106, t00002108, t00002113, t00001505`
- `attention` outputs: `t00002084, t00002086, t00002119, t00002121`
- `visipruner_similarity_check` inputs: `t00002091, t00000057, t00002128, t00002129, t00002132, t00002142, t00002141, t00002144, t00002147`
- `visipruner_similarity_check` outputs: `t00002130, t00002140, t00002145, t00002148`
- `attention_output` inputs: `t00002120, t00002088, t00002122, t00002137, t00002149, t00002067, t00002166, t00002167`
- `attention_output` outputs: `t00002121, t00002138, t00002169`
- `mlp` inputs: `t00002124, t00002149, t00002067, t00002159, t00002161, t00002164`
- `mlp` outputs: `t00002166`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer25/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer25/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer25/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.25.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.25.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.25.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.25.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.25.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.25.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.25.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.25.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.25.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.25.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.25.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.25` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.25.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.25.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.25.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.25.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.25.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.25.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.25` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002067']` outputs=`['t00002068']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002068']` outputs=`['t00002069']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002069']` outputs=`['t00002070']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002070']` outputs=`['t00002071']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002071']` outputs=`['t00002072']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002068', 't00002072']` outputs=`['t00002073']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002073']` outputs=`['t00002074']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002075', 't00002074']` outputs=`['t00002076']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002076', 't00002077']` outputs=`['t00002078']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002076', 't00002079']` outputs=`['t00002080']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002076', 't00002081']` outputs=`['t00002082']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002078']` outputs=`['t00002083']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002083']` outputs=`['t00002084']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00002080']` outputs=`['t00002085']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002085']` outputs=`['t00002086']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00002082']` outputs=`['t00002087']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002087']` outputs=`['t00002088']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002090']` outputs=`['t00002091']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002093']` outputs=`['t00002094']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002095']` outputs=`['t00002096']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002094', 't00001475']` outputs=`['t00002097']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002097']` outputs=`['t00002098']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002096', 't00001475']` outputs=`['t00002099']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002099']` outputs=`['t00002100']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002084', 't00002098']` outputs=`['t00002101']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002084']` outputs=`['t00002102']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002084']` outputs=`['t00002103']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002103']` outputs=`['t00002104']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002104', 't00002102']` outputs=`['t00002105']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002105', 't00002100']` outputs=`['t00002106']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002101', 't00002106']` outputs=`['t00002107']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002083']` outputs=`['t00002084']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002085']` outputs=`['t00002086']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002087']` outputs=`['t00002088']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002101', 't00002106']` outputs=`['t00002107']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002108', 't00002113']` outputs=`['t00002114']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002114']` outputs=`['t00002115']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00002107', 't00002115']` outputs=`['t00002116']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00002116']` outputs=`['t00002117']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00002117', 't00001505']` outputs=`['t00002118']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00002118']` outputs=`['t00002119']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00002120']` outputs=`['t00002120']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00002120', 't00002088']` outputs=`['t00002121']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00002091']` outputs=`['t00002092']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00002092']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00002125']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00002125']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00002128', 't00002129']` outputs=`['t00002130']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00002132']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00002140']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00002142', 't00002141']` outputs=`['t00002143']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00002143', 't00002144']` outputs=`['t00002145']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00002147']` outputs=`['t00002148']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune deep exit check

该层的 Visual 相关过程是 deep exit check：运行时先生成 `P=10` 的 probe index 向量 `t00002140`，再对 probe/reference 表示做 `P×Hidden` 对齐差值 `t00002143`；随后沿 Hidden 维得到每个 probe index 的 score `t00002145`，最后规约为布尔决策 `t00002148`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Probe index axis P=10 (highly compressed to 1 row)      Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
probe index vector     t00002140  P=10                ──▶   [PROBE_INDEX_VECTOR]                    ◀── arange.start 观测到的 P=10 index

                                                            ┌────────────────────────────────────────┐
selected probe rows    t00002142  P=10                ──▶   │ SELECTED_PROBE_ROWS                    │  ◀── 40:1 压缩表示 4096:10
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
reference probe rows   t00002141  P=10                ──▶   │ REFERENCE_PROBE_ROWS                   │  ◀── 与 selected 同一 P×Hidden 坐标对齐
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
delta / compare rows   t00002143  P=10                ──▶   │ PROBE_DELTA_COMPARE                    │  ◀── selected-reference 后沿 Hidden 维比较
                                                            └────────────────────────────────────────┘

Probe score axis P=10 (same 1-row compression; 1-col expanded)
                                                            ┌──────┐
score vector           t00002145  P=10                ──▶   │SCORE │  ◀── 每个 probe index 保留一个 score
                                                            └──────┘
decision scalar        t00002148  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 probe score 做 any 归约
```

Evidence: `#75 arange.start` output=`t00002140`, observed shape=`[10]`; `#78 sub.Tensor` inputs=`[t00002142,t00002141]` output=`t00002143`, observed shape=`[1,10,4096]`; `#80 cosine_similarity.default` inputs=`[t00002143,t00002144]` output=`t00002145`, observed shape=`[1,10]`; `#83 any.default` input=`t00002147` output=`t00002148`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00002120', 't00002088']` outputs=`['t00002121']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002122']` outputs=`['t00002123']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002123']` outputs=`['t00002124']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00002137']` outputs=`['t00002138']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00002124', 't00002149']` outputs=`['t00002150']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002067', 't00002150']` outputs=`['t00002151']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00002166', 't00002167']` outputs=`['t00002168']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00002151', 't00002168']` outputs=`['t00002169']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00002124', 't00002149']` outputs=`['t00002150']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002067', 't00002150']` outputs=`['t00002151']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00002151']` outputs=`['t00002152']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00002152']` outputs=`['t00002153']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00002153']` outputs=`['t00002154']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00002154']` outputs=`['t00002155']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00002155']` outputs=`['t00002156']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00002152', 't00002156']` outputs=`['t00002157']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00002157']` outputs=`['t00002158']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00002159', 't00002158']` outputs=`['t00002160']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00002160', 't00002161']` outputs=`['t00002162']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00002162']` outputs=`['t00002163']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00002160', 't00002164']` outputs=`['t00002165']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00002163', 't00002165']` outputs=`['t00002166']` -> shape=[1, 58, 11008], dtype=float16
