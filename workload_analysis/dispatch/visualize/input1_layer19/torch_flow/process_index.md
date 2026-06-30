# input1_layer19 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer19/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer19/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer19/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00001453, t00001461`
- `input_rmsnorm` outputs: `t00001462`
- `qkv_projection` inputs: `t00001462, t00001463, t00001465, t00001467`
- `qkv_projection` outputs: `t00001470, t00001472, t00001474`
- `rope` inputs: `t00001477, t00001480, t00001482, t00001475, t00001470`
- `rope` outputs: `t00001478, t00001494`
- `attention` inputs: `t00001469, t00001471, t00001473, t00001488, t00001493, t00001495, t00001500, t00001505`
- `attention` outputs: `t00001470, t00001472, t00001507, t00001509`
- `visipruner_similarity_check` inputs: `t00001478, t00000057, t00001516, t00001517, t00001520, t00001530, t00001529, t00001532, t00001535`
- `visipruner_similarity_check` outputs: `t00001518, t00001528, t00001533, t00001536`
- `attention_output` inputs: `t00001508, t00001474, t00001510, t00001525, t00001537, t00001453, t00001554, t00001555`
- `attention_output` outputs: `t00001509, t00001526, t00001557`
- `mlp` inputs: `t00001512, t00001537, t00001453, t00001547, t00001549, t00001552`
- `mlp` outputs: `t00001554`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer19/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer19/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer19/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.19.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.19.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.19.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.19.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.19.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.19.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.19.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.19.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.19.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.19.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.19.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.19.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.19.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.19.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.19.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.19.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.19.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.19` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00001453']` outputs=`['t00001454']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00001454']` outputs=`['t00001455']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00001455']` outputs=`['t00001456']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00001456']` outputs=`['t00001457']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00001457']` outputs=`['t00001458']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00001454', 't00001458']` outputs=`['t00001459']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00001459']` outputs=`['t00001460']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00001461', 't00001460']` outputs=`['t00001462']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00001462', 't00001463']` outputs=`['t00001464']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00001462', 't00001465']` outputs=`['t00001466']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00001462', 't00001467']` outputs=`['t00001468']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00001464']` outputs=`['t00001469']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00001469']` outputs=`['t00001470']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00001466']` outputs=`['t00001471']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001471']` outputs=`['t00001472']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00001468']` outputs=`['t00001473']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001473']` outputs=`['t00001474']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00001477']` outputs=`['t00001478']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00001480']` outputs=`['t00001481']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00001482']` outputs=`['t00001483']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00001481', 't00001475']` outputs=`['t00001484']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00001484']` outputs=`['t00001485']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00001483', 't00001475']` outputs=`['t00001486']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00001486']` outputs=`['t00001487']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00001470', 't00001485']` outputs=`['t00001488']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00001470']` outputs=`['t00001489']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00001470']` outputs=`['t00001490']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00001490']` outputs=`['t00001491']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00001491', 't00001489']` outputs=`['t00001492']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00001492', 't00001487']` outputs=`['t00001493']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001488', 't00001493']` outputs=`['t00001494']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00001469']` outputs=`['t00001470']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00001471']` outputs=`['t00001472']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00001473']` outputs=`['t00001474']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00001488', 't00001493']` outputs=`['t00001494']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00001495', 't00001500']` outputs=`['t00001501']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00001501']` outputs=`['t00001502']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00001494', 't00001502']` outputs=`['t00001503']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00001503']` outputs=`['t00001504']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00001504', 't00001505']` outputs=`['t00001506']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00001506']` outputs=`['t00001507']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00001508']` outputs=`['t00001508']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00001508', 't00001474']` outputs=`['t00001509']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00001478']` outputs=`['t00001479']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00001479']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00001513']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00001513']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00001516', 't00001517']` outputs=`['t00001518']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00001520']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00001528']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00001530', 't00001529']` outputs=`['t00001531']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00001531', 't00001532']` outputs=`['t00001533']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00001535']` outputs=`['t00001536']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune deep exit check

该层的 Visual 相关过程是 deep exit check：运行时先生成 `P=10` 的 probe index 向量 `t00001528`，再对 probe/reference 表示做 `P×Hidden` 对齐差值 `t00001531`；随后沿 Hidden 维得到每个 probe index 的 score `t00001533`，最后规约为布尔决策 `t00001536`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Probe index axis P=10 (highly compressed to 1 row)      Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
probe index vector     t00001528  P=10                ──▶   [PROBE_INDEX_VECTOR]                    ◀── arange.start 观测到的 P=10 index

                                                            ┌────────────────────────────────────────┐
selected probe rows    t00001530  P=10                ──▶   │ SELECTED_PROBE_ROWS                    │  ◀── 40:1 压缩表示 4096:10
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
reference probe rows   t00001529  P=10                ──▶   │ REFERENCE_PROBE_ROWS                   │  ◀── 与 selected 同一 P×Hidden 坐标对齐
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
delta / compare rows   t00001531  P=10                ──▶   │ PROBE_DELTA_COMPARE                    │  ◀── selected-reference 后沿 Hidden 维比较
                                                            └────────────────────────────────────────┘

Probe score axis P=10 (same 1-row compression; 1-col expanded)
                                                            ┌──────┐
score vector           t00001533  P=10                ──▶   │SCORE │  ◀── 每个 probe index 保留一个 score
                                                            └──────┘
decision scalar        t00001536  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 probe score 做 any 归约
```

Evidence: `#75 arange.start` output=`t00001528`, observed shape=`[10]`; `#78 sub.Tensor` inputs=`[t00001530,t00001529]` output=`t00001531`, observed shape=`[1,10,4096]`; `#80 cosine_similarity.default` inputs=`[t00001531,t00001532]` output=`t00001533`, observed shape=`[1,10]`; `#83 any.default` input=`t00001535` output=`t00001536`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00001508', 't00001474']` outputs=`['t00001509']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00001510']` outputs=`['t00001511']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00001511']` outputs=`['t00001512']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00001525']` outputs=`['t00001526']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00001512', 't00001537']` outputs=`['t00001538']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001453', 't00001538']` outputs=`['t00001539']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00001554', 't00001555']` outputs=`['t00001556']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00001539', 't00001556']` outputs=`['t00001557']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00001512', 't00001537']` outputs=`['t00001538']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00001453', 't00001538']` outputs=`['t00001539']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00001539']` outputs=`['t00001540']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00001540']` outputs=`['t00001541']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00001541']` outputs=`['t00001542']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00001542']` outputs=`['t00001543']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00001543']` outputs=`['t00001544']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00001540', 't00001544']` outputs=`['t00001545']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00001545']` outputs=`['t00001546']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00001547', 't00001546']` outputs=`['t00001548']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00001548', 't00001549']` outputs=`['t00001550']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00001550']` outputs=`['t00001551']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00001548', 't00001552']` outputs=`['t00001553']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00001551', 't00001553']` outputs=`['t00001554']` -> shape=[1, 58, 11008], dtype=float16
