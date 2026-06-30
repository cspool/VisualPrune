# input1_layer26 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer26/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002169, t00002177`
- `input_rmsnorm` outputs: `t00002178`
- `qkv_projection` inputs: `t00002178, t00002179, t00002181, t00002183`
- `qkv_projection` outputs: `t00002186, t00002188, t00002190`
- `rope` inputs: `t00002192, t00002195, t00002197, t00001475, t00002186`
- `rope` outputs: `t00002193, t00002209`
- `attention` inputs: `t00002185, t00002187, t00002189, t00002203, t00002208, t00002210, t00002215, t00001505`
- `attention` outputs: `t00002186, t00002188, t00002221, t00002223`
- `visipruner_similarity_check` inputs: `t00002193, t00000057, t00002230, t00002231, t00002234, t00002244, t00002243, t00002246, t00002249`
- `visipruner_similarity_check` outputs: `t00002232, t00002242, t00002247, t00002250`
- `attention_output` inputs: `t00002222, t00002190, t00002224, t00002239, t00002251, t00002169, t00002268, t00002269`
- `attention_output` outputs: `t00002223, t00002240, t00002271`
- `mlp` inputs: `t00002226, t00002251, t00002169, t00002261, t00002263, t00002266`
- `mlp` outputs: `t00002268`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer26/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer26/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer26/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.26.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.26.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.26.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.26.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.26.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.26.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.26.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.26.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.26.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.26.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.26.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.26` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.26.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.26.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.26.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.26.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.26.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.26.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.26` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002169']` outputs=`['t00002170']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002170']` outputs=`['t00002171']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002171']` outputs=`['t00002172']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002172']` outputs=`['t00002173']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002173']` outputs=`['t00002174']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002170', 't00002174']` outputs=`['t00002175']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002175']` outputs=`['t00002176']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002177', 't00002176']` outputs=`['t00002178']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002178', 't00002179']` outputs=`['t00002180']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002178', 't00002181']` outputs=`['t00002182']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002178', 't00002183']` outputs=`['t00002184']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002180']` outputs=`['t00002185']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002185']` outputs=`['t00002186']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00002182']` outputs=`['t00002187']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002187']` outputs=`['t00002188']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00002184']` outputs=`['t00002189']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002189']` outputs=`['t00002190']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002192']` outputs=`['t00002193']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002195']` outputs=`['t00002196']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002197']` outputs=`['t00002198']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002196', 't00001475']` outputs=`['t00002199']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002199']` outputs=`['t00002200']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002198', 't00001475']` outputs=`['t00002201']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002201']` outputs=`['t00002202']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002186', 't00002200']` outputs=`['t00002203']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002186']` outputs=`['t00002204']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002186']` outputs=`['t00002205']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002205']` outputs=`['t00002206']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002206', 't00002204']` outputs=`['t00002207']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002207', 't00002202']` outputs=`['t00002208']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002203', 't00002208']` outputs=`['t00002209']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002185']` outputs=`['t00002186']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002187']` outputs=`['t00002188']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002189']` outputs=`['t00002190']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002203', 't00002208']` outputs=`['t00002209']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002210', 't00002215']` outputs=`['t00002216']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002216']` outputs=`['t00002217']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00002209', 't00002217']` outputs=`['t00002218']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00002218']` outputs=`['t00002219']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00002219', 't00001505']` outputs=`['t00002220']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00002220']` outputs=`['t00002221']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00002222']` outputs=`['t00002222']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00002222', 't00002190']` outputs=`['t00002223']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00002193']` outputs=`['t00002194']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00002194']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00002227']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00002227']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00002230', 't00002231']` outputs=`['t00002232']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00002234']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00002242']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00002244', 't00002243']` outputs=`['t00002245']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00002245', 't00002246']` outputs=`['t00002247']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00002249']` outputs=`['t00002250']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune deep exit check

该层的 Visual 相关过程是 deep exit check：运行时先生成 `P=10` 的 probe index 向量 `t00002242`，再对 probe/reference 表示做 `P×Hidden` 对齐差值 `t00002245`；随后沿 Hidden 维得到每个 probe index 的 score `t00002247`，最后规约为布尔决策 `t00002250`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Probe index axis P=10 (highly compressed to 1 row)      Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
probe index vector     t00002242  P=10                ──▶   [PROBE_INDEX_VECTOR]                    ◀── arange.start 观测到的 P=10 index

                                                            ┌────────────────────────────────────────┐
selected probe rows    t00002244  P=10                ──▶   │ SELECTED_PROBE_ROWS                    │  ◀── 40:1 压缩表示 4096:10
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
reference probe rows   t00002243  P=10                ──▶   │ REFERENCE_PROBE_ROWS                   │  ◀── 与 selected 同一 P×Hidden 坐标对齐
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
delta / compare rows   t00002245  P=10                ──▶   │ PROBE_DELTA_COMPARE                    │  ◀── selected-reference 后沿 Hidden 维比较
                                                            └────────────────────────────────────────┘

Probe score axis P=10 (same 1-row compression; 1-col expanded)
                                                            ┌──────┐
score vector           t00002247  P=10                ──▶   │SCORE │  ◀── 每个 probe index 保留一个 score
                                                            └──────┘
decision scalar        t00002250  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 probe score 做 any 归约
```

Evidence: `#75 arange.start` output=`t00002242`, observed shape=`[10]`; `#78 sub.Tensor` inputs=`[t00002244,t00002243]` output=`t00002245`, observed shape=`[1,10,4096]`; `#80 cosine_similarity.default` inputs=`[t00002245,t00002246]` output=`t00002247`, observed shape=`[1,10]`; `#83 any.default` input=`t00002249` output=`t00002250`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00002222', 't00002190']` outputs=`['t00002223']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002224']` outputs=`['t00002225']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002225']` outputs=`['t00002226']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00002239']` outputs=`['t00002240']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00002226', 't00002251']` outputs=`['t00002252']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002169', 't00002252']` outputs=`['t00002253']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00002268', 't00002269']` outputs=`['t00002270']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00002253', 't00002270']` outputs=`['t00002271']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00002226', 't00002251']` outputs=`['t00002252']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002169', 't00002252']` outputs=`['t00002253']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00002253']` outputs=`['t00002254']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00002254']` outputs=`['t00002255']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00002255']` outputs=`['t00002256']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00002256']` outputs=`['t00002257']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00002257']` outputs=`['t00002258']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00002254', 't00002258']` outputs=`['t00002259']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00002259']` outputs=`['t00002260']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00002261', 't00002260']` outputs=`['t00002262']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00002262', 't00002263']` outputs=`['t00002264']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00002264']` outputs=`['t00002265']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00002262', 't00002266']` outputs=`['t00002267']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00002265', 't00002267']` outputs=`['t00002268']` -> shape=[1, 58, 11008], dtype=float16
