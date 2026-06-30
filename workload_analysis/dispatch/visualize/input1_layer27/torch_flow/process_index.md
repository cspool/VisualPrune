# input1_layer27 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `workload_analysis/dispatch/visualize/input1_layer27/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Dispatch Tensor ID Stage I/O

- `input_rmsnorm` inputs: `t00002271, t00002279`
- `input_rmsnorm` outputs: `t00002280`
- `qkv_projection` inputs: `t00002280, t00002281, t00002283, t00002285`
- `qkv_projection` outputs: `t00002288, t00002290, t00002292`
- `rope` inputs: `t00002294, t00002297, t00002299, t00001475, t00002288`
- `rope` outputs: `t00002295, t00002311`
- `attention` inputs: `t00002287, t00002289, t00002291, t00002305, t00002310, t00002312, t00002317, t00001505`
- `attention` outputs: `t00002288, t00002290, t00002323, t00002325`
- `visipruner_similarity_check` inputs: `t00002295, t00000057, t00002332, t00002333, t00002336, t00002346, t00002345, t00002348, t00002351`
- `visipruner_similarity_check` outputs: `t00002334, t00002344, t00002349, t00002352`
- `attention_output` inputs: `t00002324, t00002292, t00002326, t00002341, t00002353, t00002271, t00002370, t00002371`
- `attention_output` outputs: `t00002325, t00002342, t00002373`
- `mlp` inputs: `t00002328, t00002353, t00002271, t00002363, t00002365, t00002368`
- `mlp` outputs: `t00002370`

## Complete Dispatch Op Coverage

- coverage json: `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/dispatch_op_coverage.json`
- coverage csv: `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/dispatch_op_coverage.csv`
- coverage markdown: `workload_analysis/dispatch/visualize/input1_layer27/dispatch_review/dispatch_op_coverage.md`
- ops in dispatch rows: `100`
- ops listed in coverage: `100`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

| # | Op | Runtime subprocess | Module split | Tensor dataflow | Stage evidence |
|---:|---|---|---|---|---|
| 1 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 2 | `pow.Tensor_Scalar` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 3 | `mean.dim` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 4 | `add.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 5 | `rsqrt.default` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 6 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 7 | `to.dtype` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 8 | `mul.Tensor` | `model.layers.27.input_layernorm` | `True` | `True` | `input_rmsnorm` |
| 9 | `linear.default` | `model.layers.27.self_attn.q_proj` | `True` | `True` | `qkv_projection` |
| 10 | `linear.default` | `model.layers.27.self_attn.k_proj` | `True` | `True` | `qkv_projection` |
| 11 | `linear.default` | `model.layers.27.self_attn.v_proj` | `True` | `True` | `qkv_projection` |
| 12 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 13 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 14 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 15 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 16 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection` |
| 17 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `qkv_projection, attention` |
| 18 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 19 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 20 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 21 | `gt.Scalar` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 22 | `is_nonzero.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `visipruner_similarity_check` |
| 23 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 24 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 25 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 26 | `item.default` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 27 | `slice.Tensor` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `rope` |
| 28 | `to.dtype` | `model.layers.27.self_attn.rotary_emb` | `True` | `True` | `` |
| 29 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 30 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 31 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 32 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 33 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 34 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 35 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 36 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 37 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 38 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope` |
| 39 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `rope, attention` |
| 40 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 41 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 42 | `slice.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 43 | `neg.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 44 | `cat.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 45 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 46 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 47 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 48 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 49 | `div.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 50 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 51 | `softmax.int` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 52 | `to.dtype` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 53 | `dropout.default` | `model.layers.27.self_attn` | `True` | `True` | `attention` |
| 54 | `matmul.default` | `model.layers.27.self_attn` | `True` | `True` | `attention, attention_output` |
| 55 | `transpose.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 56 | `contiguous.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` |
| 57 | `reshape.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` |
| 58 | `gt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 59 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 60 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 61 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 62 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 63 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 64 | `sub.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 65 | `add.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 66 | `eq.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 67 | `is_nonzero.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 68 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 69 | `select.int` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 70 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 71 | `mul.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 72 | `permute.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 73 | `contiguous.default` | `model.layers.27.self_attn` | `True` | `True` | `attention_output` |
| 74 | `view.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 75 | `arange.start` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 76 | `index.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 77 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 78 | `sub.Tensor` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 79 | `unsqueeze.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 80 | `cosine_similarity.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 81 | `squeeze.dim` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 82 | `lt.Scalar` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 83 | `any.default` | `model.layers.27.self_attn` | `True` | `True` | `visipruner_similarity_check` |
| 84 | `item.default` | `model.layers.27.self_attn` | `True` | `True` | `` |
| 85 | `linear.default` | `model.layers.27.self_attn.o_proj` | `True` | `True` | `attention_output, mlp` |
| 86 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output, mlp` |
| 87 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 88 | `pow.Tensor_Scalar` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 89 | `mean.dim` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 90 | `add.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 91 | `rsqrt.default` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 92 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 93 | `to.dtype` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 94 | `mul.Tensor` | `model.layers.27.post_attention_layernorm` | `True` | `True` | `mlp` |
| 95 | `linear.default` | `model.layers.27.mlp.gate_proj` | `True` | `True` | `mlp` |
| 96 | `silu.default` | `model.layers.27.mlp.act_fn` | `True` | `True` | `mlp` |
| 97 | `linear.default` | `model.layers.27.mlp.up_proj` | `True` | `True` | `mlp` |
| 98 | `mul.Tensor` | `model.layers.27.mlp` | `True` | `True` | `mlp` |
| 99 | `linear.default` | `model.layers.27.mlp.down_proj` | `True` | `True` | `attention_output` |
| 100 | `add.Tensor` | `model.layers.27` | `True` | `True` | `attention_output` |

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` inputs=`['t00002271']` outputs=`['t00002272']` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` inputs=`['t00002272']` outputs=`['t00002273']` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` inputs=`['t00002273']` outputs=`['t00002274']` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` inputs=`['t00002274']` outputs=`['t00002275']` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` inputs=`['t00002275']` outputs=`['t00002276']` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` inputs=`['t00002272', 't00002276']` outputs=`['t00002277']` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` inputs=`['t00002277']` outputs=`['t00002278']` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` inputs=`['t00002279', 't00002278']` outputs=`['t00002280']` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` inputs=`['t00002280', 't00002281']` outputs=`['t00002282']` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` inputs=`['t00002280', 't00002283']` outputs=`['t00002284']` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` inputs=`['t00002280', 't00002285']` outputs=`['t00002286']` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` inputs=`['t00002282']` outputs=`['t00002287']` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` inputs=`['t00002287']` outputs=`['t00002288']` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` inputs=`['t00002284']` outputs=`['t00002289']` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002289']` outputs=`['t00002290']` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` inputs=`['t00002286']` outputs=`['t00002291']` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002291']` outputs=`['t00002292']` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` inputs=`['t00002294']` outputs=`['t00002295']` -> shape=[], dtype=int64
- `#24 slice.Tensor` inputs=`['t00002297']` outputs=`['t00002298']` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` inputs=`['t00002299']` outputs=`['t00002300']` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` inputs=`['t00002298', 't00001475']` outputs=`['t00002301']` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` inputs=`['t00002301']` outputs=`['t00002302']` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` inputs=`['t00002300', 't00001475']` outputs=`['t00002303']` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` inputs=`['t00002303']` outputs=`['t00002304']` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` inputs=`['t00002288', 't00002302']` outputs=`['t00002305']` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` inputs=`['t00002288']` outputs=`['t00002306']` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` inputs=`['t00002288']` outputs=`['t00002307']` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` inputs=`['t00002307']` outputs=`['t00002308']` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` inputs=`['t00002308', 't00002306']` outputs=`['t00002309']` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` inputs=`['t00002309', 't00002304']` outputs=`['t00002310']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002305', 't00002310']` outputs=`['t00002311']` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` inputs=`['t00002287']` outputs=`['t00002288']` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` inputs=`['t00002289']` outputs=`['t00002290']` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` inputs=`['t00002291']` outputs=`['t00002292']` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` inputs=`['t00002305', 't00002310']` outputs=`['t00002311']` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` inputs=`['t00002312', 't00002317']` outputs=`['t00002318']` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` inputs=`['t00002318']` outputs=`['t00002319']` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` inputs=`['t00002311', 't00002319']` outputs=`['t00002320']` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` inputs=`['t00002320']` outputs=`['t00002321']` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` inputs=`['t00002321', 't00001505']` outputs=`['t00002322']` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` inputs=`['t00002322']` outputs=`['t00002323']` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` inputs=`['t00002324']` outputs=`['t00002324']` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` inputs=`['t00002324', 't00002292']` outputs=`['t00002325']` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` inputs=`['t00002295']` outputs=`['t00002296']` -> shape=[], dtype=bool
- `#22 is_nonzero.default` inputs=`['t00002296']` outputs=`[]` -> False
- `#58 gt.Scalar` inputs=`['t00000057']` outputs=`['t00002329']` -> shape=[], dtype=bool
- `#59 is_nonzero.default` inputs=`['t00002329']` outputs=`[]` -> True
- `#64 sub.Tensor` inputs=`['t00002332', 't00002333']` outputs=`['t00002334']` -> shape=[], dtype=int64
- `#67 is_nonzero.default` inputs=`['t00002336']` outputs=`[]` -> True
- `#75 arange.start` inputs=`[]` outputs=`['t00002344']` -> shape=[10], dtype=int64
- `#78 sub.Tensor` inputs=`['t00002346', 't00002345']` outputs=`['t00002347']` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` inputs=`['t00002347', 't00002348']` outputs=`['t00002349']` -> shape=[1, 10], dtype=float16
- `#83 any.default` inputs=`['t00002351']` outputs=`['t00002352']` -> shape=[], dtype=bool

### `attention_output`
#### Attn 输出 Visual 相关处理字符画：VisiPrune deep exit check

该层的 Visual 相关过程是 deep exit check：运行时先生成 `P=10` 的 probe index 向量 `t00002344`，再对 probe/reference 表示做 `P×Hidden` 对齐差值 `t00002347`；随后沿 Hidden 维得到每个 probe index 的 score `t00002349`，最后规约为布尔决策 `t00002352`。普通 Attn 输出路径只保留 evidence rows，不在这里画。

```text
Probe index axis P=10 (highly compressed to 1 row)      Hidden dimension
                                                            0                                      4096
                                                            ▲                                        ▲
probe index vector     t00002344  P=10                ──▶   [PROBE_INDEX_VECTOR]                    ◀── arange.start 观测到的 P=10 index

                                                            ┌────────────────────────────────────────┐
selected probe rows    t00002346  P=10                ──▶   │ SELECTED_PROBE_ROWS                    │  ◀── 40:1 压缩表示 4096:10
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
reference probe rows   t00002345  P=10                ──▶   │ REFERENCE_PROBE_ROWS                   │  ◀── 与 selected 同一 P×Hidden 坐标对齐
                                                            └────────────────────────────────────────┘
                                                            ┌────────────────────────────────────────┐
delta / compare rows   t00002347  P=10                ──▶   │ PROBE_DELTA_COMPARE                    │  ◀── selected-reference 后沿 Hidden 维比较
                                                            └────────────────────────────────────────┘

Probe score axis P=10 (same 1-row compression; 1-col expanded)
                                                            ┌──────┐
score vector           t00002349  P=10                ──▶   │SCORE │  ◀── 每个 probe index 保留一个 score
                                                            └──────┘
decision scalar        t00002352  shape=[]            ──▶   [REDUCED_BOOL_DECISION]                  ◀── 对 probe score 做 any 归约
```

Evidence: `#75 arange.start` output=`t00002344`, observed shape=`[10]`; `#78 sub.Tensor` inputs=`[t00002346,t00002345]` output=`t00002347`, observed shape=`[1,10,4096]`; `#80 cosine_similarity.default` inputs=`[t00002347,t00002348]` output=`t00002349`, observed shape=`[1,10]`; `#83 any.default` input=`t00002351` output=`t00002352`, observed shape=`[]`.

- `#54 matmul.default` inputs=`['t00002324', 't00002292']` outputs=`['t00002325']` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` inputs=`['t00002326']` outputs=`['t00002327']` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` inputs=`['t00002327']` outputs=`['t00002328']` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` inputs=`['t00002341']` outputs=`['t00002342']` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` inputs=`['t00002328', 't00002353']` outputs=`['t00002354']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002271', 't00002354']` outputs=`['t00002355']` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` inputs=`['t00002370', 't00002371']` outputs=`['t00002372']` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` inputs=`['t00002355', 't00002372']` outputs=`['t00002373']` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` inputs=`['t00002328', 't00002353']` outputs=`['t00002354']` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` inputs=`['t00002271', 't00002354']` outputs=`['t00002355']` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` inputs=`['t00002355']` outputs=`['t00002356']` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` inputs=`['t00002356']` outputs=`['t00002357']` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` inputs=`['t00002357']` outputs=`['t00002358']` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` inputs=`['t00002358']` outputs=`['t00002359']` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` inputs=`['t00002359']` outputs=`['t00002360']` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` inputs=`['t00002356', 't00002360']` outputs=`['t00002361']` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` inputs=`['t00002361']` outputs=`['t00002362']` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` inputs=`['t00002363', 't00002362']` outputs=`['t00002364']` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` inputs=`['t00002364', 't00002365']` outputs=`['t00002366']` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` inputs=`['t00002366']` outputs=`['t00002367']` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` inputs=`['t00002364', 't00002368']` outputs=`['t00002369']` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` inputs=`['t00002367', 't00002369']` outputs=`['t00002370']` -> shape=[1, 58, 11008], dtype=float16
