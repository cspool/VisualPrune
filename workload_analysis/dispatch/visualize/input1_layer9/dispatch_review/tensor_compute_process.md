# input1_layer9 Dispatch Analysis

## Source

- rows: `97`
- phase: `prefill`
- token_state: `full_visual`
- role: `middle_probe`

## Original Dimensions Inferred From Dispatch

```json
{
  "seq": 624,
  "kv_len": 624,
  "hidden": 4096,
  "heads": 32,
  "head_dim": 128,
  "ffn": 11008,
  "visual_start": null,
  "visual_end": null,
  "tail_start": null
}
```

## Small Tensor Config

```json
{
  "seq": 16,
  "q_seq": 16,
  "kv_seq": 16,
  "hidden": 32,
  "heads": 4,
  "head_dim": 8,
  "visual_start": 3,
  "visual_end": 13,
  "tail_start": 13,
  "ffn": 64
}
```

## Reconstructed Compute Stages

1. `input_rmsnorm`: Input RMSNorm: cast/square/mean/rsqrt/multiply weight.
2. `qkv_projection`: Q/K/V projection: linear projections and head split.
3. `rope`: RoPE: gather cos/sin by position ids and rotate half channels.
4. `attention`: Attention: q @ k^T, scale, add mask, softmax.
5. `visipruner_similarity_check`: VisiPrune similarity/check: cosine-similarity probe or check sub-process.
6. `attention_output`: Attention output: attention @ value, merge heads, output projection, residual add.
7. `mlp`: MLP: post RMSNorm, SiLU gate, up projection, gated product, down projection, final residual.

## Tensor ID Data Dependencies

Data dependencies are derived from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- tensor-id producer-consumer edges: `107`
- external input tensor ids: `['t00000455', 't00000463', 't00000465', 't00000467', 't00000469', 't00000023', 't00000481', 't00000483', 't00000053', 't00000057', 't00000525', 't00000534', 't00000544', 't00000546', 't00000549', 't00000552']`
- final output tensor ids: `['t00000554']`

## Dispatch Op Coverage

Every dispatch op row must be listed in `dispatch_op_coverage.*` and covered by runtime module split plus tensor-id dataflow.

- ops in dispatch rows: `97`
- ops listed in coverage: `97`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

## Top Ops

- `mul.Tensor`: 10
- `add.Tensor`: 9
- `linear.default`: 7
- `slice.Tensor`: 7
- `to.dtype`: 7
- `select.int`: 6
- `transpose.int`: 5
- `unsqueeze.default`: 5
- `item.default`: 4
- `view.default`: 4
- `is_nonzero.default`: 3
- `cat.default`: 2
- `contiguous.default`: 2
- `gt.Scalar`: 2
- `index.Tensor`: 2
- `matmul.default`: 2
- `mean.dim`: 2
- `neg.default`: 2
- `pow.Tensor_Scalar`: 2
- `rsqrt.default`: 2
- `any.default`: 1
- `cosine_similarity.default`: 1
- `div.Tensor`: 1
- `dropout.default`: 1
- `eq.Scalar`: 1
