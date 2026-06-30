# input1_layer28 Dispatch Analysis

## Source

- rows: `83`
- phase: `prefill`
- token_state: `deep_removed`
- role: `boundary_after_prune`

## Original Dimensions Inferred From Dispatch

```json
{
  "seq": 48,
  "kv_len": 48,
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
5. `attention_output`: Attention output: attention @ value, merge heads, output projection, residual add.
6. `mlp`: MLP: post RMSNorm, SiLU gate, up projection, gated product, down projection, final residual.

## Tensor ID Data Dependencies

Data dependencies are derived from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- tensor-id producer-consumer edges: `91`
- external input tensor ids: `['t00002374', 't00002382', 't00002384', 't00002386', 't00002388', 't00002396', 't00002401', 't00002403', 't00002426', 't00000057', 't00002442', 't00002452', 't00002454', 't00002457', 't00002460']`
- final output tensor ids: `['t00002462']`

## Dispatch Op Coverage

Every dispatch op row must be listed in `dispatch_op_coverage.*` and covered by runtime module split plus tensor-id dataflow.

- ops in dispatch rows: `83`
- ops listed in coverage: `83`
- missing event_op_index values: `[]`
- missing from module_split: `[]`
- missing from tensor_dataflow: `[]`

## Top Ops

- `add.Tensor`: 10
- `mul.Tensor`: 10
- `linear.default`: 7
- `to.dtype`: 7
- `slice.Tensor`: 6
- `transpose.int`: 5
- `select.int`: 4
- `is_nonzero.default`: 3
- `view.default`: 3
- `cat.default`: 2
- `gt.Scalar`: 2
- `index.Tensor`: 2
- `item.default`: 2
- `matmul.default`: 2
- `mean.dim`: 2
- `neg.default`: 2
- `pow.Tensor_Scalar`: 2
- `rsqrt.default`: 2
- `unsqueeze.default`: 2
- `contiguous.default`: 1
- `div.Tensor`: 1
- `dropout.default`: 1
- `eq.Scalar`: 1
- `reshape.default`: 1
- `silu.default`: 1
