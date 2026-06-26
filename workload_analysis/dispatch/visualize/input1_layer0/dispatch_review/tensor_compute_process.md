# input1_layer0 Dispatch Analysis

## Source

- rows: `91`
- phase: `prefill`
- token_state: `full_visual`
- role: `shallow_or_boundary`

## Original Dimensions Inferred From Dispatch

```json
{
  "seq": 624,
  "kv_len": 624,
  "hidden": 4096,
  "heads": 32,
  "head_dim": 128,
  "ffn": 11008,
  "visual_start": 35,
  "visual_end": 64,
  "tail_start": 611
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
5. `visual_adjust`: Visual attention adjustment: apply the dispatch-proven visual clear/fold variant.
6. `attention_output`: Attention output: attention @ value, merge heads, output projection, residual add.
7. `mlp`: MLP: post RMSNorm, SiLU gate, up projection, gated product, down projection, final residual.

## Top Ops

- `slice.Tensor`: 11
- `mul.Tensor`: 9
- `add.Tensor`: 8
- `linear.default`: 7
- `to.dtype`: 7
- `item.default`: 6
- `transpose.int`: 5
- `gt.Scalar`: 3
- `is_nonzero.default`: 3
- `select.int`: 3
- `view.default`: 3
- `cat.default`: 2
- `index.Tensor`: 2
- `matmul.default`: 2
- `mean.dim`: 2
- `neg.default`: 2
- `pow.Tensor_Scalar`: 2
- `rsqrt.default`: 2
- `unsqueeze.default`: 2
- `contiguous.default`: 1
- `copy_.default`: 1
- `div.Tensor`: 1
- `dropout.default`: 1
- `fill_.Tensor`: 1
- `lift_fresh.default`: 1
