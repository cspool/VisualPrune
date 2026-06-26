# input1_layer18 Dispatch Analysis

## Source

- rows: `104`
- phase: `prefill`
- token_state: `full_visual`
- role: `middle_select;boundary_before_prune`

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

## Top Ops

- `add.Tensor`: 10
- `mul.Tensor`: 10
- `linear.default`: 7
- `slice.Tensor`: 7
- `to.dtype`: 7
- `select.int`: 6
- `unsqueeze.default`: 6
- `transpose.int`: 5
- `item.default`: 4
- `view.default`: 4
- `gt.Scalar`: 3
- `is_nonzero.default`: 3
- `cat.default`: 2
- `contiguous.default`: 2
- `index.Tensor`: 2
- `matmul.default`: 2
- `mean.dim`: 2
- `neg.default`: 2
- `pow.Tensor_Scalar`: 2
- `rsqrt.default`: 2
- `squeeze.dim`: 2
- `sub.Tensor`: 2
- `any.default`: 1
- `cosine_similarity.default`: 1
- `div.Tensor`: 1
