# input2_layer27 Reconstruction Review

This file reviews the generated small-tensor reconstruction against this layer's real dispatch trace.

## Dispatch Evidence

- rows: `76`
- phase: `decode`
- role: `decode_prune_effect`
- token_state: `middle_pruned_cache`
- q_len: `1`
- kv_len: `59`
- hidden: `4096`
- heads: `32`
- head_dim: `128`
- ffn: `11008`

## Dispatch-Derived Expected Stages

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Generated ONNX Stages

- `input_rmsnorm`: 01_input_rmsnorm.onnx: inputs={'hidden_states': [1, 32]}, outputs={'x_norm': [1, 32], 'variance': [1, 1], 'inv_rms': [1, 1]}
- `qkv_projection`: 02_qkv_projection.onnx: inputs={'x_norm': [1, 32]}, outputs={'q_heads': [4, 1, 8], 'k_heads': [4, 1, 8], 'v_heads': [4, 1, 8]}
- `rope`: 03_rope.onnx: inputs={'q_heads': [4, 1, 8], 'k_heads': [4, 1, 8], 'position_ids': [1]}, outputs={'q_rope': [4, 1, 8], 'k_current_rope': [4, 1, 8]}
- `kv_cache_concat`: 04_kv_cache_concat.onnx: inputs={'k_current_rope': [4, 1, 8], 'v_current': [4, 1, 8], 'past_k': [4, 15, 8], 'past_v': [4, 15, 8]}, outputs={'k_heads': [4, 16, 8], 'v_heads': [4, 16, 8]}
- `attention`: 05_attention.onnx: inputs={'q_rope': [4, 1, 8], 'k_heads': [4, 16, 8], 'attention_mask': [1, 1, 16]}, outputs={'raw_scores': [4, 1, 16], 'masked_scores': [4, 1, 16], 'attn': [4, 1, 16]}
- `attention_output`: 06_attention_output.onnx: inputs={'adjusted_attn': [4, 1, 16], 'v_heads': [4, 16, 8], 'hidden_states': [1, 32]}, outputs={'context': [1, 32], 'attn_out': [1, 32], 'after_attn': [1, 32]}
- `mlp`: 07_mlp.onnx: inputs={'after_attn': [1, 32]}, outputs={'gated': [1, 64], 'ffn_out': [1, 32], 'output': [1, 32]}
- `full_flow`: 08_full_flow.onnx: inputs={'hidden_states': [1, 32], 'position_ids': [1], 'attention_mask': [1, 1, 16], 'past_k': [4, 15, 8], 'past_v': [4, 15, 8]}, outputs={'output': [1, 32]}

## Generated Process Code

- `dispatch_reconstruction`: `True`
- `toy_tensor_compute`: `True`
- `layer_profile`: `True`
- `process_index`: `True`

## Alignment

- split_alignment: `aligned`
- process_alignment: `aligned`

## Verdict

- status: `match`

## Core Dispatch Evidence

### `input_rmsnorm`

- dispatch_supported: `True`
- summary: RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence.
- evidence ops:
  - `#1 to.dtype` -> shape=[1, 1, 4096], dtype=float32
  - `#2 pow.Tensor_Scalar` -> shape=[1, 1, 4096], dtype=float32
  - `#3 mean.dim` -> shape=[1, 1, 1], dtype=float32
  - `#4 add.Tensor` -> shape=[1, 1, 1], dtype=float32
  - `#5 rsqrt.default` -> shape=[1, 1, 1], dtype=float32
  - `#6 mul.Tensor` -> shape=[1, 1, 4096], dtype=float32
  - `#7 to.dtype` -> shape=[1, 1, 4096], dtype=float16
  - `#8 mul.Tensor` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`

- dispatch_supported: `True`
- summary: Q/K/V evidence is three hidden-size linear projections followed by view/transpose head split.
- evidence ops:
  - `#9 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#10 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#11 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#12 view.default` -> shape=[1, 1, 32, 128], dtype=float16
  - `#13 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
  - `#14 view.default` -> shape=[1, 1, 32, 128], dtype=float16
  - `#15 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
  - `#16 view.default` -> shape=[1, 1, 32, 128], dtype=float16
  - `#17 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16

### `rope`

- dispatch_supported: `True`
- summary: RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add.
- evidence ops:
  - `#20 add.Tensor` -> shape=[], dtype=int64
  - `#24 slice.Tensor` -> shape=[625, 128], dtype=float16
  - `#27 slice.Tensor` -> shape=[625, 128], dtype=float16
  - `#29 index.Tensor` -> shape=[1, 1, 128], dtype=float16
  - `#30 unsqueeze.default` -> shape=[1, 1, 1, 128], dtype=float16
  - `#31 index.Tensor` -> shape=[1, 1, 128], dtype=float16
  - `#32 unsqueeze.default` -> shape=[1, 1, 1, 128], dtype=float16
  - `#33 mul.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
  - `#34 slice.Tensor` -> shape=[1, 32, 1, 64], dtype=float16
  - `#35 slice.Tensor` -> shape=[1, 32, 1, 64], dtype=float16
  - `#36 neg.default` -> shape=[1, 32, 1, 64], dtype=float16
  - `#37 cat.default` -> shape=[1, 32, 1, 128], dtype=float16
  - `#38 mul.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
  - `#39 add.Tensor` -> shape=[1, 32, 1, 128], dtype=float16

### `kv_cache_concat`

- dispatch_supported: `True`
- summary: Decode cache evidence is K/V cat outputs whose sequence dimension equals dispatch kv_len.
- evidence ops:
  - `#47 cat.default` -> shape=[1, 32, 59, 128], dtype=float16
  - `#48 cat.default` -> shape=[1, 32, 59, 128], dtype=float16

### `attention`

- dispatch_supported: `True`
- summary: Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores.
- evidence ops:
  - `#13 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
  - `#15 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
  - `#17 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
  - `#39 add.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
  - `#46 add.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
  - `#49 transpose.int` -> shape=[1, 32, 128, 59], dtype=float16
  - `#50 matmul.default` -> shape=[1, 32, 1, 59], dtype=float16
  - `#51 div.Tensor` -> shape=[1, 32, 1, 59], dtype=float16
  - `#52 add.Tensor` -> shape=[1, 32, 1, 59], dtype=float16
  - `#53 softmax.int` -> shape=[1, 32, 1, 59], dtype=float32
  - `#55 dropout.default` -> shape=[1, 32, 1, 59], dtype=float16
  - `#56 matmul.default` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`

- dispatch_supported: `True`
- summary: Attention-output evidence is attn @ V, transpose/reshape, output projection, and residual add.
- evidence ops:
  - `#56 matmul.default` -> shape=[1, 32, 1, 128], dtype=float16
  - `#58 reshape.default` -> shape=[1, 1, 4096], dtype=float16
  - `#61 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#62 add.Tensor` -> shape=[1, 1, 4096], dtype=float16
  - `#75 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#76 add.Tensor` -> shape=[1, 1, 4096], dtype=float16

### `mlp`

- dispatch_supported: `True`
- summary: MLP evidence is post-attention RMSNorm, gate/up linear, SiLU, gated product, down linear, residual add.
- evidence ops:
  - `#54 to.dtype` -> shape=[1, 32, 1, 59], dtype=float16
  - `#61 linear.default` -> shape=[1, 1, 4096], dtype=float16
  - `#62 add.Tensor` -> shape=[1, 1, 4096], dtype=float16
  - `#63 to.dtype` -> shape=[1, 1, 4096], dtype=float32
  - `#64 pow.Tensor_Scalar` -> shape=[1, 1, 4096], dtype=float32
  - `#65 mean.dim` -> shape=[1, 1, 1], dtype=float32
  - `#66 add.Tensor` -> shape=[1, 1, 1], dtype=float32
  - `#67 rsqrt.default` -> shape=[1, 1, 1], dtype=float32
  - `#68 mul.Tensor` -> shape=[1, 1, 4096], dtype=float32
  - `#69 to.dtype` -> shape=[1, 1, 4096], dtype=float16
  - `#70 mul.Tensor` -> shape=[1, 1, 4096], dtype=float16
  - `#71 linear.default` -> shape=[1, 1, 11008], dtype=float16
  - `#72 silu.default` -> shape=[1, 1, 11008], dtype=float16
  - `#73 linear.default` -> shape=[1, 1, 11008], dtype=float16

