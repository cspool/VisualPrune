# input1_layer28 Reconstruction Review

This file reviews the generated small-tensor reconstruction against this layer's real dispatch trace.

## Dispatch Evidence

- rows: `83`
- phase: `prefill`
- role: `boundary_after_prune`
- token_state: `deep_removed`
- q_len: `48`
- kv_len: `48`
- hidden: `4096`
- heads: `32`
- head_dim: `128`
- ffn: `11008`

## Dispatch-Derived Expected Stages

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `attention_output`
6. `mlp`

## Generated ONNX Stages

- `input_rmsnorm`: 01_input_rmsnorm.onnx: inputs={'hidden_states': [16, 32]}, outputs={'x_norm': [16, 32], 'variance': [16, 1], 'inv_rms': [16, 1]}
- `qkv_projection`: 02_qkv_projection.onnx: inputs={'x_norm': [16, 32]}, outputs={'q_heads': [4, 16, 8], 'k_heads': [4, 16, 8], 'v_heads': [4, 16, 8]}
- `rope`: 03_rope.onnx: inputs={'q_heads': [4, 16, 8], 'k_heads': [4, 16, 8], 'position_ids': [16]}, outputs={'q_rope': [4, 16, 8], 'k_current_rope': [4, 16, 8]}
- `attention`: 04_attention.onnx: inputs={'q_rope': [4, 16, 8], 'k_heads': [4, 16, 8], 'attention_mask': [1, 16, 16]}, outputs={'raw_scores': [4, 16, 16], 'masked_scores': [4, 16, 16], 'attn': [4, 16, 16]}
- `attention_output`: 05_attention_output.onnx: inputs={'adjusted_attn': [4, 16, 16], 'v_heads': [4, 16, 8], 'hidden_states': [16, 32]}, outputs={'context': [16, 32], 'attn_out': [16, 32], 'after_attn': [16, 32]}
- `mlp`: 06_mlp.onnx: inputs={'after_attn': [16, 32]}, outputs={'gated': [16, 64], 'ffn_out': [16, 32], 'output': [16, 32]}
- `full_flow`: 07_full_flow.onnx: inputs={'hidden_states': [16, 32], 'position_ids': [16], 'attention_mask': [1, 16, 16]}, outputs={'output': [16, 32]}

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
  - `#1 to.dtype` -> shape=[1, 48, 4096], dtype=float32
  - `#2 pow.Tensor_Scalar` -> shape=[1, 48, 4096], dtype=float32
  - `#3 mean.dim` -> shape=[1, 48, 1], dtype=float32
  - `#4 add.Tensor` -> shape=[1, 48, 1], dtype=float32
  - `#5 rsqrt.default` -> shape=[1, 48, 1], dtype=float32
  - `#6 mul.Tensor` -> shape=[1, 48, 4096], dtype=float32
  - `#7 to.dtype` -> shape=[1, 48, 4096], dtype=float16
  - `#8 mul.Tensor` -> shape=[1, 48, 4096], dtype=float16

### `qkv_projection`

- dispatch_supported: `True`
- summary: Q/K/V evidence is three hidden-size linear projections followed by view/transpose head split.
- evidence ops:
  - `#9 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#10 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#11 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#12 view.default` -> shape=[1, 48, 32, 128], dtype=float16
  - `#13 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16
  - `#14 view.default` -> shape=[1, 48, 32, 128], dtype=float16
  - `#15 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16
  - `#16 view.default` -> shape=[1, 48, 32, 128], dtype=float16
  - `#17 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16

### `rope`

- dispatch_supported: `True`
- summary: RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add.
- evidence ops:
  - `#20 add.Tensor` -> shape=[], dtype=int64
  - `#24 slice.Tensor` -> shape=[624, 128], dtype=float16
  - `#27 slice.Tensor` -> shape=[624, 128], dtype=float16
  - `#29 index.Tensor` -> shape=[1, 48, 128], dtype=float16
  - `#30 unsqueeze.default` -> shape=[1, 1, 48, 128], dtype=float16
  - `#31 index.Tensor` -> shape=[1, 48, 128], dtype=float16
  - `#32 unsqueeze.default` -> shape=[1, 1, 48, 128], dtype=float16
  - `#33 mul.Tensor` -> shape=[1, 32, 48, 128], dtype=float16
  - `#34 slice.Tensor` -> shape=[1, 32, 48, 64], dtype=float16
  - `#35 slice.Tensor` -> shape=[1, 32, 48, 64], dtype=float16
  - `#36 neg.default` -> shape=[1, 32, 48, 64], dtype=float16
  - `#37 cat.default` -> shape=[1, 32, 48, 128], dtype=float16
  - `#38 mul.Tensor` -> shape=[1, 32, 48, 128], dtype=float16
  - `#39 add.Tensor` -> shape=[1, 32, 48, 128], dtype=float16

### `attention`

- dispatch_supported: `True`
- summary: Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores.
- evidence ops:
  - `#13 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16
  - `#15 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16
  - `#17 transpose.int` -> shape=[1, 32, 48, 128], dtype=float16
  - `#39 add.Tensor` -> shape=[1, 32, 48, 128], dtype=float16
  - `#46 add.Tensor` -> shape=[1, 32, 48, 128], dtype=float16
  - `#47 transpose.int` -> shape=[1, 32, 128, 48], dtype=float16
  - `#48 matmul.default` -> shape=[1, 32, 48, 48], dtype=float16
  - `#49 div.Tensor` -> shape=[1, 32, 48, 48], dtype=float16
  - `#50 add.Tensor` -> shape=[1, 32, 48, 48], dtype=float16
  - `#51 softmax.int` -> shape=[1, 32, 48, 48], dtype=float32
  - `#53 dropout.default` -> shape=[1, 32, 48, 48], dtype=float16
  - `#54 matmul.default` -> shape=[1, 32, 48, 128], dtype=float16

### `attention_output`

- dispatch_supported: `True`
- summary: Attention-output evidence is attn @ V, transpose/reshape, output projection, and residual add.
- evidence ops:
  - `#54 matmul.default` -> shape=[1, 32, 48, 128], dtype=float16
  - `#56 contiguous.default` -> shape=[1, 48, 32, 128], dtype=float16
  - `#57 reshape.default` -> shape=[1, 48, 4096], dtype=float16
  - `#68 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#69 add.Tensor` -> shape=[1, 48, 4096], dtype=float16
  - `#82 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#83 add.Tensor` -> shape=[1, 48, 4096], dtype=float16

### `mlp`

- dispatch_supported: `True`
- summary: MLP evidence is post-attention RMSNorm, gate/up linear, SiLU, gated product, down linear, residual add.
- evidence ops:
  - `#62 add.Tensor` -> shape=[], dtype=int64
  - `#63 mul.Tensor` -> shape=[], dtype=int64
  - `#65 add.Tensor` -> shape=[], dtype=int64
  - `#68 linear.default` -> shape=[1, 48, 4096], dtype=float16
  - `#69 add.Tensor` -> shape=[1, 48, 4096], dtype=float16
  - `#70 to.dtype` -> shape=[1, 48, 4096], dtype=float32
  - `#71 pow.Tensor_Scalar` -> shape=[1, 48, 4096], dtype=float32
  - `#72 mean.dim` -> shape=[1, 48, 1], dtype=float32
  - `#73 add.Tensor` -> shape=[1, 48, 1], dtype=float32
  - `#74 rsqrt.default` -> shape=[1, 48, 1], dtype=float32
  - `#75 mul.Tensor` -> shape=[1, 48, 4096], dtype=float32
  - `#76 to.dtype` -> shape=[1, 48, 4096], dtype=float16
  - `#77 mul.Tensor` -> shape=[1, 48, 4096], dtype=float16
  - `#78 linear.default` -> shape=[1, 48, 11008], dtype=float16

