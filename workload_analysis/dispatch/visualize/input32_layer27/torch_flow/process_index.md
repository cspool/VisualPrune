# input32_layer27 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer27/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer27/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input32_layer27/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `kv_cache_concat`
5. `attention`
6. `attention_output`
7. `mlp`

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` -> shape=[1, 1, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` -> shape=[1, 1, 4096], dtype=float32
- `#3 mean.dim` -> shape=[1, 1, 1], dtype=float32
- `#4 add.Tensor` -> shape=[1, 1, 1], dtype=float32
- `#5 rsqrt.default` -> shape=[1, 1, 1], dtype=float32
- `#6 mul.Tensor` -> shape=[1, 1, 4096], dtype=float32
- `#7 to.dtype` -> shape=[1, 1, 4096], dtype=float16
- `#8 mul.Tensor` -> shape=[1, 1, 4096], dtype=float16

### `qkv_projection`
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
- `#20 add.Tensor` -> shape=[], dtype=int64
- `#24 slice.Tensor` -> shape=[655, 128], dtype=float16
- `#27 slice.Tensor` -> shape=[655, 128], dtype=float16
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
- `#47 cat.default` -> shape=[1, 32, 89, 128], dtype=float16
- `#48 cat.default` -> shape=[1, 32, 89, 128], dtype=float16

### `attention`
- `#13 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 1, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
- `#46 add.Tensor` -> shape=[1, 32, 1, 128], dtype=float16
- `#49 transpose.int` -> shape=[1, 32, 128, 89], dtype=float16
- `#50 matmul.default` -> shape=[1, 32, 1, 89], dtype=float16
- `#51 div.Tensor` -> shape=[1, 32, 1, 89], dtype=float16
- `#52 add.Tensor` -> shape=[1, 32, 1, 89], dtype=float16
- `#53 softmax.int` -> shape=[1, 32, 1, 89], dtype=float32
- `#55 dropout.default` -> shape=[1, 32, 1, 89], dtype=float16
- `#56 matmul.default` -> shape=[1, 32, 1, 128], dtype=float16

### `attention_output`
- `#56 matmul.default` -> shape=[1, 32, 1, 128], dtype=float16
- `#58 reshape.default` -> shape=[1, 1, 4096], dtype=float16
- `#61 linear.default` -> shape=[1, 1, 4096], dtype=float16
- `#62 add.Tensor` -> shape=[1, 1, 4096], dtype=float16
- `#75 linear.default` -> shape=[1, 1, 4096], dtype=float16
- `#76 add.Tensor` -> shape=[1, 1, 4096], dtype=float16

### `mlp`
- `#54 to.dtype` -> shape=[1, 32, 1, 89], dtype=float16
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
