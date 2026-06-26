# input1_layer28 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer28/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer28/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer28/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `attention_output`
6. `mlp`

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` -> shape=[1, 48, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` -> shape=[1, 48, 4096], dtype=float32
- `#3 mean.dim` -> shape=[1, 48, 1], dtype=float32
- `#4 add.Tensor` -> shape=[1, 48, 1], dtype=float32
- `#5 rsqrt.default` -> shape=[1, 48, 1], dtype=float32
- `#6 mul.Tensor` -> shape=[1, 48, 4096], dtype=float32
- `#7 to.dtype` -> shape=[1, 48, 4096], dtype=float16
- `#8 mul.Tensor` -> shape=[1, 48, 4096], dtype=float16

### `qkv_projection`
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
- `#54 matmul.default` -> shape=[1, 32, 48, 128], dtype=float16
- `#56 contiguous.default` -> shape=[1, 48, 32, 128], dtype=float16
- `#57 reshape.default` -> shape=[1, 48, 4096], dtype=float16
- `#68 linear.default` -> shape=[1, 48, 4096], dtype=float16
- `#69 add.Tensor` -> shape=[1, 48, 4096], dtype=float16
- `#82 linear.default` -> shape=[1, 48, 4096], dtype=float16
- `#83 add.Tensor` -> shape=[1, 48, 4096], dtype=float16

### `mlp`
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
