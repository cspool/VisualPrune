# input1_layer5 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer5/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visual_adjust`
6. `attention_output`
7. `mlp`

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` -> shape=[1, 624, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` -> shape=[1, 624, 4096], dtype=float32
- `#3 mean.dim` -> shape=[1, 624, 1], dtype=float32
- `#4 add.Tensor` -> shape=[1, 624, 1], dtype=float32
- `#5 rsqrt.default` -> shape=[1, 624, 1], dtype=float32
- `#6 mul.Tensor` -> shape=[1, 624, 4096], dtype=float32
- `#7 to.dtype` -> shape=[1, 624, 4096], dtype=float16
- `#8 mul.Tensor` -> shape=[1, 624, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#10 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#11 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#12 view.default` -> shape=[1, 624, 32, 128], dtype=float16
- `#13 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#14 view.default` -> shape=[1, 624, 32, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#16 view.default` -> shape=[1, 624, 32, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16

### `rope`
- `#20 add.Tensor` -> shape=[], dtype=int64
- `#24 slice.Tensor` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` -> shape=[1, 624, 128], dtype=float16
- `#30 unsqueeze.default` -> shape=[1, 1, 624, 128], dtype=float16
- `#31 index.Tensor` -> shape=[1, 624, 128], dtype=float16
- `#32 unsqueeze.default` -> shape=[1, 1, 624, 128], dtype=float16
- `#33 mul.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#34 slice.Tensor` -> shape=[1, 32, 624, 64], dtype=float16
- `#35 slice.Tensor` -> shape=[1, 32, 624, 64], dtype=float16
- `#36 neg.default` -> shape=[1, 32, 624, 64], dtype=float16
- `#37 cat.default` -> shape=[1, 32, 624, 128], dtype=float16
- `#38 mul.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 624, 128], dtype=float16

### `attention`
- `#13 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 624, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#46 add.Tensor` -> shape=[1, 32, 624, 128], dtype=float16
- `#47 transpose.int` -> shape=[1, 32, 128, 624], dtype=float16
- `#48 matmul.default` -> shape=[1, 32, 624, 624], dtype=float16
- `#49 div.Tensor` -> shape=[1, 32, 624, 624], dtype=float16
- `#50 add.Tensor` -> shape=[1, 32, 624, 624], dtype=float16
- `#51 softmax.int` -> shape=[1, 32, 624, 624], dtype=float32
- `#61 dropout.default` -> shape=[1, 32, 624, 624], dtype=float16
- `#62 matmul.default` -> shape=[1, 32, 624, 128], dtype=float16

### `visual_adjust`
- `#55 lift_fresh.default` -> shape=[], dtype=float16
- `#57 slice.Tensor` -> shape=[1, 32, 13, 624], dtype=float16
- `#59 slice.Tensor` -> shape=[1, 32, 13, 576], dtype=float16
- `#60 fill_.Tensor` -> shape=[1, 32, 13, 576], dtype=float16

### `attention_output`
- `#62 matmul.default` -> shape=[1, 32, 624, 128], dtype=float16
- `#64 contiguous.default` -> shape=[1, 624, 32, 128], dtype=float16
- `#65 reshape.default` -> shape=[1, 624, 4096], dtype=float16
- `#68 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#69 add.Tensor` -> shape=[1, 624, 4096], dtype=float16
- `#82 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#83 add.Tensor` -> shape=[1, 624, 4096], dtype=float16

### `mlp`
- `#68 linear.default` -> shape=[1, 624, 4096], dtype=float16
- `#69 add.Tensor` -> shape=[1, 624, 4096], dtype=float16
- `#70 to.dtype` -> shape=[1, 624, 4096], dtype=float32
- `#71 pow.Tensor_Scalar` -> shape=[1, 624, 4096], dtype=float32
- `#72 mean.dim` -> shape=[1, 624, 1], dtype=float32
- `#73 add.Tensor` -> shape=[1, 624, 1], dtype=float32
- `#74 rsqrt.default` -> shape=[1, 624, 1], dtype=float32
- `#75 mul.Tensor` -> shape=[1, 624, 4096], dtype=float32
- `#76 to.dtype` -> shape=[1, 624, 4096], dtype=float16
- `#77 mul.Tensor` -> shape=[1, 624, 4096], dtype=float16
- `#78 linear.default` -> shape=[1, 624, 11008], dtype=float16
- `#79 silu.default` -> shape=[1, 624, 11008], dtype=float16
- `#80 linear.default` -> shape=[1, 624, 11008], dtype=float16
- `#81 mul.Tensor` -> shape=[1, 624, 11008], dtype=float16
