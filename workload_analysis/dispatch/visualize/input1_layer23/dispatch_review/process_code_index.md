# input1_layer23 Process Code Index

The ONNX files in this layer are final visualization outputs. The files below are the process outputs used to inspect or regenerate the layer-specific computation.

## Code Artifacts

- dispatch reconstruction: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer23/torch_flow/dispatch_reconstructed.py`
- runnable toy process: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer23/torch_flow/toy_tensor_compute.py`
- split small-tensor torch flow: `/workspace/VisiPrune/workload_analysis/dispatch/visualize/input1_layer23/torch_flow`

## Dispatch-Derived Stage Order

1. `input_rmsnorm`
2. `qkv_projection`
3. `rope`
4. `attention`
5. `visipruner_similarity_check`
6. `attention_output`
7. `mlp`

## Evidence Rows

### `input_rmsnorm`
- `#1 to.dtype` -> shape=[1, 58, 4096], dtype=float32
- `#2 pow.Tensor_Scalar` -> shape=[1, 58, 4096], dtype=float32
- `#3 mean.dim` -> shape=[1, 58, 1], dtype=float32
- `#4 add.Tensor` -> shape=[1, 58, 1], dtype=float32
- `#5 rsqrt.default` -> shape=[1, 58, 1], dtype=float32
- `#6 mul.Tensor` -> shape=[1, 58, 4096], dtype=float32
- `#7 to.dtype` -> shape=[1, 58, 4096], dtype=float16
- `#8 mul.Tensor` -> shape=[1, 58, 4096], dtype=float16

### `qkv_projection`
- `#9 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#10 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#11 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#12 view.default` -> shape=[1, 58, 32, 128], dtype=float16
- `#13 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16
- `#14 view.default` -> shape=[1, 58, 32, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16
- `#16 view.default` -> shape=[1, 58, 32, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16

### `rope`
- `#20 add.Tensor` -> shape=[], dtype=int64
- `#24 slice.Tensor` -> shape=[624, 128], dtype=float16
- `#27 slice.Tensor` -> shape=[624, 128], dtype=float16
- `#29 index.Tensor` -> shape=[1, 58, 128], dtype=float16
- `#30 unsqueeze.default` -> shape=[1, 1, 58, 128], dtype=float16
- `#31 index.Tensor` -> shape=[1, 58, 128], dtype=float16
- `#32 unsqueeze.default` -> shape=[1, 1, 58, 128], dtype=float16
- `#33 mul.Tensor` -> shape=[1, 32, 58, 128], dtype=float16
- `#34 slice.Tensor` -> shape=[1, 32, 58, 64], dtype=float16
- `#35 slice.Tensor` -> shape=[1, 32, 58, 64], dtype=float16
- `#36 neg.default` -> shape=[1, 32, 58, 64], dtype=float16
- `#37 cat.default` -> shape=[1, 32, 58, 128], dtype=float16
- `#38 mul.Tensor` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 58, 128], dtype=float16

### `attention`
- `#13 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16
- `#15 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16
- `#17 transpose.int` -> shape=[1, 32, 58, 128], dtype=float16
- `#39 add.Tensor` -> shape=[1, 32, 58, 128], dtype=float16
- `#46 add.Tensor` -> shape=[1, 32, 58, 128], dtype=float16
- `#47 transpose.int` -> shape=[1, 32, 128, 58], dtype=float16
- `#48 matmul.default` -> shape=[1, 32, 58, 58], dtype=float16
- `#49 div.Tensor` -> shape=[1, 32, 58, 58], dtype=float16
- `#50 add.Tensor` -> shape=[1, 32, 58, 58], dtype=float16
- `#51 softmax.int` -> shape=[1, 32, 58, 58], dtype=float32
- `#53 dropout.default` -> shape=[1, 32, 58, 58], dtype=float16
- `#54 matmul.default` -> shape=[1, 32, 58, 128], dtype=float16

### `visipruner_similarity_check`
- `#21 gt.Scalar` -> shape=[], dtype=bool
- `#22 is_nonzero.default` -> False
- `#58 gt.Scalar` -> shape=[], dtype=bool
- `#59 is_nonzero.default` -> True
- `#64 sub.Tensor` -> shape=[], dtype=int64
- `#67 is_nonzero.default` -> True
- `#75 arange.start` -> shape=[10], dtype=int64
- `#78 sub.Tensor` -> shape=[1, 10, 4096], dtype=float16
- `#80 cosine_similarity.default` -> shape=[1, 10], dtype=float16
- `#83 any.default` -> shape=[], dtype=bool

### `attention_output`
- `#54 matmul.default` -> shape=[1, 32, 58, 128], dtype=float16
- `#56 contiguous.default` -> shape=[1, 58, 32, 128], dtype=float16
- `#57 reshape.default` -> shape=[1, 58, 4096], dtype=float16
- `#73 contiguous.default` -> shape=[1, 58, 32, 128], dtype=float16
- `#85 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` -> shape=[1, 58, 4096], dtype=float16
- `#99 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#100 add.Tensor` -> shape=[1, 58, 4096], dtype=float16

### `mlp`
- `#85 linear.default` -> shape=[1, 58, 4096], dtype=float16
- `#86 add.Tensor` -> shape=[1, 58, 4096], dtype=float16
- `#87 to.dtype` -> shape=[1, 58, 4096], dtype=float32
- `#88 pow.Tensor_Scalar` -> shape=[1, 58, 4096], dtype=float32
- `#89 mean.dim` -> shape=[1, 58, 1], dtype=float32
- `#90 add.Tensor` -> shape=[1, 58, 1], dtype=float32
- `#91 rsqrt.default` -> shape=[1, 58, 1], dtype=float32
- `#92 mul.Tensor` -> shape=[1, 58, 4096], dtype=float32
- `#93 to.dtype` -> shape=[1, 58, 4096], dtype=float16
- `#94 mul.Tensor` -> shape=[1, 58, 4096], dtype=float16
- `#95 linear.default` -> shape=[1, 58, 11008], dtype=float16
- `#96 silu.default` -> shape=[1, 58, 11008], dtype=float16
- `#97 linear.default` -> shape=[1, 58, 11008], dtype=float16
- `#98 mul.Tensor` -> shape=[1, 58, 11008], dtype=float16
