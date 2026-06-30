# input1_layer13 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000852`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000853`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000854`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000855`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000852`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000856`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000857`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000858`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000860`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000860`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000860`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000862`: `#9 linear.default` -> `#12 view.default`
- `t00000867`: `#12 view.default` -> `#13 transpose.int`
- `t00000864`: `#10 linear.default` -> `#14 view.default`
- `t00000869`: `#14 view.default` -> `#15 transpose.int`
- `t00000866`: `#11 linear.default` -> `#16 view.default`
- `t00000871`: `#16 view.default` -> `#17 transpose.int`
- `t00000873`: `#18 select.int` -> `#19 select.int`
- `t00000874`: `#19 select.int` -> `#20 add.Tensor`
- `t00000875`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000876`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000875`: `#20 add.Tensor` -> `#23 item.default`
- `t00000878`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000875`: `#20 add.Tensor` -> `#26 item.default`
- `t00000880`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000878`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000881`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000880`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000883`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000868`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000882`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000868`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000868`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000887`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000888`: `#36 neg.default` -> `#37 cat.default`
- `t00000886`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000889`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000884`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000885`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000890`: `#38 mul.Tensor` -> `#39 add.Tensor`
