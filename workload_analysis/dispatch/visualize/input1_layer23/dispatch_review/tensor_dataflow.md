# input1_layer23 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00001864`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001865`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001866`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001867`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001864`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001868`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001869`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001870`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001872`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001872`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001872`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001874`: `#9 linear.default` -> `#12 view.default`
- `t00001879`: `#12 view.default` -> `#13 transpose.int`
- `t00001876`: `#10 linear.default` -> `#14 view.default`
- `t00001881`: `#14 view.default` -> `#15 transpose.int`
- `t00001878`: `#11 linear.default` -> `#16 view.default`
- `t00001883`: `#16 view.default` -> `#17 transpose.int`
- `t00001885`: `#18 select.int` -> `#19 select.int`
- `t00001886`: `#19 select.int` -> `#20 add.Tensor`
- `t00001887`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001888`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001887`: `#20 add.Tensor` -> `#23 item.default`
- `t00001890`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001887`: `#20 add.Tensor` -> `#26 item.default`
- `t00001892`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001890`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001893`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001892`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001895`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001880`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001894`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001880`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001880`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001899`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001900`: `#36 neg.default` -> `#37 cat.default`
- `t00001898`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001901`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001896`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001897`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001902`: `#38 mul.Tensor` -> `#39 add.Tensor`
