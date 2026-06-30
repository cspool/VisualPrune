# input1_layer21 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00001660`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001661`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001662`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001663`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001660`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001664`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001665`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001666`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001668`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001668`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001668`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001670`: `#9 linear.default` -> `#12 view.default`
- `t00001675`: `#12 view.default` -> `#13 transpose.int`
- `t00001672`: `#10 linear.default` -> `#14 view.default`
- `t00001677`: `#14 view.default` -> `#15 transpose.int`
- `t00001674`: `#11 linear.default` -> `#16 view.default`
- `t00001679`: `#16 view.default` -> `#17 transpose.int`
- `t00001681`: `#18 select.int` -> `#19 select.int`
- `t00001682`: `#19 select.int` -> `#20 add.Tensor`
- `t00001683`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001684`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001683`: `#20 add.Tensor` -> `#23 item.default`
- `t00001686`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001683`: `#20 add.Tensor` -> `#26 item.default`
- `t00001688`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001686`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001689`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001688`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001691`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001676`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001690`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001676`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001676`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001695`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001696`: `#36 neg.default` -> `#37 cat.default`
- `t00001694`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001697`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001692`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001693`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001698`: `#38 mul.Tensor` -> `#39 add.Tensor`
