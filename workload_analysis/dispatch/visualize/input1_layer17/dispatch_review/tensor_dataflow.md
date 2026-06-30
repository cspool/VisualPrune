# input1_layer17 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00001248`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001249`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001250`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001251`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001248`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001252`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001253`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001254`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001256`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001256`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001256`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001258`: `#9 linear.default` -> `#12 view.default`
- `t00001263`: `#12 view.default` -> `#13 transpose.int`
- `t00001260`: `#10 linear.default` -> `#14 view.default`
- `t00001265`: `#14 view.default` -> `#15 transpose.int`
- `t00001262`: `#11 linear.default` -> `#16 view.default`
- `t00001267`: `#16 view.default` -> `#17 transpose.int`
- `t00001269`: `#18 select.int` -> `#19 select.int`
- `t00001270`: `#19 select.int` -> `#20 add.Tensor`
- `t00001271`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001272`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001271`: `#20 add.Tensor` -> `#23 item.default`
- `t00001274`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001271`: `#20 add.Tensor` -> `#26 item.default`
- `t00001276`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001274`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001277`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001276`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001279`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001264`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001278`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001264`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001264`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001283`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001284`: `#36 neg.default` -> `#37 cat.default`
- `t00001282`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001285`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001280`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001281`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001286`: `#38 mul.Tensor` -> `#39 add.Tensor`
