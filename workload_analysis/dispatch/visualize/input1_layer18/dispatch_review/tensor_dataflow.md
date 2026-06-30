# input1_layer18 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `104`
- observed producer-consumer edges: `115`
- external input tensor ids: `16`
- produced tensor ids: `94`
- final output tensor ids: `2`

## First Observed Edges

- `t00001347`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001348`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001349`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001350`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001347`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001351`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001352`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001353`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001355`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001355`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001355`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001357`: `#9 linear.default` -> `#12 view.default`
- `t00001362`: `#12 view.default` -> `#13 transpose.int`
- `t00001359`: `#10 linear.default` -> `#14 view.default`
- `t00001364`: `#14 view.default` -> `#15 transpose.int`
- `t00001361`: `#11 linear.default` -> `#16 view.default`
- `t00001366`: `#16 view.default` -> `#17 transpose.int`
- `t00001368`: `#18 select.int` -> `#19 select.int`
- `t00001369`: `#19 select.int` -> `#20 add.Tensor`
- `t00001370`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001371`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001370`: `#20 add.Tensor` -> `#23 item.default`
- `t00001373`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001370`: `#20 add.Tensor` -> `#26 item.default`
- `t00001375`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001373`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001376`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001375`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001378`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001363`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001377`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001363`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001363`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001382`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001383`: `#36 neg.default` -> `#37 cat.default`
- `t00001381`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001384`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001379`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001380`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001385`: `#38 mul.Tensor` -> `#39 add.Tensor`
