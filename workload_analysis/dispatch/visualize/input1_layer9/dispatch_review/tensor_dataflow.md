# input1_layer9 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000456`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000457`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000458`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000459`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000456`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000460`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000461`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000462`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000464`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000464`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000464`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000466`: `#9 linear.default` -> `#12 view.default`
- `t00000471`: `#12 view.default` -> `#13 transpose.int`
- `t00000468`: `#10 linear.default` -> `#14 view.default`
- `t00000473`: `#14 view.default` -> `#15 transpose.int`
- `t00000470`: `#11 linear.default` -> `#16 view.default`
- `t00000475`: `#16 view.default` -> `#17 transpose.int`
- `t00000477`: `#18 select.int` -> `#19 select.int`
- `t00000478`: `#19 select.int` -> `#20 add.Tensor`
- `t00000479`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000480`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000479`: `#20 add.Tensor` -> `#23 item.default`
- `t00000482`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000479`: `#20 add.Tensor` -> `#26 item.default`
- `t00000484`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000482`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000485`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000484`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000487`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000472`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000486`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000472`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000472`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000491`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000492`: `#36 neg.default` -> `#37 cat.default`
- `t00000490`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000493`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000488`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000489`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000494`: `#38 mul.Tensor` -> `#39 add.Tensor`
