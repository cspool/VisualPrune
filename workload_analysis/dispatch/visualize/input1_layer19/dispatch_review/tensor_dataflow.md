# input1_layer19 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00001454`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001455`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001456`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001457`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001454`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001458`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001459`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001460`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001462`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001462`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001462`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001464`: `#9 linear.default` -> `#12 view.default`
- `t00001469`: `#12 view.default` -> `#13 transpose.int`
- `t00001466`: `#10 linear.default` -> `#14 view.default`
- `t00001471`: `#14 view.default` -> `#15 transpose.int`
- `t00001468`: `#11 linear.default` -> `#16 view.default`
- `t00001473`: `#16 view.default` -> `#17 transpose.int`
- `t00001476`: `#18 select.int` -> `#19 select.int`
- `t00001477`: `#19 select.int` -> `#20 add.Tensor`
- `t00001478`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001479`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001478`: `#20 add.Tensor` -> `#23 item.default`
- `t00001481`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001478`: `#20 add.Tensor` -> `#26 item.default`
- `t00001483`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001481`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001484`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001483`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001486`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001470`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001485`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001470`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001470`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001490`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001491`: `#36 neg.default` -> `#37 cat.default`
- `t00001489`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001492`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001487`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001488`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001493`: `#38 mul.Tensor` -> `#39 add.Tensor`
