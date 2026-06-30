# input2_layer18 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002464`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002465`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002466`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002467`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002464`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002468`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002469`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002470`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002471`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002471`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002471`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002472`: `#9 linear.default` -> `#12 view.default`
- `t00002475`: `#12 view.default` -> `#13 transpose.int`
- `t00002473`: `#10 linear.default` -> `#14 view.default`
- `t00002477`: `#14 view.default` -> `#15 transpose.int`
- `t00002474`: `#11 linear.default` -> `#16 view.default`
- `t00002479`: `#16 view.default` -> `#17 transpose.int`
- `t00002482`: `#18 select.int` -> `#19 select.int`
- `t00002483`: `#19 select.int` -> `#20 add.Tensor`
- `t00002484`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002485`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002484`: `#20 add.Tensor` -> `#23 item.default`
- `t00002486`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002484`: `#20 add.Tensor` -> `#26 item.default`
- `t00002487`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002486`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002488`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002487`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002490`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002476`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002489`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002476`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002476`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002494`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002495`: `#36 neg.default` -> `#37 cat.default`
- `t00002493`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002496`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002491`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002492`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002497`: `#38 mul.Tensor` -> `#39 add.Tensor`
