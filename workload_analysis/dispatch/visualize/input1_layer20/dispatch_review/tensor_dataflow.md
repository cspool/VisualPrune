# input1_layer20 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00001558`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001559`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001560`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001561`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001558`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001562`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001563`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001564`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001566`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001566`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001566`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001568`: `#9 linear.default` -> `#12 view.default`
- `t00001573`: `#12 view.default` -> `#13 transpose.int`
- `t00001570`: `#10 linear.default` -> `#14 view.default`
- `t00001575`: `#14 view.default` -> `#15 transpose.int`
- `t00001572`: `#11 linear.default` -> `#16 view.default`
- `t00001577`: `#16 view.default` -> `#17 transpose.int`
- `t00001579`: `#18 select.int` -> `#19 select.int`
- `t00001580`: `#19 select.int` -> `#20 add.Tensor`
- `t00001581`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001582`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001581`: `#20 add.Tensor` -> `#23 item.default`
- `t00001584`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001581`: `#20 add.Tensor` -> `#26 item.default`
- `t00001586`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001584`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001587`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001586`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001589`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001574`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001588`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001574`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001574`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001593`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001594`: `#36 neg.default` -> `#37 cat.default`
- `t00001592`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001595`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001590`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001591`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001596`: `#38 mul.Tensor` -> `#39 add.Tensor`
