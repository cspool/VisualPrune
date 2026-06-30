# input1_layer25 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00002068`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002069`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002070`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002071`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002068`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002072`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002073`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002074`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002076`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002076`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002076`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002078`: `#9 linear.default` -> `#12 view.default`
- `t00002083`: `#12 view.default` -> `#13 transpose.int`
- `t00002080`: `#10 linear.default` -> `#14 view.default`
- `t00002085`: `#14 view.default` -> `#15 transpose.int`
- `t00002082`: `#11 linear.default` -> `#16 view.default`
- `t00002087`: `#16 view.default` -> `#17 transpose.int`
- `t00002089`: `#18 select.int` -> `#19 select.int`
- `t00002090`: `#19 select.int` -> `#20 add.Tensor`
- `t00002091`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002092`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002091`: `#20 add.Tensor` -> `#23 item.default`
- `t00002094`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002091`: `#20 add.Tensor` -> `#26 item.default`
- `t00002096`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002094`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002097`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002096`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002099`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002084`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002098`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002084`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002084`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002103`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002104`: `#36 neg.default` -> `#37 cat.default`
- `t00002102`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002105`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002100`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002101`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002106`: `#38 mul.Tensor` -> `#39 add.Tensor`
