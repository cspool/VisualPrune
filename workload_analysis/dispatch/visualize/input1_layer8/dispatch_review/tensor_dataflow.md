# input1_layer8 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000357`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000358`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000359`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000360`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000357`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000361`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000362`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000363`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000365`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000365`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000365`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000367`: `#9 linear.default` -> `#12 view.default`
- `t00000372`: `#12 view.default` -> `#13 transpose.int`
- `t00000369`: `#10 linear.default` -> `#14 view.default`
- `t00000374`: `#14 view.default` -> `#15 transpose.int`
- `t00000371`: `#11 linear.default` -> `#16 view.default`
- `t00000376`: `#16 view.default` -> `#17 transpose.int`
- `t00000378`: `#18 select.int` -> `#19 select.int`
- `t00000379`: `#19 select.int` -> `#20 add.Tensor`
- `t00000380`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000381`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000380`: `#20 add.Tensor` -> `#23 item.default`
- `t00000383`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000380`: `#20 add.Tensor` -> `#26 item.default`
- `t00000385`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000383`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000386`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000385`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000388`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000373`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000387`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000373`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000373`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000392`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000393`: `#36 neg.default` -> `#37 cat.default`
- `t00000391`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000394`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000389`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000390`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000395`: `#38 mul.Tensor` -> `#39 add.Tensor`
