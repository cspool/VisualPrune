# input1_layer24 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00001966`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001967`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001968`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001969`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001966`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001970`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001971`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001972`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001974`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001974`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001974`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001976`: `#9 linear.default` -> `#12 view.default`
- `t00001981`: `#12 view.default` -> `#13 transpose.int`
- `t00001978`: `#10 linear.default` -> `#14 view.default`
- `t00001983`: `#14 view.default` -> `#15 transpose.int`
- `t00001980`: `#11 linear.default` -> `#16 view.default`
- `t00001985`: `#16 view.default` -> `#17 transpose.int`
- `t00001987`: `#18 select.int` -> `#19 select.int`
- `t00001988`: `#19 select.int` -> `#20 add.Tensor`
- `t00001989`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001990`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001989`: `#20 add.Tensor` -> `#23 item.default`
- `t00001992`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001989`: `#20 add.Tensor` -> `#26 item.default`
- `t00001994`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001992`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001995`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001994`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001997`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001982`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001996`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001982`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001982`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002001`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002002`: `#36 neg.default` -> `#37 cat.default`
- `t00002000`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002003`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001998`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001999`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002004`: `#38 mul.Tensor` -> `#39 add.Tensor`
