# input1_layer27 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00002272`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002273`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002274`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002275`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002272`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002276`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002277`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002278`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002280`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002280`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002280`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002282`: `#9 linear.default` -> `#12 view.default`
- `t00002287`: `#12 view.default` -> `#13 transpose.int`
- `t00002284`: `#10 linear.default` -> `#14 view.default`
- `t00002289`: `#14 view.default` -> `#15 transpose.int`
- `t00002286`: `#11 linear.default` -> `#16 view.default`
- `t00002291`: `#16 view.default` -> `#17 transpose.int`
- `t00002293`: `#18 select.int` -> `#19 select.int`
- `t00002294`: `#19 select.int` -> `#20 add.Tensor`
- `t00002295`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002296`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002295`: `#20 add.Tensor` -> `#23 item.default`
- `t00002298`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002295`: `#20 add.Tensor` -> `#26 item.default`
- `t00002300`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002298`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002301`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002300`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002303`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002288`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002302`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002288`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002288`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002307`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002308`: `#36 neg.default` -> `#37 cat.default`
- `t00002306`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002309`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002304`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002305`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002310`: `#38 mul.Tensor` -> `#39 add.Tensor`
