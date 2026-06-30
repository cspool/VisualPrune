# input1_layer26 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `100`
- observed producer-consumer edges: `111`
- external input tensor ids: `15`
- produced tensor ids: `91`
- final output tensor ids: `1`

## First Observed Edges

- `t00002170`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002171`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002172`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002173`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002170`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002174`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002175`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002176`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002178`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002178`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002178`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002180`: `#9 linear.default` -> `#12 view.default`
- `t00002185`: `#12 view.default` -> `#13 transpose.int`
- `t00002182`: `#10 linear.default` -> `#14 view.default`
- `t00002187`: `#14 view.default` -> `#15 transpose.int`
- `t00002184`: `#11 linear.default` -> `#16 view.default`
- `t00002189`: `#16 view.default` -> `#17 transpose.int`
- `t00002191`: `#18 select.int` -> `#19 select.int`
- `t00002192`: `#19 select.int` -> `#20 add.Tensor`
- `t00002193`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002194`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002193`: `#20 add.Tensor` -> `#23 item.default`
- `t00002196`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002193`: `#20 add.Tensor` -> `#26 item.default`
- `t00002198`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002196`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002199`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002198`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002201`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002186`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002200`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002186`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002186`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002205`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002206`: `#36 neg.default` -> `#37 cat.default`
- `t00002204`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002207`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002202`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002203`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002208`: `#38 mul.Tensor` -> `#39 add.Tensor`
