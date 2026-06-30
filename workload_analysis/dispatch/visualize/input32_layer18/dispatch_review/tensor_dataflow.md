# input32_layer18 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002831`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002832`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002833`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002834`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002831`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002835`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002836`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002837`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002838`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002838`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002838`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002839`: `#9 linear.default` -> `#12 view.default`
- `t00002842`: `#12 view.default` -> `#13 transpose.int`
- `t00002840`: `#10 linear.default` -> `#14 view.default`
- `t00002844`: `#14 view.default` -> `#15 transpose.int`
- `t00002841`: `#11 linear.default` -> `#16 view.default`
- `t00002846`: `#16 view.default` -> `#17 transpose.int`
- `t00002849`: `#18 select.int` -> `#19 select.int`
- `t00002850`: `#19 select.int` -> `#20 add.Tensor`
- `t00002851`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002852`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002851`: `#20 add.Tensor` -> `#23 item.default`
- `t00002853`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002851`: `#20 add.Tensor` -> `#26 item.default`
- `t00002854`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002853`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002855`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002854`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002857`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002843`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002856`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002843`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002843`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002861`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002862`: `#36 neg.default` -> `#37 cat.default`
- `t00002860`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002863`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002858`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002859`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002864`: `#38 mul.Tensor` -> `#39 add.Tensor`
