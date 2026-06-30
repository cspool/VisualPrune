# input32_layer28 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00003049`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00003050`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00003051`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00003052`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00003049`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00003053`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00003054`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00003055`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00003056`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00003056`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00003056`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00003057`: `#9 linear.default` -> `#12 view.default`
- `t00003060`: `#12 view.default` -> `#13 transpose.int`
- `t00003058`: `#10 linear.default` -> `#14 view.default`
- `t00003062`: `#14 view.default` -> `#15 transpose.int`
- `t00003059`: `#11 linear.default` -> `#16 view.default`
- `t00003064`: `#16 view.default` -> `#17 transpose.int`
- `t00003066`: `#18 select.int` -> `#19 select.int`
- `t00003067`: `#19 select.int` -> `#20 add.Tensor`
- `t00003068`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00003069`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00003068`: `#20 add.Tensor` -> `#23 item.default`
- `t00003070`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00003068`: `#20 add.Tensor` -> `#26 item.default`
- `t00003071`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00003070`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00003072`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00003071`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00003074`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00003061`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00003073`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00003061`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00003061`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00003078`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00003079`: `#36 neg.default` -> `#37 cat.default`
- `t00003077`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00003080`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00003075`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00003076`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00003081`: `#38 mul.Tensor` -> `#39 add.Tensor`
