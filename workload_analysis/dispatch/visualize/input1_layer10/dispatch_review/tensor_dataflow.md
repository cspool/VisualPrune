# input1_layer10 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000555`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000556`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000557`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000558`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000555`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000559`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000560`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000561`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000563`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000563`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000563`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000565`: `#9 linear.default` -> `#12 view.default`
- `t00000570`: `#12 view.default` -> `#13 transpose.int`
- `t00000567`: `#10 linear.default` -> `#14 view.default`
- `t00000572`: `#14 view.default` -> `#15 transpose.int`
- `t00000569`: `#11 linear.default` -> `#16 view.default`
- `t00000574`: `#16 view.default` -> `#17 transpose.int`
- `t00000576`: `#18 select.int` -> `#19 select.int`
- `t00000577`: `#19 select.int` -> `#20 add.Tensor`
- `t00000578`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000579`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000578`: `#20 add.Tensor` -> `#23 item.default`
- `t00000581`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000578`: `#20 add.Tensor` -> `#26 item.default`
- `t00000583`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000581`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000584`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000583`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000586`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000571`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000585`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000571`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000571`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000590`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000591`: `#36 neg.default` -> `#37 cat.default`
- `t00000589`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000592`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000587`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000588`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000593`: `#38 mul.Tensor` -> `#39 add.Tensor`
