# input1_layer15 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00001050`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001051`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001052`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001053`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001050`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001054`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001055`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001056`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001058`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001058`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001058`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001060`: `#9 linear.default` -> `#12 view.default`
- `t00001065`: `#12 view.default` -> `#13 transpose.int`
- `t00001062`: `#10 linear.default` -> `#14 view.default`
- `t00001067`: `#14 view.default` -> `#15 transpose.int`
- `t00001064`: `#11 linear.default` -> `#16 view.default`
- `t00001069`: `#16 view.default` -> `#17 transpose.int`
- `t00001071`: `#18 select.int` -> `#19 select.int`
- `t00001072`: `#19 select.int` -> `#20 add.Tensor`
- `t00001073`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001074`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001073`: `#20 add.Tensor` -> `#23 item.default`
- `t00001076`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001073`: `#20 add.Tensor` -> `#26 item.default`
- `t00001078`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001076`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001079`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001078`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001081`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001066`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001080`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001066`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001066`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001085`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001086`: `#36 neg.default` -> `#37 cat.default`
- `t00001084`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001087`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001082`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001083`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001088`: `#38 mul.Tensor` -> `#39 add.Tensor`
