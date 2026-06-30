# input1_layer14 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000951`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000952`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000953`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000954`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000951`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000955`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000956`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000957`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000959`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000959`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000959`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000961`: `#9 linear.default` -> `#12 view.default`
- `t00000966`: `#12 view.default` -> `#13 transpose.int`
- `t00000963`: `#10 linear.default` -> `#14 view.default`
- `t00000968`: `#14 view.default` -> `#15 transpose.int`
- `t00000965`: `#11 linear.default` -> `#16 view.default`
- `t00000970`: `#16 view.default` -> `#17 transpose.int`
- `t00000972`: `#18 select.int` -> `#19 select.int`
- `t00000973`: `#19 select.int` -> `#20 add.Tensor`
- `t00000974`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000975`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000974`: `#20 add.Tensor` -> `#23 item.default`
- `t00000977`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000974`: `#20 add.Tensor` -> `#26 item.default`
- `t00000979`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000977`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000980`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000979`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000982`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000967`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000981`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000967`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000967`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000986`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000987`: `#36 neg.default` -> `#37 cat.default`
- `t00000985`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000988`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000983`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000984`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000989`: `#38 mul.Tensor` -> `#39 add.Tensor`
