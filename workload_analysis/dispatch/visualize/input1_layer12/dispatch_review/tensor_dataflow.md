# input1_layer12 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000753`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000754`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000755`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000756`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000753`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000757`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000758`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000759`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000761`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000761`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000761`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000763`: `#9 linear.default` -> `#12 view.default`
- `t00000768`: `#12 view.default` -> `#13 transpose.int`
- `t00000765`: `#10 linear.default` -> `#14 view.default`
- `t00000770`: `#14 view.default` -> `#15 transpose.int`
- `t00000767`: `#11 linear.default` -> `#16 view.default`
- `t00000772`: `#16 view.default` -> `#17 transpose.int`
- `t00000774`: `#18 select.int` -> `#19 select.int`
- `t00000775`: `#19 select.int` -> `#20 add.Tensor`
- `t00000776`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000777`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000776`: `#20 add.Tensor` -> `#23 item.default`
- `t00000779`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000776`: `#20 add.Tensor` -> `#26 item.default`
- `t00000781`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000779`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000782`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000781`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000784`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000769`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000783`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000769`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000769`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000788`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000789`: `#36 neg.default` -> `#37 cat.default`
- `t00000787`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000790`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000785`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000786`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000791`: `#38 mul.Tensor` -> `#39 add.Tensor`
