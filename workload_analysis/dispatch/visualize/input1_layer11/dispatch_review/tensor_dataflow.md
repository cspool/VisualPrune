# input1_layer11 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00000654`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00000655`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00000656`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00000657`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00000654`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00000658`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00000659`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00000660`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00000662`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00000662`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00000662`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00000664`: `#9 linear.default` -> `#12 view.default`
- `t00000669`: `#12 view.default` -> `#13 transpose.int`
- `t00000666`: `#10 linear.default` -> `#14 view.default`
- `t00000671`: `#14 view.default` -> `#15 transpose.int`
- `t00000668`: `#11 linear.default` -> `#16 view.default`
- `t00000673`: `#16 view.default` -> `#17 transpose.int`
- `t00000675`: `#18 select.int` -> `#19 select.int`
- `t00000676`: `#19 select.int` -> `#20 add.Tensor`
- `t00000677`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00000678`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00000677`: `#20 add.Tensor` -> `#23 item.default`
- `t00000680`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00000677`: `#20 add.Tensor` -> `#26 item.default`
- `t00000682`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00000680`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00000683`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00000682`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00000685`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00000670`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00000684`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00000670`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00000670`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00000689`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00000690`: `#36 neg.default` -> `#37 cat.default`
- `t00000688`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00000691`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00000686`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00000687`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00000692`: `#38 mul.Tensor` -> `#39 add.Tensor`
