# input32_layer27 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002977`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002978`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002979`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002980`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002977`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002981`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002982`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002983`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002984`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002984`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002984`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002985`: `#9 linear.default` -> `#12 view.default`
- `t00002988`: `#12 view.default` -> `#13 transpose.int`
- `t00002986`: `#10 linear.default` -> `#14 view.default`
- `t00002990`: `#14 view.default` -> `#15 transpose.int`
- `t00002987`: `#11 linear.default` -> `#16 view.default`
- `t00002992`: `#16 view.default` -> `#17 transpose.int`
- `t00002994`: `#18 select.int` -> `#19 select.int`
- `t00002995`: `#19 select.int` -> `#20 add.Tensor`
- `t00002996`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002997`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002996`: `#20 add.Tensor` -> `#23 item.default`
- `t00002998`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002996`: `#20 add.Tensor` -> `#26 item.default`
- `t00002999`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002998`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00003000`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002999`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00003002`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002989`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00003001`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002989`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002989`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00003006`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00003007`: `#36 neg.default` -> `#37 cat.default`
- `t00003005`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00003008`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00003003`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00003004`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00003009`: `#38 mul.Tensor` -> `#39 add.Tensor`
