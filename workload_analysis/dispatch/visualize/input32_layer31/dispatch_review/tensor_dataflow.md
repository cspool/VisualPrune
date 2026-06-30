# input32_layer31 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00003122`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00003123`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00003124`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00003125`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00003122`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00003126`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00003127`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00003128`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00003129`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00003129`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00003129`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00003130`: `#9 linear.default` -> `#12 view.default`
- `t00003133`: `#12 view.default` -> `#13 transpose.int`
- `t00003131`: `#10 linear.default` -> `#14 view.default`
- `t00003135`: `#14 view.default` -> `#15 transpose.int`
- `t00003132`: `#11 linear.default` -> `#16 view.default`
- `t00003137`: `#16 view.default` -> `#17 transpose.int`
- `t00003139`: `#18 select.int` -> `#19 select.int`
- `t00003140`: `#19 select.int` -> `#20 add.Tensor`
- `t00003141`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00003142`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00003141`: `#20 add.Tensor` -> `#23 item.default`
- `t00003143`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00003141`: `#20 add.Tensor` -> `#26 item.default`
- `t00003144`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00003143`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00003145`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00003144`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00003147`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00003134`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00003146`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00003134`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00003134`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00003151`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00003152`: `#36 neg.default` -> `#37 cat.default`
- `t00003150`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00003153`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00003148`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00003149`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00003154`: `#38 mul.Tensor` -> `#39 add.Tensor`
