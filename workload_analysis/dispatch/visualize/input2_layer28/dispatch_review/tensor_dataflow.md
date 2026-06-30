# input2_layer28 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002676`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002677`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002678`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002679`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002676`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002680`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002681`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002682`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002683`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002683`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002683`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002684`: `#9 linear.default` -> `#12 view.default`
- `t00002687`: `#12 view.default` -> `#13 transpose.int`
- `t00002685`: `#10 linear.default` -> `#14 view.default`
- `t00002689`: `#14 view.default` -> `#15 transpose.int`
- `t00002686`: `#11 linear.default` -> `#16 view.default`
- `t00002691`: `#16 view.default` -> `#17 transpose.int`
- `t00002693`: `#18 select.int` -> `#19 select.int`
- `t00002694`: `#19 select.int` -> `#20 add.Tensor`
- `t00002695`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002696`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002695`: `#20 add.Tensor` -> `#23 item.default`
- `t00002697`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002695`: `#20 add.Tensor` -> `#26 item.default`
- `t00002698`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002697`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002699`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002698`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002701`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002688`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002700`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002688`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002688`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002705`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002706`: `#36 neg.default` -> `#37 cat.default`
- `t00002704`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002707`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002702`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002703`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002708`: `#38 mul.Tensor` -> `#39 add.Tensor`
