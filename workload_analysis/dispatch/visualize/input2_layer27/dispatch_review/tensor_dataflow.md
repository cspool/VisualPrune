# input2_layer27 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002606`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002607`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002608`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002609`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002606`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002610`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002611`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002612`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002613`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002613`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002613`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002614`: `#9 linear.default` -> `#12 view.default`
- `t00002617`: `#12 view.default` -> `#13 transpose.int`
- `t00002615`: `#10 linear.default` -> `#14 view.default`
- `t00002619`: `#14 view.default` -> `#15 transpose.int`
- `t00002616`: `#11 linear.default` -> `#16 view.default`
- `t00002621`: `#16 view.default` -> `#17 transpose.int`
- `t00002623`: `#18 select.int` -> `#19 select.int`
- `t00002624`: `#19 select.int` -> `#20 add.Tensor`
- `t00002625`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002626`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002625`: `#20 add.Tensor` -> `#23 item.default`
- `t00002627`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002625`: `#20 add.Tensor` -> `#26 item.default`
- `t00002628`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002627`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002629`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002628`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002631`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002618`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002630`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002618`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002618`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002635`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002636`: `#36 neg.default` -> `#37 cat.default`
- `t00002634`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002637`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002632`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002633`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002638`: `#38 mul.Tensor` -> `#39 add.Tensor`
