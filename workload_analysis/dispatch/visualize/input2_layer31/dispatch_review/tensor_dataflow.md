# input2_layer31 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002747`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002748`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002749`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002750`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002747`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002751`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002752`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002753`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002755`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002755`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002755`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002757`: `#9 linear.default` -> `#12 view.default`
- `t00002762`: `#12 view.default` -> `#13 transpose.int`
- `t00002759`: `#10 linear.default` -> `#14 view.default`
- `t00002764`: `#14 view.default` -> `#15 transpose.int`
- `t00002761`: `#11 linear.default` -> `#16 view.default`
- `t00002766`: `#16 view.default` -> `#17 transpose.int`
- `t00002768`: `#18 select.int` -> `#19 select.int`
- `t00002769`: `#19 select.int` -> `#20 add.Tensor`
- `t00002770`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002771`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002770`: `#20 add.Tensor` -> `#23 item.default`
- `t00002773`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002770`: `#20 add.Tensor` -> `#26 item.default`
- `t00002775`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002773`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002776`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002775`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002778`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002763`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002777`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002763`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002763`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002782`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002783`: `#36 neg.default` -> `#37 cat.default`
- `t00002781`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002784`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002779`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002780`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002785`: `#38 mul.Tensor` -> `#39 add.Tensor`
