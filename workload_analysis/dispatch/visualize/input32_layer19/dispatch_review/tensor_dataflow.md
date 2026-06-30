# input32_layer19 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002904`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002905`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002906`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002907`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002904`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002908`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002909`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002910`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002911`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002911`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002911`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002912`: `#9 linear.default` -> `#12 view.default`
- `t00002915`: `#12 view.default` -> `#13 transpose.int`
- `t00002913`: `#10 linear.default` -> `#14 view.default`
- `t00002917`: `#14 view.default` -> `#15 transpose.int`
- `t00002914`: `#11 linear.default` -> `#16 view.default`
- `t00002919`: `#16 view.default` -> `#17 transpose.int`
- `t00002921`: `#18 select.int` -> `#19 select.int`
- `t00002922`: `#19 select.int` -> `#20 add.Tensor`
- `t00002923`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002924`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002923`: `#20 add.Tensor` -> `#23 item.default`
- `t00002925`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002923`: `#20 add.Tensor` -> `#26 item.default`
- `t00002926`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002925`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002927`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002926`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002929`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002916`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002928`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002916`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002916`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002933`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002934`: `#36 neg.default` -> `#37 cat.default`
- `t00002932`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002935`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002930`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002931`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002936`: `#38 mul.Tensor` -> `#39 add.Tensor`
