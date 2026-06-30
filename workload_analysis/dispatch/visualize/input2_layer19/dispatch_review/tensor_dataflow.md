# input2_layer19 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `76`
- observed producer-consumer edges: `85`
- external input tensor ids: `17`
- produced tensor ids: `69`
- final output tensor ids: `1`

## First Observed Edges

- `t00002535`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002536`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002537`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002538`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002535`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002539`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002540`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002541`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002542`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002542`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002542`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002543`: `#9 linear.default` -> `#12 view.default`
- `t00002546`: `#12 view.default` -> `#13 transpose.int`
- `t00002544`: `#10 linear.default` -> `#14 view.default`
- `t00002548`: `#14 view.default` -> `#15 transpose.int`
- `t00002545`: `#11 linear.default` -> `#16 view.default`
- `t00002550`: `#16 view.default` -> `#17 transpose.int`
- `t00002552`: `#18 select.int` -> `#19 select.int`
- `t00002553`: `#19 select.int` -> `#20 add.Tensor`
- `t00002554`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002555`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002554`: `#20 add.Tensor` -> `#23 item.default`
- `t00002556`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002554`: `#20 add.Tensor` -> `#26 item.default`
- `t00002557`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002556`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002558`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002557`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002560`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002547`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002559`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002547`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002547`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002564`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002565`: `#36 neg.default` -> `#37 cat.default`
- `t00002563`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002566`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002561`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002562`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002567`: `#38 mul.Tensor` -> `#39 add.Tensor`
