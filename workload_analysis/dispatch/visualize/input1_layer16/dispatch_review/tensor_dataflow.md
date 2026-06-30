# input1_layer16 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `97`
- observed producer-consumer edges: `107`
- external input tensor ids: `16`
- produced tensor ids: `87`
- final output tensor ids: `1`

## First Observed Edges

- `t00001149`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00001150`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00001151`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00001152`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00001149`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00001153`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00001154`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00001155`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00001157`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00001157`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00001157`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00001159`: `#9 linear.default` -> `#12 view.default`
- `t00001164`: `#12 view.default` -> `#13 transpose.int`
- `t00001161`: `#10 linear.default` -> `#14 view.default`
- `t00001166`: `#14 view.default` -> `#15 transpose.int`
- `t00001163`: `#11 linear.default` -> `#16 view.default`
- `t00001168`: `#16 view.default` -> `#17 transpose.int`
- `t00001170`: `#18 select.int` -> `#19 select.int`
- `t00001171`: `#19 select.int` -> `#20 add.Tensor`
- `t00001172`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00001173`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00001172`: `#20 add.Tensor` -> `#23 item.default`
- `t00001175`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00001172`: `#20 add.Tensor` -> `#26 item.default`
- `t00001177`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00001175`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00001178`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00001177`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00001180`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00001165`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00001179`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00001165`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00001165`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00001184`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00001185`: `#36 neg.default` -> `#37 cat.default`
- `t00001183`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00001186`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00001181`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00001182`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00001187`: `#38 mul.Tensor` -> `#39 add.Tensor`
