# input1_layer28 Tensor Dataflow

This file is derived only from `input_tensor_ids` and `output_tensor_ids` in `dispatch_ops.csv`.

- ops: `83`
- observed producer-consumer edges: `91`
- external input tensor ids: `15`
- produced tensor ids: `75`
- final output tensor ids: `1`

## First Observed Edges

- `t00002375`: `#1 to.dtype` -> `#2 pow.Tensor_Scalar`
- `t00002376`: `#2 pow.Tensor_Scalar` -> `#3 mean.dim`
- `t00002377`: `#3 mean.dim` -> `#4 add.Tensor`
- `t00002378`: `#4 add.Tensor` -> `#5 rsqrt.default`
- `t00002375`: `#1 to.dtype` -> `#6 mul.Tensor`
- `t00002379`: `#5 rsqrt.default` -> `#6 mul.Tensor`
- `t00002380`: `#6 mul.Tensor` -> `#7 to.dtype`
- `t00002381`: `#7 to.dtype` -> `#8 mul.Tensor`
- `t00002383`: `#8 mul.Tensor` -> `#9 linear.default`
- `t00002383`: `#8 mul.Tensor` -> `#10 linear.default`
- `t00002383`: `#8 mul.Tensor` -> `#11 linear.default`
- `t00002385`: `#9 linear.default` -> `#12 view.default`
- `t00002390`: `#12 view.default` -> `#13 transpose.int`
- `t00002387`: `#10 linear.default` -> `#14 view.default`
- `t00002392`: `#14 view.default` -> `#15 transpose.int`
- `t00002389`: `#11 linear.default` -> `#16 view.default`
- `t00002394`: `#16 view.default` -> `#17 transpose.int`
- `t00002397`: `#18 select.int` -> `#19 select.int`
- `t00002398`: `#19 select.int` -> `#20 add.Tensor`
- `t00002399`: `#20 add.Tensor` -> `#21 gt.Scalar`
- `t00002400`: `#21 gt.Scalar` -> `#22 is_nonzero.default`
- `t00002399`: `#20 add.Tensor` -> `#23 item.default`
- `t00002402`: `#24 slice.Tensor` -> `#25 to.dtype`
- `t00002399`: `#20 add.Tensor` -> `#26 item.default`
- `t00002404`: `#27 slice.Tensor` -> `#28 to.dtype`
- `t00002402`: `#25 to.dtype` -> `#29 index.Tensor`
- `t00002405`: `#29 index.Tensor` -> `#30 unsqueeze.default`
- `t00002404`: `#28 to.dtype` -> `#31 index.Tensor`
- `t00002407`: `#31 index.Tensor` -> `#32 unsqueeze.default`
- `t00002391`: `#13 transpose.int` -> `#33 mul.Tensor`
- `t00002406`: `#30 unsqueeze.default` -> `#33 mul.Tensor`
- `t00002391`: `#13 transpose.int` -> `#34 slice.Tensor`
- `t00002391`: `#13 transpose.int` -> `#35 slice.Tensor`
- `t00002411`: `#35 slice.Tensor` -> `#36 neg.default`
- `t00002412`: `#36 neg.default` -> `#37 cat.default`
- `t00002410`: `#34 slice.Tensor` -> `#37 cat.default`
- `t00002413`: `#37 cat.default` -> `#38 mul.Tensor`
- `t00002408`: `#32 unsqueeze.default` -> `#38 mul.Tensor`
- `t00002409`: `#33 mul.Tensor` -> `#39 add.Tensor`
- `t00002414`: `#38 mul.Tensor` -> `#39 add.Tensor`
