


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32)
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(mul, dtype = torch.float16);  mul = None
    _param_constant0 = self._param_constant0
    mul_1 = torch.ops.aten.mul.Tensor(_param_constant0, _to_copy_1);  _param_constant0 = _to_copy_1 = None
    _param_constant1 = self._param_constant1
    t = torch.ops.aten.t.default(_param_constant1);  _param_constant1 = None
    view = torch.ops.aten.view.default(mul_1, [1, 4096])
    mm = torch.ops.aten.mm.default(view, t);  view = t = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [1, 1, 4096]);  mm = None
    _param_constant2 = self._param_constant2
    t_1 = torch.ops.aten.t.default(_param_constant2);  _param_constant2 = None
    view_1 = torch.ops.aten.view.default(mul_1, [1, 4096])
    mm_1 = torch.ops.aten.mm.default(view_1, t_1);  view_1 = t_1 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(mm_1, [1, 1, 4096]);  mm_1 = None
    _param_constant3 = self._param_constant3
    t_2 = torch.ops.aten.t.default(_param_constant3);  _param_constant3 = None
    view_2 = torch.ops.aten.view.default(mul_1, [1, 4096]);  mul_1 = None
    mm_2 = torch.ops.aten.mm.default(view_2, t_2);  view_2 = t_2 = None
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(mm_2, [1, 1, 4096]);  mm_2 = None
    view_3 = torch.ops.aten.view.default(_unsafe_view, [1, 1, 32, 128]);  _unsafe_view = None
    transpose = torch.ops.aten.transpose.int(view_3, 1, 2);  view_3 = None
    view_4 = torch.ops.aten.view.default(_unsafe_view_1, [1, 1, 32, 128]);  _unsafe_view_1 = None
    transpose_1 = torch.ops.aten.transpose.int(view_4, 1, 2);  view_4 = None
    view_5 = torch.ops.aten.view.default(_unsafe_view_2, [1, 1, 32, 128]);  _unsafe_view_2 = None
    transpose_2 = torch.ops.aten.transpose.int(view_5, 1, 2);  view_5 = None
    _tensor_constant0 = self._tensor_constant0
    slice_1 = torch.ops.aten.slice.Tensor(_tensor_constant0, 0, 0, 655);  _tensor_constant0 = None
    _tensor_constant1 = self._tensor_constant1
    slice_2 = torch.ops.aten.slice.Tensor(_tensor_constant1, 0, 0, 655);  _tensor_constant1 = None
    index = torch.ops.aten.index.Tensor(slice_1, [arg2_1]);  slice_1 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(index, 1);  index = None
    index_1 = torch.ops.aten.index.Tensor(slice_2, [arg2_1]);  slice_2 = arg2_1 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(index_1, 1);  index_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(transpose, unsqueeze)
    slice_3 = torch.ops.aten.slice.Tensor(transpose, 3, 0, 64)
    slice_4 = torch.ops.aten.slice.Tensor(transpose, 3, 64, 9223372036854775807);  transpose = None
    neg = torch.ops.aten.neg.default(slice_4);  slice_4 = None
    cat = torch.ops.aten.cat.default([neg, slice_3], -1);  neg = slice_3 = None
    mul_3 = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
    add_1 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
    mul_4 = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze);  unsqueeze = None
    slice_5 = torch.ops.aten.slice.Tensor(transpose_1, 3, 0, 64)
    slice_6 = torch.ops.aten.slice.Tensor(transpose_1, 3, 64, 9223372036854775807);  transpose_1 = None
    neg_1 = torch.ops.aten.neg.default(slice_6);  slice_6 = None
    cat_1 = torch.ops.aten.cat.default([neg_1, slice_5], -1);  neg_1 = slice_5 = None
    mul_5 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
    add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    _tensor_constant2 = self._tensor_constant2
    cat_2 = torch.ops.aten.cat.default([_tensor_constant2, add_2], -2);  _tensor_constant2 = add_2 = None
    _tensor_constant3 = self._tensor_constant3
    cat_3 = torch.ops.aten.cat.default([_tensor_constant3, transpose_2], -2);  _tensor_constant3 = transpose_2 = None
    transpose_3 = torch.ops.aten.transpose.int(cat_2, 2, 3)
    expand = torch.ops.aten.expand.default(add_1, [1, 32, 1, 128]);  add_1 = None
    view_6 = torch.ops.aten.view.default(expand, [32, 1, 128]);  expand = None
    expand_1 = torch.ops.aten.expand.default(transpose_3, [1, 32, 128, 89]);  transpose_3 = None
    view_7 = torch.ops.aten.view.default(expand_1, [32, 128, 89]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
    view_8 = torch.ops.aten.view.default(bmm, [1, 32, 1, 89]);  bmm = None
    div = torch.ops.aten.div.Tensor(view_8, 11.313708498984761);  view_8 = None
    add_3 = torch.ops.aten.add.Tensor(div, arg1_1);  div = arg1_1 = None
    _softmax = torch.ops.aten._softmax.default(add_3, -1, True);  add_3 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(_softmax, dtype = torch.float16);  _softmax = None
    clone = torch.ops.aten.clone.default(_to_copy_2);  _to_copy_2 = None
    expand_2 = torch.ops.aten.expand.default(clone, [1, 32, 1, 89]);  clone = None
    view_9 = torch.ops.aten.view.default(expand_2, [32, 1, 89]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(cat_3, [1, 32, 89, 128])
    view_10 = torch.ops.aten.view.default(expand_3, [32, 89, 128]);  expand_3 = None
    bmm_1 = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11 = torch.ops.aten.view.default(bmm_1, [1, 32, 1, 128]);  bmm_1 = None
    transpose_4 = torch.ops.aten.transpose.int(view_11, 1, 2);  view_11 = None
    view_12 = torch.ops.aten.view.default(transpose_4, [1, 1, 4096]);  transpose_4 = None
    _param_constant4 = self._param_constant4
    t_3 = torch.ops.aten.t.default(_param_constant4);  _param_constant4 = None
    view_13 = torch.ops.aten.view.default(view_12, [1, 4096]);  view_12 = None
    mm_3 = torch.ops.aten.mm.default(view_13, t_3);  view_13 = t_3 = None
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(mm_3, [1, 1, 4096]);  mm_3 = None
    add_4 = torch.ops.aten.add.Tensor(arg0_1, _unsafe_view_3);  arg0_1 = _unsafe_view_3 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(add_4, dtype = torch.float32)
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_3, 2)
    mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    add_5 = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_6 = torch.ops.aten.mul.Tensor(_to_copy_3, rsqrt_1);  _to_copy_3 = rsqrt_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(mul_6, dtype = torch.float16);  mul_6 = None
    _param_constant5 = self._param_constant5
    mul_7 = torch.ops.aten.mul.Tensor(_param_constant5, _to_copy_4);  _param_constant5 = _to_copy_4 = None
    _param_constant6 = self._param_constant6
    t_4 = torch.ops.aten.t.default(_param_constant6);  _param_constant6 = None
    view_14 = torch.ops.aten.view.default(mul_7, [1, 4096])
    mm_4 = torch.ops.aten.mm.default(view_14, t_4);  view_14 = t_4 = None
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(mm_4, [1, 1, 11008]);  mm_4 = None
    silu = torch.ops.aten.silu.default(_unsafe_view_4);  _unsafe_view_4 = None
    _param_constant7 = self._param_constant7
    t_5 = torch.ops.aten.t.default(_param_constant7);  _param_constant7 = None
    view_15 = torch.ops.aten.view.default(mul_7, [1, 4096]);  mul_7 = None
    mm_5 = torch.ops.aten.mm.default(view_15, t_5);  view_15 = t_5 = None
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(mm_5, [1, 1, 11008]);  mm_5 = None
    mul_8 = torch.ops.aten.mul.Tensor(silu, _unsafe_view_5);  silu = _unsafe_view_5 = None
    _param_constant8 = self._param_constant8
    t_6 = torch.ops.aten.t.default(_param_constant8);  _param_constant8 = None
    view_16 = torch.ops.aten.view.default(mul_8, [1, 11008]);  mul_8 = None
    mm_6 = torch.ops.aten.mm.default(view_16, t_6);  view_16 = t_6 = None
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(mm_6, [1, 1, 4096]);  mm_6 = None
    add_6 = torch.ops.aten.add.Tensor(add_4, _unsafe_view_6);  add_4 = _unsafe_view_6 = None
    return (add_6, {'dynamic_cache_layer': (cat_2, cat_3)}, None, 0)
    
