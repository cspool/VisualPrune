# Torch Profiler Process View

## Run Summary

- tag: `visipruner_full_1tok_stack_modules`
- config: `visipruner-full`
- torch: `2.12.0+cu132`
- profiler options: `record_shapes=True`, `with_stack=True`, `with_modules=True`
- output text: `The`

## Evidence Boundary

This file is generated from `torch.profiler` events plus explicit `record_function` scopes.
It is a process sketch, not an FX graph and not a TorchDispatch tensor-id reconstruction.
The profiler can show event timing, shapes, stacks, and Chrome trace structure, but it does not
provide producer-consumer tensor ids or complete alias/inplace evidence.

## Observed High-Level Schedule

- measured forward events: `1`
- measured layer events: `32`
- measured selection events: `21`
- prefill layers: `32`
- decode layer calls: `0`

## Top `record_function` Scopes

| scope | cpu total ms | device total ms | child ATen ops | top child ops |
|---|---:|---:|---:|---|
| profile.request | 830.694 | 66.830 | 11516 | aten::linear x371; aten::matmul x289; aten::mm x225; aten::addmm x146; aten::mul x498; aten::bmm x112; aten::conv2d x1; aten::convolution x1 |
| profile.generate_total | 783.792 | 66.762 | 11464 | aten::linear x371; aten::matmul x289; aten::mm x225; aten::addmm x146; aten::mul x498; aten::bmm x112; aten::conv2d x1; aten::convolution x1 |
| profile.forward_prefill | 272.138 | 52.429 | 8429 | aten::matmul x289; aten::linear x225; aten::mm x225; aten::mul x357; aten::add x309; aten::copy_ x265; aten::bmm x64; aten::to x297 |
| profile.layer00.prefill | 73.595 | 3.018 | 234 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x9; aten::bmm x2; aten::softmax x1; aten::_softmax x1; aten::cat x2 |
| profile.layer07.prefill | 61.288 | 2.350 | 267 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::add x9; aten::bmm x2; aten::copy_ x9; aten::cosine_similarity x1 |
| profile.layer07.prefill.attn | 60.020 | 1.043 | 204 | aten::matmul x6; aten::linear x4; aten::mm x4; aten::mul x6; aten::add x5; aten::bmm x2; aten::cosine_similarity x1; aten::softmax x1 |
| profile.value_aware_token_selection.layer07 | 58.086 | 0.082 | 55 | aten::cosine_similarity x1; aten::div x2; aten::linalg_vector_norm x2; aten::mul x2; aten::clone x3; aten::copy_ x3; aten::clamp_min_ x2; aten::contiguous x1 |
| profile.image_preprocess_cpu | 33.494 | 0.000 | 14 | aten::to x1; aten::_to_copy x1; aten::copy_ x1; aten::stack x1; aten::cat x1; aten::select x1; aten::lift_fresh x1; aten::narrow x1 |
| profile.layer19.prefill | 15.135 | 0.772 | 274 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x12; aten::cosine_similarity x1; aten::add x10; aten::copy_ x9; aten::mean x2 |
| profile.layer00.prefill.attn | 13.672 | 1.290 | 171 | aten::matmul x6; aten::linear x4; aten::mm x4; aten::bmm x2; aten::softmax x1; aten::_softmax x1; aten::cat x2; aten::mul x4 |
| profile.layer19.prefill.mlp | 11.096 | 0.448 | 29 | aten::linear x3; aten::matmul x3; aten::mm x3; aten::mul x1; aten::silu x1; aten::t x3; aten::transpose x3; aten::reshape x3 |
| profile.load_image | 8.994 | 0.000 | 0 |  |
| profile.layer00.prefill.mlp | 5.730 | 1.588 | 29 | aten::linear x3; aten::matmul x3; aten::mm x3; aten::silu x1; aten::mul x1; aten::t x3; aten::transpose x3; aten::_unsafe_view x3 |
| profile.layer20.prefill | 4.124 | 0.643 | 274 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x12; aten::cosine_similarity x1; aten::add x10; aten::copy_ x9; aten::mean x2 |
| profile.layer08.prefill | 4.090 | 2.228 | 267 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::copy_ x9; aten::bmm x2; aten::add x9; aten::to x9 |
| profile.layer18.prefill | 3.956 | 2.250 | 287 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::bmm x2; aten::add x10; aten::copy_ x9; aten::to x9 |
| profile.layer14.prefill | 3.765 | 2.229 | 267 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::bmm x2; aten::copy_ x9; aten::add x9; aten::to x9 |
| profile.layer10.prefill | 3.760 | 2.229 | 267 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::bmm x2; aten::copy_ x9; aten::add x9; aten::to x9 |
| profile.layer12.prefill | 3.639 | 2.349 | 267 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x11; aten::bmm x2; aten::copy_ x9; aten::add x9; aten::to x9 |
| profile.layer26.prefill | 3.632 | 0.643 | 274 | aten::matmul x9; aten::linear x7; aten::mm x7; aten::mul x12; aten::cosine_similarity x1; aten::add x10; aten::copy_ x9; aten::mean x2 |

## Layer 0 `prefill` Profiler-Derived Process Sketch

- layer scope: `profile.layer00.prefill`
- observed q_len: `624`
- observed kv_len: `624`
- hidden shape in: `[1, 624, 4096]`

### Tensor-Axis Sketch

```text
Token axis S=624 (compressed)                     Hidden dimension
                                                       0                                      4096
                                                       ▲                                        ▲
hidden input      profile.layer scope            ──▶   ┌────────────────────────────────────────┐
                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │
                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │
                                                       │ HIDDEN_ROWS_BEFORE_LAYER               │
                                                       └────────────────────────────────────────┘
self attention    profile.layer.attn             ──▶   ┌────────────────────────────────────────┐
                                                       │ QKV_ROPE_ATTENTION_OUTPUT              │
                                                       │ scope groups profiler events, not DAG   │
                                                       └────────────────────────────────────────┘
residual + norm   inferred layer body             ──▶   [RESIDUAL_ADD_AND_POST_ATTN_RMSNORM]
mlp               profile.layer.mlp              ──▶   ┌────────────────────────────────────────┐
                                                       │ GATE_UP_SILU_DOWN_ROWS                 │
                                                       │ profiler gives events, not data edges   │
                                                       └────────────────────────────────────────┘
layer output      layer return                    ──▶   [HIDDEN_ROWS_AFTER_LAYER]
```

### Layer Total: Top Child ATen Ops

| op | count | cpu total ms | device total ms | example input shapes |
|---|---:|---:|---:|---|
| aten::matmul | 9 | 1.332 | 2.368 | `[[1,624,4096],[4096,4096]]` |
| aten::linear | 7 | 0.803 | 2.249 | `[[1,624,4096],[4096,4096],[]]` |
| aten::mm | 7 | 0.581 | 2.249 | `[[624,4096],[4096,4096]]` |
| aten::mul | 9 | 0.283 | 0.128 | `[[1,624,4096],[1,624,1]]` |
| aten::bmm | 2 | 0.553 | 0.119 | `[[32,624,128],[32,128,624]]` |
| aten::softmax | 1 | 0.183 | 0.089 | `[[1,32,624,624],[],[]]` |
| aten::_softmax | 1 | 0.178 | 0.089 | `[[1,32,624,624],[],[]]` |
| aten::cat | 2 | 0.101 | 0.089 | `[[],[]]` |
| aten::add | 8 | 0.330 | 0.067 | `[[1,624,1],[],[]]` |
| aten::copy_ | 7 | 0.222 | 0.066 | `[[1,624,4096],[1,624,4096],[]]` |
| aten::to | 9 | 0.288 | 0.058 | `[[1,624,4096],[],[],[],[]]` |
| aten::_to_copy | 5 | 0.271 | 0.058 | `[[1,624,4096],[],[],[],[],[],[]]` |
| aten::neg | 2 | 9.948 | 0.045 | `[[1,32,624,64]]` |
| aten::fill_ | 1 | 0.058 | 0.036 | `[[1,32,589,576],[]]` |
| aten::silu | 1 | 5.164 | 0.031 | `[[1,624,11008]]` |
| aten::div | 1 | 0.211 | 0.031 | `[[1,32,624,624],[]]` |
| aten::mean | 2 | 8.810 | 0.021 | `[[1,624,4096],[],[],[]]` |
| aten::pow | 2 | 27.399 | 0.020 | `[[1,624,4096],[]]` |
| aten::item | 10 | 0.221 | 0.009 | `[[]]` |
| aten::_local_scalar_dense | 10 | 0.176 | 0.009 | `[[]]` |

### Attention Scope: Top Child ATen Ops

| op | count | cpu total ms | device total ms | example input shapes |
|---|---:|---:|---:|---|
| aten::matmul | 6 | 1.068 | 0.841 | `[[1,624,4096],[4096,4096]]` |
| aten::linear | 4 | 0.498 | 0.722 | `[[1,624,4096],[4096,4096],[]]` |
| aten::mm | 4 | 0.360 | 0.722 | `[[624,4096],[4096,4096]]` |
| aten::bmm | 2 | 0.553 | 0.119 | `[[32,624,128],[32,128,624]]` |
| aten::softmax | 1 | 0.183 | 0.089 | `[[1,32,624,624],[],[]]` |
| aten::_softmax | 1 | 0.178 | 0.089 | `[[1,32,624,624],[],[]]` |
| aten::cat | 2 | 0.101 | 0.089 | `[[],[]]` |
| aten::mul | 4 | 0.088 | 0.056 | `[[1,32,624,128],[1,1,624,128]]` |
| aten::add | 4 | 0.170 | 0.046 | `[[],[],[]]` |
| aten::neg | 2 | 9.948 | 0.045 | `[[1,32,624,64]]` |
| aten::fill_ | 1 | 0.058 | 0.036 | `[[1,32,589,576],[]]` |
| aten::copy_ | 3 | 0.072 | 0.034 | `[[1,32,624,624],[1,32,624,624],[]]` |
| aten::div | 1 | 0.211 | 0.031 | `[[1,32,624,624],[]]` |
| aten::to | 3 | 0.052 | 0.026 | `[[624,128],[],[],[],[]]` |
| aten::_to_copy | 1 | 0.047 | 0.026 | `[[1,32,624,624],[],[],[],[],[],[]]` |
| aten::item | 10 | 0.221 | 0.009 | `[[]]` |
| aten::_local_scalar_dense | 10 | 0.176 | 0.009 | `[[]]` |
| aten::contiguous | 1 | 0.049 | 0.006 | `[[1,624,32,128],[]]` |
| aten::clone | 1 | 0.046 | 0.006 | `[[1,624,32,128],[]]` |
| aten::sum | 1 | 0.133 | 0.006 | `[[1,32,13,576],[],[],[]]` |

### MLP Scope: Top Child ATen Ops

| op | count | cpu total ms | device total ms | example input shapes |
|---|---:|---:|---:|---|
| aten::linear | 3 | 0.304 | 1.527 | `[[1,624,4096],[11008,4096],[]]` |
| aten::matmul | 3 | 0.265 | 1.527 | `[[1,624,4096],[4096,11008]]` |
| aten::mm | 3 | 0.221 | 1.527 | `[[624,4096],[4096,11008]]` |
| aten::silu | 1 | 5.164 | 0.031 | `[[1,624,11008]]` |
| aten::mul | 1 | 0.022 | 0.029 | `[[1,624,11008],[1,624,11008]]` |
| aten::t | 3 | 0.028 | 0.000 | `[[11008,4096]]` |
| aten::transpose | 3 | 0.015 | 0.000 | `[[11008,4096],[],[]]` |
| aten::_unsafe_view | 3 | 0.011 | 0.000 | `[[624,11008],[]]` |
| aten::reshape | 3 | 0.009 | 0.000 | `[[1,624,4096],[]]` |
| aten::as_strided | 3 | 0.005 | 0.000 | `[[11008,4096],[],[],[]]` |
| aten::view | 3 | 0.003 | 0.000 | `[[1,624,4096],[]]` |

## What `with_stack` Produced

- profiler events with non-empty stack metadata: `0`
- ATen events with non-empty stack metadata: `0`
- total profiler events: `58269`

No non-empty Python stack metadata was observed in this eager run.
The option was enabled, but it did not provide a usable source-location
mapping for the ATen events exported by this local profiler path.

## What `with_modules` Produced

- ATen module-hierarchy rows with non-empty module metadata: `0`
- total module/stack summary rows: `1`

No non-empty module hierarchy was observed for ATen events in this eager run.
This matches the PyTorch profiler documentation caveat that `with_modules`
is TorchScript-oriented and is not a reliable eager-module ownership source.
