# Dispatch Tools Trace Level

This directory contains tools for collecting VisiPrune dispatch evidence.
The main filtered dispatch profiler is:

```text
visipruner_filtered_dispatch_profile.py
```

FX-oriented experimental tools live under:

```text
/workspace/VisiPrune/workload_analysis/fx/
```

## Current Trace Level

The filtered dispatch profiler records a PyTorch ATen dispatch-level trace.
It is lower level than a Python `nn.Module.forward` trace, but it is not a
C++ Dispatcher key trace and it is not a CUDA kernel timeline.

The practical stack is:

1. Module / Python forward level
   - Examples: `LlamaDecoderLayer.forward`, attention module forward, MLP module forward.
   - This is high-level model logic.
   - The profiler uses PyTorch `nn.Module` forward hooks only to sample the
     active runtime module stack for each captured op.

2. ATen op / Python dispatch level
   - This is the level recorded in `dispatch_ops.csv`.
   - Examples: `slice.Tensor`, `matmul.default`, `softmax.int`,
     `copy_.default`, `fill_.Tensor`.
   - `visipruner_filtered_dispatch_profile.py` uses PyTorch
     `torch.utils._python_dispatch.TorchDispatchMode` and implements
     `__torch_dispatch__` to intercept these runtime ATen ops.
   - The CSV can record op schema, input/output tensor metadata, tensor ids,
     observed producer-consumer edges inside the captured scope, aliases, and
     the sampled module stack.

3. C++ Dispatcher / DispatchKey level
   - This level explains how PyTorch's C++ Dispatcher chooses kernels through
     DispatchKeys such as `BackendSelect`, `AutogradCUDA`, or `CUDA`.
   - `TORCH_SHOW_DISPATCH_TRACE=1` is closer to this layer when available in
     the PyTorch build.
   - The filtered dispatch profiler does not use this mechanism and does not
     record DispatchKey selection chains.

4. CUDA kernel / GPU timeline level
   - This level records actual kernel launches, CUDA streams, GPU timing, and
     possible fused kernels.
   - Tools such as Nsight Systems, Nsight Compute, and `torch.profiler` are
     closer to this layer.
   - The filtered dispatch profiler does not record CUDA kernel timelines or
     wall-clock kernel latency.

## Official PyTorch Mechanisms Used Here

`visipruner_filtered_dispatch_profile.py` uses these official PyTorch
mechanisms:

- `torch.utils._python_dispatch.TorchDispatchMode`
- `__torch_dispatch__`
- `nn.Module.register_forward_pre_hook`
- `nn.Module.register_forward_hook`

The dispatch trace is entered only inside manifest-selected layer forward
calls. This is intentional: the tool collects a filtered layer-level ATen op
trace instead of wrapping the whole `model.generate()` call.

## Torch-Dispatch Terminology

Articles or documents that discuss `torch-dispatch`, `__torch_dispatch__`, or
`TorchDispatchMode` in the context of `torch.compile`, AOTAutograd, proxy
tensors, or tracing are referring to the same PyTorch dispatch interface
family used by this profiler.

The usage is different:

- In compiler internals, `__torch_dispatch__` is often used to trace,
  transform, proxy, or functionalize tensor operations as part of a compiler
  pipeline.
- In `visipruner_filtered_dispatch_profile.py`, `__torch_dispatch__` is used
  during eager `generate()` execution to record a filtered ATen op trace for
  selected layer forwards.

So the interface family is shared, but this profiler is not running the
`torch.compile` pipeline.

## Eager Execution vs `torch.compile`

The filtered dispatch profiler runs the model in eager mode. This matters
because eager and compiled execution expose different evidence.

| Aspect | Eager execution | `torch.compile` execution |
| --- | --- | --- |
| Execution model | Python runs step by step; each PyTorch op is dispatched immediately. | PyTorch tries to trace Python code into one or more graphs, then hands those graphs to a compiler backend. |
| Op visibility | The observed ATen op stream is close to the real runtime calls made by eager execution. | Original ops may be decomposed, normalized, reordered, fused, or split across graph breaks. |
| Optimization | Minimal cross-op graph optimization. | Graph-level optimization, kernel fusion, code generation, and reduced Python overhead may apply. |
| Dynamic control flow | Naturally supported and directly observable along the path actually taken. | Difficult-to-trace code can cause graph breaks; compiled execution may be segmented or fall back to eager. |
| First-run cost | No tracing or compile cost. | First run may pay tracing/guard/compile cost; later runs reuse compiled graphs if guards match. |
| Best use | Runtime evidence, debugging, dynamic path observation, dispatch-op coverage. | Performance optimization and structured graph capture for traceable computation. |

For this directory, `dispatch_ops.csv` is evidence from eager ATen dispatch
execution, not evidence from a compiled graph.

## Readable Process Reconstruction Goal

A dynamic dispatch op stream is not enough by itself when the goal is to
understand a layer's computation. The op stream also needs to be reverse-shaped
into a readable process representation.

There are two different routes.

### Route 1: Reconstruct from eager dispatch trace

This is the evidence-first route used for VisiPrune layer analysis.

1. Run the real eager `generate()` request.
2. Capture selected layer forwards with `TorchDispatchMode.__torch_dispatch__`.
3. Treat each `dispatch_ops.csv` data row as one observed ATen dispatch op.
4. Build dataflow from `input_tensor_ids` and `output_tensor_ids`.
5. Preserve alias and mutation evidence such as view/slice/select, `copy_`,
   `fill_`, shared storage, and inplace outputs.
6. Generate a low-level SSA/ATen-style process first, then group it into
   readable sub-processes.

Example low-level representation:

```python
t60 = aten.slice(t56, ...)
t61 = aten.slice(t60, ...)
t62 = aten.sum(t61, dim=-1)
t65 = aten.slice(aten.slice(t56, ...), ...)
aten.fill_(t65, 0)
t67 = aten.select(aten.slice(t56, ...), ...)
aten.copy_(t67, t62)
```

Strengths:

- Closest to the real eager runtime path.
- Does not depend on whether `torch.compile` can trace the model.
- Captures the actual dynamic branch taken for this request.
- Fits strict dispatch coverage requirements: every `event_op_index` from
  `dispatch_ops.csv` can be accounted for.
- Tensor-id producer-consumer edges, aliases, and inplace effects can be tied
  directly to runtime evidence.

Limitations:

- Raw ATen op flow is too low-level to be readable without post-processing.
- Human-readable names such as `visual_adjust`, `attention_scores`, or
  `tail_mass_fold` are interpretation layers over dispatch evidence, not
  original source names recovered from the trace.
- View, alias, and inplace semantics must be handled carefully; otherwise the
  generated process may look valid while changing the real memory behavior.

### Route 2: Generate readable code from compile/FX/export IR

This route starts from PyTorch compiler or graph-capture IR rather than from
`dispatch_ops.csv`.

Relevant mechanisms include:

- `torch.fx.GraphModule.code`, which prints Python-like `forward()` code from
  an FX graph.
- `GraphModule.to_folder()`, which dumps generated FX module code for reading
  and debugging.
- `torch.fx.experimental.proxy_tensor.make_fx`, which executes a function and
  returns an FX `GraphModule` representing captured tensor ops.
- `torch.export`, which produces an exported program with a traced tensor
  computation graph.

Strengths:

- The graph already has explicit node dependencies.
- Generated code is usually much more readable than a raw dispatch CSV.
- It can be useful as a reference for ordinary tensor computation fragments.

Limitations:

- It is not a decompiler for `dispatch_ops.csv`.
- The generated graph is the compiler/export IR, not necessarily the exact
  eager dispatch op stream.
- Ops may be decomposed, functionalized, reordered, or fused.
- Mutation and alias behavior may be rewritten, which can hide evidence that
  matters for VisiPrune visual adjustment and inplace operations.
- Dynamic token pruning, cache shape changes, and Python control flow may cause
  graph breaks or incomplete captures.

## Can Compile Codegen Be Used For Op Reconstruction?

It can be used as a rendering or auxiliary mechanism, but not as the source of
truth for dispatch-op reconstruction.

The compile/FX/export stack contains mechanisms that turn IR into Python-like
code, but it does not provide an automatic inverse compiler that takes an
arbitrary `dispatch_ops.csv` and reconstructs the original process.

For VisiPrune dispatch reconstruction, the safe strategy is:

1. Treat `dispatch_ops.csv` as the ground truth.
2. Preserve every observed `event_op_index`; no dispatch op may be omitted.
3. Derive data dependencies only from `input_tensor_ids` and
   `output_tensor_ids`.
4. Preserve alias/inplace evidence rather than silently functionalizing it
   away.
5. Build a custom SSA or FX-like intermediate representation from the dispatch
   rows.
6. Optionally use FX-style code generation to print a readable Python-like
   process.
7. Validate the printed process back against `dispatch_ops.csv` coverage and
   tensor-id producer-consumer edges.

In short: eager dispatch trace is the ground truth for fidelity; FX/export
codegen can help with readability. Compile IR is useful as an auxiliary
reference or cross-check only when it can be proven to match the captured
dispatch evidence.

## Related Tools That Are Not Used

`torch.library.opcheck` is not used by this profiler. It validates custom
operator registration, schema behavior, autograd registration, FakeTensor
support, and AOT dispatch behavior. It is not a model execution trace tool and
does not generate `dispatch_ops.csv`.

`TORCH_SHOW_DISPATCH_TRACE=1` is not used by this profiler. It is useful for
debugging C++ Dispatcher call and redispatch paths when supported by the
installed PyTorch build, but its output is console-oriented and does not include
the structured layer/module/tensor-id evidence collected here.

`torch.profiler`, Nsight Systems, and Nsight Compute are not used by this
profiler. They are useful for timing, CUDA kernel timelines, stream behavior,
and kernel-level performance analysis, but they answer a different question
from the filtered ATen dispatch evidence stored in `dispatch_ops.csv`.
