# Torch Profiler vs FX/Dispatch Summary

This run used `torch.profiler` with `record_shapes=True`, `with_stack=True`,
and `with_modules=True` on the eager VisiPrune/LLaVA generation path.

## Observed Result

- profiler events: `58269`
- key averages: `2109`
- explicit `record_function` scopes: `246`
- measured prefill layer scopes: `32`
- measured selection scopes: `21`
- non-empty `FunctionEvent.stack` rows: `0`
- non-empty module-hierarchy rows: `0`

The profiler did run the real GPU workload and produced useful timing, shape,
and Chrome-trace data. However, in this eager run the automatic stack and module
hierarchy metadata did not provide usable process ownership.

## Practical Interpretation

`torch.profiler` is useful for general first-pass performance inspection:
operator timing, CUDA timeline, input shapes, memory if enabled, and coarse
regions when the code adds `record_function` scopes.

It does not directly replace the current algorithmic trace, because it does not
record VisiPrune-specific schedule semantics such as forward id, pruning phase,
selection metadata, or algorithm-level process labels unless those are manually
annotated.

It does not directly replace TorchDispatch reconstruction, because it does not
provide stable tensor ids, producer-consumer edges, alias/inplace evidence, or a
complete low-level dataflow reconstruction.

It does not directly replace FX visualization, because it records runtime
events rather than a reusable graph with node arguments, users, and generated
graph code.

The closest profiler-based approximation is the current hybrid method:
explicit `record_function` scopes define request, forward, layer, attention,
MLP, and selection regions; profiler events then fill those regions with timing
and shape evidence. This is good for an initial algorithm survey, but it is not
a perfect substitute for FX or dispatch-based process reconstruction.
