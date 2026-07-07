# Torch Profiler Trace Experiment

This directory is a sandbox for testing whether `torch.profiler` can produce a
useful trace and a process-style visualization close to the existing FX and
TorchDispatch artifacts.

The profiler run intentionally enables:

```text
record_shapes=True
with_stack=True
with_modules=True
```

`with_modules=True` is enabled because this experiment is explicitly checking
its usefulness, but current PyTorch documents module hierarchy support as
TorchScript-oriented. For this eager VisiPrune/LLaVA path, module hierarchy may
be empty or incomplete. The script therefore also inserts explicit
`record_function` scopes around request, forward, layer, attention, MLP, and
value-selection regions.

`with_stack=True` is also treated as an observed signal rather than a guarantee.
In the local eager VisiPrune run, the exported `FunctionEvent.stack` fields were
empty for all events even though the option was enabled. That means the reliable
eager-process anchors in this experiment are the explicit `record_function`
scopes plus tensor shapes and timing events, not automatic Python stack or
module ownership metadata.

## Run

Use the workload-analysis environment wrapper:

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/torch_profile/tools/torch_profiler_generate_trace.py \
  --config visipruner-full \
  --gpu 1 \
  --max-new-tokens 2 \
  --tag visipruner_full_2tok_stack_modules
```

Outputs are written under:

```text
workload_analysis/torch_profile/traces/<tag>/
```

Expected artifacts:

```text
metadata.json
chrome_trace.json
profiler_events.csv
profiler_key_averages.csv
record_function_scopes.csv
module_stack_summary.csv
layer_events.csv
selection_events.csv
process_view.md
```

## Interpretation Boundary

This experiment is not a replacement for the existing trace stack:

- `algorithmic_trace` remains the source of VisiPrune dynamic schedule evidence:
  `forward_id`, phase, `q_len`, `kv_len`, token selection, and deep exit.
- filtered TorchDispatch remains the source of eager ATen op and tensor-id
  evidence for process reconstruction.
- FX remains the source of fixed-input low-level graph evidence when a readable
  graph is needed.

`torch.profiler` is useful for event timing, shape, memory, stack/source, and
Chrome timeline inspection. The generated `process_view.md` is deliberately
called a profiler-derived sketch because it lacks the explicit dataflow edges,
producer-consumer tensor ids, and alias/inplace evidence available in the
dispatch trace.
