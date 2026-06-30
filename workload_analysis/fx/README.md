# VisiPrune FX Tools

This directory contains experimental FX-oriented tools for readable graph
capture and code rendering. These tools are separate from the DispatchMode CSV
profiler under `workload_analysis/dispatch/tools/`.

The entry points are:

```text
fx_dynamic_trace.py
fx_trace_to_readable_code.py
fx_layer_process_reconstruct.py
```

## Purpose

The dispatch profiler records eager ATen op evidence in `dispatch_ops.csv`.
That evidence remains the ground truth for runtime op coverage. The FX tools in
this directory are for a different but related job:

- use PyTorch `make_fx` to capture selected tensor computations as FX graphs;
- save FX graph code and normalized node JSON;
- render `fx_nodes.json` into readable Python-like ATen process code.

These scripts should be treated as an experimental readable-graph capture path,
not as a replacement for the DispatchMode evidence trace.

## Evidence Boundary

The current FX path has three different evidence levels. Keep them separate
when reading generated files:

1. Runtime sampling evidence:
   selected decoder layer inputs are sampled during the real eager
   `model.generate()` call. This is the only part that observes the real
   end-to-end execution.
2. Offline fixed-input FX graph:
   after `generate()` finishes, sampled inputs are replayed through
   `make_fx(...)` to build a `GraphModule` for one observed input path.
   This graph is useful as a low-level tensor DAG.
3. Process reconstruction labels:
   files such as `fx_process_reconstruction.md` add readable process labels
   over the FX DAG. These labels are produced by reconstruction code, not by
   PyTorch FX or runtime module sampling.

FX should therefore not be treated as a source of true VisiPrune process or
module hierarchy. In this workflow it is useful for a fixed-input ATen DAG and
data-dependency view. Dispatch traces remain more reliable for "what actually
ran"; both FX and dispatch need additional evidence or manual interpretation
for high-level process/module semantics.

## Dynamic FX Trace

`fx_dynamic_trace.py` can run PyTorch `make_fx` in two modes.

Generic callable mode:

```bash
python /workspace/VisiPrune/workload_analysis/fx/fx_dynamic_trace.py \
  --target package.module:function \
  --input-spec-file input_spec.json \
  --output-dir /workspace/VisiPrune/workload_analysis/fx/traces \
  --tag my_trace
```

VisiPrune selected-layer mode:

```bash
python /workspace/VisiPrune/workload_analysis/fx/fx_dynamic_trace.py \
  --model-layer-trace \
  --layers input1_layer0,input1_layer5 \
  --gpu 1 \
  --tag fx_selected_layers
```

Layer target syntax:

- `0,5,6`: trace those layer ids for every forward where they execute.
- `input1_layer5`: trace only layer 5 in forward/input 1.
- `1:5`: equivalent compact syntax for `input1_layer5`.

Each selected layer event gets its own output directory containing:

```text
fx_graph.py
fx_graph.txt
fx_nodes.json
fx_graph_module.pt
fx_trace_metadata.json
fx_graph_module/
```

`fx_graph_module.pt` is a serialized `GraphModule` used by process
reconstruction scripts that need direct access to `GraphModule.graph`.
`fx_graph_module/` is produced with the official
`GraphModule.to_folder(..., module_name="FxLayerGraphModule")` API and contains
an importable GraphModule code package plus `state_dict.pt`. Process
reconstruction should prefer this GraphModule artifact when it needs to operate
on `GraphModule.graph` rather than the JSON node copy.

The selected-layer mode is a two-stage path:

1. Run the full VisiPrune `generate()` request in normal eager mode.
2. Runtime wrappers on decoder layers only sample matched layer inputs.
3. After `generate()` finishes, restore the original forwards and run
   `make_fx` offline on the sampled layer inputs.

This avoids executing `make_fx` inside the end-to-end forward path. Generation
always continues with the normal eager layer output. `--layer-trace-continue-with`
is kept only as a compatibility flag and does not change selected-layer runtime
execution.

### Runtime Wrapper Lifecycle

Selected-layer tracing temporarily wraps the in-memory model object. It does
not edit source files under `repo/llava`.

During setup:

- `model.forward` is wrapped to assign a monotonically increasing
  `forward_id` and infer `prefill` versus `decode` from the current sequence
  length.
- each `model.get_model().layers[i].forward` is wrapped to record
  `forward_id`, `layer_id`, `q_len`, `past_len`, `kv_len`, input/output shapes,
  and whether the event matches `--layers`.
- if the event matches, the wrapper snapshots `call_args` and `call_kwargs`.
  Tensor values are cloned, and Hugging Face `DynamicCache` key/value tensors
  are cloned recursively.

The wrapped layer then immediately calls the original layer forward:

```python
output = original_layer_forward(*call_args, **call_kwargs)
```

The real eager `generate()` output is therefore produced by the original model
logic. The wrapper only observes and snapshots inputs.

After `generate()` returns, all wrappers are restored in a `finally` block.
Only after restoration does the script run offline `make_fx(...)` over the
saved samples.

Tensor inputs and Hugging Face dynamic cache tensors are cloned at the matched
layer entry so the offline FX attempt does not read cache state mutated later
in generation.

The offline FX stage applies analysis-only specialization for fixed-input
Python control flow. It does not modify source files or the normal runtime
forward path. For each sampled layer input, the tool:

- computes fixed scalar guards such as `position_ids[0, -1] + 1`,
  `batch_size`, and `q_len`;
- temporarily binds a generated attention `forward` on the in-memory module
  instance with those scalar guard expressions replaced by Python constants;
- temporarily converts scalar tensor module attributes such as `num_images`
  and `vis_end_index` to Python integers;
- dry-runs `value_aware_token_selection` once on cloned inputs to record
  `.item()` branch decisions, then replays those decisions during `make_fx`;
- normalizes Hugging Face `DynamicCache` outputs to the current layer's K/V
  tensors so the FX graph returns tensor data rather than a Python cache object.

Each FX output records this specialization metadata in
`fx_trace_metadata.json` and the per-run manifest.

### Analysis-Only Specialization

`make_fx` captures tensor operations by executing the Python function once.
VisiPrune layer code contains Python control flow that depends on runtime
tensor values. In eager execution this is valid, but under `make_fx(real)` those
values can become tracing tensors/proxies and cannot be used directly in Python
branches or `.item()` decisions.

The tool therefore specializes one already observed input path:

- `position_ids[0, -1] + 1` is computed from the cloned runtime input and
  substituted as a Python constant in an in-memory copy of the attention
  `forward` method.
- `batch_size` and `q_len` are computed from the sampled `hidden_states` shape
  and forced back into the in-memory attention `forward` method.
- scalar module attributes such as `num_images`, `vis_end_index`, and
  `vis_half_index` are temporarily converted to Python integers when needed.
- `value_aware_token_selection` is first dry-run on cloned inputs to record
  `.item()` branch decisions. During `make_fx`, a replay implementation follows
  the recorded branch for that fixed input.
- Hugging Face `DynamicCache` outputs are normalized to the current layer's
  key/value tensors so the FX graph returns FX-compatible tensor data.

All of these replacements are reverted after each offline FX call. They are not
general model transformations; they are a traceability workaround for one
fixed sampled input. The resulting `GraphModule` represents the observed path,
not every possible branch of the original layer code.

### Fixed-Input Trace Issues Resolved

The selected-layer trace is based on the normal PyTorch/LLaVA inference
implementation. Source files under `repo/llava` are not modified. The FX tool
only patches in-memory module instances during the offline `make_fx` call, then
restores the original methods immediately.

The following fixed-input trace blockers are handled in the analysis tool:

- `make_fx` keyword arguments: this PyTorch build's `make_fx` wrapper rejects
  kwargs, so runtime kwargs are flattened into positional FX inputs and restored
  to the original keyword names inside the offline target.
- Tensor scalar Python control flow: expressions such as
  `position_ids[0, -1] + 1`, `q_len > 611`, and `self.num_images > 0` are
  deterministic for the sampled input but become tracing tensors under
  `make_fx(real)`. The tool computes those values from cloned runtime inputs and
  temporarily replaces only those Python guard expressions with constants.
- Value-aware `.item()` branches: `value_aware_token_selection` uses
  `torch.any(...).item()` to choose a branch. The tool dry-runs the function on
  cloned runtime inputs, records the branch decisions, and replays the same
  decisions during FX capture.
- Mutating KV cache: the runtime sampler clones Tensor inputs and Hugging Face
  `DynamicCache` tensors at layer entry, so the offline trace does not read
  cache state mutated later in generation.
- Python cache output: `DynamicCache` is not a valid FX graph output. The
  offline target normalizes it to the current layer's key/value tensors, and
  records this output normalization in metadata.

These transformations are intentionally analysis-only specializations of one
observed runtime path. Dispatch traces remain the runtime-op ground truth.

For this PyTorch build, the `make_fx` wrapper does not accept keyword arguments
and direct tracing of a bound method with `**kwargs` can fail argument-count
checks. The tool therefore records the original runtime kwargs, flattens them
for the FX call, and the offline FX target immediately calls the original
`layer.forward` with the same keyword names.

## Readable Code Rendering

`fx_trace_to_readable_code.py` renders `fx_nodes.json` into Python-like ATen
process code:

```bash
python /workspace/VisiPrune/workload_analysis/fx/fx_trace_to_readable_code.py \
  --trace-dir /workspace/VisiPrune/workload_analysis/fx/traces/fx_selected_layers \
  --recursive
```

This writes `fx_readable_process.py` and `fx_readable_process.md` next to each
found `fx_nodes.json`.

## GraphModule-Based Process Reconstruction

`fx_layer_process_reconstruct.py` is the process reconstruction path for saved
FX graphs:

```bash
python /workspace/VisiPrune/workload_analysis/fx/fx_layer_process_reconstruct.py \
  --trace-dir /workspace/VisiPrune/workload_analysis/fx/traces/fx_selected_layers \
  --recursive
```

This script loads `fx_graph_module.pt` with `torch.load(...,
weights_only=False)` and iterates `GraphModule.graph.nodes` directly. It does
not consume `fx_nodes.json`. For each layer trace directory it writes:

```text
fx_process_reconstruction.json
fx_process_reconstruction.md
fx_process_nodes.csv
```

For recursive runs it also writes:

```text
fx_process_reconstruction_manifest.json
fx_process_reconstruction_manifest.csv
```

FX provides an operation DAG, node users, arguments, and node metadata. It does
not provide official VisiPrune semantic process labels. The reconstruction
script therefore assigns readable process labels over the FX DAG, while keeping
the exact node names, targets, arguments, users, and node ranges visible in the
outputs for audit.

### What Process Reconstruction Actually Does

`fx_layer_process_reconstruct.py` loads the saved `GraphModule` and works from:

```python
list(graph_module.graph.nodes)
```

The script serializes each FX node's:

- index and name;
- FX op kind, such as `placeholder`, `get_attr`, `call_function`, or `output`;
- target, such as `aten.mm.default`, `aten.bmm.default`,
  `aten.index.Tensor`, or `aten._softmax.default`;
- direct input node names from `args` / `kwargs`;
- user node names from `node.users`;
- available tensor metadata.

It then assigns stage labels by pattern matching the low-level ATen DAG. The
current labels include `input_rmsnorm`, `qkv_projection`, `rope`,
`attention_scores`, `attention_output`, `visual_process`,
`output_projection`, `post_attention_rmsnorm`, `mlp`, and `layer_output`.

These labels are not recovered from runtime module hooks or FX official process
metadata. They are inferred by reconstruction rules, for example:

- first `aten.mm.default` group before attention is treated as Q/K/V
  projection;
- first `aten.index.Tensor` before attention is treated as the RoPE position
  lookup boundary;
- the first and second `aten.bmm.default` calls are treated as QK score
  computation and attention-weighted V computation;
- the first `aten.mm.default` after attention is treated as output projection;
- later `aten.mm.default` calls are treated as MLP projections;
- nodes between attention output and output projection are labeled
  `visual_process` only when such a gap exists.

This reconstruction is only a readable grouping over an FX ATen graph. It is
not proof of true VisiPrune process/module ownership.

### Practical Limitations Observed

The generated `GraphModule.graph.nodes` mostly contain low-level ATen
operations. In the current traces, most nodes do not carry enough
`nn_module_stack`, source-line, or process metadata to reconstruct the original
VisiPrune module/process hierarchy.

As a result:

- FX is useful for a fixed-input tensor DAG and direct node dependency view.
- FX is not sufficient by itself to recover reliable VisiPrune process/module
  semantics.
- `fx_process_reconstruction.md` should be read as an automatically grouped
  low-level graph, not as ground-truth process documentation.
- When FX reconstruction conflicts with dispatch evidence or source-level
  understanding, dispatch/source evidence should take priority.

Because serialized `GraphModule` objects are PyTorch-version-sensitive, run the
process script with the same analysis environment used for trace generation:

```bash
/workspace/VisiPrune/workload_analysis/env/run_with_analysis_env.sh \
  /workspace/VisiPrune/workload_analysis/fx/fx_layer_process_reconstruct.py \
  --trace-dir /workspace/VisiPrune/workload_analysis/fx/traces/fx_selected_layers \
  --recursive
```

The loader includes a narrow compatibility fix for this PyTorch 2.12
environment: it explicitly imports and attaches
`torch._higher_order_ops.triton_kernel_wrap` before `torch.load`, because FX
GraphModule deserialization can access that module through the package
attribute even when the package did not attach it by default.
