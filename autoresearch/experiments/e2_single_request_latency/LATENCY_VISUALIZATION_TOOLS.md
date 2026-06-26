# Single-Request Latency Visualization Tools

Date: 2026-06-22

## Generated Visualizations

All outputs are under:

```text
autoresearch/experiments/e2_single_request_latency/output/visualizations/
```

Artifacts:

| artifact | purpose |
|---|---|
| `single_request_latency_breakdown.png` | Static report figure: clock stage stack, Nsight kernel family stack, decode-iteration lines |
| `single_request_latency_dashboard.html` | Self-contained Plotly dashboard for local browser inspection |
| `single_request_latency_trace_chrome.json` | Chrome Trace Event JSON for Perfetto UI or `chrome://tracing` |
| `latency_stage_breakdown.csv` | Source table for request-stage breakdown |
| `kernel_family_breakdown.csv` | Source table for Nsight CUDA kernel family breakdown |
| `decode_iteration_lines.csv` | Source table for per-decode-iteration line plot |

Regenerate:

```bash
cd /workspace/VisPrune
/workspace/VisPrune/venv_profiling/bin/python \
  autoresearch/experiments/e2_single_request_latency/code/visualize_latency_breakdown.py
```

## Recommended Tool Stack

### 1. NVIDIA Nsight Systems GUI

Best for authoritative CUDA/NVTX timeline inspection.

Current files:

```text
output/nsys_dense_fa2_32tok.nsys-rep
output/nsys_visprune_full_32tok.nsys-rep
```

Open:

```bash
/opt/nvidia/nsight-systems/2026.3.1/bin/nsys-ui \
  /workspace/VisPrune/autoresearch/experiments/e2_single_request_latency/output/nsys_visprune_full_32tok.nsys-rep
```

Use this when checking:

- CPU-side NVTX ranges such as `visprune.request`, `generate_total`, `forward_decode`;
- CUDA kernel stream placement and gaps;
- GPU projection of NVTX ranges;
- exact kernel names and launch order.

Official references:

- NVIDIA Nsight Systems product page: https://developer.nvidia.com/nsight-systems
- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html

### 2. Perfetto UI / Chrome Trace Viewer

Best for shareable browser-based timeline viewing without requiring Nsight GUI.

Generated file:

```text
output/visualizations/single_request_latency_trace_chrome.json
```

Open in either:

```text
https://ui.perfetto.dev/
chrome://tracing
```

The generated JSON contains:

- process lane `dense-fa2`;
- process lane `visipruner-full`;
- NVTX range track;
- CUDA kernel family track with original kernel names in event args.

Perfetto is useful for communicating the high-level timeline, but Nsight
Systems remains the authoritative view for CUDA-specific details.

Official/open-source references:

- Perfetto project: https://perfetto.dev/
- Perfetto UI docs: https://perfetto.dev/docs/visualization/perfetto-ui
- Perfetto external trace format docs: https://perfetto.dev/docs/getting-started/other-formats
- Perfetto source: https://github.com/google/perfetto

### 3. Plotly / Matplotlib Local Figures

Best for reports, papers, and fast side-by-side comparison.

Generated files:

```text
output/visualizations/single_request_latency_breakdown.png
output/visualizations/single_request_latency_dashboard.html
```

Current local versions:

```text
plotly 6.8.0
matplotlib 3.11.0
```

The dashboard shows:

1. clock request latency breakdown;
2. Nsight CUDA kernel family duration;
3. VisiPrune decode iteration breakdown.

This view is less detailed than Nsight/Perfetto, but it is better for quickly
answering whether the request is dominated by decode, prefill, or kernel family
time.

### 4. PyTorch Profiler / TensorBoard

Best as an alternative future instrumentation path when operator-level PyTorch
events and memory accounting are more important than Nsight's CUDA timeline.

It is not the primary artifact for this experiment because the existing
pipeline already records NVTX ranges and Nsight Systems traces. If needed, a
future profiling pass can wrap the measured request in `torch.profiler.profile`
and export a Chrome trace JSON.

Official references:

- PyTorch Profiler API: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch profiler recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- `export_chrome_trace`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.export_chrome_trace.html

## Practical Recommendation

Use the tools in this order:

1. Open `single_request_latency_breakdown.png` or
   `single_request_latency_dashboard.html` for the summary.
2. Open `single_request_latency_trace_chrome.json` in Perfetto UI when sharing a
   browser timeline is enough.
3. Open `.nsys-rep` in Nsight Systems GUI for the definitive CUDA/NVTX view.
4. Add PyTorch Profiler only if a future question requires PyTorch operator
   stack traces, memory timeline, or `record_function` scope attribution beyond
   the current NVTX ranges.

## Current Visualization Conclusion

The generated views show the same conclusion as the raw numbers:

- single-request latency is dominated by `forward_decode`;
- VisiPrune's request is slower than dense-FA2 in clock timing;
- CUDA kernel total is slightly lower for VisiPrune, but GEMV remains the
  dominant kernel family;
- each decode iteration repeats a stable GEMV-heavy pattern rather than having
  one isolated bad token.
