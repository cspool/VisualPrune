# Analysis Summary

Generated with:

```bash
GPU=1 TOKENS=32 /workspace/VisiPrune/workload_analysis/algorithmic_trace/runners/run_full_forward.sh
```

The current complete run used real forward-driven traces for both VisiPrune and
dense-eager LLaVA.

## Algorithmic Trace

Main output:

`/workspace/VisiPrune/workload_analysis/algorithmic_trace/traces/fresh_forward_visipruner_full_32tok/algorithmic_trace.json`

CSV outputs:

- `layer_trace.csv`
- `selection_trace.csv`
- `operator_flops.csv`

Fresh forward token schedule:

- Prompt tokens before image expansion: `49`
- Visual tokens: `576`
- Prefill length: `624`
- Fixed prefix tokens before image span: `35`
- Text suffix tokens after image span: `13`
- Middle selection layer: `18`
- Selected visual tokens: `10`
- Layers `0-18`: sequence length `624`
- Layers `19-27`: sequence length `58`
- Layers `28-31`: sequence length `48`
- Decode forwards from real generate: `31`

Theoretical FLOP summary, counting multiply and add as 2 FLOPs:

- Full VLM actual model path: `6.044742909952e12`
- Full VLM with ideal last-token-only lm_head: `6.032422141952e12`
- Prefill LLM: `5.222644449280e12`
- Decode LLM: `4.24409661440e11`
- CLIP vision tower: `3.81918216192e11`
- Multimodal projector: `2.4159191040e10`

This excludes model loading, image I/O, CPU preprocessing, host-device transfer,
CUDA launch overhead, compiler/runtime fusion, memory traffic, and communication.

## Open-Tool Dense Baseline

Main output:

`/workspace/VisiPrune/workload_analysis/open_tool_dense_baseline/dense_baseline/open_tool_fit_report.json`

Generated files:

- `dense_equivalent_config.json`
- `llm_analysis_dense_summary.json`
- `llm_viewer_dense_summary.json`
- `dense_tool_comparison.csv`

Interpretation:

- `llm-analysis` provides a dense language-backbone inference lower-bound style
  latency/memory estimate.
- `LLM-Viewer` provides dense language-backbone operator/roofline style output.
- These tools do not natively represent VisiPrune's data-dependent middle
  selection and deep visual-token exit, nor LLaVA's CLIP/projector path.
- Use algorithmic trace as the authoritative VisiPrune workload trace; use open-tool dense baseline as
  dense LLaMA-context comparison.

## Fresh Dense-Eager Comparison

Dense trace:

`/workspace/VisiPrune/workload_analysis/algorithmic_trace/traces/fresh_forward_dense_eager_32tok/algorithmic_trace.json`

Comparison outputs:

- `/workspace/VisiPrune/workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok.json`
- `/workspace/VisiPrune/workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok_phase.csv`
- `/workspace/VisiPrune/workload_analysis/algorithmic_trace/comparisons/fresh_visipruner_vs_dense_32tok_ops.csv`

Result:

- Dense-eager full VLM actual model path: `9.275895808000e12` FLOPs
- VisiPrune full VLM actual model path: `6.044742909952e12` FLOPs
- Saved: `3.231152898048e12` FLOPs
- Saved percentage: `34.83%`
