# E2 SAME_INPUT Single-Request Latency

This directory now keeps only the SAME_INPUT-oriented single-request latency
evidence path. Legacy dense-only, smoke, GEMV, Triton-only, and visualization
experiments were removed from this directory.

## Evidence Package

Root-level reports remain as the human-readable index. The corresponding
report package directories under `output/` contain report copies plus the raw
artifacts used by each report.

Primary layer-wise report packages:

```text
output/dense_fa2/
output/visipruner_full_eager/
output/visipruner_full_vp_fa/
output/three_way_summary/
```

Strict process-wise artifacts for VisiPrune eager full use a separate package:

```text
output/visipruner_full_eager_process_wise/
output/visipruner_full_eager_process_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md
output/visipruner_full_eager_process_wise/SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json
output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv
```

Process-level NVTX instrumentation handoff for VisiPrune eager full:

```text
FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
```

The `FA2` filename for the VisiPruner accelerated variant is a legacy filename.
When the implementation is `visipruner-full-vp-fa`, prose and interpretation
should call it VP-FA.

## Workflow

Use `$visipruner-same-input-layer-wise-workflow` for the current workflow. It describes how
to recover the FX-matched input contract, run Nsight layer profiling, check
launch-owned CUPTI attribution, and package the layer-wise report without a
separate clock sampling requirement.

Main orchestration script:

```text
code/run_same_input_three_way_layer_retest.sh
```

The script runs the three variants on the same image, prompt, token count, GPU,
and warmup policy:

```text
dense-fa2
visipruner-full
visipruner-full-vp-fa
```

## Kept Scripts

Current SAME_INPUT scripts:

```text
code/profile_visprune_single_request.py
code/run_same_input_three_way_layer_retest.sh
code/run_clock_layer_profile_single_request.sh
code/run_nsys_layer_profile_single_request.sh
code/analyze_layer_nsys.py
code/generate_layer_performance_report.py
code/audit_same_input_three_way_layer_retest.py
code/generate_process_performance_breakdown.py
```

`run_clock_layer_profile_single_request.sh` and
`run_nsys_layer_profile_single_request.sh` are generic runners kept because the
SAME_INPUT orchestration script calls them with deterministic `sameinput_*`
tags.

## Kept Raw Outputs

Current output package convention:

```text
output/dense_fa2/
output/visipruner_full_eager/
output/visipruner_full_eager_process_wise/
output/visipruner_full_vp_fa/
output/three_way_summary/
```

These packages include report copies, clock JSON/range/layer-event files,
Nsight `.nsys-rep` and SQLite exports, Nsight stats CSVs, layer-kernel
breakdown CSV/JSON files, SAME_INPUT audit files, or strict process
attribution CSV/JSON files associated with the report. Process-wise files stay
under `output/visipruner_full_eager_process_wise/`; their source Nsight trace
and SQLite stay under `output/visipruner_full_eager/`.

## Interpretation

Keep these metrics separate:

```text
clock end-to-end latency
NVTX CPU range duration
CUPTI launch-owned kernel sum
```

The CUPTI launch-owned kernel sum is built from CUDA runtime calls whose CPU
start timestamp falls inside the NVTX CPU range, joined to CUPTI kernels by
`correlationId`. It is not GPU-time overlap with the NVTX range, and it is not
an end-to-end latency metric.
