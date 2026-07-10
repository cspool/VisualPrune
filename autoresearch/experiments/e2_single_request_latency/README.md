# E2 SAME_INPUT Single-Request Latency

This directory now keeps only the SAME_INPUT-oriented single-request latency
evidence path. Legacy dense-only, smoke, GEMV, Triton-only, and visualization
experiments were removed from this directory.

## Evidence Package

This README is the root-level human-readable index. New report package
directories are expected under `output/`. Historical reference artifacts are
kept under `output_bk/`; treat them as examples for paths and fields because
they are known to have gaps.

Primary layer-wise report packages:

```text
output/dense_fa2/
output/visipruner_full_eager_layer_wise/
output/visipruner_full_vp_fa/
output/three_way_summary/
```

Strict process-wise artifacts for VisiPrune eager full use a separate package:

```text
output/visipruner_full_eager_process_wise/
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok.nsys-rep
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok.sqlite
output/visipruner_full_eager_process_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md
output/visipruner_full_eager_process_wise/SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv
output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json
output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv
```

Process-wise GPU hardware diagnostics use a separate package:

```text
output/visipruner_full_eager_process_wise_ncu/
output/visipruner_full_eager_process_wise_ncu/ncu_<layer>_<phase>_<process>.ncu-rep
output/visipruner_full_eager_process_wise_ncu/ncu_<layer>_<phase>_<process>_metrics.csv
output/visipruner_full_eager_process_wise_ncu/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_NCU_REPORT.md
```

Process-level NVTX instrumentation handoff for VisiPrune eager full:

```text
output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
```

The `FA2` filename for the VisiPruner accelerated variant is a legacy filename.
When the implementation is `visipruner-full-vp-fa`, prose and interpretation
should call it VP-FA.

## Workflow

Use `VISIPRUNER_FULL_EAGER_PROCESS_ATTRIBUTION_WORKFLOW.md` as the index for
the VisiPruner full eager attribution flow. The executable runbooks are:

```text
workflows/01_layer_wise_end_to_end_trace.md
workflows/02_representative_fx_process_wise_trace.md
workflows/03_process_gpu_hardware_trace.md
workflows/04_full_layer_fx_process_wise_estimate.md
```

The relevant skills are `$visipruner-same-input-layer-wise-workflow`,
`$visipruner-fx-process-nvtx-instrumentation`,
`$visipruner-process-performance-breakdown`, and
`$visipruner-segmented-process-attribution`.

All performance-collection workflows require `GPU=1`; GPU 1 currently has
lighter load and lower expected measurement interference.

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
code/generate_segmented_process_attribution.py
```

`run_clock_layer_profile_single_request.sh` and
`run_nsys_layer_profile_single_request.sh` are generic runners kept because the
SAME_INPUT orchestration script calls them with deterministic `sameinput_*`
tags.

## Kept Raw Outputs

Current output package convention:

```text
output/dense_fa2/
output/visipruner_full_eager_layer_wise/
output/visipruner_full_eager_process_wise/
output/visipruner_full_eager_process_wise_ncu/
output/visipruner_full_vp_fa/
output/three_way_summary/
```

These packages include reports, clock JSON/range/layer-event files,
Nsight `.nsys-rep` and SQLite exports, Nsight stats CSVs, layer-kernel
breakdown CSV/JSON files, SAME_INPUT audit files, or strict process
attribution CSV/JSON files associated with the report. The expected
process-wise package should keep its own process-level Nsight `.nsys-rep` and
SQLite, because derived CSV/report files alone cannot recover process GPU
launch order.

Historical reference packages, if needed, are under:

```text
output_bk/
```

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
