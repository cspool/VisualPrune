# 02 Representative FX Process-wise Trace

目标：对代表 input-layer 采集 strict FX process-wise trace，得到 direct process timing、kernel families、validation status、GPU kernel launch order。

## Skills

```text
$visipruner-fx-process-nvtx-instrumentation
$visipruner-process-performance-breakdown
```

## Required GPU

```text
GPU=1
```

GPU 1 当前负载更轻，性能干扰更小。此 workflow 需要和 workflow 01 使用同一 GPU 策略。

## Expected Output Directory

```text
output/visipruner_full_eager_process_wise/
```

`output_bk/visipruner_full_eager_process_wise_bk/` 只作为字段和报告结构参考。该备份缺少原始 process-level `.nsys-rep/.sqlite`，不能恢复 GPU launch order。

## Possible Scripts

```text
code/profile_visprune_single_request.py
code/run_nsys_layer_profile_single_request.sh
code/analyze_layer_nsys.py
code/generate_process_performance_breakdown.py
```

## Required Handoff

```text
output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
```

The handoff must define process/fragment NVTX names, `aggregation_key`, FX nodes/op families, expected kernel families, same-input contract, and unresolved risks.

## Command Template

```bash
cd /workspace/VisiPrune

ROOT_DIR=/workspace/VisiPrune \
CONFIG=visipruner-full \
TAG=nsys_sameinput_visipruner_full_eager_32tok \
OUTPUT_DIR=/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise \
REPORT_DIR=/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise \
PROCESS_REPORT_DIR=/workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise \
FX_PROCESS_PROFILE=on \
MAX_NEW_TOKENS=32 \
WARMUP_ITERS=1 \
GPU=1 \
bash /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh
```

If a valid process-level sqlite already exists for the same input/code state, regenerate reports with:

```bash
python /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/code/generate_process_performance_breakdown.py \
  --sqlite /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok.sqlite \
  --layer-events /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_layer_events.csv \
  --handoff /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md \
  --variant visipruner-full-eager \
  --display-name "VisiPruner Full Eager" \
  --output-csv /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv \
  --output-json /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json \
  --process-csv /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv \
  --report /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md \
  --aggregate-report /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency/output/visipruner_full_eager_process_wise/SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
```

## Expected Files

```text
FX_PROCESS_NVTX_INSTRUMENTATION_HANDOFF.md
nsys_sameinput_visipruner_full_eager_32tok.json
nsys_sameinput_visipruner_full_eager_32tok.nsys-rep
nsys_sameinput_visipruner_full_eager_32tok.sqlite
nsys_sameinput_visipruner_full_eager_32tok_layer_events.csv
nsys_sameinput_visipruner_full_eager_32tok_stats_*.csv
SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md
SAME_INPUT_PROCESS_WISE_PERFORMANCE_BREAKDOWN.md
nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.csv
nsys_sameinput_visipruner_full_eager_32tok_process_nvtx_kernel_breakdown.json
same_input_visipruner_full_eager_process_attribution.csv
process_gpu_timeline.csv
process_kernel_launch_order.csv
```

`process_gpu_timeline.csv` and `process_kernel_launch_order.csv` should be derived from the same process-level sqlite. They must include process NVTX start/end, runtime call start/end, CUPTI kernel start/end, stream, correlationId, kernel name/family, and `gpu_order_basis`.

## Checks

```bash
cd /workspace/VisiPrune/autoresearch/experiments/e2_single_request_latency

sqlite3 output/visipruner_full_eager_process_wise/nsys_sameinput_visipruner_full_eager_32tok.sqlite \
  "select count(*) from NVTX_EVENTS where text like 'visprune.fx_process%';"

rg -n 'Representative Layer Process GPU Execution Order|gpu_order_basis|first_launch_owned_gpu_kernel_start' \
  output/visipruner_full_eager_process_wise/SAME_INPUT_VISIPRUNER_FULL_EAGER_PROCESS_WISE_PERFORMANCE_REPORT.md \
  output/visipruner_full_eager_process_wise/same_input_visipruner_full_eager_process_attribution.csv
```

## Constraints

- No `visprune.fx_process.*` ranges means no strict process-wise report.
- The report must include per-parent-layer process launch order sorted by first launch-owned CUPTI kernel GPU start.
- No-kernel rows fall back to first CUDA runtime call start, then process NVTX start; expose this in `gpu_order_basis`.
- Do not infer GPU order from old CSV/report files that lack timestamp fields.
