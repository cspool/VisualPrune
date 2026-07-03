#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
REPORT_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/generate_triton_vpfa_layer_report.py"

CLOCK_TAG="${CLOCK_TAG:-clock_triton_vpfa_layer_32tok}"
NSYS_TAG="${NSYS_TAG:-nsys_triton_vpfa_layer_32tok}"
REPORT_PATH="${REPORT_PATH:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/TRITON_VPFA_LAYER_PERFORMANCE_REPORT.md}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
  --clock-json "${OUTPUT_DIR}/${CLOCK_TAG}.json" \
  --clock-ranges "${OUTPUT_DIR}/${CLOCK_TAG}_ranges.csv" \
  --layer-events "${OUTPUT_DIR}/${CLOCK_TAG}_layer_events.csv" \
  --nsys-layer-csv "${OUTPUT_DIR}/${NSYS_TAG}_layer_kernel_breakdown.csv" \
  --output "${REPORT_PATH}"
