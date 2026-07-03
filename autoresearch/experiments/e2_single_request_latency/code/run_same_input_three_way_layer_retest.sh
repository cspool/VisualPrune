#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
REPORT_DIR="${REPORT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency}"
CLOCK_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/run_clock_layer_profile_single_request.sh"
NSYS_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh"
REPORT_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/generate_layer_performance_report.py"
AUDIT_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/audit_same_input_three_way_layer_retest.py"

COMMON_IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg}"
COMMON_PROMPT="${PROMPT:-Describe the image briefly.}"
COMMON_TOKENS="${MAX_NEW_TOKENS:-32}"
COMMON_GPU="${GPU:-1}"
COMMON_WARMUP="${WARMUP_ITERS:-1}"

mkdir -p "${OUTPUT_DIR}" "${REPORT_DIR}"
cd "${ROOT_DIR}"

preflight_cuda() {
  CUDA_VISIBLE_DEVICES="${COMMON_GPU}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

if not torch.cuda.is_available():
    print(
        "CUDA preflight failed: torch.cuda.is_available() is false "
        f"with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        file=sys.stderr,
    )
    raise SystemExit(2)
if torch.cuda.device_count() < 1:
    print("CUDA preflight failed: no CUDA devices visible", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight ok: {torch.cuda.get_device_name(0)}")
PY
}

run_one() {
  local config="$1"
  local tag="$2"
  local title="$3"
  local report="$4"
  local clock_tag="clock_${tag}_${COMMON_TOKENS}tok"
  local nsys_tag="nsys_${tag}_${COMMON_TOKENS}tok"

  echo "START ${tag} clock"
  CONFIG="${config}" \
    IMAGE_PATH="${COMMON_IMAGE_PATH}" \
    PROMPT="${COMMON_PROMPT}" \
    MAX_NEW_TOKENS="${COMMON_TOKENS}" \
    TAG="${clock_tag}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    GPU="${COMMON_GPU}" \
    WARMUP_ITERS="${COMMON_WARMUP}" \
    "${CLOCK_SCRIPT}" >"${OUTPUT_DIR}/${clock_tag}.log" 2>&1
  echo "DONE ${tag} clock"

  echo "START ${tag} nsys"
  CONFIG="${config}" \
    IMAGE_PATH="${COMMON_IMAGE_PATH}" \
    PROMPT="${COMMON_PROMPT}" \
    MAX_NEW_TOKENS="${COMMON_TOKENS}" \
    TAG="${nsys_tag}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    GPU="${COMMON_GPU}" \
    WARMUP_ITERS="${COMMON_WARMUP}" \
    "${NSYS_SCRIPT}" >"${OUTPUT_DIR}/${nsys_tag}.log" 2>&1
  echo "DONE ${tag} nsys"

  "${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --clock-json "${OUTPUT_DIR}/${clock_tag}.json" \
    --clock-ranges "${OUTPUT_DIR}/${clock_tag}_ranges.csv" \
    --layer-events "${OUTPUT_DIR}/${clock_tag}_layer_events.csv" \
    --nsys-layer-csv "${OUTPUT_DIR}/${nsys_tag}_layer_kernel_breakdown.csv" \
    --output "${report}" \
    --title "${title}"
}

preflight_cuda

run_one \
  dense-fa2 \
  sameinput_dense_fa2 \
  "Same-input dense-FA2 layer performance report" \
  "${REPORT_DIR}/SAME_INPUT_DENSE_FA2_LAYER_PERFORMANCE_REPORT.md"

run_one \
  visipruner-full \
  sameinput_visipruner_full_eager \
  "Same-input eager VisiPruner full layer performance report" \
  "${REPORT_DIR}/SAME_INPUT_VISIPRUNER_FULL_EAGER_LAYER_PERFORMANCE_REPORT.md"

run_one \
  visipruner-full-fa2 \
  sameinput_visipruner_full_fa2 \
  "Same-input VisiPruner-FA2 layer performance report" \
  "${REPORT_DIR}/SAME_INPUT_VISIPRUNER_FULL_FA2_LAYER_PERFORMANCE_REPORT.md"

"${PYTHON_BIN}" "${AUDIT_SCRIPT}" \
  --root-dir "${ROOT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --report-dir "${REPORT_DIR}" \
  --tokens "${COMMON_TOKENS}" \
  --expected-image "${COMMON_IMAGE_PATH}" \
  --expected-prompt "${COMMON_PROMPT}" \
  --json-output "${OUTPUT_DIR}/same_input_three_way_audit.json"

echo "SAME_INPUT_THREE_WAY_LAYER_RETEST_DONE"
