#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
DENSE_PACKAGE_DIR="${DENSE_PACKAGE_DIR:-${OUTPUT_DIR}/dense_fa2}"
EAGER_PACKAGE_DIR="${EAGER_PACKAGE_DIR:-${OUTPUT_DIR}/visipruner_full_eager}"
VPFA_PACKAGE_DIR="${VPFA_PACKAGE_DIR:-${OUTPUT_DIR}/visipruner_full_vp_fa}"
THREE_WAY_PACKAGE_DIR="${THREE_WAY_PACKAGE_DIR:-${OUTPUT_DIR}/three_way_summary}"
CLOCK_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/run_clock_layer_profile_single_request.sh"
NSYS_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/run_nsys_layer_profile_single_request.sh"
REPORT_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/generate_layer_performance_report.py"
AUDIT_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/audit_same_input_three_way_layer_retest.py"

COMMON_IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg}"
COMMON_PROMPT="${PROMPT:-Describe the image briefly.}"
COMMON_TOKENS="${MAX_NEW_TOKENS:-32}"
COMMON_GPU="${GPU:-1}"
COMMON_WARMUP="${WARMUP_ITERS:-1}"

mkdir -p "${OUTPUT_DIR}" "${DENSE_PACKAGE_DIR}" "${EAGER_PACKAGE_DIR}" "${VPFA_PACKAGE_DIR}" "${THREE_WAY_PACKAGE_DIR}"
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
  local package_dir="$4"
  local report_name="$5"
  local fx_process_profile="${6:-${FX_PROCESS_PROFILE:-off}}"
  local clock_tag="clock_${tag}_${COMMON_TOKENS}tok"
  local nsys_tag="nsys_${tag}_${COMMON_TOKENS}tok"
  local report="${package_dir}/${report_name}"

  mkdir -p "${package_dir}"

  echo "START ${tag} clock"
  CONFIG="${config}" \
    IMAGE_PATH="${COMMON_IMAGE_PATH}" \
    PROMPT="${COMMON_PROMPT}" \
    MAX_NEW_TOKENS="${COMMON_TOKENS}" \
    TAG="${clock_tag}" \
    OUTPUT_DIR="${package_dir}" \
    GPU="${COMMON_GPU}" \
    WARMUP_ITERS="${COMMON_WARMUP}" \
    FX_PROCESS_PROFILE="${fx_process_profile}" \
    "${CLOCK_SCRIPT}" >"${package_dir}/${clock_tag}.log" 2>&1
  echo "DONE ${tag} clock"

  echo "START ${tag} nsys"
  CONFIG="${config}" \
    IMAGE_PATH="${COMMON_IMAGE_PATH}" \
    PROMPT="${COMMON_PROMPT}" \
    MAX_NEW_TOKENS="${COMMON_TOKENS}" \
    TAG="${nsys_tag}" \
    OUTPUT_DIR="${package_dir}" \
    GPU="${COMMON_GPU}" \
    WARMUP_ITERS="${COMMON_WARMUP}" \
    FX_PROCESS_PROFILE="${fx_process_profile}" \
    "${NSYS_SCRIPT}" >"${package_dir}/${nsys_tag}.log" 2>&1
  echo "DONE ${tag} nsys"

  "${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --clock-json "${package_dir}/${clock_tag}.json" \
    --clock-ranges "${package_dir}/${clock_tag}_ranges.csv" \
    --layer-events "${package_dir}/${clock_tag}_layer_events.csv" \
    --nsys-layer-csv "${package_dir}/${nsys_tag}_layer_kernel_breakdown.csv" \
    --output "${report}" \
    --title "${title}"
}

preflight_cuda

run_one \
  dense-fa2 \
  sameinput_dense_fa2 \
  "Same-input dense-FA2 layer performance report" \
  "${DENSE_PACKAGE_DIR}" \
  "SAME_INPUT_DENSE_FA2_LAYER_PERFORMANCE_REPORT.md"

run_one \
  visipruner-full \
  sameinput_visipruner_full_eager \
  "Same-input eager VisiPruner full layer performance report" \
  "${EAGER_PACKAGE_DIR}" \
  "SAME_INPUT_VISIPRUNER_FULL_EAGER_LAYER_PERFORMANCE_REPORT.md"

run_one \
  visipruner-full-vp-fa \
  sameinput_visipruner_full_vpfa \
  "Same-input VisiPruner VP-FA layer performance report" \
  "${VPFA_PACKAGE_DIR}" \
  "SAME_INPUT_VISIPRUNER_FULL_FA2_LAYER_PERFORMANCE_REPORT.md"

"${PYTHON_BIN}" "${AUDIT_SCRIPT}" \
  --root-dir "${ROOT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tokens "${COMMON_TOKENS}" \
  --expected-image "${COMMON_IMAGE_PATH}" \
  --expected-prompt "${COMMON_PROMPT}" \
  --json-output "${THREE_WAY_PACKAGE_DIR}/same_input_three_way_audit.json"

echo "SAME_INPUT_THREE_WAY_LAYER_RETEST_DONE"
