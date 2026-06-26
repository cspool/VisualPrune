#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/2026.3.1/bin/nsys}"
TAG="${TAG:-nsys_dense_fa2_32tok}"

export HF_HOME="${VISPRUNE_HF_HOME:-${ROOT_DIR}/models}"
export HUGGINGFACE_HUB_CACHE="${VISPRUNE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -n "${FLASH_ATTN_SITE_PACKAGES:-}" && -d "${FLASH_ATTN_SITE_PACKAGES}" ]]; then
  export PYTHONPATH="${FLASH_ATTN_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

"${NSYS_BIN}" profile \
  --trace=cuda,nvtx,cublas,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --stats=true \
  --output "${OUTPUT_DIR}/${TAG}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
    --config "${CONFIG:-dense-fa2}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --warmup-iters "${WARMUP_ITERS:-1}" \
    --gpu "${GPU:-1}" \
    --sync-timing off \
    --nvtx on \
    --cuda-profiler-api \
    --tag "${TAG}"

"${NSYS_BIN}" stats \
  --report nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum \
  --format csv \
  --force-export true \
  --force-overwrite true \
  --output "${OUTPUT_DIR}/${TAG}_stats" \
  "${OUTPUT_DIR}/${TAG}.nsys-rep"
