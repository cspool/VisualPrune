#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py"
ANALYZE_SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/analyze_layer_nsys.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/2026.3.1/bin/nsys}"
CONFIG_NAME="${CONFIG:-dense-fa2}"
TOKENS="${MAX_NEW_TOKENS:-32}"
CONFIG_TAG="${CONFIG_NAME//-/_}"
TAG="${TAG:-nsys_${CONFIG_TAG}_layer_${TOKENS}tok}"

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
    --config "${CONFIG_NAME}" \
    --image-path "${IMAGE_PATH:-${ROOT_DIR}/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg}" \
    --prompt "${PROMPT:-Describe the image briefly.}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens "${TOKENS}" \
    --warmup-iters "${WARMUP_ITERS:-1}" \
    --gpu "${GPU:-1}" \
    --sync-timing off \
    --nvtx on \
    --cuda-profiler-api \
    --layer-profile \
    --tag "${TAG}"

"${NSYS_BIN}" stats \
  --report nvtx_sum,nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum \
  --format csv \
  --force-export true \
  --force-overwrite true \
  --output "${OUTPUT_DIR}/${TAG}_stats" \
  "${OUTPUT_DIR}/${TAG}.nsys-rep"

"${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
  --sqlite "${OUTPUT_DIR}/${TAG}.sqlite" \
  --layer-events "${OUTPUT_DIR}/${TAG}_layer_events.csv" \
  --output "${OUTPUT_DIR}/${TAG}_layer_kernel_breakdown.json"
