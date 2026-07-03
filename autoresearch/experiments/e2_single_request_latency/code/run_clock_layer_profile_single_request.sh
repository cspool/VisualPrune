#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/output}"
CONFIG_NAME="${CONFIG:-dense-fa2}"
TOKENS="${MAX_NEW_TOKENS:-32}"
CONFIG_TAG="${CONFIG_NAME//-/_}"
TAG="${TAG:-clock_${CONFIG_TAG}_layer_${TOKENS}tok}"

export HF_HOME="${VISPRUNE_HF_HOME:-${ROOT_DIR}/models}"
export HUGGINGFACE_HUB_CACHE="${VISPRUNE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -n "${FLASH_ATTN_SITE_PACKAGES:-}" && -d "${FLASH_ATTN_SITE_PACKAGES}" ]]; then
  export PYTHONPATH="${FLASH_ATTN_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" "${SCRIPT}" \
  --config "${CONFIG_NAME}" \
  --image-path "${IMAGE_PATH:-${ROOT_DIR}/autoresearch/data/benchmark_images/002901d9d194c4fb.jpg}" \
  --prompt "${PROMPT:-Describe the image briefly.}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-new-tokens "${TOKENS}" \
  --warmup-iters "${WARMUP_ITERS:-1}" \
  --gpu "${GPU:-1}" \
  --sync-timing on \
  --nvtx on \
  --layer-profile \
  --tag "${TAG}"
