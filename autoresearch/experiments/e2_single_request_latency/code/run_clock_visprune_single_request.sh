#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisPrune}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv_profiling/bin/python}"
SCRIPT="${ROOT_DIR}/autoresearch/experiments/e2_single_request_latency/code/profile_visprune_single_request.py"

export HF_HOME="${VISPRUNE_HF_HOME:-${ROOT_DIR}/models}"
export HUGGINGFACE_HUB_CACHE="${VISPRUNE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" "${SCRIPT}" \
  --config "${CONFIG:-visipruner-full}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
  --warmup-iters "${WARMUP_ITERS:-1}" \
  --gpu "${GPU:-1}" \
  --sync-timing on \
  --nvtx on \
  --tag "${TAG:-clock_visprune_full_32tok}"
