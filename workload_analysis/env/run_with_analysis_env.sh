#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_ANALYSIS_DIR="${WORKLOAD_ANALYSIS_DIR:-/workspace/VisiPrune/workload_analysis}"
RUNNER_PYTHON="${RUNNER_PYTHON:-/workspace/VisiPrune/venv_profiling/bin/python}"

export HF_HOME="${HF_HOME:-/workspace/VisiPrune/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKLOAD_ANALYSIS_DIR}/vendor/pip_cache}"
export PYTHONPATH="${WORKLOAD_ANALYSIS_DIR}/vendor/python:${WORKLOAD_ANALYSIS_DIR}/external/llm-analysis:${WORKLOAD_ANALYSIS_DIR}/external/llm-viewer:${PYTHONPATH:-}"

exec "${RUNNER_PYTHON}" "$@"
