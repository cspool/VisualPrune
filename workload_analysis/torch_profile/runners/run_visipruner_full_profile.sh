#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/VisiPrune}"
GPU="${GPU:-1}"
TOKENS="${TOKENS:-2}"
TAG="${TAG:-visipruner_full_${TOKENS}tok_stack_modules}"

"${ROOT_DIR}/workload_analysis/env/run_with_analysis_env.sh" \
  "${ROOT_DIR}/workload_analysis/torch_profile/tools/torch_profiler_generate_trace.py" \
  --config visipruner-full \
  --gpu "${GPU}" \
  --max-new-tokens "${TOKENS}" \
  --tag "${TAG}"

