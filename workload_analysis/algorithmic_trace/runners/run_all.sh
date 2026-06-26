#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_ANALYSIS_DIR="${WORKLOAD_ANALYSIS_DIR:-/workspace/VisiPrune/workload_analysis}"
RUNNER="${WORKLOAD_ANALYSIS_DIR}/env/run_with_analysis_env.sh"

if [ "$#" -gt 0 ]; then
  echo "run_all.sh uses the GPU-free e2 reconstruction path; ignoring extra args: $*" >&2
fi

RECONSTRUCT="${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/tools/reconstruct_algorithmic_trace_from_e2.py"
if [ ! -f "${RECONSTRUCT}" ]; then
  echo "missing e2 reconstruction script: ${RECONSTRUCT}" >&2
  exit 1
fi

"${RUNNER}" "${RECONSTRUCT}"
"${RUNNER}" "${WORKLOAD_ANALYSIS_DIR}/open_tool_dense_baseline/tools/open_tool_compare.py"
