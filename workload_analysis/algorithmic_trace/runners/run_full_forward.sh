#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_ANALYSIS_DIR="${WORKLOAD_ANALYSIS_DIR:-/workspace/VisiPrune/workload_analysis}"
RUNNER="${WORKLOAD_ANALYSIS_DIR}/env/run_with_analysis_env.sh"
GPU="${GPU:-1}"
TOKENS="${TOKENS:-32}"

VIS_TRACE="${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/traces/fresh_forward_visipruner_full_${TOKENS}tok/algorithmic_trace.json"
DENSE_TRACE="${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/traces/fresh_forward_dense_eager_${TOKENS}tok/algorithmic_trace.json"

"${RUNNER}" "${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/tools/visipruner_algorithmic_trace.py" \
  --config visipruner-full \
  --max-new-tokens "${TOKENS}" \
  --gpu "${GPU}" \
  --tag "fresh_forward_visipruner_full_${TOKENS}tok"

"${RUNNER}" "${WORKLOAD_ANALYSIS_DIR}/open_tool_dense_baseline/tools/open_tool_compare.py" \
  --trace "${VIS_TRACE}"

"${RUNNER}" "${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/tools/visipruner_algorithmic_trace.py" \
  --config dense-eager \
  --max-new-tokens "${TOKENS}" \
  --gpu "${GPU}" \
  --tag "fresh_forward_dense_eager_${TOKENS}tok"

"${RUNNER}" "${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/tools/compare_algorithmic_traces.py" \
  --baseline "${DENSE_TRACE}" \
  --candidate "${VIS_TRACE}" \
  --tag "fresh_visipruner_vs_dense_${TOKENS}tok"

printf '%s\n' "${VIS_TRACE}" > "${WORKLOAD_ANALYSIS_DIR}/algorithmic_trace/traces/latest_algorithmic_trace_path.txt"
