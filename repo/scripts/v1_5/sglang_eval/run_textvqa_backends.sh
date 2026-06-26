#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-textvqa}" exec "${SCRIPT_DIR}/run_eval_backends.sh" "$@"
