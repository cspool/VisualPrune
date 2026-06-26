#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/config.env" >&2
  exit 2
fi

CONFIG_FILE="$1"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
set +a

exec "${SCRIPT_DIR}/run_eval_backends.sh"
