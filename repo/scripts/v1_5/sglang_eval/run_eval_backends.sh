#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SGLANG_DIR="${SGLANG_DIR:-${ROOT_DIR}/sglang}"
PYTHON_BIN="${PYTHON_BIN:-${SGLANG_DIR}/.venv/bin/python}"

TASK="${TASK:-textvqa}"
DATASET_NAME="${DATASET_NAME-}"
QUESTION_FILE="${QUESTION_FILE-}"
IMAGE_FOLDER="${IMAGE_FOLDER-}"
OUTPUT_DIR="${OUTPUT_DIR-}"
IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH-}"
IMAGE_FIELD="${IMAGE_FIELD:-image}"
QUESTION_FIELD="${QUESTION_FIELD:-text}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT_DIR}/models/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234}"
MODEL_VIEW_DIR="${MODEL_VIEW_DIR:-${ROOT_DIR}/models/sglang/llava-v1.5-7b}"
CLIP_PREPROCESSOR_CONFIG="${CLIP_PREPROCESSOR_CONFIG:-${ROOT_DIR}/models/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/preprocessor_config.json}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_ID="${MODEL_ID:-llava-v1.5-7b-sglang}"

BACKENDS="${BACKENDS:-triton torch_native dsa-tilelang}"
PORT_BASE="${PORT_BASE:-30000}"
HOST="${HOST:-127.0.0.1}"
LIMIT="${LIMIT:-16}"
OFFSET="${OFFSET:-0}"
WARMUP="${WARMUP:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
DTYPE="${DTYPE:-float16}"
CONTINUE_ON_BACKEND_FAIL="${CONTINUE_ON_BACKEND_FAIL:-1}"
PRUNING_CONFIG="${PRUNING_CONFIG:-{\"mode\":[\"shallow\",\"middle\",\"deep\"],\"shallow_mid_layer\":6,\"layer_threshold\":0.995,\"tokens_threshold\":0.2}}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

task_defaults() {
  case "${TASK}" in
    textvqa)
      DATASET_NAME="${DATASET_NAME:-textvqa}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/textvqa/train_images}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/textvqa/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    mme)
      DATASET_NAME="${DATASET_NAME:-mme}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/MME/llava_mme.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/MME/MME_Benchmark_release_version}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/MME/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    gqa)
      DATASET_NAME="${DATASET_NAME:-gqa}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/gqa/data/images}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/gqa/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    pope)
      DATASET_NAME="${DATASET_NAME:-pope}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/pope/llava_pope_test.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/pope/val2014}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/pope/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-2}"
      ;;
    mmvet|mm-vet)
      DATASET_NAME="${DATASET_NAME:-mmvet}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/mm-vet/llava-mm-vet.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/mmvet/mm-vet/images}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/mm-vet/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    vqav2)
      DATASET_NAME="${DATASET_NAME:-vqav2}"
      QUESTION_FILE="${QUESTION_FILE:-${ROOT_DIR}/repo/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl}"
      IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT_DIR}/repo/playground/data/eval/vqav2/test2015}"
      OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/repo/playground/data/eval/vqav2/answers/sglang_backends}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    custom)
      DATASET_NAME="${DATASET_NAME:-custom}"
      IMAGE_SEARCH_DEPTH="${IMAGE_SEARCH_DEPTH:-0}"
      ;;
    *)
      echo "Unsupported TASK=${TASK}. Use textvqa, mme, gqa, pope, mmvet, vqav2, or custom." >&2
      exit 2
      ;;
  esac

  if [[ -z "${QUESTION_FILE}" || -z "${IMAGE_FOLDER}" || -z "${OUTPUT_DIR}" ]]; then
    echo "QUESTION_FILE, IMAGE_FOLDER, and OUTPUT_DIR are required for TASK=${TASK}" >&2
    exit 2
  fi
}

prepare_model_path() {
  if [[ -n "${MODEL_PATH}" ]]; then
    echo "${MODEL_PATH}"
    return 0
  fi
  if [[ -f "${BASE_MODEL_PATH}/preprocessor_config.json" ]]; then
    echo "${BASE_MODEL_PATH}"
    return 0
  fi
  if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "Base model path not found: ${BASE_MODEL_PATH}" >&2
    return 1
  fi
  if [[ ! -f "${CLIP_PREPROCESSOR_CONFIG}" ]]; then
    echo "CLIP preprocessor config not found: ${CLIP_PREPROCESSOR_CONFIG}" >&2
    return 1
  fi
  mkdir -p "${MODEL_VIEW_DIR}"
  find "${MODEL_VIEW_DIR}" -mindepth 1 -maxdepth 1 -type l -delete
  for item in "${BASE_MODEL_PATH}"/*; do
    ln -sfn "${item}" "${MODEL_VIEW_DIR}/$(basename "${item}")"
  done
  ln -sfn "${CLIP_PREPROCESSOR_CONFIG}" "${MODEL_VIEW_DIR}/preprocessor_config.json"
  echo "${MODEL_VIEW_DIR}"
}

backend_args() {
  case "$1" in
    triton)
      printf '%s\n' --attention-backend triton --sampling-backend pytorch
      ;;
    torch_native)
      printf '%s\n' --attention-backend torch_native --sampling-backend pytorch
      ;;
    flashinfer)
      printf '%s\n' --attention-backend flashinfer --sampling-backend flashinfer
      ;;
    dsa-tilelang|tilelang)
      printf '%s\n' --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --sampling-backend pytorch
      ;;
    nsa-tilelang)
      printf '%s\n' --attention-backend nsa --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --sampling-backend pytorch
      ;;
    *)
      echo "Unknown backend: $1" >&2
      return 2
      ;;
  esac
}

wait_for_server() {
  local base_url="$1"
  local pid="$2"
  for _ in $(seq 1 "${SERVER_WAIT_SECONDS}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "${base_url}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

write_backend_status() {
  local backend="$1"
  local status="$2"
  local message="$3"
  local file="${OUTPUT_DIR}/${backend}.status.json"
  "${PYTHON_BIN}" - "$backend" "$status" "$message" "$file" <<'PY'
import json
import sys
from datetime import datetime, timezone

backend, status, message, file_name = sys.argv[1:5]
payload = {
    "backend": backend,
    "status": status,
    "message": message,
    "time": datetime.now(timezone.utc).isoformat(),
}
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

classify_server_failure() {
  local server_log="$1"
  if grep -q "DSA backend only supports DeepSeek DSA" "${server_log}" 2>/dev/null; then
    printf '%s\t%s\n' "unsupported" "DSA backend only supports DeepSeek DSA for this model"
    return 0
  fi
  if grep -q "NSA backend only supports" "${server_log}" 2>/dev/null; then
    printf '%s\t%s\n' "unsupported" "NSA backend is not supported for this model"
    return 0
  fi
  printf '%s\t%s\n' "server_failed" "server did not become healthy"
}

aggregate_summaries() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
statuses = {}
for path in sorted(out_dir.glob("*.status.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    statuses[data.get("backend") or path.stem.replace(".status", "")] = data

seen = set()
for path in sorted(out_dir.glob("*.metrics.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    backend = data.get("backend")
    status = statuses.get(backend, {})
    seen.add(backend)
    rows.append({
        "backend": backend,
        "status": status.get("status", "ok"),
        "message": status.get("message", ""),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "request_throughput_rps": data.get("request_throughput_rps"),
        "mean_latency_s": data.get("mean_latency_s"),
        "p95_latency_s": data.get("p95_latency_s"),
        "completion_tokens_per_s": data.get("completion_tokens_per_s"),
        "completion_tokens": data.get("completion_tokens"),
        "wall_time_s": data.get("wall_time_s"),
    })

for backend, status in sorted(statuses.items()):
    if backend in seen:
        continue
    rows.append({
        "backend": backend,
        "status": status.get("status", "unknown"),
        "message": status.get("message", ""),
        "completed": 0,
        "failed": "",
        "request_throughput_rps": "",
        "mean_latency_s": "",
        "p95_latency_s": "",
        "completion_tokens_per_s": "",
        "completion_tokens": "",
        "wall_time_s": "",
    })

csv_path = out_dir / "summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "backend",
        "status",
        "message",
        "completed",
        "failed",
        "request_throughput_rps",
        "mean_latency_s",
        "p95_latency_s",
        "completion_tokens_per_s",
        "completion_tokens",
        "wall_time_s",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {csv_path}")
PY
}

task_defaults
mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  echo "If CUDA or the environment is missing, run: conda activate cu132" >&2
  exit 1
fi

MODEL_PATH="$(prepare_model_path)"

server_pid=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT

backend_index=0
for backend in ${BACKENDS}; do
  port=$((PORT_BASE + backend_index))
  backend_index=$((backend_index + 1))
  base_url="http://${HOST}:${port}"
  answers_file="${OUTPUT_DIR}/${backend}.answers.jsonl"
  metrics_file="${OUTPUT_DIR}/${backend}.metrics.json"
  server_log="${OUTPUT_DIR}/${backend}.server.log"
  client_log="${OUTPUT_DIR}/${backend}.client.log"

  mapfile -t backend_cli < <(backend_args "${backend}")

  echo "Starting task=${TASK} backend=${backend} port=${port}"
  "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${port}" \
    --enable-multimodal \
    --served-model-name default \
    --dtype "${DTYPE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    "${backend_cli[@]}" \
    ${SERVER_EXTRA_ARGS:-} \
    >"${server_log}" 2>&1 &
  server_pid=$!

  if ! wait_for_server "${base_url}" "${server_pid}"; then
    echo "Backend ${backend} did not become healthy. See ${server_log}" >&2
    IFS=$'\t' read -r failure_status failure_message < <(classify_server_failure "${server_log}")
    write_backend_status "${backend}" "${failure_status}" "${failure_message}"
    cleanup_server
    if [[ "${CONTINUE_ON_BACKEND_FAIL}" != "1" ]]; then
      exit 1
    fi
    continue
  fi

  client_args=(
    "${SCRIPT_DIR}/bench_vqa_sglang.py"
    --base-url "${base_url}/v1"
    --model default
    --model-id "${MODEL_ID}"
    --question-file "${QUESTION_FILE}"
    --image-folder "${IMAGE_FOLDER}"
    --answers-file "${answers_file}"
    --metrics-file "${metrics_file}"
    --dataset-name "${DATASET_NAME}"
    --backend-label "${backend}"
    --image-field "${IMAGE_FIELD}"
    --question-field "${QUESTION_FIELD}"
    --image-search-depth "${IMAGE_SEARCH_DEPTH}"
    --offset "${OFFSET}"
    --limit "${LIMIT}"
    --warmup "${WARMUP}"
    --concurrency "${CONCURRENCY}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --pruning-config "${PRUNING_CONFIG}"
  )
  if [[ -n "${TOP_P}" ]]; then
    client_args+=(--top-p "${TOP_P}")
  fi

  echo "Running task=${TASK} workload backend=${backend}"
  if "${PYTHON_BIN}" "${client_args[@]}" >"${client_log}" 2>&1; then
    write_backend_status "${backend}" "ok" "completed"
  else
    echo "Client failed for backend ${backend}. See ${client_log}" >&2
    write_backend_status "${backend}" "client_failed" "client command failed"
    if [[ "${CONTINUE_ON_BACKEND_FAIL}" != "1" ]]; then
      cleanup_server
      exit 1
    fi
  fi

  cleanup_server
  sleep 3
done

aggregate_summaries
