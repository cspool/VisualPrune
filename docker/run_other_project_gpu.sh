#!/usr/bin/env bash
set -euo pipefail

# ── CUDA env selection ──────────────────────────────────────────────────────
# The all-in-one image contains three conda environments:
#   cu128 (torch 2.11.0+cu128), cu130 (torch 2.12.0+cu130), cu132 (torch 2.12.0+cu132)
# Set CUDA_DEFAULT to choose which env is active on container start.
# Inside the container, switch anytime with: source /opt/cuda-env.sh <name>
CUDA_DEFAULT="${CUDA_DEFAULT:-cu132}"

case "${CUDA_DEFAULT}" in
  cu128|cu130|cu132) ;;
  *)
    echo "Unknown CUDA_DEFAULT: ${CUDA_DEFAULT}" >&2
    echo "Valid: cu128 cu130 cu132" >&2
    exit 1
    ;;
esac

IMAGE_NAME="${IMAGE_NAME:-auto-research-gpu:latest}"
CUDA_SHORT="13.2"  # base image CUDA (toolkit version, NOT PyTorch CUDA)

# ── Paths and names ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DEFAULT_PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
PROJECT_NAME="${PROJECT_NAME:-$(basename "${PROJECT_DIR}")}"
CONTAINER_NAME="${CONTAINER_NAME:-${PROJECT_NAME}_dev}"
CACHE_DIR="${CACHE_DIR:-/data3/${PROJECT_NAME}_docker_cache}"
HOST_HOME="${HOST_HOME:-${HOME}}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
HOST_GROUP="${HOST_GROUP:-$(id -gn)}"
CONTAINER_HOME="${CONTAINER_HOME:-${HOST_HOME}}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/data3/${PROJECT_NAME}_docker_cache}"
CONTAINER_HOME_CACHE_DIR="${CONTAINER_HOME_CACHE_DIR:-${CACHE_DIR}/home/${HOST_USER}}"
NPM_GLOBAL_DIR="${NPM_GLOBAL_DIR:-${CACHE_DIR}/npm-global}"
CONTAINER_NPM_GLOBAL_DIR="${CONTAINER_NPM_GLOBAL_DIR:-${CONTAINER_CACHE_DIR}/npm-global}"
HOST_NVIDIA_DIR="${HOST_NVIDIA_DIR:-/opt/nvidia}"
CONTAINER_NVIDIA_DIR="${CONTAINER_NVIDIA_DIR:-/opt/nvidia}"
HOST_NVIDIA_MODE="${HOST_NVIDIA_MODE:-ro,rshared}"
CLAUDE_CONFIG_MODE="${CLAUDE_CONFIG_MODE:-rw}"
CODEX_CONFIG_MODE="${CODEX_CONFIG_MODE:-rw}"
SKILL_SYMLINK_TARGET_MODE="${SKILL_SYMLINK_TARGET_MODE:-${CLAUDE_SYMLINK_TARGET_MODE:-ro}}"
BLOCKED_PATH_MODE="${BLOCKED_PATH_MODE:-ro,rshared}"
SHARE_HOST_HOME_PACKAGES="${SHARE_HOST_HOME_PACKAGES:-1}"
RECREATE="${RECREATE:-0}"

mkdir -p "${CACHE_DIR}"

if [ ! -w "${CACHE_DIR}" ]; then
  echo "Cache dir is not writable by ${HOST_USER}:${HOST_GROUP} (${HOST_UID}:${HOST_GID}): ${CACHE_DIR}" >&2
  echo "Please chown it or set CACHE_DIR to a writable path." >&2
  exit 1
fi

HOST_HOME_REAL="$(readlink -f -- "${HOST_HOME}")"
CONTAINER_HOME_CACHE_REAL="$(realpath -m -- "${CONTAINER_HOME_CACHE_DIR}")"

path_is_under() {
  local path="$1"
  local prefix="$2"

  [ "${path}" = "${prefix}" ] || [[ "${path}" == "${prefix}"/* ]]
}

reject_forbidden_home_cache() {
  if path_is_under "${CONTAINER_HOME_CACHE_REAL}" "${HOST_HOME_REAL}"; then
    echo "Refusing to use a host-home path as container HOME cache: ${CONTAINER_HOME_CACHE_DIR}" >&2
    echo "Set CONTAINER_HOME_CACHE_DIR outside ${HOST_HOME}." >&2
    exit 1
  fi
}

blocked_mount_root_for_path() {
  local source_real="$1"
  local blocked_path
  local source_parent

  if [ "${source_real}" = "${HOST_HOME_REAL}" ]; then
    printf '%s\n' "${HOST_HOME_REAL}"
    return 0
  fi

  for blocked_path in \
    "${HOST_HOME_REAL}/.claude/plugins/cache" \
    "${HOST_HOME_REAL}/.codex/plugins/cache" \
    "${HOST_HOME_REAL}/.cache" \
    "${HOST_HOME_REAL}/.conda" \
    "${HOST_HOME_REAL}/.cargo" \
    "${HOST_HOME_REAL}/.deno" \
    "${HOST_HOME_REAL}/.gem" \
    "${HOST_HOME_REAL}/.gradle" \
    "${HOST_HOME_REAL}/.ivy2" \
    "${HOST_HOME_REAL}/.julia" \
    "${HOST_HOME_REAL}/.local" \
    "${HOST_HOME_REAL}/.m2" \
    "${HOST_HOME_REAL}/.mamba" \
    "${HOST_HOME_REAL}/.micromamba" \
    "${HOST_HOME_REAL}/.npm" \
    "${HOST_HOME_REAL}/.nvm" \
    "${HOST_HOME_REAL}/.pyenv" \
    "${HOST_HOME_REAL}/.rbenv" \
    "${HOST_HOME_REAL}/.rustup" \
    "${HOST_HOME_REAL}/.venv" \
    "${HOST_HOME_REAL}/.virtualenvs" \
    "${HOST_HOME_REAL}/anaconda3" \
    "${HOST_HOME_REAL}/miniconda3" \
    "${HOST_HOME_REAL}/venv"
  do
    if path_is_under "${source_real}" "${blocked_path}"; then
      printf '%s\n' "${blocked_path}"
      return 0
    fi
  done

  case "/${source_real#/}/" in
    */node_modules/*)
      printf '%s\n' "${source_real%%/node_modules/*}/node_modules"
      return 0
      ;;
    */site-packages/*)
      printf '%s\n' "${source_real%%/site-packages/*}/site-packages"
      return 0
      ;;
    */dist-packages/*)
      printf '%s\n' "${source_real%%/dist-packages/*}/dist-packages"
      return 0
      ;;
    */__pypackages__/*)
      printf '%s\n' "${source_real%%/__pypackages__/*}/__pypackages__"
      return 0
      ;;
    */.conda/*)
      printf '%s\n' "${source_real%%/.conda/*}/.conda"
      return 0
      ;;
    */.nox/*)
      printf '%s\n' "${source_real%%/.nox/*}/.nox"
      return 0
      ;;
    */.tox/*)
      printf '%s\n' "${source_real%%/.tox/*}/.tox"
      return 0
      ;;
    */.venv/*)
      printf '%s\n' "${source_real%%/.venv/*}/.venv"
      return 0
      ;;
    */env/*)
      printf '%s\n' "${source_real%%/env/*}/env"
      return 0
      ;;
    */venv/*)
      printf '%s\n' "${source_real%%/venv/*}/venv"
      return 0
      ;;
    */venv_*/*)
      source_parent="${source_real}"
      while [ "${source_parent}" != "/" ]; do
        case "$(basename -- "${source_parent}")" in
          venv_*)
            printf '%s\n' "${source_parent}"
            return 0
            ;;
        esac
        source_parent="$(dirname -- "${source_parent}")"
      done
      return 1
      ;;
  esac

  return 1
}

mount_target_for_root() {
  local source_real="$1"
  local target_path="$2"
  local source_root="$3"
  local relative_path
  local target_prefix_len

  relative_path="${source_real#"${source_root}"}"
  if [ -z "${relative_path}" ]; then
    printf '%s\n' "${target_path}"
    return
  fi

  target_prefix_len=$((${#target_path} - ${#relative_path}))
  if [ "${target_prefix_len}" -ge 0 ] && [ "${target_path:${target_prefix_len}}" = "${relative_path}" ]; then
    printf '%s\n' "${target_path:0:${target_prefix_len}}"
    return
  fi

  container_path_for_host_path "${source_root}"
}

reject_forbidden_home_cache

mkdir -p "${CONTAINER_HOME_CACHE_DIR}" "${NPM_GLOBAL_DIR}" "${CACHE_DIR}/xdg-cache"

build_nvidia_path_prefix() {
  local host_dir
  local rel_path
  local container_dir
  local path_prefix=""

  if [ ! -d "${HOST_NVIDIA_DIR}" ]; then
    return
  fi

  for host_dir in \
    "${HOST_NVIDIA_DIR}/bin" \
    "${HOST_NVIDIA_DIR}"/nsight-compute/* \
    "${HOST_NVIDIA_DIR}"/nsight-systems/*/bin
  do
    [ -d "${host_dir}" ] || continue
    rel_path="${host_dir#"${HOST_NVIDIA_DIR}"}"
    container_dir="${CONTAINER_NVIDIA_DIR}${rel_path}"
    path_prefix="${container_dir}${path_prefix:+:${path_prefix}}"
  done

  printf '%s\n' "${path_prefix}"
}

NVIDIA_PATH_PREFIX="$(build_nvidia_path_prefix)"

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

if container_exists; then
  if [ "${RECREATE}" = "1" ]; then
    docker rm -f "${CONTAINER_NAME}"
  else
    if container_running; then
      echo "Container ${CONTAINER_NAME} is already running."
      echo "Attach with: docker exec -it ${CONTAINER_NAME} /bin/bash"
    else
      echo "Container ${CONTAINER_NAME} already exists but is stopped."
      echo "Start it with: docker start -ai ${CONTAINER_NAME}"
    fi
    echo "To recreate it with the mounts in this script, run:"
    echo "  RECREATE=1 $0"
    exit 1
  fi
fi

docker_args=(
  run
  -itd
  --name "${CONTAINER_NAME}"
  --user "${HOST_UID}:${HOST_GID}"
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e USER="${HOST_USER}"
  -e LOGNAME="${HOST_USER}"
  -e HOME="${CONTAINER_HOME}"
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e HF_HOME="${CONTAINER_CACHE_DIR}/huggingface"
  -e TORCH_HOME="${CONTAINER_CACHE_DIR}/torch"
  -e TRITON_CACHE_DIR="${CONTAINER_CACHE_DIR}/triton"
  -e XDG_CACHE_HOME="${CONTAINER_CACHE_DIR}/xdg-cache"
  -e NPM_CONFIG_PREFIX="${CONTAINER_NPM_GLOBAL_DIR}"
  -e CODEX_HOME="${CONTAINER_HOME}/.codex"
  -e PATH="${NVIDIA_PATH_PREFIX:+${NVIDIA_PATH_PREFIX}:}${CONTAINER_HOME}/.local/bin:${CONTAINER_NPM_GLOBAL_DIR}/bin:/opt/conda/bin:/opt/conda/condabin:/usr/local/cuda-${CUDA_SHORT}/bin:/usr/local/bin:/usr/bin:/bin"
  -e CUDA_DEFAULT="${CUDA_DEFAULT}"
  -v "${PROJECT_DIR}:/workspace/${PROJECT_NAME}"
  -v "${CONTAINER_HOME_CACHE_DIR}:${CONTAINER_HOME}"
  -v "${CACHE_DIR}:${CONTAINER_CACHE_DIR}"
  -v "${NPM_GLOBAL_DIR}:${CONTAINER_NPM_GLOBAL_DIR}"
  -v /etc/passwd:/etc/passwd:ro
  -v /etc/group:/etc/group:ro
  -w "/workspace/${PROJECT_NAME}"
)

for group_id in $(id -G); do
  if [ "${group_id}" != "${HOST_GID}" ]; then
    docker_args+=(--group-add "${group_id}")
  fi
done

declare -A ADDED_MOUNTS=()
declare -a SYMLINK_SCAN_ROOTS=()

prepare_container_home_target() {
  local target_path="$1"
  local source_real="$2"
  local relative_path
  local cache_target

  if [[ "${target_path}" != "${CONTAINER_HOME}/"* ]]; then
    return
  fi

  relative_path="${target_path#"${CONTAINER_HOME}/"}"
  cache_target="${CONTAINER_HOME_CACHE_DIR}/${relative_path}"

  if [ -d "${source_real}" ]; then
    mkdir -p "${cache_target}"
  else
    mkdir -p "$(dirname -- "${cache_target}")"
    touch "${cache_target}"
  fi
}

add_mount_if_exists() {
  local source_path="$1"
  local target_path="$2"
  local mode="${3:-}"
  local scan_symlinks="${4:-0}"
  local source_real
  local blocked_mount_root
  local blocked_mount="0"
  local mount_key
  local mount_spec

  if [ ! -e "${source_path}" ]; then
    echo "Skip missing mount source: ${source_path}"
    return
  fi

  source_real="$(readlink -f -- "${source_path}")"
  if blocked_mount_root="$(blocked_mount_root_for_path "${source_real}")"; then
    target_path="$(mount_target_for_root "${source_real}" "${target_path}" "${blocked_mount_root}")"
    source_real="${blocked_mount_root}"
    mode="${BLOCKED_PATH_MODE}"
    blocked_mount="1"
  fi

  mount_key="${source_real}=>${target_path}"
  if [ -n "${ADDED_MOUNTS[${mount_key}]:-}" ]; then
    return
  fi
  ADDED_MOUNTS["${mount_key}"]=1

  if [ "${blocked_mount}" = "1" ]; then
    echo "Mounting blocked host package/home path read-only shared: ${source_real} -> ${target_path}" >&2
  fi

  prepare_container_home_target "${target_path}" "${source_real}"

  mount_spec="${source_real}:${target_path}"
  if [ -n "${mode}" ]; then
    mount_spec="${mount_spec}:${mode}"
  fi
  docker_args+=(-v "${mount_spec}")

  if [ "${scan_symlinks}" = "1" ] && [ -d "${source_real}" ]; then
    SYMLINK_SCAN_ROOTS+=("${source_real}")
  fi
}

add_host_home_package_mounts() {
  local package_dir

  if [ "${SHARE_HOST_HOME_PACKAGES}" != "1" ]; then
    return
  fi

  for package_dir in \
    ".cache" \
    ".cargo" \
    ".conda" \
    ".deno" \
    ".gem" \
    ".gradle" \
    ".ivy2" \
    ".julia" \
    ".local" \
    ".m2" \
    ".mamba" \
    ".micromamba" \
    ".npm" \
    ".nvm" \
    ".pyenv" \
    ".rbenv" \
    ".rustup" \
    ".venv" \
    ".virtualenvs" \
    "anaconda3" \
    "miniconda3" \
    "venv"
  do
    if [ -e "${HOST_HOME}/${package_dir}" ]; then
      add_mount_if_exists \
        "${HOST_HOME}/${package_dir}" \
        "${CONTAINER_HOME}/${package_dir}" \
        "${BLOCKED_PATH_MODE}"
    fi
  done
}

add_nvidia_tool_mounts() {
  add_mount_if_exists \
    "${HOST_NVIDIA_DIR}" \
    "${CONTAINER_NVIDIA_DIR}" \
    "${HOST_NVIDIA_MODE}"
}

container_path_for_host_path() {
  local host_path="$1"
  local relative_path

  if path_is_under "${host_path}" "${HOST_HOME_REAL}"; then
    relative_path="${host_path#"${HOST_HOME_REAL}"}"
    printf '%s%s\n' "${CONTAINER_HOME}" "${relative_path}"
  else
    printf '%s\n' "${host_path}"
  fi
}

skill_mount_root_for_path() {
  local path="$1"

  case "${path}" in
    */skills/*)
      printf '%s\n' "${path%%/skills/*}/skills"
      ;;
    */skills)
      printf '%s\n' "${path}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

add_symlink_target_mount() {
  local link_path="$1"
  local raw_target
  local target_real
  local link_container_path
  local target_container_path
  local source_root
  local target_root

  raw_target="$(readlink -- "${link_path}")"
  target_real="$(readlink -f -- "${link_path}" 2>/dev/null || true)"
  if [ -z "${target_real}" ] || [ ! -e "${target_real}" ]; then
    echo "Skip broken Claude skill symlink: ${link_path} -> ${raw_target}" >&2
    return
  fi

  link_container_path="$(container_path_for_host_path "${link_path}")"
  if [[ "${raw_target}" = /* ]]; then
    target_container_path="${raw_target}"
  else
    target_container_path="$(realpath -m -- "$(dirname -- "${link_container_path}")/${raw_target}")"
  fi

  source_root="$(skill_mount_root_for_path "${target_real}")"
  target_root="$(skill_mount_root_for_path "${target_container_path}")"
  add_mount_if_exists "${source_root}" "${target_root}" "${SKILL_SYMLINK_TARGET_MODE}" "1"
}

scan_registered_symlink_roots() {
  local scan_index=0
  local root
  local root_real
  local link_path
  declare -A SCANNED_ROOTS=()

  while [ "${scan_index}" -lt "${#SYMLINK_SCAN_ROOTS[@]}" ]; do
    root="${SYMLINK_SCAN_ROOTS[${scan_index}]}"
    scan_index=$((scan_index + 1))
    root_real="$(readlink -f -- "${root}" 2>/dev/null || true)"

    if [ -z "${root_real}" ] || [ -n "${SCANNED_ROOTS[${root_real}]:-}" ]; then
      continue
    fi
    SCANNED_ROOTS["${root_real}"]=1

    while IFS= read -r -d '' link_path; do
      add_symlink_target_mount "${link_path}"
    done < <(find "${root_real}" -type l -print0 2>/dev/null)
  done
}

add_claude_code_mounts() {
  add_mount_if_exists \
    "${HOST_HOME}/.claude" \
    "${CONTAINER_HOME}/.claude" \
    "${CLAUDE_CONFIG_MODE}" \
    "1"

  add_mount_if_exists \
    "${HOST_HOME}/.claude.json" \
    "${CONTAINER_HOME}/.claude.json" \
    "${CLAUDE_CONFIG_MODE}"

  add_mount_if_exists \
    "${HOST_HOME}/.bashrc" \
    "${CONTAINER_HOME}/.bashrc" \
    "ro"
}

add_codex_mounts() {
  add_mount_if_exists \
    "${HOST_HOME}/.codex" \
    "${CONTAINER_HOME}/.codex" \
    "${CODEX_CONFIG_MODE}" \
    "1"
}

add_nvidia_tool_mounts
add_claude_code_mounts
add_codex_mounts
add_host_home_package_mounts
scan_registered_symlink_roots

docker "${docker_args[@]}" \
  "${IMAGE_NAME}" \
  bash -lc '
    source /opt/conda/etc/profile.d/conda.sh
    source /opt/cuda-env.sh "${CUDA_DEFAULT}"
    mkdir -p "${XDG_CACHE_HOME}" "${NPM_CONFIG_PREFIX}"
    npm install -g @openai/codex@latest || command -v codex >/dev/null 2>&1
    npm install -g @anthropic-ai/claude-code@latest || command -v claude >/dev/null 2>&1
    exec /bin/bash
  '
