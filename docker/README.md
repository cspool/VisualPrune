# AutoResearch GPU Docker Environment

## Design

One image, three PyTorch builds. PyTorch wheels from `download.pytorch.org` bundle
their own CUDA runtime libraries, so a single CUDA 13.2 base can host all three.
Each version lives in its own conda environment.

| Conda Env | PyTorch | CUDA Toolkit (wheel) |
|-----------|---------|---------------------|
| `cu128` | 2.11.0+cu128 | 12.8 |
| `cu130` | 2.12.0+cu130 | 13.0 |
| `cu132` | 2.12.0+cu132 | 13.2 (default) |

Base image: `nvidia/cuda:13.2.1-devel-ubuntu22.04`. All three are fully supported
on RTX 4090 (Ada Lovelace, CC 8.9). Host driver ≥570 is required.

## Build

```bash
docker build --network host -f docker/Dockerfile.gpu -t auto-research-gpu:latest .
```

`.` is the build context — the set of files the Docker daemon can access
during the build. `COPY` / `ADD` can only pull from inside this path.
The current Dockerfile has no `COPY`/`ADD`, but keeping `.` as context
is harmless and avoids surprises if `COPY` is added later.

## Run

```bash
# Default: cu132 env active
bash docker/run_other_project_gpu.sh

# Start with cu128 active
CUDA_DEFAULT=cu128 bash docker/run_other_project_gpu.sh

# Start with cu130 active
CUDA_DEFAULT=cu130 bash docker/run_other_project_gpu.sh
```

### Switch CUDA inside the container

```bash
source /opt/cuda-env.sh cu128   # → torch 2.11.0+cu128
source /opt/cuda-env.sh cu130   # → torch 2.12.0+cu130
source /opt/cuda-env.sh cu132   # → torch 2.12.0+cu132 (default)
```

Or directly:

```bash
conda activate cu128
```

### Long-lived container

```bash
RECREATE=1 bash docker/run_other_project_gpu.sh
# Detach: Ctrl+P Ctrl+Q
# Re-attach:
sudo docker exec -it VisPrune_dev bash
```

## Startup Parameters Reference

`docker/run_other_project_gpu.sh` builds the `docker run` command from
environment variables. Pass overrides before the script name, for example:

```bash
CUDA_DEFAULT=cu128 RECREATE=1 bash docker/run_other_project_gpu.sh
```

Mounts and user options are fixed when the container is created. After changing
mount-related variables or the script, recreate the container:

```bash
RECREATE=1 bash docker/run_other_project_gpu.sh
```

### Runtime flags

The fixed runtime options are:

```text
docker run -itd
--name ${CONTAINER_NAME}
--gpus all
--network host
--ipc host
--ulimit memlock=-1
--ulimit stack=67108864
-w /workspace/${PROJECT_NAME}
```

`--gpus all` exposes all GPUs through the NVIDIA container runtime.
`--network host` and `--ipc host` are useful for distributed training,
Jupyter/dashboard ports, and large shared-memory workloads. The script does
not add `--privileged` by default.

### Core writable mounts

```text
-v ${PROJECT_DIR}:/workspace/${PROJECT_NAME}
-v ${CONTAINER_HOME_CACHE_DIR}:${CONTAINER_HOME}
-v ${CACHE_DIR}:${CONTAINER_CACHE_DIR}
-v ${NPM_GLOBAL_DIR}:${CONTAINER_NPM_GLOBAL_DIR}
```

These mounts are read-write by default:

| Mount | Default purpose |
|-------|-----------------|
| `${PROJECT_DIR}:/workspace/${PROJECT_NAME}` | Current project workspace. Code, outputs, project-local `venv`, `.venv`, and `node_modules` are not hidden. |
| `${CONTAINER_HOME_CACHE_DIR}:${CONTAINER_HOME}` | Container-owned HOME backing directory. This is intentionally not the host home directory. |
| `${CACHE_DIR}:${CONTAINER_CACHE_DIR}` | Shared cache root for `HF_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, and `XDG_CACHE_HOME`. |
| `${NPM_GLOBAL_DIR}:${CONTAINER_NPM_GLOBAL_DIR}` | Writable npm global prefix used for tools such as Codex CLI and Claude Code. |

`CACHE_DIR` is this launcher's host-side writable state directory, not Docker's
daemon storage for images and containers. By default it lives under
`${DOCKER_CACHE_ROOT:-/data3/docker_cache}/${PROJECT_NAME}` so cache/state for
different projects can be managed under one root. It stores the backing
directory for container `$HOME`, `xdg-cache`, npm global tools, and ML/tool
caches such as Hugging Face, Torch, and Triton.

`CONTAINER_HOME_CACHE_DIR` must stay outside `${HOST_HOME}`. This prevents the
container HOME mount from accidentally becoming the real host HOME.

### Read-only host software and identity mounts

```text
-v ${HOST_NVIDIA_DIR}:${CONTAINER_NVIDIA_DIR}:ro,rshared
-v /etc/passwd:/etc/passwd:ro
-v /etc/group:/etc/group:ro
```

`${HOST_NVIDIA_DIR:-/opt/nvidia}` is mounted read-only when present, and the
script prepends detected Nsight Compute / Nsight Systems paths under that mount
to `PATH`. `/etc/passwd` and `/etc/group` are read-only identity maps so user
and group names resolve correctly inside the container.

When `SHARE_HOST_HOME_PACKAGES=1` (default), common host-home package and tool
directories are also mounted read-only using `${BLOCKED_PATH_MODE:-ro,rshared}`
if they exist:

```text
${HOST_HOME}/.cache
${HOST_HOME}/.local
${HOST_HOME}/.conda, .mamba, .micromamba
${HOST_HOME}/.npm, .nvm
${HOST_HOME}/.cargo, .rustup
${HOST_HOME}/.venv, .virtualenvs, venv
${HOST_HOME}/anaconda3, miniconda3
node_modules, site-packages, dist-packages, __pypackages__
```

This gives containers shared read access to host packages or host-installed
tools while preventing multi-user containers from overwriting the shared host
copy.

Host-home read-only binds use a bypass mount point rather than replacing the
same path under container `$HOME`:

```text
${HOST_HOME}/.local     -> ${HOST_HOME_READONLY}/.local:${BLOCKED_PATH_MODE}
${HOST_HOME}/.conda     -> ${HOST_HOME_READONLY}/.conda:${BLOCKED_PATH_MODE}
${HOST_HOME}/miniconda3 -> ${HOST_HOME_READONLY}/miniconda3:${BLOCKED_PATH_MODE}
${HOST_HOME}/.bashrc    -> ${HOST_HOME_READONLY}/.bashrc:ro
```

`HOST_HOME_READONLY` is set inside the container to
`${CONTAINER_HOST_HOME_DIR}`. This keeps container-owned paths such as
`${CONTAINER_HOME}/.local`, `${CONTAINER_HOME}/.cache`, and
`${CONTAINER_HOME}/.conda` writable, while host copies remain available for
read-only inspection or reuse. To preserve access to host user-installed
commands without hiding the writable container `.local`, the script adds the
bypass bin directory as a PATH fallback:

```text
PATH=${CONTAINER_HOME}/.local/bin:${HOST_HOME_READONLY}/.local/bin:${CONTAINER_NPM_GLOBAL_DIR}/bin:...
```

This also prevents host `.bashrc` or host `miniconda3` conda initialization from
overriding the conda environment selected by the container image. Disable these
mounts with:

```bash
SHARE_HOST_HOME_PACKAGES=0 RECREATE=1 bash docker/run_other_project_gpu.sh
```

### Claude, Codex, and skill mounts

The helper does **not** mount the whole host home directory. `${CONTAINER_HOME}`
is backed by `${CONTAINER_HOME_CACHE_DIR}`, then selected Claude/Codex state is
mounted from `${HOST_HOME}`:

```text
${HOST_HOME}/.claude -> ${CONTAINER_HOME}/.claude:${CLAUDE_CONFIG_MODE:-rw}
${HOST_HOME}/.claude.json -> ${CONTAINER_HOME}/.claude.json:${CLAUDE_CONFIG_MODE:-rw}
${HOST_HOME}/.codex -> ${CONTAINER_HOME}/.codex:${CODEX_CONFIG_MODE:-rw}
${HOST_HOME}/.bashrc -> ${HOST_HOME_READONLY}/.bashrc:ro
```

Claude and Codex config mounts are read-write by default so the tools can update
settings, auth state, skills, and files such as `config.toml`. `.bashrc` stays
read-only and is mounted through the bypass path so it does not override the
container shell startup files.

Skill symlink targets under the mounted Claude/Codex trees are resolved
automatically and mounted read-only by default with
`${SKILL_SYMLINK_TARGET_MODE:-ro}`. For example, a link such as
`~/.claude/skills/foo -> ~/.orchestra/skills/foo` causes the helper to mount the
target `skills` root, not the whole host home. Claude/Codex plugin caches such
as `${HOST_HOME}/.claude/plugins/cache` and `${HOST_HOME}/.codex/plugins/cache`
are protected by `${BLOCKED_PATH_MODE:-ro,rshared}`.

### Root login and host-side permissions

By default, the helper logs into the container as root:

```text
CONTAINER_LOGIN_USER=root
--user 0:${HOST_GID}
```

It also adds all supplementary host groups with `--group-add`. This keeps root
convenience inside the container while preserving access to group-owned host
files.

Before startup, the script applies ACLs to writable shared directories when
`ROOT_LOGIN_ACL=1`:

```text
u:${HOST_USER}:rwX
d:u:${HOST_USER}:rwX
```

Files created by container root in `${PROJECT_DIR}`, `${CACHE_DIR}`,
`${CONTAINER_HOME_CACHE_DIR}`, `${NPM_GLOBAL_DIR}`, and other writable helper
mounts inherit permissions for the normal host user. Writable Claude/Codex
config directories, `${HOST_HOME}/.claude` and `${HOST_HOME}/.codex`, use the
same recursive ACL handling as the project directory when they are mounted
read-write. They may still show `root:${HOST_GROUP}` ownership, or a remapped
owner such as `nobody`, but the host user can read and edit them through the
ACL.

The first startup shell sets `umask 0002`. New `docker exec ... bash` shells may
report their default umask, so the inherited ACL is the main host-side access
guarantee. To use the old non-root login model:

```bash
CONTAINER_LOGIN_USER=host RECREATE=1 bash docker/run_other_project_gpu.sh
```

### Common variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CUDA_DEFAULT` | `cu132` | Active conda env on startup. Valid: `cu128`, `cu130`, `cu132`. |
| `IMAGE_NAME` | `auto-research-gpu:latest` | Docker image to run. |
| `PROJECT_DIR` | repo root containing `docker/` | Host project directory mounted read-write. |
| `PROJECT_NAME` | `basename ${PROJECT_DIR}` | Workspace directory name under `/workspace`. |
| `CONTAINER_NAME` | `${PROJECT_NAME}_dev` | Long-lived container name. |
| `DOCKER_CACHE_ROOT` | `/data3/docker_cache` | Parent directory for project-specific Docker launcher caches. |
| `CACHE_DIR` | `${DOCKER_CACHE_ROOT}/${PROJECT_NAME}` | Host cache root for this project. |
| `HOST_HOME` | `${HOME}` | Source for selected host-home config and read-only package mounts. |
| `CONTAINER_HOME` | `${HOST_HOME}` | Path used as `HOME` inside the container. Backed by `${CONTAINER_HOME_CACHE_DIR}`. |
| `CONTAINER_HOME_CACHE_DIR` | `${CACHE_DIR}/home/${HOST_USER}` | Host directory backing container HOME. Must not be inside `${HOST_HOME}`. |
| `CONTAINER_HOST_HOME_DIR` | `${CONTAINER_HOME}/.host-home` | Container path where read-only host-home bypass mounts are exposed. Exported as `HOST_HOME_READONLY`. |
| `CONTAINER_CACHE_DIR` | `${CACHE_DIR}` | Container cache path used by ML/tool caches. |
| `CONTAINER_LOGIN_USER` | `root` | `root` runs as UID 0 with host GID; `host` runs as `${HOST_UID}:${HOST_GID}`. |
| `CLAUDE_CONFIG_MODE` | `rw` | Mode for `.claude` and `.claude.json` mounts. |
| `CODEX_CONFIG_MODE` | `rw` | Mode for `.codex` mount. |
| `SKILL_SYMLINK_TARGET_MODE` | `ro` | Mode for resolved skill symlink target roots. |
| `BLOCKED_PATH_MODE` | `ro,rshared` | Mode for protected host package/cache/plugin roots. |
| `SHARE_HOST_HOME_PACKAGES` | `1` | Mount common host package/tool directories read-only when present. |
| `ROOT_LOGIN_ACL` | `1` | Apply host-user ACLs to writable mounts for root-login containers. |
| `ROOT_LOGIN_ACL_RECURSIVE` | `1` | Apply default ACLs recursively to directories managed by the helper. |
| `HOST_NVIDIA_DIR` | `/opt/nvidia` | Host NVIDIA tools directory. |
| `CONTAINER_NVIDIA_DIR` | `/opt/nvidia` | Container path for host NVIDIA tools. |
| `HOST_NVIDIA_MODE` | `ro,rshared` | Mode for the NVIDIA tools mount. |
| `RECREATE` | `0` | Set to `1` to remove and recreate an existing container. |

The script also passes these environment variables into the container:

```text
HOME, USER, LOGNAME, HF_HOME, TORCH_HOME, TRITON_CACHE_DIR,
XDG_CACHE_HOME, NPM_CONFIG_PREFIX, CODEX_HOME, HOST_HOME_READONLY,
CUDA_DEFAULT, PATH
```

## Verify GPU

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Expected (depends on active conda env):

```text
torch 2.12.0+cu132     # (or 2.11.0+cu128 / 2.12.0+cu130)
cuda_available True
device_count 2
device0 NVIDIA GeForce RTX 4090
```

Quick cross-check all three envs:

```bash
for env in cu128 cu130 cu132; do
  conda run -n $env python3 -c "import torch; print(f'{env}: {torch.__version__}  cuda={torch.cuda.is_available()}')"
done
```

## Configure Codex

Codex state is already mounted at `${CODEX_HOME:-${HOME}/.codex}`. Inside the
container:

```bash
mkdir -p "${CODEX_HOME}"
cat >> "${CODEX_HOME}/config.toml" <<'EOF'
sandbox_mode = "danger-full-access"
approval_policy = "on-request"
EOF
codex -C "$PWD"
```

## Profiling Mode

`docker/run_other_project_gpu.sh` does not add `--privileged` by default. It
does provide:

```bash
-e NVIDIA_DRIVER_CAPABILITIES=all               # Full NVIDIA driver access
-v /opt/nvidia:/opt/nvidia:ro,rshared           # Host Nsight tools
```

Avoid `--privileged` whenever possible. It grants broad host-level access and is
dangerous on shared machines. Only consider adding it for a short-lived,
isolated profiling/debugging container when narrower Docker flags cannot solve
the problem, then recreate the container.

### If ncu still shows ERR_NVGPUCTRPERM

This is a host-side driver restriction. Run on the **host machine** (once):

```bash
# On the HOST machine
sudo bash docker/fix_gpu_perf_counters.sh
sudo reboot
```

### Verify GPU performance counters

After reboot, verify inside the container:

```bash
ncu --launch-count 1 --launch-skip 0 python -c "
import torch
a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
b = torch.randn(128, 128, device='cuda', dtype=torch.float16)
torch.cuda.synchronize()
c = torch.matmul(a, b)
torch.cuda.synchronize()
"
# Expected: normal profiling output, NO "ERR_NVGPUCTRPERM"
```
