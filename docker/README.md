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
source /opt/venv/cu128/bin/activate
```

### Long-lived container

```bash
RECREATE=1 bash docker/run_other_project_gpu.sh
# Detach: Ctrl+P Ctrl+Q
# Re-attach:
sudo docker exec -it VisPrune_dev bash
```

## Bind Mounts

```text
-v ${PROJECT_DIR}:/workspace/${PROJECT_NAME}
-v ${CONTAINER_HOME_CACHE_DIR}:${CONTAINER_HOME}
-v ${CACHE_DIR}:${CONTAINER_CACHE_DIR}
-v ${NPM_GLOBAL_DIR}:${CONTAINER_NPM_GLOBAL_DIR}
-v ${HOST_NVIDIA_DIR}:${CONTAINER_NVIDIA_DIR}:ro,rshared
-v /etc/passwd:/etc/passwd:ro
-v /etc/group:/etc/group:ro
```

The helper does **not** mount the host home directory. `${CONTAINER_HOME}` is an
empty container-owned home cache. The current project is mounted as-is, while
Claude/Codex state is mounted at directory granularity from `${HOST_HOME}`:

```text
${HOST_HOME}/.claude -> ${CONTAINER_HOME}/.claude
${HOST_HOME}/.claude.json -> ${CONTAINER_HOME}/.claude.json
${HOST_HOME}/.codex -> ${CONTAINER_HOME}/.codex
${HOST_HOME}/.bashrc -> ${CONTAINER_HOME}/.bashrc:ro
```

Directory-level mounts let Claude and Codex update their own settings, auth
state, skills, and files such as `config.toml` without running into single-file
bind mount replacement failures. Skill symlink targets under the mounted
Claude/Codex trees are still resolved automatically and mounted read-only by
default, so links like `~/.claude/skills/foo -> ~/.orchestra/skills/...` still
work without exposing the rest of home.

The script treats other host-home content or host package environments from
`${HOST_HOME}` as blocked paths, such as `${HOST_HOME}/.local`,
`${HOST_HOME}/.cache`, conda/mamba dirs, npm/nvm dirs, venvs, `node_modules`,
`site-packages`, and `dist-packages`. Claude/Codex plugin caches such as
`${HOST_HOME}/.claude/plugins/cache` and `${HOST_HOME}/.codex/plugins/cache`
are treated the same way. If one of these paths is needed by a configured
Claude/Codex mount or skill symlink, the helper promotes the leaf path to a
high-level protected root, deduplicates it, and bind-mounts that root as
`ro,rshared`, so the container can read shared content but cannot edit or delete
the host copy.

`${PROJECT_DIR}` is bind-mounted as-is. The helper does not hide project-local
directories such as `venv`, `.venv`, or `node_modules`.

Host NVIDIA tools under `${HOST_NVIDIA_DIR:-/opt/nvidia}` are mounted read-only
to `${CONTAINER_NVIDIA_DIR:-/opt/nvidia}` when present. The container `PATH`
is extended with detected Nsight Compute and Nsight Systems tool directories
under that mount.

Mounts are fixed when a container is created. After changing this script,
recreate the container:

```bash
RECREATE=1 bash docker/run_other_project_gpu.sh
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

Inside the container:

```bash
mkdir -p "${CODEX_HOME:-/data3/auto_research_codex_home}"
cat >> "${CODEX_HOME:-/data3/auto_research_codex_home}/config.toml" <<'EOF'
sandbox_mode = "danger-full-access"
approval_policy = "on-request"
EOF
codex -C /workspace/auto_research
```

## Profiling Mode

Container runs with `--privileged` by default — all Linux capabilities and
GPU performance counter access are granted.

`docker/run_other_project_gpu.sh` already provides:
```bash
--privileged                                    # All Linux capabilities
-e NVIDIA_DRIVER_CAPABILITIES=all               # Full NVIDIA driver access
-v /opt/nvidia:/opt/nvidia:ro,rshared           # Host Nsight tools
```

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
