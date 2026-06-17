#!/usr/bin/env bash
# =============================================================================
# fix_gpu_perf_counters.sh
# Run ONCE on the HOST machine (NOT in Docker) to enable GPU perf counters.
#
# What this does:
#   Enables NVIDIA GPU performance counters for all users (not just root).
#   Without this, ncu/nsys fail with ERR_NVGPUCTRPERM inside containers.
#
# Usage:
#   sudo bash docker/fix_gpu_perf_counters.sh
#   sudo reboot
# =============================================================================
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "ERROR: This script must be run as root on the HOST machine."
    echo ""
    echo "  sudo bash docker/fix_gpu_perf_counters.sh"
    exit 1
fi

echo "=== NVIDIA GPU Performance Counter Unlock ==="
echo ""

# 1. Enable persistence mode (keeps GPU driver loaded even when idle)
echo "[1/3] Enabling persistence mode..."
nvidia-smi -pm 1
echo "  ✓ Persistence mode enabled"
echo ""

# 2. Allow non-admin users to access GPU performance counters
#    The actual kernel parameter name is RmProfilingAdminOnly (NOT NVreg_RestrictProfilingToAdminUsers)
#    Check with: cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
echo "[2/3] Configuring kernel module to allow non-admin profiling..."
MODPROBE_FILE="/etc/modprobe.d/nvidia-profiling.conf"
cat > "${MODPROBE_FILE}" << 'EOF'
# Allow non-admin users to access NVIDIA GPU performance counters.
# Required for ncu (Nsight Compute) and nsys (Nsight Systems) to work
# inside Docker containers and for non-root users.
#
# The actual kernel parameter controlling this is RmProfilingAdminOnly.
# NVreg_RestrictProfilingToAdminUsers is an alias that some driver versions use.
# Setting both to be safe.
options nvidia NVreg_RestrictProfilingToAdminUsers=0 RmProfilingAdminOnly=0
EOF
echo "  ✓ Written: ${MODPROBE_FILE}"
echo ""

# 3. Update initramfs to include the new module config
echo "[3/3] Updating initramfs..."
if command -v update-initramfs &>/dev/null; then
    update-initramfs -u
    echo "  ✓ initramfs updated"
elif command -v dracut &>/dev/null; then
    dracut --force
    echo "  ✓ dracut initramfs rebuilt"
else
    echo "  ⚠ Could not find update-initramfs or dracut."
    echo "    Please rebuild your initramfs manually."
fi
echo ""

echo "=== DONE ==="
echo ""
echo "ACTION REQUIRED: Reboot the host machine for changes to take effect."
echo ""
echo "  sudo reboot"
echo ""
echo "After reboot, verify inside the container:"
echo ""
echo "  ncu --launch-count 1 --launch-skip 0 python -c \""
echo "  import torch"
echo "  a = torch.randn(128, 128, device='cuda', dtype=torch.float16)"
echo "  b = torch.randn(128, 128, device='cuda', dtype=torch.float16)"
echo "  c = torch.matmul(a, b)"
echo "  torch.cuda.synchronize()"
echo "  \""
echo ""
echo "Expected: Normal profiling output. NO 'ERR_NVGPUCTRPERM'."
