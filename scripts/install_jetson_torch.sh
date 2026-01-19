#!/bin/bash
# Install CUDA-enabled PyTorch on NVIDIA Jetson (JetPack 6 / L4T R36.x)
#
# This script installs PyTorch with CUDA support from NVIDIA's wheel repository.
# Run this after `uv sync` to replace the CPU-only torch with CUDA-enabled version.
#
# Usage: ./scripts/install_jetson_torch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if we're on a Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Error: This script is for NVIDIA Jetson devices only."
    echo "       /etc/nv_tegra_release not found."
    exit 1
fi

echo "Detected Jetson device:"
head -1 /etc/nv_tegra_release

# Detect JetPack version from L4T release
L4T_VERSION=$(head -1 /etc/nv_tegra_release | grep -oP 'R\K[0-9]+')
echo "L4T major version: R${L4T_VERSION}"

# Set the appropriate NVIDIA wheel index
if [ "$L4T_VERSION" -ge 36 ]; then
    # JetPack 6.x (L4T R36.x)
    NVIDIA_INDEX="https://developer.download.nvidia.com/compute/redist/jp/v60"
    echo "Using JetPack 6 wheel index"
elif [ "$L4T_VERSION" -ge 35 ]; then
    # JetPack 5.x (L4T R35.x)
    NVIDIA_INDEX="https://developer.download.nvidia.com/compute/redist/jp/v512"
    echo "Using JetPack 5 wheel index"
else
    echo "Error: Unsupported L4T version R${L4T_VERSION}"
    echo "       This script supports JetPack 5.x (L4T R35.x) and JetPack 6.x (L4T R36.x)"
    exit 1
fi

cd "$PROJECT_DIR"

echo ""
echo "Installing CUDA-enabled PyTorch for Jetson..."
echo ""

# Get system Python version
SYSTEM_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "System Python: $SYSTEM_PY_VER"

# Check current venv Python version
if [ -d ".venv" ]; then
    VENV_PY_VER=$(.venv/bin/python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    echo "Venv Python: $VENV_PY_VER"
    
    if [ "$SYSTEM_PY_VER" != "$VENV_PY_VER" ]; then
        echo ""
        echo "WARNING: Virtual environment Python version ($VENV_PY_VER) doesn't match system Python ($SYSTEM_PY_VER)"
        echo "System PyTorch packages will not work with a different Python version."
        echo ""
        echo "Options:"
        echo "  1. Recreate venv with system Python (recommended)"
        echo "  2. Exit and manually create venv with: uv venv --python python3"
        echo ""
        read -p "Recreate venv with system Python? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing venv..."
            rm -rf .venv
            echo "Creating new venv with system Python..."
            uv venv --python python3
            echo "Syncing dependencies..."
            uv sync
        else
            echo "Exiting. Please recreate venv manually and run this script again."
            exit 0
        fi
    fi
fi

# For Jetson, the most reliable method is using system packages
echo ""
echo "Installing PyTorch and torchvision from system packages..."
echo "This requires sudo access."
echo ""

sudo apt-get update
sudo apt-get install -y python3-pip python3-torch python3-torchvision

echo ""
echo "Configuring venv to use system site-packages..."

# Enable system site-packages in the venv
VENV_PYVENV_CFG=".venv/pyvenv.cfg"
if [ -f "$VENV_PYVENV_CFG" ]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$VENV_PYVENV_CFG"
    echo "Enabled system site-packages in venv configuration"
else
    echo "Warning: pyvenv.cfg not found"
fi

echo ""
echo "Verifying installation..."
uv run python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Warning: CUDA not available after installation')
"

echo ""
echo "Done! CUDA-enabled PyTorch is now installed."
