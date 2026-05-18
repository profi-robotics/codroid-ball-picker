#!/usr/bin/env bash
# Install CUDA-enabled PyTorch on NVIDIA Jetson.
#
# This script keeps uv from installing PyPI CPU/incompatible CUDA torch wheels
# and installs the project torch extra from the Jetson AI Lab JP6/CUDA 12.6
# index configured in pyproject.toml.
#
# Usage:
#   ./scripts/install_jetson_torch.sh

set -euo pipefail

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

cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required."
    exit 1
fi

SYSTEM_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
VENV_PY_VER=""
if [ -x ".venv/bin/python3" ]; then
    VENV_PY_VER=$(.venv/bin/python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
fi

if [ ! -d ".venv" ] || [ "${VENV_PY_VER}" != "${SYSTEM_PY_VER}" ]; then
    if [ -d ".venv" ] && [ "${JETSON_RECREATE_VENV:-1}" != "1" ]; then
        echo "Error: .venv uses Python ${VENV_PY_VER}, but Jetson torch wheels target system Python ${SYSTEM_PY_VER}."
        echo "Set JETSON_RECREATE_VENV=1 or remove .venv before running this script."
        exit 1
    fi
    echo "Creating project environment with system Python ${SYSTEM_PY_VER}."
    rm -rf .venv
    uv venv --python python3
fi

echo "Syncing project with Jetson CUDA torch packages."
uv sync --no-dev --frozen --inexact --extra torch

echo ""
echo "Verifying CUDA availability..."
uv run python scripts/check_ml_gpu.py --require-jetson --require-cuda

echo ""
echo "Done. CUDA-enabled PyTorch is installed and visible to the project."
