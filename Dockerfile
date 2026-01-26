# Multi-platform container for Jetson and development
# Uses Ubuntu base compatible with both ARM64 (Jetson) and AMD64 (development)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# System deps for opencv, depthai, and general build tooling
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  git \
  libglib2.0-0 \
  libgl1-mesa-glx \
  libusb-1.0-0 \
  python3-dev \
  python3-pip \
  python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Install uv (latest version)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy manifests first for layer caching
COPY codroid-ball-picker/pyproject.toml ./
COPY codroid-ball-picker/README.md ./

# Copy local dependencies that are referenced in pyproject.toml
COPY codroid-api ./codroid-api
COPY onrobot-api ./onrobot-api

# Update pyproject.toml to use correct local paths for the container
RUN sed -i 's|path = "../codroid-api"|path = "./codroid-api"|g' pyproject.toml && \
  sed -i 's|path = "../onrobot-api"|path = "./onrobot-api"|g' pyproject.toml

# Copy the minimal source code needed for package build
COPY codroid-ball-picker/app ./app
COPY codroid-ball-picker/vision ./vision

# Install Python dependencies using uv with local packages (regenerate lock)
RUN uv sync --no-dev

# For Jetson: Install CUDA-optimized PyTorch (JetPack 6.x compatible)
# This overrides the CPU-only version that might be installed by dependencies
RUN if [ "$(uname -m)" = "aarch64" ]; then \
  uv pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60; \
  fi

# Now copy the remaining application files
COPY codroid-ball-picker/run_codroid_picker.py ./
COPY codroid-ball-picker/assets ./assets

# Expose port
EXPOSE 8050

# Run the Dash UI
CMD ["uv", "run", "python", "run_codroid_picker.py", "--host", "0.0.0.0", "--port", "8050"]
