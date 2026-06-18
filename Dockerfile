# syntax=docker/dockerfile:1.7

# Jetson Orin Nano / JetPack 6.x runtime image.
# r36.4.0 is the current public L4T JetPack base for JetPack 6.2.x systems.
ARG JETSON_BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0

FROM ${JETSON_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Runtime libraries cover OpenCV, DepthAI USB access, PyTorch/YOLO inference,
# and small diagnostics that are useful when bringing up hardware remotely.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libopenblas0 \
      libsm6 \
      libusb-1.0-0 \
      libxext6 \
      libxrender1 \
      pkg-config \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      udev \
      usbutils && \
    rm -rf /var/lib/apt/lists/*

RUN python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "Jetson image must use Python 3.10 because the configured "
        "Jetson GPU PyTorch wheels are cp310 builds."
    )
PY

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy manifests first for dependency-layer caching. The GitHub workflow builds
# from a parent context that contains this repo plus the private local API repos.
COPY codroid-ball-picker/pyproject.toml codroid-ball-picker/uv.lock codroid-ball-picker/README.md ./
COPY codroid-api ./codroid-api
COPY onrobot-api ./onrobot-api

# The source checkout uses sibling editable dependencies. Inside the image those
# repos live under /app, so rewrite both the manifest and lockfile paths before
# syncing.
RUN sed -i \
      -e 's|path = "../codroid-api"|path = "./codroid-api"|g' \
      -e 's|path = "../onrobot-api"|path = "./onrobot-api"|g' \
      pyproject.toml && \
    sed -i \
      -e 's|editable = "../codroid-api"|editable = "./codroid-api"|g' \
      -e 's|editable = "../onrobot-api"|editable = "./onrobot-api"|g' \
      uv.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY codroid-ball-picker/app ./app
COPY codroid-ball-picker/vision ./vision
COPY codroid-ball-picker/assets ./assets
COPY codroid-ball-picker/data/markers ./data/markers
COPY codroid-ball-picker/scripts/capture_depth_snapshot_once.py ./scripts/capture_depth_snapshot_once.py
COPY codroid-ball-picker/run_codroid_picker.py ./
COPY codroid-ball-picker/run_codroid_picker_supervised.py ./

RUN python -m compileall \
      app \
      vision \
      scripts/capture_depth_snapshot_once.py \
      run_codroid_picker.py \
      run_codroid_picker_supervised.py && \
    python - <<'PY'
import cv2
import depthai
import torch
import torchvision

print("opencv", cv2.__version__)
print("depthai", depthai.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
PY

EXPOSE 8050

CMD [ \
      "python", \
      "run_codroid_picker_supervised.py", \
      "--host", \
      "0.0.0.0", \
      "--port", \
      "8050" \
    ]
