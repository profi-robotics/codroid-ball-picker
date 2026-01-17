# Jetson Orin Nano deployment container
# Uses NVIDIA L4T PyTorch base (JetPack 5.x / L4T R35.x)

FROM nvcr.io/nvidia/l4t-pytorch:r35.4.1-py3

ARG UV_VERSION=0.4.0
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

# Install uv (pinned)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --version ${UV_VERSION}

# Copy manifests first for layer caching
COPY pyproject.toml uv.lock ./ 

# Install Python dependencies into the project venv
RUN uv sync --frozen --no-dev

# Now copy the application code
COPY . .

EXPOSE 8050

# Run the Dash UI
CMD ["uv", "run", "python", "run_codroid_picker.py", "--host", "0.0.0.0", "--port", "8050"]
