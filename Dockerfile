# CPU/Raspberry Pi image for the FastAPI rewrite.

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  git \
  libglib2.0-0 \
  libusb-1.0-0 \
  libgl1 \
  libjpeg62-turbo \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY codroid-ball-picker/pyproject.toml ./
COPY codroid-ball-picker/uv.lock ./
COPY codroid-ball-picker/README.md ./
COPY codroid-api ./codroid-api
COPY onrobot-api ./onrobot-api

RUN sed -i 's|path = "../codroid-api"|path = "./codroid-api"|g' pyproject.toml && \
  sed -i 's|path = "../onrobot-api"|path = "./onrobot-api"|g' pyproject.toml

COPY codroid-ball-picker/codroid_picker ./codroid_picker
COPY codroid-ball-picker/vision ./vision
COPY codroid-ball-picker/assets ./assets
COPY codroid-ball-picker/data/markers ./data/markers
COPY codroid-ball-picker/run_codroid_picker.py ./

RUN uv sync --no-dev --frozen

EXPOSE 8050

CMD ["uv", "run", "uvicorn", "codroid_picker.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8050", "--workers", "1"]

