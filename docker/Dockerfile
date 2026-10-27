# syntax=docker/dockerfile:1
# Remi WebUI | NVIDIA GPU
#
# Built from this checkout: the build context is the repository root.
#
#   docker build -f docker/Dockerfile -t remi-webui .
#
# Two stages: the web interface (webui/dist) is built with Node.js and the
# lockfile, then the server image gets this checkout, Python 3.13, PyTorch,
# requirements.txt and the built SPA (no Node.js at runtime). Nothing is cloned:
# the image runs exactly the code of the checkout it was built from.
# tests/test_remi_docker.py keeps this file in step with the launcher.

ARG CUDA_VERSION=13.0.0
ARG NODE_VERSION=24

# region The web interface

FROM node:${NODE_VERSION}-slim AS webui

# The pnpm major the launcher falls back to (modules/launch_utils.PNPM_FALLBACK)
ARG PNPM_VERSION=12
RUN npm install --global "pnpm@${PNPM_VERSION}"

WORKDIR /src/webui

# Dependencies first (cached while only the sources change)
COPY webui/package.json webui/pnpm-lock.yaml webui/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY webui/ ./
RUN pnpm run build

# endregion

# region The server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# The PyTorch build the launcher installs on Linux (modules/launch_utils.default_torch_command)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PACKAGES="torch==2.14.0+cu130 torchvision==0.29.0+cu130"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_NO_CACHE=1 \
    TORCH_INDEX_URL=${TORCH_INDEX_URL} \
    REMI_SKIP_WEBUI_BUILD=1 \
    REMI_DATA_DIR=/home/remi/data

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies (git: extension installs and updates from the Extensions page)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Non-Root User (UID 99 / GID 100 = nobody:users)
RUN groupadd -g 100 users 2>/dev/null || true \
    && useradd -u 99 -g 100 -d /home/remi -m -s /bin/bash remi \
    && mkdir -p /home/remi/app /home/remi/data \
    && chown -R 99:100 /home/remi

WORKDIR /home/remi/app
USER remi

# Python 3.13 Virtual ENVironment
RUN uv venv venv --python 3.13 --seed

ENV VIRTUAL_ENV="/home/remi/app/venv" \
    PATH="/home/remi/app/venv/bin:$PATH" \
    PYTHON="/home/remi/app/venv/bin/python3.13"

# PyTorch, then the WebUI's requirements (Gradio-free)
RUN uv pip install ${TORCH_PACKAGES} --extra-index-url "${TORCH_INDEX_URL}"
COPY --chown=99:100 requirements.txt ./
RUN uv pip install -r requirements.txt

# This checkout (.dockerignore leaves out models, outputs, venvs and webui/node_modules)
# and the web interface built above
COPY --chown=99:100 . ./
COPY --from=webui --chown=99:100 /src/webui/dist ./webui/dist

# The built-in extensions' installers, once at build time (the GPU test needs a GPU, which
# a build does not have); the same launcher runs them again on start, as a quick check.
RUN python launch.py --exit --skip-torch-cuda-test --data-dir /tmp/remi-build \
    && rm -rf /tmp/remi-build

# Runtime
RUN chmod +x docker/entrypoint.sh

EXPOSE 7860

# config.json, styles.csv, user.css, remi-data/ (queue, history, prompt tools),
# models/, output/ and extensions/ all live in the data folder
VOLUME ["/home/remi/data"]

# /internal/ping answers without a login, also with --auth
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=15 \
    CMD curl -fsS http://localhost:7860/internal/ping || exit 1

ENTRYPOINT ["/home/remi/app/docker/entrypoint.sh"]

# endregion
