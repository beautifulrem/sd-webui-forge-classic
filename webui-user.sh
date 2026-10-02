#!/usr/bin/env bash

# export PYTHON=
# export GIT=
# export VENV_DIR=

# export TORCH_COMMAND="pip install torch torchvision"

export COMMANDLINE_ARGS="${COMMANDLINE_ARGS:---uv}"

# --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

exec "$(dirname "$0")/webui.sh" "$@"
