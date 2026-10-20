#!/usr/bin/env bash
# Build the interpreter of tools/record_legacy_ui.py (Gradio 4.40 and the requirements of the
# last Gradio-era revision, tools/legacy-ui-requirements.txt), then run the recorder's
# --self-test: it records a known script in that environment and compares the record with
# the script's committed fixtures.
#
#   tools/legacy_ui_env.sh                 # into ../remi-gradio-env (next to the checkout)
#   tools/legacy_ui_env.sh /path/to/env    # anywhere outside the checkout and outside /tmp
#
# Needs the network once (PyPI). Uses uv when it is installed, else python3.13 -m venv + pip
# ($LEGACY_PYTHON picks the base interpreter: 3.10 to 3.13). The WebUI's own environment is
# never touched: it must stay Gradio-free (tools/check.sh "env:gradio-free"). Afterwards
# `REMI_LEGACY_PYTHON=<env>/bin/python tools/check.sh --only legacy-ui` repeats the self-test.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT/../remi-gradio-env}"
REQUIREMENTS="$ROOT/tools/legacy-ui-requirements.txt"
BASE_PYTHON="${LEGACY_PYTHON:-python3.13}"

if command -v uv >/dev/null 2>&1; then
  [ -x "$ENV_DIR/bin/python" ] || uv venv --python "$BASE_PYTHON" "$ENV_DIR"
  uv pip install --python "$ENV_DIR/bin/python" -r "$REQUIREMENTS"
else
  [ -x "$ENV_DIR/bin/python" ] || "$BASE_PYTHON" -m venv "$ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
fi

cd "$ROOT"
"$ENV_DIR/bin/python" tools/record_legacy_ui.py --self-test
echo "Legacy recorder ready: $ENV_DIR/bin/python tools/record_legacy_ui.py --help"
