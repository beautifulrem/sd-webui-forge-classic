#!/usr/bin/env bash
# Build the interpreter of tools/record_legacy_ui.py (the Gradio of the last Gradio-era revision
# and that revision's requirements), then run the recorder's --self-test: it records a known
# script in that environment and compares the record with the script's committed fixtures.
#
#   tools/legacy_ui_env.sh                 # into ../remi-gradio-env (next to the checkout)
#   tools/legacy_ui_env.sh /path/to/env    # anywhere outside the checkout and outside /tmp
#
# No file of the checkout lists those requirements: `record_legacy_ui.py --requirements` derives
# them from the git history (LEGACY_REV's requirements.txt, the Gradio pins of the last launcher
# that installed Gradio, Pillow 10) into <env>/legacy-ui-requirements.txt, outside the checkout.
# Needs the full history and the network once (PyPI). Uses uv when it is installed, else
# python3.13 -m venv + pip ($LEGACY_PYTHON picks the base interpreter: 3.10 to 3.13). The
# WebUI's own environment is never touched: it must stay Gradio-free (tools/check.sh
# "env:gradio-free"). Afterwards `REMI_LEGACY_PYTHON=<env>/bin/python tools/check.sh --only
# legacy-ui` repeats the self-test.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT/../remi-gradio-env}"
BASE_PYTHON="${LEGACY_PYTHON:-python3.13}"

case "$(cd "$(dirname "$ENV_DIR")" 2>/dev/null && pwd)/" in
  "$ROOT"/*) echo "$ENV_DIR is inside the checkout: pick a folder outside it" >&2; exit 2 ;;
esac

if command -v uv >/dev/null 2>&1; then
  [ -x "$ENV_DIR/bin/python" ] || uv venv --python "$BASE_PYTHON" "$ENV_DIR"
else
  [ -x "$ENV_DIR/bin/python" ] || "$BASE_PYTHON" -m venv "$ENV_DIR"
fi

# --requirements needs only the standard library and git: the new interpreter can run it.
REQUIREMENTS="$ENV_DIR/legacy-ui-requirements.txt"
(cd "$ROOT" && "$ENV_DIR/bin/python" tools/record_legacy_ui.py --requirements) >"$REQUIREMENTS"

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$ENV_DIR/bin/python" -r "$REQUIREMENTS"
else
  "$ENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
fi

cd "$ROOT"
"$ENV_DIR/bin/python" tools/record_legacy_ui.py --self-test
echo "Legacy recorder ready: $ENV_DIR/bin/python tools/record_legacy_ui.py --help"
