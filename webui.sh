#!/usr/bin/env bash

set -e

WEBUI_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$WEBUI_DIR"

# Load optional settings
if [ -f "webui.settings.sh" ]; then
    source webui.settings.sh
fi

# Defaults
VENV_DIR="${VENV_DIR:-$WEBUI_DIR/venv}"
SD_WEBUI_RESTART="tmp/restart"
ERROR_REPORTING="FALSE"

mkdir -p tmp

# Keep existing environments and explicit interpreter overrides. On a fresh
# install, prefer 3.13 instead of macOS's system python3 (often 3.9).
if [ -z "${PYTHON:-}" ]; then
    if [ "$VENV_DIR" != "-" ] && [ "${SKIP_VENV:-}" != "1" ] && [ -x "$VENV_DIR/bin/python" ]; then
        PYTHON="$VENV_DIR/bin/python"
    elif command -v python3.13 >/dev/null 2>&1; then
        PYTHON=$(command -v python3.13)
    elif command -v uv >/dev/null 2>&1; then
        PYTHON=$(uv python find --offline 3.13 2>tmp/stderr.txt) || PYTHON=""
        if [ -z "$PYTHON" ] && [ "$VENV_DIR" != "-" ] && [ "${SKIP_VENV:-}" != "1" ]; then
            uv venv "$VENV_DIR" --python 3.13 --seed >tmp/stdout.txt 2>tmp/stderr.txt || goto_show_logs=1
            PYTHON="$VENV_DIR/bin/python"
        fi
    elif python3 -c 'import sys; assert sys.version_info[:2] == (3, 13)' >tmp/stdout.txt 2>tmp/stderr.txt; then
        PYTHON=python3
    fi
fi

if [ -z "${PYTHON:-}" ]; then
    echo "Python 3.13 was not found. Install Python 3.13 or uv, or set PYTHON explicitly in webui.settings.sh."
    goto_show_logs=1
elif ! "$PYTHON" -c "" >tmp/stdout.txt 2>tmp/stderr.txt; then
    echo "Couldn't launch python: $PYTHON"
    goto_show_logs=1
fi

# Venv handling
if [ -z "$goto_show_logs" ]; then
    if [ "$VENV_DIR" = "-" ] || [ "$SKIP_VENV" = "1" ]; then
        :
    else
        if [ -x "$VENV_DIR/bin/python" ]; then
            :
        else
            PYTHON_FULLNAME=$("$PYTHON" -c "import sys; print(sys.executable)")
            echo "Creating venv in directory $VENV_DIR using python $PYTHON_FULLNAME"

            "$PYTHON_FULLNAME" -m venv "$VENV_DIR" >tmp/stdout.txt 2>tmp/stderr.txt || {
                echo "Unable to create venv in directory \"$VENV_DIR\""
                goto_show_logs=1
            }
        fi

        if [ -z "$goto_show_logs" ]; then
            "$VENV_DIR/bin/python" -m pip install --upgrade pip || \
                echo "Warning: Failed to upgrade PIP version"

            # Activate venv
            # shellcheck disable=SC1091
            source "$VENV_DIR/bin/activate"
            PYTHON="$VENV_DIR/bin/python"
            echo "venv $PYTHON"
        fi
    fi
fi

# Check the selected environment, not an unrelated system Python.
if [ -z "$goto_show_logs" ]; then
    if command -v uv >/dev/null 2>&1; then
        :
    elif ! "$PYTHON" -m pip --help >tmp/stdout.txt 2>tmp/stderr.txt; then
        echo "Couldn't launch pip"
        goto_show_logs=1
    fi
fi

# Launch
if [ -z "$goto_show_logs" ]; then
    "$PYTHON" launch.py "$@"

    if [ -f "$SD_WEBUI_RESTART" ]; then
        exec "$0" "$@"
    fi

    exit 0
fi

# Show logs
echo
echo "exit code: $?"

if [ -s tmp/stdout.txt ]; then
    echo
    echo "stdout:"
    cat tmp/stdout.txt
fi

if [ -s tmp/stderr.txt ]; then
    echo
    echo "stderr:"
    cat tmp/stderr.txt
fi

echo
echo "Launch Unsuccessful! Exiting..."
exit 1
