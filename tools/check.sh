#!/usr/bin/env bash
# Every quality gate of the Remi WebUI in one command (scorecard dimension 8).
#
#   tools/check.sh                    # everything
#   tools/check.sh --skip e2e         # leave steps out (comma-separated names or prefixes)
#   tools/check.sh --only ruff,pytest # run only these steps (prefixes: "pytest" = core + extensions)
#   tools/check.sh --list             # list the steps
#
# Steps: ruff (Remi code strict, test suites, whole tree for syntax errors) - pytest for tests/
# and for every extensions-builtin/*/tests directory separately - OpenAPI and settings
# contracts - i18n coverage (server zh_CN catalog, SPA chrome catalog) - pnpm install
# (frozen lockfile), build (tsc + vite), oxlint, vitest, Playwright end-to-end tests and
# axe scans against the mock backend. Steps keep running after a failure; the script ends
# with a summary table and exits 1 when any step failed.
#
# Python: $PYTHON, else venv/ or .venv/ in the repo, else the shared test env
# /tmp/remi-testenv, else python3 (it needs requirements.txt plus pytest; ruff is taken
# from that environment, then PATH, then `uvx ruff`). Logs: $REMI_CHECK_LOGS (default
# a temporary directory, printed at the end). The real-model smoke test is separate:
# tools/smoke_real.py.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

ONLY=""
SKIP=""
LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:-}"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    --skip) SKIP="${2:-}"; shift 2 ;;
    --skip=*) SKIP="${1#*=}"; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# region Environment

pick_python() {
  local candidate
  for candidate in "${PYTHON:-}" "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" /tmp/remi-testenv/bin/python "$(command -v python3 || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    if "$candidate" -c "import fastapi, pytest, torch" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Python steps fail (with this message) when there is none; frontend-only runs do not need it.
PY="$(pick_python)" || PY=""
py() {
  if [ -z "$PY" ]; then
    echo "No Python with the WebUI requirements and pytest found: set PYTHON=/path/to/python" >&2
    return 2
  fi
  "$PY" "$@"
}

if [ -n "$PY" ] && "$PY" -m ruff --version >/dev/null 2>&1; then
  RUFF=("$PY" -m ruff)
elif command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
else
  RUFF=(uvx ruff)
fi

LOGS="${REMI_CHECK_LOGS:-$(mktemp -d "${TMPDIR:-/tmp}/remi-check.XXXXXX")}"
mkdir -p "$LOGS"
# Port of the e2e test's own mock dev server: away from a developer's `pnpm dev` (5173/5199).
export REMI_E2E_PORT="${REMI_E2E_PORT:-5299}"
export PYTHONDONTWRITEBYTECODE=1

# endregion

# region Step bookkeeping

NAMES=()
RESULTS=()
TIMES=()

matches() { # matches <name> <comma list>: exact name or prefix
  local item
  local IFS=,
  for item in $2; do
    [ -n "$item" ] || continue
    case "$1" in "$item"|"$item":*|"$item"-*) return 0 ;; esac
  done
  return 1
}

wanted() {
  if [ -n "$ONLY" ] && ! matches "$1" "$ONLY"; then return 1; fi
  if [ -n "$SKIP" ] && matches "$1" "$SKIP"; then return 1; fi
  return 0
}

# step <name> <command...>: runs the command with its output in $LOGS/<name>.log
step() {
  local name="$1"
  shift
  if [ "$LIST" = 1 ]; then
    echo "$name"
    return 0
  fi
  wanted "$name" || return 0
  local log="$LOGS/${name//[:\/]/_}.log"
  local start end status
  printf '==> %-44s ' "$name"
  start=$(date +%s)
  "$@" >"$log" 2>&1
  status=$?
  end=$(date +%s)
  NAMES+=("$name")
  TIMES+=("$((end - start))")
  if [ $status -eq 0 ]; then
    RESULTS+=("pass")
    echo "pass ($((end - start))s)"
  else
    RESULTS+=("FAIL")
    echo "FAIL ($((end - start))s, exit $status)"
    tail -n 40 "$log" | sed 's/^/    | /'
  fi
}

# pytest that also fails when nothing was collected (pytest exits 5).
run_pytest() {
  py -m pytest "$@" -q -p no:cacheprovider -W ignore::DeprecationWarning
}

# endregion

# region Python

step ruff:remi "${RUFF[@]}" check remi tools webui/tools tests/remi_*.py tests/test_remi_*.py
step ruff:tests "${RUFF[@]}" check tests extensions-builtin/*/tests
step ruff:syntax "${RUFF[@]}" check . --select E9 --output-format concise

step pytest:tests run_pytest tests
for dir in extensions-builtin/*/tests; do
  [ -d "$dir" ] || continue
  ext="${dir#extensions-builtin/}"
  step "pytest:${ext%/tests}" run_pytest "$dir"
done

step contract:openapi py tools/export_openapi.py --check
step contract:settings py tools/dump_settings.py --check

step i18n:server run_pytest tests/test_remi_core_zh_cn.py

# endregion

# region Frontend

pnpm_in_webui() { (cd "$ROOT/webui" && pnpm "$@"); }

step pnpm:install pnpm_in_webui install --frozen-lockfile
step i18n:spa pnpm_in_webui exec vitest run src/locales
step pnpm:build pnpm_in_webui build
step pnpm:lint pnpm_in_webui lint
step pnpm:vitest pnpm_in_webui exec vitest run
step e2e pnpm_in_webui exec playwright test --reporter=line

# endregion

[ "$LIST" = 1 ] && exit 0

# region Summary

failed=0
echo
printf '%-44s %-6s %8s\n' "step" "result" "seconds"
printf '%-44s %-6s %8s\n' "--------------------------------------------" "------" "--------"
for i in "${!NAMES[@]}"; do
  printf '%-44s %-6s %8s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"
  [ "${RESULTS[$i]}" = "pass" ] || failed=$((failed + 1))
done
echo
echo "Python: ${PY:-none found}"
echo "Logs:   $LOGS"
if [ ${#NAMES[@]} -eq 0 ]; then
  echo "No step matched --only/--skip." >&2
  exit 2
fi
if [ $failed -gt 0 ]; then
  echo "$failed of ${#NAMES[@]} steps failed."
  exit 1
fi
echo "All ${#NAMES[@]} steps passed."

# endregion
