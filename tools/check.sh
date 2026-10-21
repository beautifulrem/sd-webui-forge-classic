#!/usr/bin/env bash
# Every quality gate of the Remi WebUI in one command (scorecard dimension 8).
#
#   tools/check.sh                    # everything
#   tools/check.sh --skip e2e         # leave steps out (comma-separated names or prefixes)
#   tools/check.sh --only ruff,pytest # run only these steps (prefixes: "pytest" = core + extensions)
#   tools/check.sh --list             # list the steps
#   tools/check.sh --strict           # a skipped step (a missing optional interpreter) fails the run
#
# Steps: a Gradio-free Python - ruff (Remi code strict, test suites, the built-in extensions'
# scripts and libraries, the upstream-origin runtime, the whole tree for syntax errors and
# Gradio imports) - pytest for
# tests/ (with a coverage floor for remi/) and for every extensions-builtin/*/tests directory
# separately, then a coverage floor for each Remi file outside remi/ over all of them - OpenAPI,
# settings and tab schema contracts, upstream line endings - the legacy ui() recorder's self-test
# (skipped, with the reason, without its Gradio interpreter) - i18n coverage (server zh_CN catalog, SPA
# chrome catalog) - pnpm install (frozen lockfile), build (tsc + vite), the JS bundle budget,
# oxlint, vitest (with coverage floors), Playwright end-to-end tests and axe scans against the
# mock backend.
# Steps keep running after a failure; the script ends with a summary table and exits 1
# when any step failed (or, with --strict, was skipped).
#
# Python: $PYTHON, else venv/ or .venv/ in the repo, else python3 - the first one with
# requirements.txt and requirements-dev.txt (pytest, pytest-cov) installed, preferring one
# without Gradio (env:gradio-free fails when the chosen one can import Gradio: the WebUI
# must start without it). Ruff is taken from that environment, then PATH, then `uvx ruff`.
# Logs: $REMI_CHECK_LOGS (default a temporary directory, printed at the end). The
# real-model smoke test is separate: tools/smoke_real.py.
# legacy-ui (the legacy ui() recorder's self-test) needs the recorder's own Gradio 4.40 interpreter,
# built once with the network by tools/legacy_ui_env.sh: $REMI_LEGACY_PYTHON, else the one that
# script builds by default (../remi-gradio-env/bin/python). Without it the step is reported as
# "skip" with the reason (a failure when named in --only, or with --strict). So does
# legacy-ui:fixtures, which records every committed legacy ui() fixture again and compares it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

ONLY=""
SKIP=""
LIST=0
STRICT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    --skip) SKIP="${2:-}"; shift 2 ;;
    --skip=*) SKIP="${1#*=}"; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# region Environment

GRADIO_PROBE='import importlib.util, sys; found = [name for name in ("gradio", "gradio_client", "gradio_rangeslider") if importlib.util.find_spec(name)]; print(" ".join(found)); sys.exit(1 if found else 0)'

pick_python() {
  # $PYTHON is used as given; otherwise the first candidate without Gradio, else the first one.
  local candidate fallback=""
  if [ -n "${PYTHON:-}" ]; then
    echo "$PYTHON"
    return 0
  fi
  for candidate in "$ROOT/venv/bin/python" "$ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    "$candidate" -c "import fastapi, pytest, pytest_cov, torch" >/dev/null 2>&1 || continue
    if "$candidate" -c "$GRADIO_PROBE" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
    [ -n "$fallback" ] || fallback="$candidate"
  done
  [ -n "$fallback" ] && echo "$fallback"
}

# Python steps fail (with this message) when there is none; frontend-only runs do not need it.
PY="$(pick_python)"
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

# The legacy ui() recorder's Gradio 4.40 interpreter (tools/legacy_ui_env.sh's default location).
LEGACY_DEFAULT="$(cd "$ROOT/.." && pwd)/remi-gradio-env/bin/python"
if [ -n "${REMI_LEGACY_PYTHON:-}" ]; then
  LEGACY_PY="$REMI_LEGACY_PYTHON"
elif [ -x "$LEGACY_DEFAULT" ]; then
  LEGACY_PY="$LEGACY_DEFAULT"
else
  LEGACY_PY=""
fi

LOGS="${REMI_CHECK_LOGS:-$(mktemp -d "${TMPDIR:-/tmp}/remi-check.XXXXXX")}"
mkdir -p "$LOGS"
# pytest-cov's data file goes with the logs, not into the repository.
export COVERAGE_FILE="$LOGS/.coverage"
# Port of the e2e test's own mock dev server: away from a developer's `pnpm dev` (5173/5199).
export REMI_E2E_PORT="${REMI_E2E_PORT:-5299}"
export PYTHONDONTWRITEBYTECODE=1

# endregion

# region Step bookkeeping

NAMES=()
RESULTS=()
TIMES=()
SKIPPED=() # "<name>: <reason>" of each skipped step

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

# skip_step <name> <reason>: a step that cannot run here (a missing optional interpreter). It is
# still listed, and the summary reports it as "skip" with the reason; --strict counts it as failed.
skip_step() {
  local name="$1" reason="$2"
  if [ "$LIST" = 1 ]; then
    echo "$name"
    return 0
  fi
  wanted "$name" || return 0
  printf '==> %-44s skip (%s)\n' "$name" "$reason"
  NAMES+=("$name")
  TIMES+=("0")
  RESULTS+=("skip")
  SKIPPED+=("$name: $reason")
}

# pytest that also fails when nothing was collected (pytest exits 5).
run_pytest() {
  py -m pytest "$@" -q -p no:cacheprovider -W ignore::DeprecationWarning
}

# endregion

# region Python

gradio_free() {
  local found
  if found="$(py -c "$GRADIO_PROBE")"; then
    echo "$PY cannot import Gradio"
    return 0
  fi
  [ -n "$found" ] || return 2
  echo "$PY can import: $found" >&2
  echo "The WebUI is Gradio-free; run the checks with an interpreter without Gradio (set PYTHON=...) so a Gradio import cannot hide." >&2
  return 1
}

step env:gradio-free gradio_free

# Which rules apply to which file is pyproject.toml's business (strict for Remi's files).
step ruff:remi "${RUFF[@]}" check remi tools webui/tools docs tests/remi_*.py tests/test_remi_*.py modules/extra_networks_pages.py modules/redact.py
step ruff:tests "${RUFF[@]}" check tests extensions-builtin/*/tests
step ruff:extensions "${RUFF[@]}" check extensions-builtin/*/scripts extensions-builtin/*/lib_*
# The upstream-origin runtime Remi edits (legacy rules); modules_forge/packages is vendored as is.
step ruff:core "${RUFF[@]}" check modules modules_forge backend scripts launch.py webui.py --extend-exclude modules_forge/packages
step ruff:syntax "${RUFF[@]}" check . --select E9,TID251 --output-format concise

# Coverage floors (tools/check_coverage.py): remi/ as a whole (the measured line coverage of
# tests/, minus a small margin), and each Remi file outside remi/ on its own (pyproject.toml's
# strict list: modules/redact.py, the ControlNet unit editor, ...). Those files are tested by
# different suites, so every pytest step measures them into a data file of its own
# ($LOGS/.coverage.pytest_<name>) and pytest:coverage combines the data of this log folder.
COVERAGE_FLOOR_PYTHON=92
COVERAGE_FLOOR_STRICT=90
coverage_data() { echo "$LOGS/.coverage.${1//[:\/]/_}"; }
# pytest_cov <step name> <pytest args...>: run_pytest measuring the Remi files outside remi/
pytest_cov() {
  local data sources
  data="$(coverage_data "$1")"
  shift
  sources="$(py tools/check_coverage.py sources)" || return 2
  rm -f "$data"
  # shellcheck disable=SC2086 # one --cov=<source> per word
  COVERAGE_FILE="$data" run_pytest "$@" $sources --cov-report=
}
pytest_tests() {
  local status=0
  pytest_cov pytest:tests tests --cov=remi || status=$?
  py tools/check_coverage.py remi --data "$(coverage_data pytest:tests)" --floor "$COVERAGE_FLOOR_PYTHON" || status=$((status ? status : 1))
  return $status
}
step pytest:tests pytest_tests
for dir in extensions-builtin/*/tests; do
  [ -d "$dir" ] || continue
  ext="${dir#extensions-builtin/}"
  step "pytest:${ext%/tests}" pytest_cov "pytest:${ext%/tests}" "$dir"
done
strict_coverage() {
  py tools/check_coverage.py strict --floor "$COVERAGE_FLOOR_STRICT" --data "$LOGS"/.coverage.pytest_*
}
step pytest:coverage strict_coverage

step contract:openapi py tools/export_openapi.py --check
step contract:settings py tools/dump_settings.py --check
# The real tab schemas the SPA contract tests (webui/src/ui/contract.test.tsx) render.
step contract:schemas py webui/tools/dump_schemas.py --check
# Files shared with Forge Neo keep upstream's CRLF line endings (clean merges of Neo updates).
step contract:line-endings py tools/check_line_endings.py
# The legacy ui() recorder end to end: records a known script in the last Gradio-era revision and
# compares the record with its committed fixtures (docs/remi-webui/upstream-absorption.md §2.5).
# Its interpreter is optional (Gradio 4.40, built once with the network): without one the step is
# a "skip" with the reason, unless --only names it (then it fails with the same message).
LEGACY_MISSING="no Gradio 4.40 interpreter: build it with tools/legacy_ui_env.sh (to $LEGACY_DEFAULT) or set REMI_LEGACY_PYTHON"
legacy_ui_self_test() {
  if [ -z "$LEGACY_PY" ]; then
    echo "$LEGACY_MISSING" >&2
    return 2
  fi
  "$LEGACY_PY" tools/record_legacy_ui.py --self-test --report "$LOGS/legacy-ui.json"
}
if [ -n "$LEGACY_PY" ] || { [ -n "$ONLY" ] && matches legacy-ui "$ONLY"; }; then
  step legacy-ui legacy_ui_self_test
else
  skip_step legacy-ui "$LEGACY_MISSING"
fi
# Every committed legacy ui() fixture recorded again in that revision, each with its format and the
# WebUI context its ui() needs (tools/legacy_ui_recordings.py), and compared value by value.
legacy_ui_fixtures() {
  if [ -z "$LEGACY_PY" ]; then
    echo "$LEGACY_MISSING" >&2
    return 2
  fi
  "$LEGACY_PY" tools/record_legacy_ui.py --fixtures --check --report "$LOGS/legacy-ui-fixtures.json"
}
if [ -n "$LEGACY_PY" ] || { [ -n "$ONLY" ] && matches legacy-ui:fixtures "$ONLY"; }; then
  step legacy-ui:fixtures legacy_ui_fixtures
else
  skip_step legacy-ui:fixtures "$LEGACY_MISSING"
fi

step i18n:server run_pytest tests/test_remi_core_zh_cn.py

# endregion

# region Frontend

pnpm_in_webui() { (cd "$ROOT/webui" && pnpm "$@"); }

step pnpm:install pnpm_in_webui install --frozen-lockfile
step i18n:spa pnpm_in_webui exec vitest run src/locales
step pnpm:build pnpm_in_webui build
# The JS budget (scorecard dimension 5): a manifest build of the same sources, checked against
# webui/tools/perf/budget.json (initial JS, every route and lazy entry, recorded exceptions).
bundle_budget() {
  local out="$LOGS/bundle-dist"
  pnpm_in_webui exec vite build --manifest --outDir "$out" --emptyOutDir --logLevel warn &&
    (cd "$ROOT/webui" && node tools/perf/bundle.mjs "$out" --check --json "$LOGS/bundle.json")
}
step perf:bundle bundle_budget
step pnpm:lint pnpm_in_webui lint
step pnpm:vitest pnpm_in_webui exec vitest run --coverage
step e2e pnpm_in_webui exec playwright test --reporter=line

# endregion

[ "$LIST" = 1 ] && exit 0

# region Summary

failed=0
skipped=0
echo
printf '%-44s %-6s %8s\n' "step" "result" "seconds"
printf '%-44s %-6s %8s\n' "--------------------------------------------" "------" "--------"
for i in "${!NAMES[@]}"; do
  printf '%-44s %-6s %8s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"
  case "${RESULTS[$i]}" in
    pass) ;;
    skip) skipped=$((skipped + 1)) ;;
    *) failed=$((failed + 1)) ;;
  esac
done
echo
echo "Python: ${PY:-none found}"
echo "Legacy: ${LEGACY_PY:-none found}"
echo "Logs:   $LOGS"
if [ ${#NAMES[@]} -eq 0 ]; then
  echo "No step matched --only/--skip." >&2
  exit 2
fi
if [ $skipped -gt 0 ]; then
  echo "Skipped (not run, not passed):"
  printf '  %s\n' "${SKIPPED[@]}"
fi
passed=$((${#NAMES[@]} - failed - skipped))
if [ $failed -gt 0 ]; then
  echo "$failed of ${#NAMES[@]} steps failed ($passed passed, $skipped skipped)."
  exit 1
fi
if [ $skipped -gt 0 ]; then
  echo "$passed of ${#NAMES[@]} steps passed, $skipped skipped."
  [ "$STRICT" = 1 ] && { echo "--strict: a skipped step fails the run." >&2; exit 1; }
  exit 0
fi
echo "All ${#NAMES[@]} steps passed."

# endregion
