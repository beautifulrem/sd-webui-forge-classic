# Remi WebUI — tests and quality gates

One command runs every gate (scorecard dimension 8):

```sh
tools/check.sh                      # everything, ~15 min
tools/check.sh --skip e2e           # leave steps out (names or prefixes, comma-separated)
tools/check.sh --only ruff,pytest   # only these steps ("pytest" = core + every extension)
tools/check.sh --list               # step names
```

It keeps going after a failure, prints the last lines of each failing step's
log, ends with a table (step, result, seconds) and exits 1 when any step
failed. Logs go to `$REMI_CHECK_LOGS` (default: a temporary folder, printed at
the end).

| step | what it runs |
|---|---|
| `env:gradio-free` | fails when the chosen interpreter can import `gradio`, `gradio_client` or `gradio_rangeslider` |
| `ruff:remi` | ruff with the strict rule set on Remi's own Python: `remi/`, `tools/`, `webui/tools/`, `tests/remi_*.py`, `tests/test_remi_*.py`, `modules/extra_networks_pages.py`, `modules/redact.py` |
| `ruff:tests` | ruff on `tests/` and every `extensions-builtin/*/tests` (strict rules on the Remi files, Forge's legacy rules on the rest) |
| `ruff:extensions` | ruff on every `extensions-builtin/*/scripts` and `extensions-builtin/*/lib_*` (strict rules on the files Remi wrote, such as `lib_controlnet/unit_param.py`; legacy rules on the rest) |
| `ruff:core` | ruff with the legacy rules on the upstream-origin runtime Remi edits: `modules/`, `modules_forge/` (not the vendored `packages/`), `backend/`, `scripts/`, `launch.py`, `webui.py` |
| `ruff:syntax` | the whole tree (including vendored third-party code) for syntax errors (`E9`) and Gradio imports (`TID251`) |
| `pytest:tests` | `pytest tests --cov=remi`, failing below 91 % line coverage of `remi/` (measured 92 %), or on a leaked resource (below) |
| `pytest:<extension>` | `pytest extensions-builtin/<extension>/tests`, one run per directory (they share module names); a directory that collects nothing fails |
| `contract:openapi` | `tools/export_openapi.py --check`: `webui/src/api/openapi.json` matches the server |
| `contract:settings` | `tools/dump_settings.py --check`: the SPA's settings contract matches `GET /api/v2/options` |
| `contract:schemas` | `webui/tools/dump_schemas.py --check`: the tab schema fixtures the SPA contract tests render (`webui/src/api/mock/contract/{txt2img,img2img,extras}.json`) match `tab_schema()` |
| `contract:line-endings` | `tools/check_line_endings.py`: the files shared with Forge Neo that upstream commits with CRLF (`tools/upstream_crlf.txt`) still have CRLF, so Neo merges stay line-by-line ([upstream-absorption.md §3](upstream-absorption.md#3-procedure)) |
| `i18n:server` | `tests/test_remi_core_zh_cn.py`: the zh_CN catalog covers every core string, glossary, language pack and extension catalogs agree |
| `pnpm:install` | `pnpm install --frozen-lockfile` in `webui/` |
| `i18n:spa` | `src/locales/locales.test.ts`: every chrome key has a zh-CN translation with the same placeholders |
| `pnpm:build` | `tsc -b` (app, Vite config, e2e specs) and the production build |
| `pnpm:lint` | oxlint with `--deny-warnings --report-unused-disable-directives` (correctness and suspicious categories, React hooks rules, `typescript/no-explicit-any`; `webui/.oxlintrc.json`) |
| `pnpm:vitest` | the Vitest suites (jsdom) with V8 coverage, failing below the floors in `webui/vite.config.ts` (lines 83 %, statements 77 %, functions 72 %, branches 68 %; measured 84.5 / 79.0 / 73.1 / 69.0 %) |
| `e2e` | Playwright against the mock backend, including the axe scans |

**Python.** `$PYTHON`, else `venv/` or `.venv/` in the repo, else
`python3` — of those that import `fastapi`, `pytest`, `pytest_cov` and `torch`
(`requirements.txt` plus `requirements-dev.txt`), the first one without Gradio,
else the first one; the `env:gradio-free` step fails when the chosen
interpreter can import `gradio`, `gradio_client` or `gradio_rangeslider`
(the WebUI must start and its suites must pass without them). Ruff comes
from that environment, then `PATH`, then `uvx ruff`.

**Leaked resources fail the Python suites.** `pyproject.toml` turns
`ResourceWarning` (an unclosed SQLite connection, file or socket) and
pytest's `PytestUnraisableExceptionWarning` (the same warning raised while the
garbage collector frees the object) into errors for every pytest run, so a
leak fails `pytest:*` instead of passing silently (Python hides
`ResourceWarning` by default). The failing test is the one during which the
object was freed; `PYTHONTRACEMALLOC=20` shows where it was allocated. Close
SQLite connections explicitly (`contextlib.closing`, or a store's `close()`):
`with sqlite3.connect(...)` only commits or rolls back.

**Ruff configuration** (`pyproject.toml`). Upstream-origin code keeps Forge's
legacy rules (`N805`, `S307`, `S102`, `T`, `W`, `F`) and its existing style;
Remi's own files also get `E`, `I`, `B`, `UP`, `SIM`, `C4`, `RUF`, `PIE`,
`PLE` (a negated `per-file-ignores` glob lists them and switches the strict
rules off elsewhere: a new Python file Remi adds outside `remi/`, `tools/` or
`tests/(test_)remi_*.py` goes into that list). `print` (`T20`) is allowed in
the upstream-origin runtime and the extensions' scripts and libraries, which
report to the console as Forge does; Remi's own code logs. `TID251` bans
`gradio`, `gradio_client` and `gradio_rangeslider` imports everywhere, and a
`from tests import ...` (a site-packages `tests` package, as ultralytics
installs, would shadow the repository's: load helpers by path). Long lines
are the house style (black at 1024). The coverage data file goes to the log
folder (`COVERAGE_FILE`), never into the repository.

## End-to-end tests (`webui/tests/e2e`)

`pnpm test:e2e` (or `pnpm exec playwright test`) starts Vite in mock mode on
`$REMI_E2E_PORT` (default 5199; `tools/check.sh` uses 5299) unless a server
already answers there, then runs:

- `generate.spec.ts`: generate with live preview → result and history,
  Escape interrupts, batch count N → N results in batch order (one seed
  each), Alt+Enter skips the current batch and the job goes on, Generate
  forever re-queues until stopped, Generate several times, prompt editor keys
  and bracket warning, txt2img ↔ img2img, the image viewer, send to img2img,
  the phone layout. The mock backend runs batch count and Skip as the server
  does (`src/api/mock/server.ts`).
- `canvas.spec.ts`: inpaint (paint a mask, undo / redo, the job carries the
  mask), sketch, zoom / fit / full screen, uploaded masks per mode, drop and
  paste, image size, phone scrolling.
- `library.spec.ts`: gallery search and the viewer with the parameters,
  compare, PNG Info → txt2img, queue (pause, add twice, keyboard reorder,
  resume).
- `flows.spec.ts`: settings change applied with Mod+S and discard, the
  networks drawer (Alt+N, search, insert a LoRA), queue cancel, infotext read
  from the prompt, PNG Info from a pasted image, an extension page and its
  action, the Extensions page, zh-CN without a reload, a right-to-left
  locale (`<html dir="rtl">`, the rail on the right), `user.css` applied over
  the app's styles, light / dark (kept across reloads), command palette
  (navigate, theme), skip link, focus rings.
- `extensions.spec.ts`: extension management against the mock admin API:
  install from a URL (validation, the folder conflict, listed afterwards),
  load the index and install from it, Check for updates (progress, Stop,
  summary, Update buttons), restore a backup (cancel, then confirm: the
  reconnect overlay and the page reload).
- `extras.spec.ts`: Extras (upload a source, pick the upscaler and the scale,
  process, then the result size, the before / after comparison, its infotext
  and the viewer; the job carries the values by name), a txt2img result sent
  to Extras, and generation with Hires. fix and with the Refiner (the values
  reach the job, the infotext and the result's parameter list), and the Hires.
  fix target controls dimmed while they do not apply (`inactive_if`).
- `a11y.spec.ts`: axe-core (`@axe-core/playwright`, WCAG 2.x A/AA + best
  practices) on every page in light and dark, plus open states (networks
  drawer, result, image viewer, gallery details, PNG Info result, queue with
  jobs, inpaint editor, settings with unsaved changes, menus and the command
  palette) and the main pages on a phone. Any serious or critical violation
  fails the test with the rule, the element and the fix summary.

Before the workers start, `global-setup.ts` opens every page once in one
browser, so Vite's on-demand transforms are cached and the workers' first
pages do not race for them. Most flow tests also fail on console errors or uncaught page errors
(`watchConsole` in `helpers.ts`). Workers: 3 locally, 2 on CI
(`REMI_E2E_WORKERS` overrides); CI retries once.

## Real-server smoke test

```sh
$PYTHON tools/smoke_real.py   # --python $PYTHON --port 7879 --out /tmp/remi-smoke --baseline tests/fixtures/parity --report docs/research/smoke/<date>.json
```

`$PYTHON` is an interpreter with requirements.txt and without Gradio: the
first check fails when the server's interpreter (`--python`, default the one
running the script) can import `gradio`, `gradio_client` or
`gradio_rangeslider`, so a passing run proves a Gradio-free start and
generation. Starts `launch.py --api` with the real model in an isolated data folder (a
copy of `config.json`, models from `models/`), then checks txt2img parity
against the pre-rewrite baseline (the committed fixture
`tests/fixtures/parity/anima_turbo_txt2img_seed12345.png` by default) through `/sdapi/v1/txt2img` and
`/api/v2/jobs` (pixel-identical image, infotext equal except `Version`),
img2img at denoising 0.5, inpaint (the unmasked half unchanged, the masked
half changed) and an Extras Lanczos upscale, stops the server and prints a
table (exit 1 on a failed check, 2 when the server did not start). It
refuses to run while another `launch.py` holds the model. Not part of
`tools/check.sh` or CI (it needs the model and a GPU / MPS).
`--report <file>.json` also writes the run as JSON (date, git revision and
whether the tree had uncommitted changes, the server interpreter and the
Gradio packages it can import, which must be none, and every check): commit
it under `docs/research/smoke/` as the evidence that the WebUI starts and
generates without Gradio (scorecard dimension 1).

`--features` also checks the Anima features that need LoRA files and the
paste round-trips (about five more minutes). It writes two tiny deterministic
test LoRAs for the checkpoint (`tools/anima_test_loras.py`, rank 4, only the
checkpoint's safetensors header is read) into the temporary data folder and
starts the server with `--lora-dir` there, then checks at 512x512: a LoRA
changes the image; LoRA Layer Weight at block weight 0 undoes it (pixel-identical
to the prompt without the tag, whose leftover `, ` is a token too); LoRA Stage
Scheduler changes it; Anima FreeFuse routes the two LoRAs (infotext
`Anima FreeFuse: 2 subjects ...`); Sparse Attention records its settings on a
CUDA GPU with sol_attn and elsewhere leaves the image unchanged with the job
warning "... was skipped: CUDA device is required"; inpaint with "Masked
content: latent noise" finishes and keeps the unmasked half. Each feature
job's infotext (and Prompt Anchor's, PP Suite's and the Refiner's) goes through `POST /api/v2/infotext/parse`
without problems and, queued again, gives the same pixels and infotext. So do
three runs where a feature replaced part of the request (an Anima Flow
sampler's scheduler lock, Dynamic Shift's Automatic schedule and the Anima
Hires Guard's clamp of a 2x Hires target): the infotext records the
replacement ("... replaced scheduler: Simple", "Anima Hires adjustments"),
the paste restores the request and the pasted job replaces it again. Without
a model, `tests/test_infotext_replaced_settings_roundtrip.py` checks the
same byte for byte (the API's processing object, the scripts' hooks,
`get_sigmas` and `create_infotext`), and `tests/test_refiner_switch.py`
switches between two tiny synthetic checkpoints that differ: the refiner
pass changes the result from the switch step on and records
`Refiner: <name> [hash], Refiner switch at`.
For the Refiner it also writes a second, genuinely different checkpoint into
the data folder (`tools/anima_test_loras.py --refiner`: a copy of the
checkpoint with a third seeded low-rank delta merged into the same layers,
about 4 GB, passed with `--ckpt-dirs`; `--no-refiner-checkpoint` skips it and
uses an installed one): with it as the Refiner at switch 0.5 the image must
change, the infotext record `Refiner: remi-test-refiner [<hash>], Refiner
switch at: 0.5`, and the base checkpoint be selected again afterwards.
The same LoRAs can be written into a LoRA folder by hand:
`$PYTHON tools/anima_test_loras.py models/Stable-diffusion/<anima>.safetensors models/Lora`.

## Browser audit of the real server

`docs/remi-webui/integration-checklist.md` records the full walk-through
(every page and built-in extension panel, light / dark, en / zh-CN, three
window sizes). Its automated part runs against a running server (not part of
`tools/check.sh`: it needs the server and about 20 minutes):

```sh
cd webui
node tools/audit/real-server.mjs --base http://127.0.0.1:7860 --out /tmp/audit --expand 1 --axe all
```

It fails on console errors / warnings, failed requests, axe serious /
critical violations, page overflow, wrapped tab strips, clipped text, stray
scrollbars and scrolled layout boxes, and lists Latin-only text in zh-CN for
review. `tall-shot.mjs` (full-height screenshots with every section open),
`axe-one.mjs` (one page's violations in detail) and `find-text.mjs` (where a
text is rendered) help with what it finds.

## Performance measurements

`docs/research/ui-performance.md` records the scorecard's performance budgets
with the method and raw numbers. The tools are not part of `tools/check.sh`
(they need the real model, synthetic fixtures and about an hour):
`tools/perf/fixtures.py` (5,000 LoRAs, 10,000 gallery images, the 300-token
prompt), `tools/perf/serve_traced.py` (the server with timing probes),
`tools/perf/startup.py` (old vs new startup), `webui/tools/perf/bundle.mjs`
(bundle budgets from the Vite manifest) and `webui/tools/perf/browser.mjs`
(Playwright + CDP: DOM size, INP, scrolling, preview latency, generation
overhead).

## CI

`.github/workflows/check.yml` runs `tools/check.sh` in two jobs: Python
(CPU torch, `requirements.txt` + `requirements-dev.txt`, no Gradio; steps
`env`, `ruff`, `pytest` with the coverage floor, `contract`, `i18n:server`)
and the SPA (install, build, lint, Vitest with coverage, Playwright with
Chromium; steps `pnpm`, `i18n:spa`, `e2e`). Logs and the Playwright report
are uploaded when a job fails. The workflow passes `actionlint`; it runs on
pushes to `remi-webui`, `main` and `neo`, on pull requests and by hand
(`workflow_dispatch`), so a run exists once the branch is pushed to GitHub.
Until then the same steps run locally with the command CI uses:

```sh
PYTHON=/path/to/gradio-free/python tools/check.sh --only env,ruff,pytest,contract,i18n:server
tools/check.sh --only pnpm,i18n:spa,e2e
```
