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
| `ruff:remi` | ruff with the strict rule set on Remi's own Python: `remi/`, `tools/`, `webui/tools/`, `docs/` (any example or measurement script there), `tests/remi_*.py`, `tests/test_remi_*.py`, `modules/extra_networks_pages.py`, `modules/redact.py` |
| `ruff:tests` | ruff on `tests/` and every `extensions-builtin/*/tests` (strict rules on the Remi files, Forge's legacy rules on the rest) |
| `ruff:extensions` | ruff on every `extensions-builtin/*/scripts` and `extensions-builtin/*/lib_*` (strict rules on the files Remi wrote, such as `lib_controlnet/unit_param.py`; legacy rules on the rest) |
| `ruff:core` | ruff with the legacy rules on the upstream-origin runtime Remi edits: `modules/`, `modules_forge/` (not the vendored `packages/`), `backend/`, `scripts/`, `launch.py`, `webui.py` |
| `ruff:syntax` | the whole tree (including vendored third-party code) for syntax errors (`E9`) and Gradio imports (`TID251`) |
| `pytest:tests` | `pytest tests --cov=remi`, failing below 92 % line coverage of `remi/` (measured 93.6 %), or on a leaked resource (below) |
| `pytest:<extension>` | `pytest extensions-builtin/<extension>/tests`, one run per directory (they share module names); a directory that collects nothing fails |
| `contract:openapi` | `tools/export_openapi.py --check`: `webui/src/api/openapi.json` matches the server |
| `contract:settings` | `tools/dump_settings.py --check`: the SPA's settings contract matches `GET /api/v2/options` |
| `contract:schemas` | `webui/tools/dump_schemas.py --check`: the tab schema fixtures the SPA contract tests render (`webui/src/api/mock/contract/{txt2img,img2img,extras}.json`) match `tab_schema()` |
| `contract:line-endings` | `tools/check_line_endings.py`: the files shared with Forge Neo that upstream commits with CRLF (`tools/upstream_crlf.txt`) still have CRLF, so Neo merges stay line-by-line ([upstream-absorption.md §3](upstream-absorption.md#3-procedure)) |
| `legacy-ui` | only with `$REMI_LEGACY_PYTHON` (the Gradio 4.40 interpreter `tools/legacy_ui_env.sh` builds, network needed once), or when named in `--only`: `tools/record_legacy_ui.py --self-test` records Spectrum's Gradio `ui()` at its last Gradio revision and compares it with the committed fixtures ([upstream-absorption.md §2.5](upstream-absorption.md#25-a-new-built-in-extension-upstream)) |
| `i18n:server` | `tests/test_remi_core_zh_cn.py`: the zh_CN catalog covers every core string, glossary, language pack and extension catalogs agree |
| `pnpm:install` | `pnpm install --frozen-lockfile` in `webui/` |
| `i18n:spa` | `src/locales/locales.test.ts`: every chrome key has a zh-CN translation with the same placeholders |
| `pnpm:build` | `tsc -b` (app, Vite config, e2e specs) and the production build |
| `perf:bundle` | a `--manifest` build of the same sources into the log directory, checked by `webui/tools/perf/bundle.mjs --check` against `webui/tools/perf/budget.json`: the initial JS ≤ 300 KB gzip, every route and every other lazy entry ≤ 150 KB on top of it, the exempt editor chunks and the recorded exception (the default Generate route, capped); the report is `bundle.json` in the logs (`docs/research/ui-performance.md`, "Bundle") |
| `pnpm:lint` | oxlint with `--deny-warnings --report-unused-disable-directives` (correctness and suspicious categories, React hooks rules, `typescript/no-explicit-any`; `webui/.oxlintrc.json`) |
| `pnpm:vitest` | the Vitest suites (jsdom) with V8 coverage, failing below the floors in `webui/vite.config.ts` (lines 85 %, statements 79.5 %, functions 75 %, branches 69.5 %; measured 86.1 / 80.6 / 76.2 / 70.6 %) |
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
  and bracket warning, txt2img ↔ img2img, the image viewer, send to img2img (with the "Resize by" size of the sent result),
  the phone layout. The mock backend runs batch count and Skip as the server
  does (`src/api/mock/server.ts`).
- `canvas.spec.ts`: inpaint (paint a mask, undo / redo, the job carries the
  mask), sketch, zoom / fit / full screen, the shortcuts of an editor that
  appears under a resting pointer, uploaded masks per mode, drop and
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
- `browser-apis.spec.ts` (`@real`, also run against a real server, see
  "Browser features against the real server"): the screen wake lock while
  generating, the notification permission prompt and the notification and
  sound in the background, gamepad navigation and the live preview in the
  viewer, the "Check for updates" feedback.
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

## Every extension end to end on a tiny model

`tests/test_remi_extensions_e2e.py` (part of `pytest:tests`, 2-5 minutes on
the CPU) runs every built-in extension and every script of `scripts/` and
`modules/processing_scripts/` through the real server, without the real model
and without network access:

- `tools/tiny_anima.py` writes a checkpoint of the real Anima architecture at
  toy size (28 DiT blocks of width 64, a 2-layer Qwen3 text encoder with
  Anima's tokenizers, the Qwen-Image VAE at base width 8; seeded weights,
  about 40 MB) and loads it as the real `Anima` engine (the stock loader only
  accepts the real sizes, so `install()` wraps `forge_loader` for files it
  wrote). Its `write_models()` adds a second checkpoint (the Refiner's), the
  test LoRAs and LoKr of `tools/anima_test_loras.py`, an Anima Control-LLLite
  (ControlNet) and a YOLOv8n ADetailer detector built from Ultralytics'
  bundled config, all seeded.
- `tests/remi_extensions_e2e.py` starts the WebUI in a fresh interpreter
  (`--cpu`, every built-in extension, the server's startup callbacks), serves
  the real app and job runner, and queues each case through
  `POST /api/v2/jobs` with the extension off and on.
- For each case the test checks that the job completes with only the expected
  warnings and no script error, that the extension's hooks ran with the
  arguments the schema converted (a choice's index where `run()` takes one),
  that the infotext gains its keys, that the image changes (or, for the
  extensions that must not change it, is the same pixels: Prompt Blacklist,
  Never OOM, Sparse Attention off CUDA, prompt comments), and that pasting the
  infotext into a fresh form (`POST /api/v2/infotext/parse`) and queueing it
  gives the same pixels and infotext. Some cases check more: LoRA Layer Weight
  at block weight 0 gives the pixels of `<lora:x:0>` (LoRA and LoKR tabs),
  LoRA Control's weight schedule gives neither the LoRA at 1 nor at 0.
- ControlNet runs with the Anima Control-LLLite and the canny, tile_colorfix
  and inpaint_only preprocessors (so `sd_forge_controlllite` and the
  `forge_*` preprocessor folders run too), the LoRA cases go through
  `sd_forge_lora`. `NOT_RUN` in the test lists what is not run, with the reason
  (UI-only extensions and pages; the reference preprocessor and IP-Adapter,
  which patch the SD 1.5 / SDXL UNet's attention; Torch Compile and
  RadialAttention, CUDA only; PiD, a separate model); a new extension
  directory or script fails the test until it is one or the other.

The cases are data (`CASES` in `tests/remi_extensions_e2e.py`); one or a few
run alone with

```sh
$PYTHON tests/remi_extensions_e2e.py /tmp/remi-e2e /tmp/remi-e2e.json --only dave,refiner,adetailer
```

and the same models serve the whole WebUI for trying an extension by hand, in
the browser or through `/api/v2` and `/sdapi/v1`, with a model that loads in a
second:

```sh
$PYTHON tools/tiny_anima.py serve --data-dir /tmp/remi-tiny --port 7899 --cpu --api
```

## Real-server smoke test

```sh
$PYTHON tools/smoke_real.py   # [--features] [--models] --python $PYTHON --port 7879 --out /tmp/remi-smoke --baseline tests/fixtures/parity --report docs/research/smoke/<date>.json
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
test LoRAs and a test LoKr for the checkpoint (`tools/anima_test_loras.py`,
rank 4; the LoKr has `lokr_w1` (4x4) and a rank-4 `lokr_w2_a` / `lokr_w2_b`
per layer, the layout LyCORIS LoKr trainers write; only the checkpoint's
safetensors header is read) into the temporary data folder and
starts the server with `--lora-dir` there, then checks at 512x512: a LoRA
changes the image; LoRA Layer Weight at block weight 0 undoes it (pixel-identical
to the prompt without the tag, whose leftover `, ` is a token too); the LoKr
changes the image, and Layer Weight's LoKR tab at `all=1.0` reproduces
`<lora:remi-test-lokr:1>` and at `all=0` `<lora:remi-test-lokr:0>`
pixel for pixel (the tag's own weight is ignored under the tab), `0-13=0`
gives a third image that its pasted infotext regenerates, the infotext
records `Anima LoRA layer lokr enabled: True` and `Anima LoRA layer lokr
blocks: ...`, and the LoRA tab at 0 leaves the LoKr alone. Without a model,
`extensions-builtin/anima-lora-layer-weight/tests/test_lokr_patches.py`
takes the same LoKr (written for a tiny checkpoint) through Forge's loader
(`load_lora` → comfy's `LoKrAdapter`), the panel's `add_patches` hook and
`merge_lora_to_weight`: `all=1` gives exactly the weights of the tag at 1,
`all=0` those of the tag at 0, a block or module at 0 leaves exactly that
part, each tab touches only its adapter kind, and the infotext keys are
written; LoRA Stage
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
The same LoRAs (and the LoKr) can be written into a LoRA folder by hand:
`$PYTHON tools/anima_test_loras.py models/Stable-diffusion/<anima>.safetensors models/Lora`.

`--models` checks the features that need model files, with the ones
installed in `models/` (a missing one fails its check; nothing is
downloaded), on the parity baseline request (768x768, seed 12345, so the
first pass is the baseline image):

| check | model | passes when |
|---|---|---|
| Extras ESRGAN | the ESRGAN file in `models/ESRGAN` (an anime one first; `RealESRGAN_x4plus_anime_6B.pth`), `/sdapi/v1/extra-single-image` at 2x | the result is 1536x1536, downscaled it differs from the source by a mean below 8, it is not a plain Lanczos 2x (mean difference above 2: a model that fails to load leaves the Lanczos resize), and the info records `Postprocess upscale by: 2` and `Postprocess upscaler: <name>` |
| ControlNet | preprocessor `lineart_standard (from white bg & black line)` and the lineart model in `models/ControlNet` (`anima-lllite-lineart-1`), through `/sdapi/v1/txt2img` `alwayson_scripts` on a generated line drawing | the infotext's `ControlNet 0` records `Module: lineart_standard ...` and `Model: <model>`, the image differs from the baseline (mean above 2), and its edges follow the drawing: their correlation with the drawing's lines is at least 0.05 above the baseline's (about -0.04) |
| ADetailer | `models/adetailer/face_yolov8n.pt` (`ad_enable`, `ad_model`) through `/api/v2/jobs` | the detector, run by the smoke test itself, finds one face in the baseline image; the job's infotext records `ADetailer model: face_yolov8n.pt` and `ADetailer Version`; inside the face box the image changed (mean above 1), farther than 32 px from it it did not (mean below 0.05); the pasted infotext regenerates the image |

Without the Anima model, `tests/test_smoke_real_models.py` runs these checks
against a fake server (the requests they send, a silent fallback failing
them, missing models reported) and, when the files are installed, checks the
inputs they rely on: `face_yolov8n.pt` finds the one face in the baseline
image, and the ESRGAN model loads with spandrel (the WebUI's loader) and
upscales by its scale keeping the content.

Sparse Attention Integrated needs CUDA (sol_attn); on MPS and CPU the job
says so ("Sparse Attention Integrated was skipped: CUDA device is
required") and the image is unchanged, which `--features` checks. The
Prompt Blacklist panel only acts through its button (it rewrites the
prompt in the form), so it never changes a queued generation by design.

### Browser features against the real server (`--browser`)

Some SPA features go through browser APIs that cannot be watched on a real
screen: the screen wake lock while generating
(`prevent_screen_sleep_during_generation`), the notification permission
prompt ("Notify me when images are ready"), the browser notification and
the `notification.mp3` sound at `notification_volume` when images are ready
with the page in the background, gamepad navigation in the viewer
(`js_modal_lightbox_gamepad`), the live preview in the viewer
(`js_live_preview_in_modal_lightbox`), and the extensions' "Check for
updates" feedback (a toast "No user extensions to check" when none is
enabled and installed from a repository, else the progress).
`webui/tests/e2e/browser-apis.spec.ts` checks each with recording probes in
place of `navigator.wakeLock`, `Notification`, `Audio` and
`navigator.getGamepads`, and puts the page in the background by its
visibility and focus. Its tests are tagged `@real`: they run with the mock
backend in `e2e`, and against a real server that generates with its model:

```sh
cd webui && pnpm build && cd ..                        # the server serves webui/dist
$PYTHON tools/smoke_real.py --browser [--features] ...  # starts its own server, then runs them
$PYTHON tools/smoke_browser.py --base-url http://127.0.0.1:7860   # or against a WebUI already running
```

Both run `pnpm exec playwright test` with `REMI_E2E_BASE_URL` set, which
makes `webui/playwright.config.ts` run only the `@real` tests, one at a
time, against that server (no Vite, 5-minute timeout per test for a first
model load) and report one row per test (the JSON report and failure traces
go to `--out`). The tests change no server setting: client options and
user assets are overridden for the page only (`overrideBootstrap` in
`tests/e2e/helpers.ts` rewrites `GET /api/v2/app`); they generate a few
512x512 images into the server's outputs. The gamepad reading and the sound
are also unit-tested (`features/lightbox/gamepad.test.ts`,
`features/progress/effects.test.ts`).

### Last recorded run

`--report docs/research/smoke/<date>.json` keeps every run's checks with the
date and revision; the latest is the record. `docs/research/smoke/2026-10-20.json`
(revision `ef242b67` with uncommitted changes, `--features`, MPS, the server
interpreter without Gradio): 70 of 70 checks passed, among them txt2img
parity through `/sdapi/v1` and `/api/v2/jobs` (mean 0, max 0, infotext equal),
img2img, inpaint (unmasked mean difference 0.000), Extras Lanczos 1.5x, LoRA
Layer Weight at 0 (mean 0, max 0), Stage Scheduler, FreeFuse, the Sparse
Attention skip warning, the Refiner with a second checkpoint and every paste
round-trip. The LoKR-tab checks and `--models` were added after that run:
the next run with the model (`$PYTHON tools/smoke_real.py --features
--models --report docs/research/smoke/<date>.json`) records them, and its
summary replaces this paragraph.

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
