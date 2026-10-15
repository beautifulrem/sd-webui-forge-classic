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
| `ruff:remi` | ruff with the strict rule set on Remi's own Python: `remi/`, `tools/`, `webui/tools/`, `tests/remi_*.py`, `tests/test_remi_*.py` |
| `ruff:tests` | ruff on `tests/` and every `extensions-builtin/*/tests` (strict rules on the Remi files, Forge's legacy rules on the rest) |
| `ruff:syntax` | the whole tree for syntax errors (`E9`) |
| `pytest:tests` | `pytest tests` |
| `pytest:<extension>` | `pytest extensions-builtin/<extension>/tests`, one run per directory (they share module names); a directory that collects nothing fails |
| `contract:openapi` | `tools/export_openapi.py --check`: `webui/src/api/openapi.json` matches the server |
| `contract:settings` | `tools/dump_settings.py --check`: the SPA's settings contract matches `GET /api/v2/options` |
| `i18n:server` | `tests/test_remi_core_zh_cn.py`: the zh_CN catalog covers every core string, glossary, language pack and extension catalogs agree |
| `pnpm:install` | `pnpm install --frozen-lockfile` in `webui/` |
| `i18n:spa` | `src/locales/locales.test.ts`: every chrome key has a zh-CN translation with the same placeholders |
| `pnpm:build` | `tsc -b` (app, Vite config, e2e specs) and the production build |
| `pnpm:lint` | oxlint |
| `pnpm:vitest` | the Vitest suites (jsdom) |
| `e2e` | Playwright against the mock backend, including the axe scans |

**Python.** `$PYTHON`, else `venv/` or `.venv/` in the repo, else
`/tmp/remi-testenv`, else `python3` — the first one that imports `fastapi`,
`pytest` and `torch` (requirements.txt plus pytest). Ruff comes from that
environment, then `PATH`, then `uvx ruff`.

**Ruff configuration** (`pyproject.toml`). Upstream-origin code keeps Forge's
legacy rules (`N805`, `S307`, `S102`, `T`, `W`, `F`) and its existing style;
Remi's own files also get `E`, `I`, `B`, `UP`, `SIM`, `C4`, `RUF`, `PIE`,
`PLE` (a negated `per-file-ignores` glob switches them off elsewhere). Long
lines are the house style (black at 1024).

## End-to-end tests (`webui/tests/e2e`)

`pnpm test:e2e` (or `pnpm exec playwright test`) starts Vite in mock mode on
`$REMI_E2E_PORT` (default 5199; `tools/check.sh` uses 5299) unless a server
already answers there, then runs:

- `generate.spec.ts`: generate with live preview → result and history,
  Escape interrupts, prompt editor keys and bracket warning, txt2img ↔ img2img,
  the image viewer, send to img2img, the phone layout.
- `canvas.spec.ts`: inpaint (paint a mask, undo / redo, the job carries the
  mask), sketch, zoom / fit / full screen, uploaded masks per mode, drop and
  paste, image size, phone scrolling.
- `library.spec.ts`: gallery search and the viewer with the parameters,
  compare, PNG Info → txt2img, queue (pause, add twice, keyboard reorder,
  resume).
- `flows.spec.ts`: settings change applied with Mod+S and discard, the
  networks drawer (Alt+N, search, insert a LoRA), queue cancel, infotext read
  from the prompt, PNG Info from a pasted image, an extension page and its
  action, the Extensions page, zh-CN without a reload, light / dark (kept
  across reloads), command palette (navigate, theme), skip link, focus rings.
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
/tmp/remi-testenv/bin/python tools/smoke_real.py   # --port 7879 --out /tmp/remi-smoke --baseline /tmp/remi-parity
```

Starts `launch.py --api` with the real model in an isolated data folder (a
copy of `config.json`, models from `models/`), then checks txt2img parity
against the pre-rewrite baseline through `/sdapi/v1/txt2img` and
`/api/v2/jobs` (pixel-identical image, infotext equal except `Version`),
img2img at denoising 0.5, inpaint (the unmasked half unchanged, the masked
half changed) and an Extras Lanczos upscale, stops the server and prints a
table (exit 1 on a failed check, 2 when the server did not start). It
refuses to run while another `launch.py` holds the model. Not part of
`tools/check.sh` or CI (it needs the model and a GPU / MPS).

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
(CPU torch; ruff, pytest, contracts, i18n) and the SPA (install, build, lint,
Vitest, Playwright with Chromium). Logs and the Playwright report are
uploaded when a job fails.
