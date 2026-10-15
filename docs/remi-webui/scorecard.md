# Remi WebUI rewrite — scorecard

The rewrite is accepted when an independent reviewer scores **100 on every
dimension** below. A score of 100 means "every criterion in the row holds, with
evidence the reviewer reproduced", not "good enough". Each criterion that fails
costs points proportional to its user impact. The reviewer records the evidence
(commands and outputs, screenshots, file:line) for every deduction and every
criterion checked.

## Dimensions

### 1. Gradio removal and architecture
- No `gradio` / `gradio_client` / `gradio_rangeslider` import anywhere in the
  repository, including built-in extensions and tests. `requirements*.txt` and
  `modules/launch_utils.py` do not install them. The WebUI starts and generates
  in an environment where Gradio is not installed.
- The layers in `architecture.md` hold: `remi.ui` declares parameters,
  `remi.server` serves REST/WS, the SPA renders schemas; no server-rendered HTML
  UI and no DOM coupling between server and client.
- No dead legacy code: old `modules/ui*.py`, `javascript/`, `script.js`,
  `style.css`, the Gradio templates in `html/`, `forge_canvas`, the
  `remi/ui/legacy*` bridge, and Gradio-only callbacks are gone or have a
  documented non-UI reason to stay.
- Contract documents match the code.

### 2. Built-in extension adaptation
- Every directory in `extensions-builtin/` and every file in `scripts/` and
  `modules/processing_scripts/` loads, exposes its full functionality in the new
  UI (params, pages, actions, widgets), and keeps its processing behaviour.
- For each extension with UI: argument order and defaults match the legacy
  snapshot, or a documented test covers the intentional change. Infotext keys,
  paste, and presets still work.
- JS-only extensions' features are reimplemented in the SPA or documented as
  merged into core features.
- Extension tests run and pass (none silently collect zero tests when tests
  exist).

### 3. Generation correctness
- txt2img with the baseline parameters is pixel-identical to the pre-rewrite
  baseline (`/tmp/remi-parity`), and its infotext matches.
- img2img, inpaint (mask respected), sketch, extras, hires fix, and refiner
  produce correct results. Every Anima feature produces the expected effect and
  infotext.
- Interrupt, skip, queue order, repeat, and forever behave correctly. Infotext
  paste round-trips restore all values.
- `/sdapi/v1` request/response compatibility holds, including `alwayson_scripts`.

### 4. Feature completeness (SPA)
- Every item of `inventory/core-frontend-features.md` §10 is implemented, or
  explicitly replaced by a better equivalent that is documented and justified.
- Pages: Generate (txt2img/img2img with canvas), Extras, PNG Info, Gallery,
  Networks, Queue, Settings, Extensions, and extension pages all work against
  the real server.

### 5. Performance
Measured and recorded in `docs/research/ui-performance.md` with the method and
raw numbers:
- Initial JS ≤ 300 KB gzip; no route chunk > 150 KB gzip except the lazy
  canvas/editor chunks, which are justified.
- Generate view: ≤ 3,000 DOM nodes with all built-in extensions loaded; INP
  p95 ≤ 50 ms while typing a 300-token prompt.
- 5,000 network cards and 10,000 gallery items scroll at 60 fps.
- Preview frame latency p95 ≤ 150 ms. UI overhead on generation speed ≤ 1 %.
- Server startup no slower than before the rewrite.

### 6. Visual design and UX
Assessed in a real browser (light and dark, desktop and narrow widths):
- Consistent spacing, typography, color, and component styling on every page.
- No overflow, clipping, or wrapped tab strips. The image area has a sensible
  size at 1280×720 and 1920×1080.
- Clear hierarchy, discoverable features, empty and loading states, and toast
  and error states.
- Keyboard shortcuts documented in-app. No console errors or warnings.

### 7. Accessibility and internationalisation
- axe reports no serious or critical violations on any page.
- Full keyboard operation, visible focus, labelled controls, and correct roles.
- zh-CN covers 100 % of core and built-in-extension strings, verified by a
  test. Switching languages needs no reload. No mixed-language labels in zh-CN
  except technical names.

### 8. Tests and quality gates
- Python and frontend suites pass. New code has meaningful unit tests.
- Playwright end-to-end tests cover the main flows against mock mode, and a
  documented real-server smoke test exists.
- Lint and type checks are clean (`pnpm build`, `oxlint`, `ruff` on new Python
  code).
- One documented command runs every check (for example `tools/check.sh`).

### 9. Security
- Auth covers REST, WS, files, and pages. WebSocket checks the Origin.
- Files are served only by opaque id. Uploads are size- and type-limited.
- No path traversal. No XSS through infotext, markdown, metadata, or extension
  HTML.
- No external network calls without explicit user configuration. Git and
  extension-install inputs are validated.

### 10. Documentation and upstream absorption
- The README (en and zh-CN) describes the new UI, installation (including the
  prebuilt-SPA or Node requirement), and launch flags.
- `docs/remi-webui/` covers the architecture, schema format, frontend
  conventions, an extension developer guide (porting from A1111 `ui()` to
  `params()`), and the API.
- An upstream-absorption guide maps upstream Forge Neo UI and runtime files to
  their Remi equivalents and gives a repeatable procedure for applying Neo
  updates.
