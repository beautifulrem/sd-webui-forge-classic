# Remi WebUI — browser integration checklist

The full walk-through of the real server in a real browser (scorecard
dimensions 2, 3, 4, 6 and 7), with the result and the evidence for every
item. Re-run it after UI changes; the automated part is one command.

**Setup (2026-10-16).** `launch.py --skip-prepare-environment --skip-install
--api --port 7860 --data-dir /tmp/remi-int/data --model-ref models` from the
Gradio-free environment (`/tmp/remi-nogradio-env`), an isolated data folder
with a copy of `config.json`, every built-in extension, the real model
(`anima-turbo-v1.1` + `qwen_image_vae` + `qwen_3_06b_base`, MPS on a MacBook
Air M4). The production build of the SPA (`pnpm build`).

- **Interactive:** Chrome (Claude in Chrome), window 1280×720 (viewport
  1148×698), light and dark, English and zh-CN.
- **Automated:** `webui/tools/audit/real-server.mjs` (Playwright + axe-core)
  opens every page — the 9 built-in ones, both extension pages and a missing
  page — at **1280×720, 1920×1080 and 420×860**, light and dark, en and zh-CN,
  with every collapsed section of the main area opened (`--expand`), and
  records console errors and warnings, failed requests, axe serious /
  critical violations, page overflow, wrapped tab strips, clipped text, stray
  scrollbars, layout boxes that were scrolled, and Latin-only text in zh-CN.
  `webui/tools/audit/tall-shot.mjs` takes full-height screenshots (inner
  scroll panes included), `webui/tools/audit/axe-one.mjs` prints the
  details of one page's violations and `webui/tools/audit/find-text.mjs`
  finds where a text (e.g. an untranslated word) is rendered.

```sh
cd webui
node tools/audit/real-server.mjs --base http://127.0.0.1:7860 --out /tmp/audit --expand 1 --axe all
node tools/audit/tall-shot.mjs txt2img /tmp/t2i.png light zh-CN 1280 9000
```

Result legend: **pass**, **fixed** (a defect found here, fixed and
re-verified), **n/a** (not testable here; the reason is given).

## Final automated sweep

`real-server.mjs --expand 1 --axe all` after all fixes: **144 views
(12 pages × 3 sizes × 2 themes × 2 locales), 0 with findings** — no console
error or warning, no failed request, no axe serious / critical violation, no
page overflow, no wrapped tab strip, no clipped text, no stray scrollbar.
(The command above writes `report.json` and a screenshot of every view to
`--out`.)
The zh-CN Latin-only texts left are technical names: model / file names,
`LoRA`, `VAE`, `ControlNet`, `ADetailer`, sampler and scheduler names
(`Euler`, `ER SDE`, `Beta`), architecture names (`FLUX`, `QWEN`, …), `API`,
`Wiki`, `png` / `jpg`, model and product names (`Anima`, `Base / Aesthetic`,
`Turbo`, `Flux.1-Kontext`, `Wan 2.2 I2V`, …), booru site names and their own
labels (`Account → Options`, `api_key`, `user_id`) and the JoyCaption model
id.

The first sweep (before the fixes) found 26 of 144 views with findings
(defects D1–D6 below); a later full sweep with every section open found 12
(D24), and its zh-CN review D25; both fixed and swept again: 0.

## Generation correctness (dimension 3)

| Item | Result | Evidence |
|---|---|---|
| txt2img with the baseline parameters, through the UI (infotext pasted into the prompt, "Paste parameters", Generate) | pass | output `00000-12345.png` is pixel-identical to `/tmp/remi-parity/baseline_1.png` (numpy equality on 768×768×3); infotext equal except `Version` |
| Live preview, progress, ETA in the title, "Starting…" before the first step | pass / fixed (D14) | screenshots during the job: preview frames, `[38%] ETA 1:49 · Generate` |
| Interrupt (Esc) keeps the partial result | pass | job ended interrupted, partial image in History |
| Generate forever, then stop from the menu | pass | two more images, the toggle unchecked, queue `size 0, running null` after the current job |
| Queue another while running (8 queued jobs), queue badge | pass / fixed (D7) | `8 of yours queued`; badge only on Queue |
| Restore after a reload during a job | pass | after `location.reload()`: step 2/8, preview and Interrupt back |
| Hires. fix 512 → 768 (Latent, 4 steps) | pass | 768×768 output; infotext `Hires upscale: 1.5, Hires steps: 4, Hires upscaler: Latent` |
| img2img inpaint (mask painted on the stage, "Only masked") | pass | rows outside the mask: mean difference 0.0 (bottom) / 2.1 (rows next to the mask); masked band 6.1; infotext `Inpaint area: Only masked, Masked area padding: 32` |
| Sketch (blank canvas, red strokes, denoise 0.5) | pass | result keeps the strokes redrawn (screenshot); job `mode: Sketch` |
| Extras: Send to Extras, Lanczos ×2 | pass | 1024×1024 result with the compare slider; infotext `Postprocess upscale by: 2.0, Postprocess upscaler: Lanczos` |
| X/Y/Z plot (Seed: 12345, 777) | pass | 1024×512 grid with `Seed: 12345` / `Seed: 777` labels |
| Send to img2img / inpaint / extras from the stage menu | pass | "Sent to Extras" toast; the image and its parameters arrive in the target form (steps, sampler, seed) |
| `/sdapi/v1` and `alwayson_scripts` | n/a here | covered by `tools/smoke_real.py` and the API tests (not repeated in the browser) |

## Built-in extensions in the browser (dimension 2)

Each panel was opened, operated and (where it applies) run in a 512×512, 8-step
job from the UI; "effect" compares the output with the plain run
(`00003-12345.png`, mean absolute pixel difference) and checks the infotext.

| Extension | Result | Evidence |
|---|---|---|
| Dynamic Shift | pass | infotext `DynShift enabled: True, DynShift start: 5.0, …`; image changed |
| Detail Daemon (daemon I active, amount 0.5) | pass | infotext `Detail Daemon: "D1:1,0,both,0.5,…"`, mean diff 23.6 |
| Anima DAVE Diversity | pass | infotext `Anima DAVE enabled: True …`, mean diff 52.4 |
| Anima Cross-Attention Boost | pass | infotext `Anima xattn boost …`, mean diff 57.9 |
| Anima Guidance & Corrections (all sub-features off) | pass | infotext `Anima guidance enabled: True …`, image unchanged as expected |
| Spectrum Integrated (warm-up 6 of 8 steps) | pass | infotext `spec_enabled: True, …`; no skipped step at 8 steps, image unchanged as expected |
| Sparse Attention Integrated | pending re-check (reference implementation on MPS) | the fused Sol-Attn kernels need NVIDIA sm_80+ or ROCm with matrix cores; on this MPS machine the run now uses Comfy Kitchen's reference implementation of the same algorithm (job warning "Sparse Attention Integrated used the reference implementation (CUDA device is required): slower than regular attention"), so the image changes and the infotext records `Sparse attention tau/range/min tokens/extra tokens` - to be confirmed with `tools/smoke_real.py --features` (it checks the change, the infotext and the paste round-trip). On the CPU: `extensions-builtin/sd_forge_sparse/tests/test_sparse_attention.py` (the reference equals Comfy Kitchen's and differs from dense attention) |
| Post Processing Suite (Sepia 1.0) | pass | infotext `PP Suite: "{"sepia":…}"`, mean diff 19.3; "Apply PP → Send to img2img" shows the selected image |
| Hires. fix + Anima Hires Guard | pass | infotext `Anima Hires guard enabled: True, Anima Hires policy: Clamp unsafe values, …` |
| ADetailer | fixed (D13) / model n/a | runs; every detector in `models/adetailer` is truncated (`.pt` zips without a central directory, `.tflite` "not a valid Flatbuffer") — Forge renamed two of them to `*.corrupted` while loading, the names were restored afterwards. The failure is now reported to the user (toast + job warnings). The effect (detection, inpainting the box, infotext, paste) runs end to end with a seeded YOLOv8n detector in `tests/test_remi_extensions_e2e.py` |
| X/Y/Z plot, Prompt matrix, Prompts from file | pass (X/Y/Z run) | grid above; the others render and validate |
| Anima resolution helper (fit to source, presets, aspect) | pass / fixed (D11) | "Size set to 1536 x 1536 (source 512 x 512)." in English and zh-CN |
| Anchor prompt, Prompt Blacklist, Prompt rescaler, Remi presets, Artist mixer, LoRA layer weight / stage scheduler, FreeFuse, Regional, ControlNet, IP-Adapter / preprocessors, ImageStitch, Never OOM, Refiner, Soft inpainting, Lock img2img steps, MultiDiffusion, negpip, compile, PID, Radial | pass (render, operate, axe) / n/a (effect) | every panel open in the expanded sweep without findings; their effects need LoRA / ControlNet / IP-Adapter / refiner weights that are not in `models/` — covered by the extensions' own tests (`tools/check.sh`). Since then every one with a processing effect also runs end to end through `/api/v2/jobs` on the tiny Anima model with seeded test LoRAs, LoKr, Refiner checkpoint and Control-LLLite (effect, infotext keys, hook arguments, paste round trip: `tests/test_remi_extensions_e2e.py`, testing.md "Every extension end to end on a tiny model"); IP-Adapter, compile, PID and Radial cannot run on Anima / without CUDA, and the test's `NOT_RUN` says why |
| Prompt Workshop page (Build Prompt, autocomplete, booru grabber UI) | pass / fixed (D10, D12) | built prompt `masterpiece, best quality, yuru yuri` |
| JoyCaption page | pass (UI) / n/a (captioning) | the ~17 GB model is not downloaded; loading it was not triggered |
| Extensions page (installed / built-in list, filters) | pass / fixed (D8) | 45 built-in; "Check for updates" and the index were not used (network) |

## SPA features (dimension 4, `inventory/core-frontend-features.md` §10)

| §10 item | Result | Evidence |
|---|---|---|
| A. Tabs, theme, reactive options, reload UI, footer / server page | pass | nav rail / bottom bar; Settings → Server and models (Reload UI, unload / reload model, versions) |
| A. Extension JS hooks / compatibility globals | replaced | documented in `architecture.md` (no Gradio JS runtime; extensions declare params / widgets) |
| B. Attention Ctrl/⌘+↑/↓, tag move Alt+←/→, undo/redo | pass | `(silver:1.1)` then ⌘Z restores; `masterpiece` moved after `best quality` |
| B. Token counters | pass | `27/512` after the model loaded (`23/75` before) |
| B. Paste parameters, clear, placeholders | pass | "Parameters applied" toast |
| B. Styles (select, apply, editor dialog) | pass (render) | the select and editor button render without findings in the sweep; the editor dialog itself is covered by the e2e suite, not operated here |
| C. Generate / Interrupt / Skip / forever / queue another, Ctrl+Enter, Esc | pass | see generation table |
| C. Progress in the title, wake lock, restore | pass | title `[25%] ETA 32s · Generate · Remi` |
| C. Hires preview "from 512×512 to 768×768" | pass | Hires. fix panel |
| C. Hires target greying (`onCalcResolutionHires`) | pass (mock e2e) | `inactive_if` on the three target sliders (`remi/core/forms.py`): "Upscale by" dims while a resize size is set, a resize size dims while it is 0; dimmed controls stay editable. `extras.spec.ts` "Hires. fix dims the target controls that do not apply…" (states, opacity 0.5, job values and `Hires resize: 1536x1536`), `Renderer.test.tsx`, `test_remi_ui_input_flags.py` |
| C. Live preview without flicker (double buffer; crossfade replaced) | pass (vitest) | `pages/generate/LivePreview.tsx`: a decoded frame is stacked over the previous one, without a fade (a 150 ms fade drew ~12 frames per step, frontend.md "Motion", ui-performance.md); `LivePreview.test.tsx`, e2e `generate.spec.ts` live preview tests |
| C. Width/height swap, aspect presets, W x H paste | pass (render) | Size card renders and takes typed sizes; swap / paste covered by vitest, not operated here |
| D. Lightbox (←/→, Home/End, zoom, S, T, Esc) | pass / fixed (D9) | ←/→ ("1 of 23 → 2 of 23"), Esc and the toolbar operated; zoom keys covered by vitest |
| D. Send to img2img / inpaint / extras / txt2img | pass | above |
| D. Notifications (permission from the Generate menu) | pass (present) | menu item "Notify me when images are ready"; the permission prompt was not triggered |
| E. Networks page and drawer (Alt+N), search, empty state, checkpoint card | pass / fixed (D5, D9) | "No LoRA found — Put files into the model folder, then refresh."; checkpoint card selected |
| F. Settings search, filters, apply (⌘S), Reload UI banner, sysinfo, licenses | pass / fixed (D3, D4) | "Settings applied"; sysinfo loads in ~1.2 s |
| F. Help for the VAE / text encoder modules (the Gradio "?" link) | pass (vitest) | the model bar's modules popover says which checkpoints need them and links to the download guide; `ModelBar.test.tsx` |
| G. Canvas: upload, blank canvas, brush / eraser / fill / lasso, undo, zoom, mask overlay, inpaint area overlay | pass / fixed (D15) | "512×512 · Crop and Resize", "Inpaint area 431×431" |
| H. Localization: zh-CN switch without a reload | pass / fixed (D6, D11, D25) | a `window` marker survived the switch from the appearance menu, `<html lang="zh-CN">`; toasts in zh-CN |
| H. Right-to-left localizations | pass (mock e2e) | `<html dir>` from the locale (`i18n/index.ts` `textDirection`), Base UI `DirectionProvider`; `flows.spec.ts` "lays the app out right to left…" (`ar`: `dir="rtl"`, the rail on the right, back to `ltr` in English), `i18n.test.tsx` |
| A. `user.css` from the data folder | pass (mock e2e) | `app/user-stylesheet.ts` links it at the end of `<body>`; `flows.spec.ts` "applies user.css…" (a routed `user.css` sets the body outline and beats the app's body background, also after a lazily loaded page), `user-stylesheet.test.ts` |
| I. Resizable panes, collapsible sections, sticky run controls | pass | sections opened / closed throughout, run controls stay on top of the scrolled pane; pane resizing not dragged here (e2e) |
| J. Extensions tab | pass | above |
| J. Install from URL / from the index, Check for updates with progress, backup restore with confirmation | pass (mock e2e) | `extensions.spec.ts` against the mock admin API (`api/mock/admin.ts`): URL validation, install, the 409 for an existing folder, the new extension listed; index loaded and an entry installed; the progress bar ("Checked n of m", Stop) then the summary toast and Update buttons; restore cancelled (nothing happens), then confirmed (reconnect overlay, page reload). Not run against GitHub (network, changes the install) |
| Gallery (filters, search, viewer with details, PNG Info) | pass | 24 images, details panel, PNG Info page with all metadata |
| Queue (running card, up next, history with filters) | pass / fixed (D13, D25) | running card with progress, "No jobs are waiting", history rows with warnings and localized durations |
| Keyboard shortcuts documented in-app | fixed (D16) | `?` dialog |

## Visual design, UX, accessibility (dimensions 6 and 7)

| Item | Result | Evidence |
|---|---|---|
| No console errors or warnings on any page | pass | sweep: 0 |
| axe: no serious / critical violation, light and dark, every page, sections open | fixed (D1–D3) | sweep: 0 |
| No overflow, clipping, wrapped tab strips | fixed (D4, D5, D10, D15) | sweep: 0 |
| Image area at 1280×720 / 1920×1080 | pass | the stage (between the parameters and the history panes) measures 563×612 px at 1280×720, 870×972 at 1920×1080 and 396×508 at 420×860 (stacked) |
| Narrow (420 px): bottom bar, stacked layout | pass | screenshots of every page at 420×860, light / dark, en / zh-CN |
| Empty / loading / error states | pass / fixed (D13, D14) | gallery "No images yet", queue "No jobs are waiting", networks empty, skeletons, job warnings |
| zh-CN: no mixed-language labels except technical names | fixed (D6) | Latin-only list above |
| Keyboard operation, visible focus | pass | ⌘K palette, `?` dialog, focus rings in screenshots; e2e `flows.spec.ts` |

## Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| D1 | img2img: radio options of the resize mode had no accessible name (axe `aria-toggle-field-name`): their ids contained the option value with spaces, which breaks `aria-labelledby` | index-based option ids in `Choice.tsx`, `MultiChoice.tsx` and the model bar's module list |
| D2 | Contrast (axe): the ⌘Enter key cap on the primary button, setting keys (`samples_save`) in dark, the settings nav headings, destructive buttons, `text-success` / `text-warning` in light | key cap outlined; `text-muted-foreground` without opacity; light `--destructive`, `--success`, `--warning` darkened to ≥ 4.5:1 |
| D3 | Links in running text only distinguished by colour (axe `link-in-text-block`, JoyCaption page, setting comments) | always underlined (`rich-text.tsx`, `SettingRow.tsx`) |
| D4 | Settings filter ("Changed / Unsaved / Pinned") truncated to "已…" in zh-CN; segmented controls wrapped | `Segmented` is an equal-column grid on one row; the settings filter never shrinks |
| D5 | Wrapped tab strip on Extras (4 tabs in a 450 px pane); Networks / Extensions / canvas option bars cut off without a hint | schema tab strips scroll sideways instead of wrapping; a shared `useScrollFade` fades the cut edge and keeps the selected tab in view (`lib/use-scroll-fade.ts`) |
| D6 | zh-CN: untranslated core strings (`Callbacks`, `Checkpoint Merger`, `Quicksettings`, `Sysinfo`, ADetailer labels, `Automatic` in the model bar, setting comments such as "(e.g. for X/Y/Z Plot)", …) | catalogs updated (core, language pack, ADetailer, ControlNet, Spectrum, soft inpainting); the model bar translates `Automatic`; setting HTML comments are looked up in the catalog (`translateHtml`) |
| D7 | The queue count badge showed on every nav item | `useQueueStatus(false)` no longer returns another observer's cached data |
| D8 | A stray vertical scrollbar on the Extensions tab strip | scrollbar hidden; the audit now detects stray scrollbars |
| D9 | Alt+N opened the networks drawer beneath the image viewer, which then could not be closed with Esc | Alt+N is ignored while another dialog is open |
| D10 | Prompt Workshop: the two tag columns did not line up (rows align controls to the bottom, columns too) | containers in a row align to the top (`remi-cell-top`) |
| D11 | Untranslatable f-string toasts (resolution helper, Spectrum presets); WS `toast` events were never translated | `ui.message` keys with zh-CN entries; the client translated WS toasts like action toasts (the WS `toast` event was removed later: nothing on the server sent one, toasts come in action patches) |
| D12 | An autocomplete list answered after the focus left stayed open over the next field | closed when the editor is not focused |
| D13 | A script hook that raised (e.g. ADetailer with a broken model, or without `ultralytics`) failed silently: the job "completed" with no trace | the queue collects `modules.errors.report` during a job as **warnings** (job API `warnings`, `job.completed.warnings`); the submitting client shows a toast, History shows them |
| D14 | "Step 0 of 0" before the first step; the previous job's last preview (and progress) shown for a new job until its first step | "Starting…"; the preview encoder and progress monitor wait for the job's `begin()` (`JobRef.begun`) |
| D15 | Focusing or scrolling to a control inside a pane could scroll `<main>` (overflow hidden) and push the whole layout up: absolutely positioned screen-reader text escaped the scroll pane | `main` is `overflow-clip`; the scroll panes are positioned; the audit flags scrolled `overflow:hidden` boxes |
| D16 | No in-app list of the keyboard shortcuts (scorecard 6) | Keyboard shortcuts dialog: `?`, the command palette and the appearance menu |
| D17 | Field and toggle labels truncated with an ellipsis in two-column rows ("高分辨率修复…", "Temperatu…") | labels wrap to two lines (hyphenated), full text in the tooltip |
| D18 | Mixed button styles in Post Processing Suite (Delete as a bare ghost button), Artist mixer delete / reset not marked destructive | `danger` variant |
| D19 | "1 unsaved changes", "1 items", "Imported 1 favorites", "Used: 1 times" | count-neutral wording ("Unsaved changes: 1", …) |
| D20 | Native blue search-clear button next to the app's own clear button (Settings) | the native one follows the theme, and is hidden where the app has its own |
| D21 | Gallery search placeholder cut off at 1280 px | wider minimum |
| D22 | The launch log hid `--port`: `--api` was taken for an abbreviation of `--api-auth` | exact flags of the launch parser are never secrets (`modules/redact.py`, test) |
| D23 | The networks page's form / prompt switches were truncated in the header ("txt2…", "Pro…") below 2xl | they get their own row under the header below 2xl |
| D24 | Dark theme: destructive buttons at 4.43:1 (axe), found by the second full sweep | dark `--destructive` lightened |
| D25 | zh-CN: ControlNet's "None" preprocessor / model, queue durations ("3 min 14 s"), Extras jobs titled "extras", ImageStitch help naming the tabs `txt2img` / `img2img` | translated (`None`, localized duration units, Extras jobs titled by their form, 文生图 / 图生图) |

## Open issues carried over (all verified)

| # | Issue | Status |
|---|---|---|
| 1 | Artist mixer: a generation made its panel the schema default, so API / imported jobs inherited the last run's artists | fixed: the defaults are the built-in ones; nothing is written by a generation (the browser draft remembers the user's panel); `tests/test_panel_defaults.py` |
| 2 | Server toasts English-only | fixed: toasts use catalog keys / `ui.message` (D11); zh-CN verified in the browser ("尺寸已设为 1536 x 1536（源图 512 x 512）。") |
| 3 | Language pack zh_CN partial English ("Preset") | fixed: "Preset" → 预设 and the D6 entries; the core catalog wins (`test_remi_core_zh_cn.py`) |
| 4 | `test_internal_routes` after runtime tests | verified fixed: `pytest tests/test_remi_server_runtime.py tests/test_remi_server_jobs.py tests/test_remi_server_app.py` → 68 passed |
| 5 | Canvas zh strings (Denoising Strength, Batch Count, Scale, CFG Scale, Seed) | verified: 重绘幅度, 生成批次, 缩放, 提示词相关性 CFG, 种子 in `/api/v2/i18n/zh_CN` |
| 6 | Prompt rescale's manual tool as script args | fixed: `remi.ui` tool inputs (`arg=False`: rendered, handed to actions, never a processing argument — `ScriptLayout.tools`); the rescaler's args are the legacy three again |
| 7 | vitest flakiness | verified: `maxWorkers: "50%"`, 20 s test / 30 s hook timeouts; `tools/check.sh` green |

## Not covered here

- Effects of extensions that need weights absent from `models/` (LoRA,
  ControlNet, IP-Adapter, refiner checkpoint, JoyCaption, working ADetailer
  detectors): their processing is covered by their own test suites.
- Extension install / update and the extension index against the real
  network (the flows themselves run offline in `extensions.spec.ts`), real
  browser notifications and the wake lock (need user permission prompts).
- Multi-user / `--auth` flows (covered by `test_remi_server_auth.py` and
  `test_remi_security.py`).

## Quality gates after the fixes

`tools/check.sh` (server stopped): **all 49 steps passed** — `pytest tests`
1159 passed, every `extensions-builtin/*/tests` directory passed, ruff,
OpenAPI / settings contracts, both i18n checks, `pnpm build`, oxlint, Vitest
867 passed (122 files), Playwright e2e 70 passed.
