# Remi WebUI architecture (Gradio-free rewrite)

Status: accepted, 2026-10-15. Supersedes the "keep Gradio as /legacy" parts of
[the replatform plan](../research/webui-replatform-plan.zh-CN.md): Remi is not
an incremental variant of Forge Neo. Neo updates are reviewed and applied by
hand, so upstream UI files carry no merge-preservation weight. The Gradio UI is
deleted, not frozen.

## Goals

- No `gradio` import anywhere at runtime or install time.
- One declarative parameter model (`remi.ui`) drives the SPA, the HTTP API,
  infotext paste, presets, conflict locks and saved queue items.
- Processing code is untouched: `scripts.Script` keeps every processing hook,
  and a script still receives its arguments positionally, in declaration order.
- Every built-in extension is ported to the new API.
- React SPA (`webui/`) with a design system, lazy routes, a binary WebSocket
  preview, and no DOM coupling to the server.

## Layers

```
webui/ (React 19 + Vite 8 + Tailwind 4 + shadcn/Base UI, built to webui/dist)
   │  REST /api/v2/*  ·  WebSocket /api/v2/ws  ·  static /assets, /files/*
remi/server/   FastAPI app factory, routes, auth, static SPA, WS hub
remi/jobs/     persistent queue (SQLite) + single worker on modules_forge.main_thread
remi/ui/       parameter/layout DSL, schema export, arg layout, settings schema, actions
remi/core/     generation forms (txt2img/img2img/extras), tab schema, infotext parse, conflicts, presets
remi/pages/    extension pages (former on_ui_tabs)
modules/       processing pipeline, scripts runtime, options (Gradio removed)
backend/       inference (unchanged)
```

`webui.py` builds the app with `remi.server.create_app()` and runs uvicorn. The
A1111 `/sdapi/v1` API (`modules/api/api.py`) stays mounted, now resolving
`alwayson_scripts` through `remi.ui` layouts.

## `remi.ui` — parameters and layout

Inputs (each has a `name` unique within its script; that name is the param id):

| Class | Value | Notes |
|---|---|---|
| `Toggle(name, default=False)` | bool | |
| `Slider(name, minimum, maximum, default, step)` | float/int | `int` when the bounds, step and a static default are all Python `int`s (`Slider(0.0, 2.0, 1.0, 1.0)` is a float slider); `integer=` overrides |
| `Number(name, default, minimum=None, maximum=None, step=None, precision=None)` | float/int | |
| `Text(name, default="", lines=1, max_lines=None, placeholder=None, prompt=False)` | str | `prompt=True` → CodeMirror prompt editor (weights, LoRA, tag completion) |
| `Choice(name, choices, default, style="dropdown" or "radio", allow_custom=False)` | str/int | `choices` list (values or `(label, value)`) or a callable resolved per request; `type="index"` gives the script the index (the wire value stays the choice value); a `None` default is the first choice |
| `MultiChoice(name, choices, default=())` | list | dropdown multiselect or `style="checkboxes"`; `type="index"` gives the script indices |
| `Seed(name, default=-1)` | int | dice / recycle buttons; `-1` or `0 … 2^63-1` |
| `Color(name, default="#000000")` | str | |
| `Image(name, kind="image" or "mask" or "sketch", source=..., mode=None)` | upload id → PIL at the script | images are uploaded once (`POST /api/v2/uploads`); `source` names the image a mask/sketch layer is painted over (a mask without `source` is an uploaded mask image); masks are alpha-coded with a luminance fallback (`nodes.mask_channel`, schema-format.md §1.3); `mode` converts (e.g. `"RGBA"`) |
| `ImageList(name)` | list of upload ids → list of PIL | batch inputs |
| `File(name, accept=...)` | upload id → path | |
| `Json(name, default)` | any JSON | escape hatch for structured widgets |
| `Custom(name, widget, default, props={})` | JSON | a frontend widget registered in `webui/src/widgets` (stage timeline, block-weight grid, ControlNet unit, …) |
| `State(name, default)` | JSON | hidden; not rendered |

Common keyword arguments for inputs: `label`, `info`, `visible_if`, `enabled_if`,
`infotext` (infotext key), `paste` (callable `params -> value` for derived
paste), `preset` (Anima preset key), `exclusive=(group, member)` (Toggle only),
`on_change="action"`, `width` (`"auto"`, `"fill"`, or a 1–12 column span),
`hidden=True` (submitted, never rendered: e.g. the choice a `Tabs` is bound to).
Sliders clamp; `Number` rejects out-of-range values; NaN/Infinity are always
rejected, also inside `Json`/`Custom`/`State` values; toggles take booleans,
`0`/`1` and `"true"`/`"false"` only (see schema-format.md §3). A request body
that is not an object of named values is a `ValidationError` too.

Layout: `Section(title, children, enable=Toggle | None, open=bool, exclusive=...)`
(a section with `enable` is the former `InputAccordion`), `Row`, `Column`,
`Group(title=None, open=True)`, `Tabs(Tab(title, children, value=...)..., bind="param")`
(a bound strip selects the `Choice` named by `bind`), `Markdown(text)`,
`Html(text)`, `Separator()`, `Button(label, action, variant)`, `Gallery(name)`
(output-only), `Output(name, kind="text" or "html" or "image" or "json")`, and
`Slot(section)` (core forms only: where the scripts of a UI category render).
Containers take children positionally and/or as `children=[...]`; `None`
children are skipped.

Conditions (`visible_if`, `enabled_if`) are small JSON predicates evaluated on
the client: `{"param": "mode", "in": ["a", "b"]}`, `{"param": "enable"}`
(truthy), `eq`/`ne` (Python `==`: `True == 1`), `gt`/`gte`/`lt`/`lte` (finite
numbers only), `{"all": [...]}`, `{"any": [...]}`,
`{"not": {...}}`, and `true`/`false`; helpers `ui.param(...)`, `ui.all_of`,
`ui.any_of`, `ui.not_`. Params from
another script or the core form are addressed as `"script_id.param"` or
`"core.param"`.

Labels are English strings. The SPA translates them through the locale
catalogs (`localizations/*.json`, keyed by English text), so extensions get
zh-CN without code changes; Remi ships `localizations/zh_CN.json` complete for
every built-in string.

### Actions

`@ui.action(name)` on a script or page registers a server callable that
receives `(self, ctx, values: dict)` (the script's current values; `ctx.core`
is the generation form, `ctx.scripts` the other scripts' values) and returns a patch:
`{param: {"value"?, "choices"?, "visible"?, "enabled"?, "label"?, "info"?,
"minimum"?, "maximum"?, "step"?}}` (another owner's param as `"owner.param"`),
optionally with `{"__toast__": {"level", "message"}}`,
`{"__outputs__": {name: value}}` and `{"__settings__": {setting: value}}`
(settings the action already changed on the server; the client only
refreshes its copy).
`remi.ui.call_action(owner, name, ActionContext(...), values)` runs and checks
one. Buttons call actions, and `on_change`
triggers them. Actions run outside the generation lock unless declared
`gpu=True`, in which case they queue behind generation.

### Scripts

```python
class Script(scripts.Script):
    def title(self): ...
    def show(self, is_img2img): ...      # unchanged
    def params(self, is_img2img):        # replaces ui(); returns a layout node
        return ui.Section(self.title(), enable=ui.Toggle("enable", False), children=[...])
    def process(self, p, enable, blocks, ...):   # unchanged, positional in declaration order
```

The runner builds a `ScriptLayout` per script and tab from `params()`, giving
`args_from`/`args_to`, defaults, the infotext map and preset map, without
rendering anything. A script whose argument order differs from its visual
order sets `param_order = ("steps", "sampler_name", ...)` (the Sampler
script). Each script gets a stable owner id (`script_id`, the slug of its
title: `seed`, `sampler`, `refiner`, …; `core` and `settings` are reserved,
so a script titled "Core" is `core_2`). Selectable scripts are the core
`script` choice plus each script shown with `visible_if: {"param":
"core.script", "eq": title}`; slot 0 of `p.script_args` is still the index.
A script whose `params()` raises (or, for a legacy script, whose `ui()`
fails in the throwaway build) is disabled: `ScriptRunner.disable_script`
removes it from the runner (no slots, no schema entry, no hooks) and keeps
the reason in `runner.disabled_scripts`; Extras operations likewise.

`ScriptRunner` API: `build_layouts()` (in `initialize_scripts`),
`layout_for(script)`, `prepare_layouts()`, `ensure_arg_slots()`,
`script_args_from_values(selected, {script_id: values})`,
`script_values_from_args(args)`, `scripts_schema()`, `paste_values(params)`,
`preset_bindings()`, `preset_values(label)`, `disable_script(script, reason)`.

Until the Gradio UI is deleted: scripts not yet ported keep `ui()`; their
params are positional (`#0`, `#1`, …), typed from their Gradio controls by
class name (`remi/ui/legacy.py`: index-type dropdowns/radios/checkbox groups
become `type="index"` choices, `gr.Number` keeps its precision, `gr.State`
keeps its object; unknown controls are `Json`), built in a throwaway
`gr.Blocks` (with `modules.gradio_extensions` applied) when no UI did. A
param the request leaves out, or sends equal to its exported default, is
what Gradio passed for the untouched control (`control.preprocess(value)`,
a fresh copy for `gr.State`). A ported script
without `ui()` gets an approximate Gradio rendering (`remi/ui/legacy_gradio.py`)
so it keeps its slots there; when the Gradio UI is built it owns the slot
assignment and the named-value path follows it. The built-in processing
scripts keep both `ui()` and `params()` (tests assert they match) until then.
`ui()` and `on_before_component`/`on_after_component` are then removed.

Anima helpers become declarative keyword arguments: `preset=` (or
`preset=(field, value)` for a localized value), `infotext=`, and
`exclusive=(group, member)` (groups and their headless priority in
`anima_feature_conflicts.EXCLUSIVE_GROUPS`). `anima_support.table_params(rows,
root)` applies a `SCRIPT_ARGS` table (infotext keys incl. `PasteKey`, preset
fields) to a `params()` tree and returns the `param_order`;
`AnimaPatchPanel.panel_params(root)` does it for the patch panels (inputs named
after the preset fields' last part). The Gradio-path functions remain for
unported extensions.

### Pages (former `on_ui_tabs`)

`remi.pages.register(Page(id, title, icon, params=..., actions=..., widget=None))`.
A page is either schema-rendered or a dedicated frontend view
(`widget="agent-scheduler"`), which is implemented in `webui/src/pages/`. Its schema (`GET /api/v2/pages/{id}`) is schema-format.md §8.

### Settings

`OptionInfo` keeps its signature, but `component` refers to `remi.ui` widget
kinds (`ui.Slider`, `ui.Choice`, `ui.MultiChoice`, `ui.Color`, `ui.Text`,
`ui.State`, …) with Gradio-style `component_args` plus `style` (`"radio"`,
`"checkboxes"`) and `allow_custom`. Gradio classes of unported extensions are
mapped by name. The settings schema is exported from `shared.opts.data_labels`
by `remi.ui.settings.settings_schema` (`.info()`, `.link()`, `.html()`,
`.needs_restart()`, `.needs_reload_ui()` are kept structured). While the
Gradio settings page exists, `OptionInfo.component` returns the Gradio
equivalent (`OptionInfo.widget` is what was given).

## `remi.core` — generation forms

`remi/core/forms.py` declares the `core` owner of each tab (txt2img, img2img,
extras) with the processing parameter names (`prompt`, `cfg_scale`,
`enable_hr`, `hr_*`, img2img `mode`, `init_img`/`sketch`/`mask` canvas layers,
`mask_upload`, `resize_tab`, `batch_*`, extras `mode`/`image`/`image_batch`/…),
infotext keys and preset fields; the user's category order decides the layout
and `Slot`s place the scripts. The current UI preset (`forge_preset`) sets the
core defaults, labels, bounds and visibility (`presets.apply_form_patch` of
`forge_preset_patch`) and the Sampler script's defaults (callable defaults),
for the schema and validation alike. `remi/core/tabs.py` produces
`GET /schema/{tab}` and validates a request into processing args (coercing
once; `forms.check_core` adds the cross-param rules: the images each img2img
mode needs, the Extras source). Entry points (named values; the caller runs
them on the main thread, one job at a time):

- `modules.txt2img.run_txt2img(core, scripts, job)` / `prepare_txt2img`,
- `modules.img2img.run_img2img(core, scripts, job)` / `prepare_img2img`,
- `modules.postprocessing.run_extras_named(core, scripts, job)` (named so as
  not to clash with the A1111 API's `run_extras`),

with `remi.core.JobContext(id_task, user, resolve_upload, extra)`. They raise
`remi.ui.ValidationError` (every bad value) before anything runs, set up
`shared.state` and the progress registry for the job (`remi.core.job.job_state`:
`state.begin(job=id_task)`, `progress.start_task`, and the interrupt/skip
flags cleared afterwards, as `wrap_gradio_gpu_call` did), and return the
`Processed` (extras: images and info HTML). `resolve_upload` should give batch
uploads their original file name as `image.filename` (output names). The
Gradio handlers delegate to the same processing builders.

`remi.core.infotext.parse_infotext(text, tab=None)` → `{tab, core, scripts,
params, problems}` from `parse_generation_parameters`, the `on_infotext_pasted`
callbacks and the layouts (unread differing settings become
`override_settings`; no tab: `detect_tab`; empty text: the last generation's
`params.txt`; schema-format.md §7). Core actions (`remi/core/actions.py`,
inputs validated): `swap_dimensions`, `hr_resolution` (both coerce only the
params they read), `__conflicts__` (`remi.core.conflicts.conflict_patch`: the
exclusive groups first, then the pure
`anima_feature_conflicts.resolve_conflict_locks` over roles found by preset
field, repeated while a rule switches a group member off; one merged entry
per control), `__preset__` (Anima preset patch, unlocking the conflict
controls) and `__forge_preset__` (`gpu=True`: `presets.apply_forge_preset`
switches checkpoint, modules and dtype through `main_entry` as the old preset
radio did, then the form patch of `forge_preset_patch` plus `__settings__`).

Gradio-free imports: `remi.*`, `modules.shared`, `modules.shared_options`,
`modules.scripts` and the option registrations import no Gradio.
`modules.processing` still does, through `modules_forge.main_entry` (model
selection state next to the Gradio model bar) until that file is split when
the Gradio UI is deleted; `remi.core` reads `main_entry` only when something
already loaded it, except `apply_forge_preset`.

## HTTP API v2 (`/api/v2`)

All bodies and responses are Pydantic models (`remi/server/models.py`), so
`/openapi.json` describes the API; `tools/export_openapi.py` writes it to
`webui/src/api/openapi.json` for the generated TS client (`--check` fails when
it is stale; a test runs it). Rejected values answer 422 with the
schema-format.md §3 body; other errors `{"detail": str}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/app` | bootstrap: version, features, tabs, pages, locales, default locale, options subset, models summary, `not_ported` (scripts / Extras operations / extension tabs still on Gradio), `auth`, `queue` |
| GET | `/schema/{tab}` | `tab` ∈ `txt2img`, `img2img`, `extras`: core form + scripts (layout trees, defaults) |
| GET | `/pages`, `/pages/{id}` | extension pages and one page's schema (schema-format.md §8; read from `remi.pages` when present: `registered()`, `get(id)`, `page_schema(id)`) |
| POST | `/actions/{owner}/{action}` | run an action, `owner` = script id, `page:<id>`, `core`, or `settings` (`refresh:<key>`); `gpu=True` actions run behind generation on the main thread |
| POST | `/rules/derive` | `{tab, core, scripts, selected?}` → the core `__conflicts__` patch (Anima conflict locks) |
| GET | `/presets`, `/presets/{name}?tab=` | Anima and UI preset names; an Anima preset's form patch |
| POST | `/jobs` | `{tab, core, scripts, priority?, repeat? (1–100 copies), forever?}` → `{job_id, job_ids, seq}` (201); validated first |
| GET | `/jobs?cursor=&limit=&status=` | history and queue, newest first: `{items, next_cursor, total}` |
| GET / PATCH / DELETE | `/jobs/{id}` | detail (incl. `core`, `scripts`, `layout_hash`); `{priority?, forever?}`; cancel a running job or remove a queued/finished one |
| POST | `/jobs/{id}/interrupt`, `/skip`, `/retry` | interrupt keeps the results (`interrupted`) or cancels a queued job; retry queues a copy |
| GET / POST | `/queue`, `/queue/items`, `/queue/reorder`, `/queue/pause`, `/queue/resume`, `/queue/stop-forever` | queue state (`{size, paused, running}`), queued jobs in run order, ordering (within each priority; at most 500 ids), pause (persisted), generate-forever off |
| POST | `/interrupt` | interrupt whatever generates (also `/sdapi`); `{clear_queue?}` |
| POST | `/uploads` | multipart `file` → `{id, width, height, name, mime, size}` (201); content-sniffed, size-limited while the body streams in (also chunked bodies; setting `remi_upload_max_mb`, live), deduplicated |
| GET | `/files/{id}?download=`, `/thumbs/{id}?w=` | files and WebP thumbnails by opaque id only (ETag; uploads and outputs immutable for a year, network previews revalidated: their id changes with the file) |
| GET / DELETE / POST | `/outputs?cursor=&q=&tab=`, `/outputs/{id}`, `/outputs/refresh` | gallery (search: every term in the infotext or name), remove (`delete_file`; without it the file stays and the folder scan skips it from then on), index the output folders |
| GET | `/networks`, `/networks/{kind}?q=&cursor=&sort=&order=` | kinds (`lora`, `embeddings`, `checkpoints`, `vae`, `text_encoders`) and items (paths relative to their folder, preview file id) |
| GET / PATCH | `/networks/{kind}/{name}/meta` | file + user metadata; edit the user metadata (`<name>.json`) |
| PUT / POST | `/networks/{kind}/{name}/preview`, `/networks/{kind}/refresh` | preview from a file id; re-scan |
| GET / PATCH | `/options` | settings schema and values; PATCH validates every value first (any invalid: 422 lists them all, nothing applied), then applies them, running `onchange`; an `onchange` that fails leaves its key unchanged while the others stay applied (saved, broadcast) and the answer is 422 listing the failures; model keys go through the model manager |
| POST | `/infotext/parse`, `/infotext/image` | infotext (or an uploaded/generated image's PNG info) → named values (schema-format.md §7) |
| GET | `/i18n/{locale}` | merged locale catalog (`zh-CN` or `zh_CN`; ETag) |
| GET / POST | `/models`, `/models/refresh`, `/models/unload` | checkpoints, VAE/TE modules, samplers, schedulers, upscalers, styles, low-bits dtypes, UI presets |
| GET / PUT / DELETE / POST | `/styles`, `/styles/{name}`, `/styles/apply` | prompt styles CRUD; write styles into prompts |
| POST | `/outputs/save`, `/outputs/open-folder` | gallery "Save" / "Save zip" (`{ids, zip}` → copies in `outdir_save` through the save path, `log.csv`, zip as a file id); open an output folder on the WebUI machine (403 with `--hide-ui-dir-config`) |
| GET / PATCH / POST | `/extensions`, `/extensions/check-updates`, `/extensions/update`, `/extensions/install` | installed extensions; enable/disable (`{disabled?, disable_all?}`), check for / apply updates (config-state backup first), install from a git URL; writes refused (403) when extension access is disabled; changes apply after `/server/restart` |
| GET | `/user-assets/{name}` | `user.css` (data folder) and `notification.mp3` (WebUI folder) when present; bootstrap `user_assets` lists them |
| POST | `/prompt/tokens` | token count of one prompt (`null` without a model; not queued) |
| GET / POST | `/auth/status`, `/auth/login`, `/auth/logout` | login (session cookie) |
| POST | `/server/reload`, `/server/restart` | rebuild the app in-process ("Reload UI"); restart the process (when the launcher can; answers first, then the serve loop stops the server, interrupts the running job and exits) |
| GET | `/sysinfo` | system information |

`/internal/{ping,progress,pending-tasks,quicksettings-hint,profile-startup,sysinfo,sysinfo-download}`
stay for API clients; queued jobs appear in `/internal/progress` as
`task(remi-<job id>)`. `/sdapi/v1` (`modules/api/api.py`, with `--api` or
`--nowebui`) is mounted on the same app; it reads default script arguments
and infotext through the `remi.ui` layouts.

## WebSocket `/api/v2/ws`

JSON text frames, flat envelope `{"type", ...fields}`: on connect `hello{features,
version}` then the current `queue.changed` (and `job.started`/`job.progress`
of a running job); `job.queued{job_id, seq, tab, position}`,
`job.started{job_id, seq, tab}`, `job.progress{job_id, seq, tab, step, steps,
eta, job_no (0-based), job_count, progress, text}`, `job.completed{job_id, seq,
tab, status ("completed"|"interrupted"|"cancelled"), outputs, infotext}`,
`job.failed{job_id, seq, tab, status, error, errors?}`, `queue.changed{size,
paused, running}`, `options.changed{keys}`, `networks.changed{kind}`, `toast`,
`model.loading{stage, model}`, `model.loaded{model, error?}`, `pong`,
`resync{dropped}` (the client's queue overflowed: the backlog is replaced by a
fresh snapshot, `queue.changed` and the running job; the client refetches
what it shows, e.g. job history). A failed send closes the socket. The
client may send `hello{features}` (previews are then only sent in the formats it
lists) and `ping`. Binary frames carry previews: little-endian header `u8 type=1,
u8 fmt (1=jpeg, 2=webp), u32 job_seq, u32 step`, then the image bytes. A
dedicated encoder thread keeps only the newest frame (setting
`live_preview_fps`, default 6; JPEG when `live_previews_image_format` is
JPEG, else WebP; nothing is encoded while no client is connected). Each
client has a bounded event queue and one preview slot.

## Server (`remi/server`, `remi/jobs`)

- `create_app(services=, auth=, webui=, sdapi=)`; `webui.py` runs it with
  uvicorn on a background thread (`remi/server/launch.py`): before each build
  it runs `before_ui` callbacks and `remi.server.runtime.prepare()` (extension
  settings, script runners, model selection), then `app_started(None, app)`
  under `extension_route_guard`. `--nowebui` serves the APIs without the SPA
  (port 7861, loopback unless `--listen`/`--server-name`). "Reload UI" holds
  the queue first (waiting for the running job while the old app still
  serves, so it can be watched and interrupted), stops the server (open
  requests get 10 s), re-runs `initialize_rest`, rebuilds and serves, then
  releases the queue.
- Startup and every module a job runs import no Gradio (a test starts the
  WebUI with Gradio blocked). Scripts still on `ui()`: with Gradio importable
  their `ui()` runs once in a throwaway `gr.Blocks` for positional defaults;
  without it they are disabled with a log line, as are script files that
  import Gradio at load time. All are listed in bootstrap `not_ported`.
- Jobs: the worker claims the next job, then waits for the generation lock;
  only with the lock is the job current (`job.started`, progress, previews,
  `model.loading`). Interrupt/skip/cancel of a job that has not begun are
  recorded and applied right after its `shared.state.begin()` (an interrupt
  while waiting for the lock finishes it as `interrupted`, nothing run); they
  never touch another caller's generation. A generate-forever job finishing
  while the queue is paused is queued again for the resume.
- Services (process-wide, survive reloads; data in `<data path>/remi-data`):
  event bus, file index (`files.db`, `uploads/`, `outputs/` for images the
  save path did not write — temporary, `thumbs/`), job store (`jobs.db`,
  WAL), the single worker (generation lock `call_queue.queue_lock`, then
  `modules_forge.main_thread`), progress monitor and preview encoder.
  Housekeeping (at start, then hourly) prunes uploads and unsaved outputs
  older than `remi_upload_retention_days` and indexes the output folders.
- Auth: `--auth user:pass[,…]` / `--auth-file` (aliases `--gradio-auth`,
  `--gradio-auth-path`). With logins, `/api/v2`, `/internal`, the WebSocket,
  files and the SPA need the signed session cookie (`POST /auth/login`;
  the shell shows a built-in login page) or HTTP Basic. `/api/v2` and
  `/internal` refuse rebound `Host`s and cross-site writes/WebSocket
  handshakes (`extension_route_guard`, whose bind address is
  `remi.server.launch.bind_host`, also under `--nowebui`); `/docs`, `/redoc`
  and `/openapi.json` need the login too; `/sdapi` gets the host and
  cross-site checks and keeps `--api-auth` as its login. Logging out revokes
  the session cookie (a denylist next to the secret); cookies are `Secure`
  with `--tls-*` or when the request is HTTPS (a TLS proxy);
  extension routes keep the route guard (`app.auth_dependency` resolves the
  login). `--listen`, `--port`, `--server-name`, `--subpath` (root path,
  with or without a prefix-stripping proxy), `--tls-keyfile/--tls-certfile`,
  `--cors-allow-origins(-regex)`.
- SPA: `webui/dist` is the router fallback (extension routes win): hashed
  `/assets` cached for a year, `index.html` no-cache and served for client
  routes, with `window.__REMI_ROOT__` and prefixed asset URLs under
  `--subpath`. `launch.py` builds it when missing or when its sources or
  lockfile changed since the build (a stamp in `dist/`), with pnpm and the
  lockfile (installed pnpm, corepack, or `npx pnpm@12`; without Node a
  stale build is used with a warning; `REMI_SKIP_WEBUI_BUILD=1` skips), and
  no longer installs Gradio.

## Frontend

- `webui/src/api` — generated client + `useWs()` store.
- `webui/src/ui` — the param renderer: one component per input kind, layout nodes,
  condition evaluator, and per-tab form stores (Zustand, persisted drafts).
- `webui/src/widgets` — custom widgets by name.
- `webui/src/features` — generate-view features: prompt editor (with
  `registerCompletionSource()` for tag completion), session results and
  history, lightbox, run controller (Generate / Interrupt / Skip, forever,
  repeat), model bar, send-to, infotext (paste, image metadata).
- `webui/src/pages` — generate (txt2img/img2img share one view), extras,
  png-info, gallery, networks, queue, settings, extension pages.
- What the generate view relies on (frontend.md "Generate page"): bootstrap
  `options` also carries `ctrl_enter_interrupt`,
  `prevent_screen_sleep_during_generation`, `send_seed` and `send_size`
  (missing keys read as the A1111 defaults); "generate forever" and "generate
  N times" are client-driven, one `POST /jobs` per run, so every run takes the
  form as it is then (the server's `repeat` / `forever` stay for API
  clients); "Send to" puts output file ids straight into image inputs; image
  drops read PNG/JPEG/WebP parameters in the browser and fall back to
  `/infotext/image`.
- `webui/src/canvas` — Konva editor for img2img sketch/inpaint/masks; the only
  module that imports Konva.
- i18n: i18next with the server catalogs; language switch without reload.

## Built-in extension porting rule

Each extension: replace `ui()` with `params()` (same input order), delete its
`javascript/` and `style.css` unless the behaviour moves to a widget, move
`on_ui_tabs` to a `Page`, keep processing code and tests, and add a schema test
(`params()` arg count and order equal the processing signature). Extensions
whose UI was JavaScript-only (tag completion, prompt-all-in-one, agent
scheduler, mobile, bracket checker, lock-img2img-steps, prompt queue, repeat
generate) become core SPA features or dedicated pages.
