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
| `RangeSlider(name, minimum, maximum, default=None, step=0.01, integer=None)` | `[low, high]` | Gradio's RangeSlider (the `range-slider` widget); integer like `Slider`; values are clamped and ordered; the default is the full range |
| `Number(name, default, minimum=None, maximum=None, step=None, precision=None)` | float/int | |
| `Text(name, default="", lines=1, max_lines=None, placeholder=None, prompt=False)` | str | `prompt=True` → CodeMirror prompt editor (weights, LoRA, tag completion) |
| `Choice(name, choices, default, style="dropdown" or "radio", allow_custom=False)` | str/int | `choices` list (values or `(label, value)`) or a callable resolved per request; `type="index"` gives the script the index (the wire value stays the choice value); a `None` default is the first choice; `same=` (server side) is a key function under which a value that is no choice still names exactly one, e.g. `ui.without_model_hash` for checkpoint titles that gain `" [hash]"` once it is computed |
| `MultiChoice(name, choices, default=())` | list | dropdown multiselect or `style="checkboxes"`; `type="index"` gives the script indices |
| `Seed(name, default=-1)` | int | dice / recycle buttons; `-1` or `0 … 2^63-1` |
| `Color(name, default="#000000")` | str | |
| `Image(name, kind="image" or "mask" or "sketch", source=..., mode=None)` | upload id → PIL at the script | images are uploaded once (`POST /api/v2/uploads`); `source` names the image a mask/sketch layer is painted over (a mask without `source` is an uploaded mask image); masks are alpha-coded with a luminance fallback (`nodes.mask_channel`, schema-format.md §1.3); `mode` converts (e.g. `"RGBA"`) |
| `ImageList(name, widget=None, props={})` | list of upload ids → list of PIL | batch inputs; `widget` names a frontend widget that edits the list (e.g. `image-list`: select, replace, remove, reorder) |
| `File(name, accept=...)` | upload id → path | |
| `Json(name, default)` | any JSON | escape hatch for structured widgets |
| `Custom(name, widget, default, props={})` | JSON | a frontend widget registered in `webui/src/widgets` (stage timeline, block-weight grid, ControlNet unit, …) |
| `State(name, default)` | JSON | hidden; not rendered |

Common keyword arguments for inputs: `label`, `info`, `visible_if`, `enabled_if`,
`inactive_if` (shown dimmed but editable while true: the value has no effect then),
`infotext` (infotext key), `paste` (callable `params -> value` for derived
paste), `preset` (Anima preset key), `exclusive=(group, member)` (Toggle only),
`on_change="action"` (after any change: edits, pastes, bulk resets),
`on_input="action"` (after the user's own edit of the control only, like Gradio's
`.input`: for derived updates a pasted infotext already records), `width` (`"auto"`, `"fill"`, or a 1–12 column span),
`hidden=True` (submitted, never rendered: e.g. the choice a `Tabs` is bound to),
`persist=False` (kept out of browser drafts: every page load starts from the
current default, e.g. a copy of a setting), `submit=False` (the form holds
it for actions only; jobs never carry it, processing gets the default) and
`arg=False` (a tool input: rendered and handed to actions, not a processing
argument at all — `ScriptLayout.tools`, outside the positional order).
`Toggle(locks="owner.param")` protects that param from programmatic writes
(paste, Send to, reused jobs, presets, action patches) while it is on.
Action status texts are translated by the client: return `ui.message(text,
**params)` (catalog key with `{{name}}` placeholders) for dynamic ones.
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
one. Buttons call actions, and `on_change` / `on_input`
trigger them. Actions run outside the generation lock unless declared
`gpu=True`, in which case they queue behind generation.

Uploads and outputs reach an action through `ctx.files` (`ui.ActionFiles`,
which the server fills with its file store): `image(id)`, `image_size(id)`,
`get(id)`, `path(id)`, `store_image(image, name)` and `store_bytes(data,
name)` (extension-guide.md "Files"). Scripts, pages and built-in extensions
use only `remi.ui` / `remi.pages`: nothing outside `remi/` (and the few
listed WebUI glue modules, such as `modules/util.py`'s `open_folder`)
imports `remi.server` or reaches the server's services through
`ctx.request`; `tests/test_remi_layers.py` enforces it.

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
A script whose `params()` raises, or that only has A1111's `ui()` (Gradio
controls, not supported), is disabled: `ScriptRunner.disable_script`
removes it from the runner (no slots, no schema entry, no hooks) and keeps
the reason in `runner.disabled_scripts`; Extras operations likewise
(`scripts_postproc.disabled_scripts`). A script file that fails to import
(e.g. it imports Gradio) is skipped and kept in
`modules.scripts.scripts_not_loaded` with the reason. All three are listed
in bootstrap `not_ported`.

`ScriptRunner` API: `build_layouts()` (in `initialize_scripts`),
`layout_for(script)`, `ensure_arg_slots()`,
`script_args_from_values(selected, {script_id: values})` (a script's values
may also be a list in argument order), `script_values_from_args(args)`,
`scripts_schema()`, `paste_values(params)`, `preset_bindings()`,
`preset_values(label)`, `disable_script(script, reason)`, and A1111's
`set_named_arg(args, script_name, param, value)` (by param name). Slots are
assigned in A1111's order (UI category, then sorting priority), which
positional `/sdapi` clients and saved jobs rely on.

Anima helpers become declarative keyword arguments: `preset=` (or
`preset=(field, value)` for a localized value), `infotext=`, and
`exclusive=(group, member)` (groups and their headless priority in
`anima_feature_conflicts.EXCLUSIVE_GROUPS`). `anima_support.table_params(rows,
root)` applies a `SCRIPT_ARGS` table (infotext keys incl. `PasteKey`, preset
fields) to a `params()` tree and returns the `param_order`;
`AnimaPatchPanel.panel_params(root)` does it for the patch panels (inputs named
after the preset fields' last part).

### Pages (former `on_ui_tabs`)

`remi.pages.register(Page(id, title, icon, params=..., actions=..., widget=None,
send_to=[SendTarget(param, label)], on_load=None))`, called when the
extension's script is imported (`remi/pages`). A page is either
schema-rendered or a dedicated frontend view (a `widget` name), which is
implemented in `webui/src/pages/custom/`. `params` is a layout node or a
callable building one per schema request (choices read from files);
subclasses may override `params()` and declare `@ui.action` methods, or
pass `actions=` (an object or module whose actions become the page's). The
owner id is `page:<id>`; `Page.values(ctx, values)` validates an action's
values and resolves uploads (images → PIL, files → paths) through the
server's file store. `on_load` names an action the SPA runs each time the
page is shown; `send_to` declares what other views can send here (PNG Info
offers "Send to <label>" for image targets). The server reads
`registered()`, `get(id)` and `page_schema(id)` (`GET /api/v2/pages/{id}`,
schema-format.md §8). `registered()` lists the pages as the navigation shows
them: in the "UI Tab Order" setting's order, without those in "Hide UI Tabs"
(both store page titles, like the built-in tabs'; `include_hidden=True`:
all). Page titles join `shared.tab_names` (those settings' choices) when the
runtime is prepared. A page is dropped when its scripts are unloaded
("Reload UI"), so disabling an extension removes its page.

Long work runs as a page **task**: `Page.start_task(ctx, name, work,
gpu=False)` answers the task's status at once (`__outputs__["task:<name>"]`)
and runs `work(task)` on a thread (`gpu=True`: holding the generation lock,
so it queues behind generation); `task.progress(fraction, text, **params)`,
`task.publish(**outputs)` (partial outputs, e.g. a log), `task.check()`
(raises once cancelled). Every page has the actions `__task__` (`{task, id?,
resume?}` → status, plus the work's result patch for the poll naming the
finished task's `id`, or for a `resume` while no view has received it, e.g.
after a reload) and `__cancel__` (`{task}`); one task per page, name and user
runs at a time; `Page.busy(*names)` tells whether any user's task of the page
is queued or running (JoyCaption refuses "Unload" then). The `task-runner` widget starts and follows them.
The built-in pages: JoyCaption (`sd-webui-joycaption-modified`) and Prompt
Workshop (`sd-forge-prompt-workshop`).

### Settings

`OptionInfo` keeps its signature, but `component` refers to `remi.ui` widget
kinds (`ui.Slider`, `ui.Choice`, `ui.MultiChoice`, `ui.Color`, `ui.Text`,
`ui.State`, …) with A1111-style `component_args` plus `style` (`"radio"`,
`"checkboxes"`) and `allow_custom`; any other component (an A1111 Gradio
class) is ignored and the widget picked from the default's type. The
settings schema is exported from `shared.opts.data_labels` by
`remi.ui.settings.settings_schema` (`.info()`, `.link()`, `.choice_info()`,
`default_order`, `.html()`, `.needs_restart()`, `.needs_reload_ui()` are
kept structured; `.js()`, a
client-side link of the old settings page, is accepted and ignored).
`on_ui_settings` callbacks run before each app is built.

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
flags cleared afterwards), and return the
`Processed` (extras: images and info HTML). `resolve_upload` should give batch
uploads their original file name as `image.filename` (output names).

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
per control), `__preset__` (Anima preset patch over the sent form values,
with `conflict_patch` run again on the result: locks, mode choices) and `__forge_preset__` (`gpu=True`: `presets.apply_forge_preset`
switches checkpoint, modules and dtype through `main_entry` as the old preset
radio did, then the form patch of `forge_preset_patch` plus `__settings__`).

Model selection (checkpoint, VAE / text encoder modules, "Diffusion in Low
Bits" and the loading parameters) lives in `modules_forge.main_entry`; it
imports the processing pipeline, so `remi.core` reads it only when something
already loaded it, except `apply_forge_preset`.

## HTTP API v2 (`/api/v2`)

All bodies and responses are Pydantic models (`remi/server/models.py`), so
`/openapi.json` describes the API; `tools/export_openapi.py` writes it to
`webui/src/api/openapi.json` for the generated TS client (`--check` fails when
it is stale; a test runs it). Rejected values answer 422 with the
schema-format.md §3 body; other errors `{"detail": str}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/app` | bootstrap: version, features, tabs, pages, locales, default locale, options subset, models summary, `not_ported` (script files, scripts and Extras operations that could not be loaded, with the reason: an import error, A1111's `ui()` without `params()`, a failing `params()`), `auth`, `queue` |
| GET | `/schema/{tab}` | `tab` ∈ `txt2img`, `img2img`, `extras`: core form + scripts (layout trees, defaults) |
| GET | `/pages`, `/pages/{id}` | extension pages and one page's schema (schema-format.md §8; read from `remi.pages` when present: `registered()`, `get(id)`, `page_schema(id)`) |
| POST | `/actions/{owner}/{action}` | run an action, `owner` = script id, `page:<id>`, `core`, or `settings` (`refresh:<key>`); `gpu=True` actions run behind generation on the main thread |
| POST | `/rules/derive` | `{tab, core, scripts, selected?}` → the core `__conflicts__` patch (Anima conflict locks) |
| GET | `/presets`, `/presets/{name}?tab=` | Anima and UI preset names; an Anima preset's form patch |
| POST | `/jobs` | `{tab, core, scripts, priority?, repeat? (1–100 copies), forever?}` → `{job_id, job_ids, seq}` (201); validated first |
| GET | `/jobs?cursor=&limit=&status=&q=&starred=&tab=` | history and queue, newest first by submission time (imported jobs keep theirs, so old Agent Scheduler history sorts below recent jobs): `{items, next_cursor, total}`; `q`: every word in the prompt summary, infotext, parameters or import notes. A job carries `starred`, `replayable`, `name`, `source` (importer) and `notes` |
| GET / PATCH / DELETE | `/jobs/{id}` | detail (incl. `core`, `scripts`, `layout_hash`); `{priority?, forever?}`; cancel a running job or remove a queued/finished one |
| POST | `/jobs/{id}/interrupt`, `/skip`, `/retry` | interrupt keeps the results (`interrupted`) or cancels a queued job; retry queues a copy |
| POST / PUT | `/jobs/{id}/duplicate`, `/jobs/{id}/params` | queue a copy (right after the job while it is queued, same priority; else at the end); replace a queued job's `{core, scripts}` (validated like `POST /jobs`; 409 once it started). Both 409 for a job that is not `replayable` (`remi/server/routes/queue_tools.py`) |
| POST | `/jobs/star`, `/jobs/clear-history` | `{ids, starred}`; delete finished jobs (`{keep_starred}`, default true; their images stay) |
| POST | `/queue/move` | `{id, after?, before?, priority?}`: put a queued job right after `after` (else right before `before`) in its (new) priority group; neighbours that are no longer queued are ignored. O(1) whatever the queue length (the SPA's drag / keyboard / "move to top" use it); 409 when the job is not queued any more |
| GET / POST | `/queue/import`, `/queue/import/{agent-scheduler\|prompt-queue}` | sources found on this machine (`{id, label, available, count, path}`; `path` is null under `--hide-ui-dir-config`); import one (`remi/jobs/importers.py`, idempotent, read-only on the source): Agent Scheduler tasks (pending → queued, finished → history with their images, bookmarks → stars; core args renamed to the core params; positional script args named through the current `ScriptLayout`s, signed or not, read with an allow-list unpickler; arguments past the last slot are kept as `opaque` and the job is stored `replayable: false`, never run, retried or duplicated; values the form rejects → `cancelled` with the errors, to edit), Prompt Queue pending prompts (form defaults + prompt); task images are indexed only when they are image files inside the configured output folders that were not removed from the gallery. The queue is paused when something was queued. Answer: a report with counts and per-task outcomes and notes |
| GET / POST | `/queue`, `/queue/items`, `/queue/reorder`, `/queue/pause`, `/queue/resume`, `/queue/stop-forever` | queue state (`{size, paused, running}`), queued jobs in run order, ordering (within each priority; at most 500 ids), pause (persisted), generate-forever off |
| POST | `/interrupt` | interrupt whatever generates (also `/sdapi`); `{clear_queue?}` |
| POST | `/uploads` | multipart `file` → `{id, width, height, name, mime, size}` (201); content-sniffed, size-limited while the body streams in (also chunked bodies; setting `remi_upload_max_mb`, live), deduplicated |
| GET | `/files/{id}?download=`, `/thumbs/{id}?w=` | files and WebP thumbnails by opaque id only (ETag; uploads and outputs immutable for a year, network previews revalidated: their id changes with the file) |
| GET / DELETE / POST | `/outputs?cursor=&q=&tab=`, `/outputs/{id}`, `/outputs/refresh` | gallery (search: every term in the infotext or name), remove (`delete_file`; without it the file stays and the folder scan skips it from then on), index the output folders |
| GET | `/networks` | kinds in display order: `{id, title, action (prompt / checkpoint / module / none), negative, fields (user-metadata keys the editor offers), builtin}`; built-ins `lora` (incl. LyCORIS), `embeddings`, `checkpoints`, `vae`, `text_encoders`, plus every extension `ExtraNetworksPage` (generic adapter of `list_items()`; Gradio-free registration with `remi.networks.register_extra_page`) |
| GET | `/networks/{kind}?q=&folder=&arch=&compatible=&sort=&order=&offset=&cursor=&limit=` | items (`remi.networks`): search (every word in name, path, alias, hash, description, activation text, tags), folder (and subfolders), architectures (`unknown` included), `compatible` (the UI preset's architecture or unknown — the *effective* architecture: the user's `sd version`, else the one detected from the header; upstream only used the user's choice), items in "."-folders as `extra_networks_hidden_models` says (Always / When searched: a search of ≥ 4 characters / Never), sort `name`/`path`/`created`/`date` (modified)/`size`, windows of ≤ 500 from `offset` (or the opaque `cursor`). An item: name, title, path/folder (relative), preview file id + `preview_type` (`image`, or `video` for `.preview.mp4` / `.webm`, served by `/files` only) + blurred `placeholder` (16 px WebP data URI of image previews, computed in the background, `<remi-data>/networks/placeholders.json`, pruned with the extracted covers on a full rescan), description, `prompt` / `negative_prompt` (LoRA: preferred weight or the default multiplier, activation text), activation text, preferred weight, negative text, tags, hash, alias, `sd_version`, `arch` (user choice, else the `modelspec` / kohya header), `subtype` (LoRA / LyCORIS / DoRA), `value` (checkpoint title, module file), mtime, created, size, `editable` (false for extension items without a model file: no sidecar, preview or folder), `hidden` |
| GET | `/networks/{kind}/facets?q=&folder=&arch=&compatible=` | folder tree with counts (no "."-folders when `extra_networks_show_hidden_directories` is off) and architecture counts (each facet ignores its own filter), `folder_total` (the tree's "All": every filter but the folder), current architecture |
| GET / PATCH | `/networks/{kind}/{name}/meta` | file header (embedded images left out), `training` facts, `dataset_tags` (kohya `ss_tag_frequency`), user metadata; edit the `<model>.json` sidecar (sd_forge_lora's format; known keys type-checked, 422) |
| PUT / POST | `/networks/{kind}/{name}/preview`, `/networks/{kind}/{name}/open-folder`, `/networks/{kind}/refresh`, `/networks/refresh` | preview from an upload / output file id (`<model>.preview.png`; 404 for a missing or non-image file, 422 for one that cannot be decoded); open the folder on the WebUI machine (403 with `--hide-ui-dir-config`; bootstrap lists the `folders.open` feature when allowed); re-scan (`/sdapi/v1/refresh-loras` also tells clients: `networks.changed`). Edits, previews and folders of items without a model file answer 409. `name` may contain `/`. Embedded `ssmd_cover_images` become previews (extracted to `<remi-data>/networks/covers`) |
| GET / PATCH | `/options` | settings schema, values and built-in `defaults`; PATCH validates every value first (any invalid: 422 lists them all, nothing applied), then applies them, running `onchange`; an `onchange` that fails leaves its key unchanged while the others stay applied (saved, broadcast) and the answer is 422 listing the failures; model keys go through the model manager |
| POST | `/infotext/parse`, `/infotext/image` | infotext (or an uploaded/generated image's PNG info) → named values (schema-format.md §7) |
| GET | `/i18n/{locale}` | merged locale catalog (`zh-CN` or `zh_CN`; ETag) |
| GET / POST | `/models`, `/models/refresh`, `/models/unload`, `/models/reload` | checkpoints, VAE/TE modules, samplers, schedulers, upscalers, styles, low-bits dtypes, UI presets; unload / load the selected model now (behind generation; `/models/reload` `{force?}` loads again even when loaded, emits `model.loading` / `model.loaded`) |
| GET / POST | `/models/loaded`, `/models/hashes` | the weights the memory manager holds (`{checkpoint, items: [{name, memory_mb, device}]}`); hash every checkpoint in the background (`POST {threads}` → 202, 409 while running; `GET` = progress `{running, done, total, failed, current, finished_at}`) |
| GET / PUT / DELETE / POST | `/styles`, `/styles/{name}`, `/styles/apply` | prompt styles CRUD; write styles into prompts |
| GET | `/gallery?cursor=&limit=&q=&category=&model=&since=&until=&starred=&sort=`, `/gallery/facets`, `/gallery/{id}` | the Gallery (`remi/library.py`, side table `gallery_meta` in `files.db`, filled lazily: model from the infotext, category `txt2img`/`img2img`/`extras`/`grids`/`saved`/`other` from the job or the output folder, star): every word of `q` in the infotext, file name or model; ISO `since` (inclusive) / `until` (exclusive); sort `newest`/`oldest`/`name`/`largest`; facets = models and categories with counts, date bounds, totals. `total` is counted for the first page only (no `cursor`). Items whose file cannot be found are left out of the page but stay indexed (an unmounted drive keeps its images and stars); only when the file's folder exists (the file was deleted) is the row dropped, without marking the path removed. The side table is synced only after `files.db` changed (`PRAGMA data_version`) |
| POST | `/gallery/star`, `/gallery/delete`, `/gallery/zip`, `/gallery/{id}/reveal` | star (`{ids ≤ 2000, starred}`); remove from the gallery (`delete_files` also deletes the files); pack originals into a zip (file id, kept an hour); show the file in the file manager of the WebUI machine (loopback clients that came through no proxy — any `X-Forwarded-*` / `Forwarded` / `X-Real-IP` header refuses; a same-machine tunnel that adds none passes as local, so use `--hide-ui-dir-config` when sharing that way; 403 with `--hide-ui-dir-config`). Zips skip files deleted meanwhile; zips older than an hour and their file rows are pruned by the hourly housekeeping. Changes emit `outputs.changed{ids}` (the SPA patches its cached pages; ids it changed itself are not refetched) |
| POST | `/pnginfo` | `{file_id}` (upload or output) → PNG Info: `infotext`, parsed `params`, format / mode / size, `entries` (text chunks, comments, stealth info, every EXIF tag incl. the Exif / GPS / Interop IFDs, binary values as `<N bytes>`), `chunks` (PNG chunk list, IDAT runs merged, text keywords) (`remi/image_meta.py`) |
| POST | `/outputs/save`, `/outputs/open-folder` | gallery "Save" / "Save zip" (`{ids, zip}` → copies in `outdir_save` through the save path, `log.csv`, zip as a file id); open an output folder on the WebUI machine (403 with `--hide-ui-dir-config`) |
| GET / PATCH / POST / DELETE | `/extensions`, `/extensions/check-updates`, `/extensions/{name}/check`, `/extensions/update`, `/extensions/install`, `/extensions/{name}` | installed extensions (`remote` without credentials, `remote_url`, `commit_url`, `restartable`, `disabled_by`, `index_url`); enable/disable (`{disabled?, disable_all?}`), check for updates (all, or one at a time for progress; a failed fetch is the status `unknown (remote error)`), apply updates (config-state backup first; 502 names those git could not update), install from a git URL (502 when git cannot clone), uninstall (not built-in ones); logic in `remi/extensions_admin.py`; git never prompts (`GIT_TERMINAL_PROMPT=0`, ssh `BatchMode=yes`, no terminal) and is bounded by timeouts; one git operation per repository at a time; writes refused (403) when extension access is disabled; changes load with `/server/reload`. Errors are `{detail, template, params}`: `template` is the English message with `{{name}}` placeholders, the key clients translate |
| GET | `/extensions/index?url=&refresh=` | an extension index (default `WEBUI_EXTENSIONS_INDEX` or the A1111 list; fetched by the server, cached 10 min; 403 without extension access, 502 when unreachable): entries with `installed`, tag descriptions |
| GET / POST / DELETE | `/extensions/config-states`, `/extensions/config-states/{id}`, `/extensions/config-states/{id}/restore` | config-state backups (`config_states/*.json`): list, save `{name}`, detail (each extension saved vs. now; the WebUI commit for reference), delete, restore `{what: "extensions"}` → `{reload}` (each extension is reset to its commit during the next reload). The WebUI checkout is never reset: old backups point at upstream commits and `git reset --hard` would drop the working tree |
| GET | `/user-assets/{name}` | `user.css` (data folder) and `notification.mp3` (WebUI folder) when present; bootstrap `user_assets` lists them |
| POST | `/prompt/tokens` | token count of one prompt (`null` without a model; not queued) |
| GET / POST / PUT / PATCH / DELETE | `/prompt-tools/config`, `/complete`, `/usage`, `/lookup`, `/translate`, `/generate`, `/translation-key`, `/groups`, `/refresh`, `/history[/{id}]`, `/collections[/{id}]`, `/favorites[/{id}]`, `/favorites-export`, `/favorites-import` | prompt tools (`remi/prompt_tools`, frontend.md "Prompt tools"): tag completion, usage ranking, chip lookups, translation through a configured provider (its API key write-only, kept in the prompt tools store), prompt writing from a description by the OpenAI-compatible provider (`/generate`: 409 without that provider, 413 over 4,000 characters, 502 when it fails), quick tag groups, prompt history (recorded from `job.queued`), favorites (paged) |
| GET / POST | `/auth/status`, `/auth/login`, `/auth/logout` | login (session cookie) |
| POST | `/server/reload`, `/server/restart` | rebuild the app in-process ("Reload UI"); restart the process (when the launcher can; answers first, then the serve loop stops the server, interrupts the running job and exits) |
| GET / POST | `/sysinfo`, `/sysinfo/check` | system information; whether a sysinfo file's checksum matches (`{text}` → `{valid}`) |
| GET | `/server/status` | `{restartable, extension_access, settings_frozen, model_loaded, version}` |
| GET | `/licenses` | `html/licenses.html` as data (groups of `{id, name, url, description, license_url}`; the SPA fetches license texts itself) |
| POST | `/options/validate` | setting values checked like `PATCH /options` but not applied (settings import) |

`/internal/{ping,progress,pending-tasks,quicksettings-hint,profile-startup,sysinfo,sysinfo-download}` (`ping` answers `{instance}`, an id that differs for every app: it changes with each reload or restart)
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
tab, status ("completed"|"interrupted"|"cancelled"), outputs, infotext,
warnings?}` (`warnings`: notes on script hooks that raised during the run and
were skipped, and the processing's `p.comment()` notes — e.g. a feature
skipped on this device — also kept on the job),
`job.failed{job_id, seq, tab, status, error, errors?}`, `queue.changed{size,
paused, running}`, `options.changed{keys}`, `networks.changed{kind}`,
`outputs.changed{ids}` (gallery stars / removals), `toast`,
`model.loading{stage, model}`, `model.loaded{model, error?}`, `pong`,
`resync{dropped}` (the client's queue overflowed: the backlog is replaced by a
fresh snapshot, `queue.changed` and the running job; the client refetches
what it shows, e.g. job history). A failed send closes the socket. The
client may send `hello{features}` (previews are then only sent in the formats it
lists) and `ping`. Binary frames carry previews: little-endian header `u8 type=1,
u8 fmt (1=jpeg, 2=webp), u32 job_seq, u32 step`, then the image bytes (the
step restarts for each image of a batch; the client orders frames by
arrival). Only the newest preview is encoded: `shared.state`'s
`live_preview_listeners` call the encoder when a preview is assigned; a
small one (the RGB / Approx NN previews, up to 256 × 256 px for JPEG,
96 × 96 for WebP) is encoded and handed to the hubs right there on the
generation thread (0.1-0.5 ms), larger ones and those inside a throttle wait
by the encoder thread (`remi.realtime.PreviewEncoder`), so the sampler does
not hand the interpreter to another thread for every step; `live_preview_fps`
(default 6) only throttles (JPEG when `live_previews_image_format` is JPEG,
else WebP; nothing is encoded while no client is connected). The SPA draws a step once: a
`job.progress` that arrives while previews come every step waits for the
next frame (`webui/src/api/progress-gate.ts`), and both are applied in one
animation frame once the image has decoded (`webui/src/api/paint-batch.ts`).
Each
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
- Gradio is not a dependency: nothing imports it and `requirements.txt` /
  `launch_utils` do not install it (see "Gradio removal" below). Scripts
  without `params()` and script files that fail to import are disabled with
  a log line and listed in bootstrap `not_ported`.
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
- Auth: `--auth user:pass[,…]` / `--auth-file`. With logins, `/api/v2`, `/internal`, the WebSocket,
  files and the SPA need the signed session cookie (`POST /auth/login`;
  the shell shows a built-in login page) or HTTP Basic. `/api/v2` and
  `/internal` refuse rebound `Host`s and cross-site writes/WebSocket
  handshakes (`extension_route_guard`, whose bind address is
  `remi.server.launch.bind_host`, also under `--nowebui`); `/docs`, `/redoc`
  and `/openapi.json` need the login too; `/sdapi` gets the host and
  cross-site checks, and its login is `--api-auth` when set, else the WebUI
  login (clients on this machine exempt only with
  `--api-no-auth-on-localhost`). Failed logins (form or Basic) are throttled;
  cross-site subresource `GET`s are refused (fetch metadata); responses carry
  `nosniff` and `Referrer-Policy`, HTML pages a hash-based CSP
  (`security.md`). Logging out revokes
  the session cookie (a denylist next to the secret); cookies are `Secure`
  with `--tls-*` or when the request is HTTPS (a TLS proxy);
  extension routes keep the route guard (`app.auth_dependency` resolves the
  login). `--listen`, `--port`, `--server-name`, `--subpath` (root path,
  with or without a prefix-stripping proxy), `--tls-keyfile/--tls-certfile`,
  `--cors-allow-origins(-regex)`.
- SPA: `webui/dist` is the router fallback (extension routes win): hashed
  `/assets` cached for a year and sent from the build's precompressed `.br` /
  `.gz` copies when the client accepts them, `index.html` no-cache and served for client
  routes, with `window.__REMI_ROOT__` and prefixed asset URLs under
  `--subpath`. `launch.py` builds it when missing or when its sources or
  lockfile changed since the build (a stamp in `dist/`), with pnpm and the
  lockfile (installed pnpm, corepack, or `npx pnpm@12`; without Node a
  stale build is used with a warning; `REMI_SKIP_WEBUI_BUILD=1` skips).

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
generate) become core SPA features or dedicated pages. The Agent Scheduler,
Prompt Queue and Repeat Generate are the core Queue (`/jobs`, the Queue page,
"Add to queue" next to Generate, client-driven forever / repeat N): their
folders are gone except `agent-scheduler/preload.py` (its command-line flags,
so launch commands keep working; `--agent-scheduler-sqlite-file` tells the
importer where the database is), and their data comes over with
`POST /queue/import/{source}`.

**Decision: their HTTP APIs are not kept.** The inventory listed the 34
`/agent-scheduler/v1/*` routes and the 8 `/prompt-queue/*` routes to keep;
no compatibility shim is provided. Agent Scheduler's API queued Gradio
handler calls (positional `fn_inputs`, a `task_id` hijacked from
`randomId`) and its callbacks; Prompt Queue's was the state of a
client-side loop that clicked Generate. Neither maps onto `/jobs` without
re-creating the positional coupling Remi removes, and no extension in this
repository calls them. Scripts and tools use `POST /api/v2/jobs` (named
`core` / `scripts` params, `priority`, `repeat`, `forever`), `/queue/*`
and the `job.*` WebSocket events instead; the A1111 `/sdapi/v1` API stays
for clients written against it.

The same holds for the two JavaScript-only prompt extensions: the 16
`/tacapi/v1/*` routes of Tag Autocomplete and the 46 `/physton_prompt/*`
routes of Prompt All-in-One (inventory rows 25 and 28) are not kept. They
were private back ends of each extension's own browser script (tag files
read through Gradio's `file=` route, a Vue app's storage, 21 translator
back ends, a runtime package installer); their only client is gone, and
`/api/v2/prompt-tools` (`remi/server/routes/prompt_tools.py`) serves the
SPA's completion and prompt tools instead. A request to any of the four
retired prefixes gets `410 Gone` naming the replacement
(`remi/server/spa.py` `RETIRED_APIS`, `tests/test_remi_server_app.py`)
rather than the SPA shell, unless an installed extension registers a route
there itself.

**Decision: what the JavaScript-only extensions keep.** Their features are
core SPA features now; each extension's README lists, setting by setting
or feature by feature, what was kept, what merged into a core feature, and
what was dropped and why, and a test keeps the setting lists complete:

- Tag Autocomplete (`extensions-builtin/sd-webui-tagcomplete-neo/README.md`,
  `tests/test_remi_prompt_tools.py`): 41 of its 58 `tac_*` settings are
  kept with their keys (dotted keys renamed with `_`, saved values moved);
  the 17 others are explained (no third-party lookups such as CivitAI,
  no `$style` variables without an expander, one keymap and one
  contrast-checked palette shared by the editor).
- Prompt All-in-One (`extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md`):
  tag view, history, favorites, tag groups and LLM prompt writing are
  kept; quality presets, highlighting, the token counter and the keyword
  blacklist merged into Styles / Prompt Anchor, the prompt editor and the
  Prompt Blacklist extension; vendor-specific translators, mbart50,
  translation while typing, mouse hotkeys and the package installer are
  dropped (the reasons are in the README).
- Agent Scheduler (`extensions-builtin/agent-scheduler/README.md`,
  `tests/test_remi_server_jobs.py`): history retention, re-queueing failed
  jobs and the action when the queue is done are Queue settings with their
  old keys; the layout settings (button and tab placement, page size,
  checkpoint dropdown, pause flag, remappable shortcut) are dropped and
  explained there; "Add to queue" has the fixed shortcut Ctrl/Cmd+Shift+Enter.

## Gradio removal

Gradio, `gradio_client` and `gradio_rangeslider` are gone: no import, no
requirement, no launcher install step. `tests/test_dependencies.py` keeps it
that way: every module-level third-party import of the WebUI's own code
(`modules`, `modules_forge`, `remi`, `scripts`, `extensions-builtin`) must
come from `requirements.txt` or the packages it pulls in, and the suites run
in an environment with nothing else installed.

Deleted (legacy UI only): `modules/ui.py` and the `modules/ui_*.py` UI
modules (checkpoint merger, common, components, extensions, extra-network
cards and metadata editors, Gradio extensions, load/save, postprocessing,
prompt styles, settings, tempdir, top row), `modules/gradio_extensions.py`,
`modules/shared_gradio_themes.py`, `modules/anima_lora_ui.py`,
`modules_forge/forge_canvas/`, `javascript/`, `script.js`, `style.css`, the
`html/extra-networks-*` templates, `html/card-no-preview.jpg` and
`html/footer.html`, `remi/ui/legacy.py` and `remi/ui/legacy_gradio.py` (the
positional `#i` layouts of unported scripts; schema entries no longer carry
`legacy`), `Script.ui()` / `ScriptPostprocessing.ui()` and the runners'
Gradio build path (`setup_ui`, `create_script_ui`, `inputs`, `controls`,
`infotext_fields`, `paste_field_names`, `group`, `elem_id`), the component
hooks (`Script.before_component` / `after_component` /
`on_before_component` / `on_after_component`), the Gradio handlers of
txt2img / img2img / Extras / PNG Info, the `call_queue.wrap_gradio_*`
wrappers and `progress.restore_progress`, the paste / "Send to" wiring of
`infotext_utils` (`PasteField`, `ParamBinding`, `connect_paste*`),
`main_entry`'s model bar (`make_checkpoint_manager_ui`, `forge_main_entry`,
`on_preset_change`, `on_refresh_checkpoint`), the Gradio exclusive-group
wiring of `anima_feature_conflicts`, the Gradio preset-control registry of
`anima_presets` and `anima_support.register_table_fields`, the settings
`gradio_theme`, `gradio_themes_cache`, `show_gradio_deprecation_warnings`
and `enable_reloading_ui_scripts`, the command-line flags `--gradio-debug`,
`--gradio-allowed-path`, `--no-gradio-queue` and the `--gradio-auth` /
`--gradio-auth-path` aliases (use `--auth` / `--auth-file`), and the
`GRADIO_SERVER_NAME` environment variable (use `--listen` / `--server-name`).

Callbacks (`modules/script_callbacks.py`): `on_ui_tabs` (extension tabs are
`remi.pages`), `on_ui_train_tabs` and `on_before_component` /
`on_after_component` are removed; an A1111 extension that uses them builds a
Gradio UI and fails to import, so it is listed in `not_ported`.
`on_ui_settings` (registers settings) and `on_before_ui` (registers
extra-network pages, X/Y/Z axes, … before each app build) are kept;
`on_app_started` receives `demo=None`.

Kept, each for a non-UI reason:

| What | Why |
|---|---|
| `modules/ui_extra_networks.py` | three-line import alias of `modules/extra_networks_pages.py`: A1111 extensions register their extra-network pages through `modules.ui_extra_networks.register_page`; the Networks page lists them (`remi.networks`) |
| `modules/extra_networks_pages.py` | the data API of extra-network pages (`ExtraNetworksPage.list_items()`, `register_page`), without any HTML |
| `modules_forge/main_entry.py` | model selection state and changes (`checkpoint_change`, `modules_change`, `dtype_change`, `refresh_model_loading_parameters`) used by processing, Hires. fix, the refiner, X/Y/Z and the server |
| `modules/extras.py` `run_modelmerger` | the checkpoint merge itself, run by the Checkpoint Merger page (`remi/core/merger.py`, registered by `scripts/checkpoint_merger.py`) as a page task with `progress=`; returns its status message |
| `html/licenses.html` | read by `GET /api/v2/licenses` |
| `modules/util.plaintext_to_html` | the A1111 `/sdapi/v1/extra-single-image` and `extra-batch-images` responses carry `html_info` (HTML) for API compatibility; Remi's own API and UI get plain infotext (`run_extras_named`) |
| `OptionInfo.js()`, `OptionInfo(comment_before=, comment_after=)`, `OptionInfo.html()` | A1111 extension API: the two arguments and `.html()` become the setting's `html` help (for third-party options; Remi's own options use `.info()`, `.link()`, `.choice_info()` and `default_order`), `.js()` is ignored |
| `util.open_folder`, `ScriptRunner.set_named_arg` | A1111 helpers extensions call; Gradio-free now (`set_named_arg` takes param names) |
| `--use-textbox-seed`, `--add-stop-route`, `--ngrok-region`, `--ui-config-file` | accepted so existing launch commands keep working; they do nothing (`--ui-config-file`: form drafts live in the browser, defaults come from `params()` and the UI presets) |
| `--share` | accepted so existing launch commands keep working, but there is no public tunnel: the server logs a warning at startup (use `--listen`, a reverse proxy or `--ngrok`). Like those, it marks the server as non-local (`cmd_opts.webui_is_non_local`), so the protections of an exposed server stay on (installing extensions needs `--enable-insecure-extension-access`; the browser is not opened automatically) |
| `legacy_ui_<tab>.json` test fixtures | the argument contract recorded from each script's Gradio `ui()` before its port (the control class is the `legacy_control` key in some); the ports are tested against them. New ones are recorded with `tools/record_legacy_ui.py` (in the format of `tests/remi_port_harness.describe`), which builds the `ui()` in an export of the last Gradio-era revision, run by an interpreter of its own (`tools/legacy-ui-requirements.txt`: that revision's requirements plus Gradio 4.40) |

Settings: every core option (`modules/shared_options.py`) and every UI
preset option (`modules_forge/presets.py`) has a reader — processing, the
server, a script or the SPA (`CLIENT_OPTION_DEFAULTS`, sent in the bootstrap
`options`). The Gradio-only settings (theme, accordion, card layout, brush
colour, gallery height, …) were removed — frontend.md ("Gradio-era settings
that were removed") lists each with the SPA control that replaces it;
`tests/test_remi_option_consumers.py` fails when a setting has no reader (its `KEPT` list, empty today, would name
any setting kept on purpose with its reason).
