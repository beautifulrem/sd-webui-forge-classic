# Absorbing Forge Neo updates into Remi

Remi started as a downstream branch of
[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
(`neo`) and still shares its **runtime**: `backend/`, the processing
pipeline, samplers, model loading, the scripts runtime, the extensions'
processing code and the `/sdapi/v1` API. Its **user interface** no longer
shares anything: the Gradio UI (`modules/ui*.py`, `javascript/`, `html/`,
`style.css`, `script.js`, ForgeCanvas, every `ui()`) was deleted and replaced
by the declarative parameter model (`remi/ui`), the FastAPI server
(`remi/server`, `remi/jobs`) and the React SPA (`webui/`)
([architecture.md](architecture.md)).

So an upstream update is two kinds of work:

- **runtime changes** are merged as before (Git merge, conflicts resolved in
  favour of Remi's Anima hooks; see [remi-architecture-notes.md](../remi-architecture-notes.md));
- **UI changes** cannot be merged: their *intent* is translated into
  `params()`, `remi.core` forms, actions, settings widgets or SPA code, and
  the upstream file stays deleted.

History of earlier syncs:
[2026-10-02](../upstream-sync-2026-10-02.md) (selective port from reviewed
diffs) and [2026-10-14](../upstream-sync-2026-10-14.md) (Git merge of `neo`
at `46af880c`, the last upstream commit absorbed before the UI rewrite).
Each new sync adds a `docs/upstream-sync-YYYY-MM-DD.md` in the same format.

## 1. Where upstream files went

| upstream file(s) | in Remi | what to do with an upstream change |
|---|---|---|
| `backend/**`, `modules_forge/packages/**` | same files (with Remi's Anima hooks) | merge |
| `modules/processing.py`, `modules/sd_models.py`, `modules/sd_samplers*.py`, `modules/sd_schedulers.py`, `modules/sd_vae*.py`, `modules/sd_unet.py`, `modules/prompt_parser.py`, `modules/images.py`, `modules/masking.py`, `modules/rng*.py`, `modules/devices.py`, `modules/upscaler*.py`, `modules/*_model.py`, `modules/textual_inversion/**`, `modules/extra_networks.py` | same files | merge; check Remi's hooks listed in remi-architecture-notes.md |
| `modules/scripts.py`, `modules/scripts_postprocessing.py`, `modules/scripts_auto_postprocessing.py` | same files: hooks unchanged, `ui()` / Gradio wiring replaced by `params()` layouts (`build_layouts`, `script_args_from_values`, `disable_script`) | merge hook and runner changes; translate anything about controls into the layout code |
| `modules/script_callbacks.py` | same file without the UI-building callbacks (`on_ui_tabs`, `on_ui_train_tabs`, `on_before_component`, `on_after_component`) | merge non-UI callbacks; a new UI callback has no equivalent (see §3) |
| `modules/txt2img.py`, `modules/img2img.py`, `modules/postprocessing.py` | same files; entry points take named values (`run_txt2img(core, scripts, job)`, `run_img2img`, `run_extras_named`); the Gradio handler signatures are gone | merge changes to how `p` is built; a new argument of the old handler is a new core param (§2.3) |
| `modules/infotext_utils.py` | same file (parsing, `create_override_settings_dict`); paste bindings are `infotext=` / `paste=` on inputs and `remi/core/infotext.py` | merge parsing changes; new paste fields of `ui.py` become `infotext=` keys in `remi/core/forms.py` |
| `modules/shared_state.py` | same file plus `State.live_preview_listeners`: called after each `assign_current_image`, it wakes Remi's preview encoder (`remi/realtime.py`) so a live preview leaves as soon as it exists | merge; keep the listener list and its call in `assign_current_image` |
| `modules/errors.py` | same file; while a job runs, `remi/jobs/queue.py` (`script_errors`) wraps `errors.report(message, *, exc_info)` to turn "Error running <hook>: <script>" reports into the job's warnings | merge; keep `report`'s name and signature (or adapt `script_errors`) |
| `modules/call_queue.py`, `modules/progress.py` | `queue_lock` and the progress registry kept for `/sdapi` and `/internal/progress`; jobs run in `remi/jobs` (queue, worker) with `remi/core/job.py` (`job_state`, what `wrap_gradio_gpu_call` did) | merge lock / progress fixes into both places |
| `modules/shared_options.py`, `modules_forge/shared_options.py`, options in extensions | same files; `component` is a `remi.ui` class (`ui.Slider`, `ui.Choice`, …) | merge; convert `gr.*` components (§2.2) |
| `modules/shared_items.py`, `modules/shared.py`, `modules/options.py`, `modules/shared_cmd_options.py`, `modules/cmd_args.py` | same files (no Gradio types; `--auth` etc. added, Gradio-only flags are no-ops) | merge; a new flag that configures Gradio needs a server equivalent or a no-op |
| `modules/api/api.py`, `modules/api/models.py` | same files, mounted on the Remi app; script args and infotext go through the `remi.ui` layouts (`init_default_script_args`, `apply_infotext`) | merge; keep the layout-based script-argument code |
| `modules/launch_utils.py`, `launch.py`, `webui.py`, `requirements*.txt`, `webui.sh`, `webui.bat` | same files: no Gradio install, `build_webui()` builds the SPA, `webui.py` serves `remi.server` | merge installer changes; drop Gradio pins and Gradio launch code |
| `modules/initialize.py`, `modules/initialize_util.py` | same files without Gradio setup | merge |
| `modules_forge/main_entry.py` | the model-selection logic only (`checkpoint_change`, `modules_change`, `dtype_change`, loading parameters); the model bar is the SPA's `features/model-bar` with `/api/v2/models` and `remi/core/presets.py` | merge selection logic; translate model-bar UI changes into the SPA |
| `modules_forge/presets.py` | same file; applied to forms by `remi/core/presets.py` (`apply_forge_preset`, `forge_preset_patch`) | merge; a new preset value that changes a control becomes a form patch entry |
| `modules/ui.py` (txt2img / img2img / PNG Info tabs, paste fields, top-level layout) | `remi/core/forms.py` (core params, infotext keys, presets, slots), `remi/core/tabs.py` (schema, validation), `remi/core/actions.py` (size swap, Hires resolution), `webui/src/pages/generate`, `features/*` | translate (§2.3) |
| `modules/ui_toprow.py` | `webui/src/features/prompt-editor` (prompt fields, token counter, styles), `features/progress` (Generate / Interrupt / Skip, forever) | translate |
| `modules/ui_common.py` | `webui/src/features/send-to` (result actions), `POST /api/v2/outputs/save`, `features/lightbox` | translate |
| `modules/ui_components.py`, `modules/gradio_extensions.py`, `modules/ui_gradio_extensions.py`, `modules/shared_gradio_themes.py` | `remi.ui` nodes (`InputAccordion` → `Section(enable=)`, `ToolButton` → `Button(variant="tool")`, `FormRow` → `Row`, `DropdownMulti` → `MultiChoice`), the SPA design system and appearance menu | none (a new component is a new node kind or widget only if a feature needs it) |
| `modules/ui_postprocessing.py` | `remi/core/forms.py` / `remi/core/extras.py` (the `extras` form), `webui/src/pages/extras` | translate |
| `modules/ui_extra_networks*.py`, `html/extra-networks-*.html`, `javascript/extraNetworks.js` | `remi/networks.py`, `remi/network_metadata.py`, `/api/v2/networks/*`, `webui/src/features/networks`; the extension-facing `ExtraNetworksPage` data API is Gradio-free in `modules/extra_networks_pages.py` (`modules/ui_extra_networks.py` re-exports it for A1111 extensions; `remi.networks` adapts registered pages) | translate UI; merge changes to item data (`list_items`, metadata) |
| `modules/ui_extensions.py`, `javascript/extensions.js` | `remi/extensions_admin.py`, `/api/v2/extensions/*`, `webui/src/pages/extensions` | translate (git logic changes: port into `extensions_admin.py`) |
| `modules/ui_settings.py`, `javascript/settings.js`, `javascript/ui_settings_hints.js` | `remi/ui/settings.py` (settings schema), `remi/server/options.py`, `PATCH /api/v2/options`, `webui/src/pages/settings` | translate |
| `modules/ui_prompt_styles.py` | `/api/v2/styles`, the style editor dialog in `features/prompt-editor` | translate |
| `modules/ui_loadsave.py` (`ui-config.json`) | not carried over: forms start from schema defaults, UI presets and browser drafts | none |
| `modules/ui_tempdir.py` | `remi/files.py` (uploads, outputs, thumbnails by opaque id) | translate if it is about saving or serving files |
| `modules/ui_checkpoint_merger.py` | no merger view in the SPA; `modules/extras.run_modelmerger` is the processing code | merge processing changes in `extras.py`; a UI for it would be a new page |
| `modules_forge/forge_canvas/**` | `webui/src/canvas` (Konva editor) | translate (upstream's 2026-10-02 single-click drawing is already covered) |
| `javascript/*.js` (other) | SPA features: `edit-attention.js` / `edit-order.js` / `promptField.js` / `token-counters.js` → prompt editor; `imageviewer*.js` → lightbox; `progressbar.js` → WebSocket progress; `contextMenus.js` (generate forever) → run controller; `dragdrop.js` / `imageDragAndDrop.js` → drop zones; `aspectRatioOverlay.js` / `imageMaskFix.js` → canvas; `hires_fix.js` → `hr_resolution` action; `localization.js` → i18n; `notification.js` → completion notifications; `resizeHandle.js` → resizable panes; `ui.js` → app shell | translate |
| `html/licenses.html` | `GET /api/v2/licenses` (parsed into data; kept in `html/`) | merge the license list |
| `html/footer.html`, `html/ui.webp`, other `html/*` | none | none |
| `scripts/*.py`, `modules/processing_scripts/*.py` | same files; `ui()` replaced by `params()` with the same arguments | merge processing; translate `ui()` changes into `params()` (§2.1) |
| `extensions-builtin/<upstream ext>/` (ControlNet, LoRA, MultiDiffusion, Spectrum, …) | same folders; `ui()` → `params()`, `on_ui_tabs` → `remi.pages`, extension JS → widgets | merge `lib_*` / processing; translate UI (§2.1); upstream `mobile` and `prompt-bracket-checker` are SPA features and stay deleted |

## 2. Translating each kind of change

### 2.1 A script or extension changes its `ui()`

Upstream adds, removes or reorders controls in a `ui()` (a built-in script,
a processing script, a bundled extension).

1. Read the upstream diff of `ui()` **and** of the processing hook signature
   (`process`, `process_before_every_sampling`, …). The hook signature is the
   contract: `params()` must declare the inputs in exactly that positional
   order (use `param_order` when the visual order differs).
2. Edit `params()`: the component mapping is in
   [extension-guide.md §14](extension-guide.md#14-porting-an-a1111--forge-extension)
   (`gr.Slider` → `ui.Slider`, `gr.Radio` → `ui.Choice(style="radio")`,
   `.change` visibility → `visible_if`, buttons → actions, …). Carry the
   infotext key (`infotext=`), paste logic (`paste=`) and Anima preset field.
3. Update the extension's tests: the argument-order assertion
   (`layout.names` / `values_to_args({})`) and, for ports with a legacy
   snapshot, list the change in `assert_matches_snapshot(..., changes=...)`
   instead of re-recording the snapshot.
4. Add the new strings to `localizations/zh_CN.json` (the localization test
   fails otherwise).
5. If a control moved between scripts or a positional index changed,
   imported Agent Scheduler tasks and API clients are affected: keep the
   order compatible with upstream's `ui()` so `/sdapi` `alwayson_scripts`
   payloads written for Forge Neo still work.

### 2.2 A new or changed setting

Upstream `shared_options.py` uses Gradio classes. Convert while merging:

| upstream | Remi |
|---|---|
| `gr.Checkbox` / no component with a `bool` default | `ui.Toggle` or no component |
| `gr.Slider, {"minimum", "maximum", "step"}` | `ui.Slider, {same}` |
| `gr.Number` | `ui.Number` |
| `gr.Textbox` | `ui.Text` (`{"lines": n}`) |
| `gr.Dropdown, {"choices"}` | `ui.Choice, {"choices"}` (`"allow_custom": True` for an editable dropdown) |
| `gr.Radio, {"choices"}` | `ui.Choice, {"choices", "style": "radio"}` |
| `ui_components.DropdownMulti`, `gr.Dropdown(multiselect=True)` | `ui.MultiChoice` |
| `gr.CheckboxGroup` | `ui.MultiChoice, {"style": "checkboxes"}` |
| `gr.ColorPicker` | `ui.Color` |
| `gr.HTML` | `ui.Html` |

Keep `.info()`, `.link()`, `.needs_restart()`, `.needs_reload_ui()`,
`refresh=` and `onchange=`. Then check the setting in the SPA (Settings page
search), add its label and help to `localizations/zh_CN.json`, and, if the
SPA itself must react to it (a client-side behaviour such as a lightbox
option), add the key to the bootstrap `options` subset in
`remi/server/routes/core.py` and implement it in `webui/src`. Settings that
only configured Gradio (themes, queue, `gradio_*`) have no effect in Remi:
leave them out or keep them hidden.

### 2.3 A change to the generation forms (`modules/ui.py`)

A new core control (a new txt2img/img2img parameter) touches:

1. `modules/processing.py` / `txt2img.py` / `img2img.py`: merged as usual.
2. `remi/core/forms.py`: the param (name = the processing attribute), its
   infotext key, preset field, slot/category, UI-preset patch if any.
3. `remi/core/tabs.py` / `forms.check_core`: cross-param rules, if any.
4. `modules/api/api.py` mapping (`INFOTEXT_API_FIELDS`) when the infotext
   key differs from the API field.
5. The SPA only when the control needs more than the schema renderer (the
   curated generate view frames blocks by content: `pages/generate/curate.ts`).
6. Refresh what the frontend tests and the typed client read, from the
   repository root: `python webui/tools/dump_schemas.py` (the tab schemas),
   `python tools/dump_settings.py` for settings (`tools/check.sh` runs it
   with `--check`), and `python tools/export_openapi.py` followed by
   `pnpm gen:api` in `webui/` when a route or model changed (frontend.md
   "Scripts").

### 2.4 A JavaScript or styling change

Find the SPA feature that replaced the file (§1 table, or
[inventory/core-frontend-features.md](inventory/core-frontend-features.md),
which maps every upstream JS function to its feature) and implement the
behaviour there with a test. Cosmetic upstream CSS changes are ignored: the
SPA has its own design system.

### 2.5 A new built-in extension upstream

Port it like Remi's own ports ([extension-guide.md §14](extension-guide.md#14-porting-an-a1111--forge-extension),
architecture.md "Built-in extension porting rule"): snapshot its `ui()` with
the recording CLI of `tests/remi_port_harness.py` in an environment that
still has Gradio (the pre-rewrite checkout), write `params()`, move tabs to
pages and JavaScript to widgets or core features, add `zh_CN` strings and
the parity / localization / no-Gradio tests, and list it in
[inventory/extensions.md](inventory/extensions.md).

### 2.6 A new upstream model family, sampler or scheduler

Merge. The SPA picks up new samplers, schedulers, checkpoints, modules and UI
presets from `/api/v2/models` and the forms' callable choices; a new UI
preset (`forge_preset`) needs its defaults in `modules_forge/presets.py` and
appears in the model bar automatically.

## 3. Procedure

```bash
git fetch upstream                                   # upstream = https://github.com/Haoming02/sd-webui-forge-classic.git
BASE=$(git merge-base HEAD upstream/neo)             # last absorbed upstream commit (46af880c after the 2026-10-14 sync)
git log --oneline --no-merges "$BASE"..upstream/neo  # what is new

# Review by path group (bash or zsh)
review() { echo "== $1"; shift; git diff --stat "$BASE" upstream/neo -- "$@"; }
review runtime  backend modules_forge/packages modules/processing.py 'modules/sd_*.py' modules/prompt_parser.py \
                modules/images.py modules/masking.py modules/textual_inversion modules/extra_networks.py modules/devices.py
review runner   modules/scripts.py modules/scripts_postprocessing.py modules/script_callbacks.py modules/txt2img.py \
                modules/img2img.py modules/postprocessing.py modules/infotext_utils.py modules/call_queue.py modules/progress.py modules/api
review settings modules/shared_options.py modules_forge/shared_options.py modules/shared_items.py modules/cmd_args.py \
                modules_forge/presets.py modules_forge/main_entry.py
review ui       modules/ui.py 'modules/ui_*.py' modules/gradio_extensions.py modules/shared_gradio_themes.py \
                javascript html style.css script.js modules_forge/forge_canvas
review scripts  scripts modules/processing_scripts extensions-builtin
review install  launch.py webui.py webui.sh webui.bat modules/launch_utils.py modules/initialize.py modules/initialize_util.py requirements.txt

git diff "$BASE" upstream/neo -- modules/ui.py 'modules/ui_*.py' javascript html modules_forge/forge_canvas  # translated, not merged
git diff "$BASE" upstream/neo -G'def ui\(|gr\.' -- scripts modules/processing_scripts extensions-builtin    # ui() changes
git diff --name-status --diff-filter=A "$BASE" upstream/neo                                            # new files
```

Then merge on a branch:

```bash
git switch -c sync/neo-$(date +%Y-%m-%d)
git merge --no-ff --no-commit upstream/neo

git status --short | grep '^DU'     # upstream edited a file Remi deleted (modules/ui*.py, javascript/, …):
                                   # translate the change (§2), then: git rm <file>
git diff --name-only --diff-filter=A --cached | grep -E '^(javascript/|html/|modules/ui|style\.css|script\.js)'
                                   # new upstream UI files: translate, then git rm them
git diff --name-only --diff-filter=U                       # content conflicts: resolve keeping Remi's hooks
rg -n --glob '*.py' '^\s*(import gradio|from gradio)|gradio_rangeslider' .  # must print nothing
```

Resolve, translate, test (`tools/check.sh` runs every Python suite, the
OpenAPI check, i18n coverage and the frontend build, lint and tests; then the
real-model smoke test `tools/smoke_real.py` on a GPU or Apple Silicon
machine), write the sync note, and commit the merge. Because it is a real
merge, `git log HEAD..upstream/neo` is empty afterwards and the next sync
starts from the new merge base.

A selective port (cherry-picking reviewed commits, as on 2026-10-02) is
possible when only a few fixes are wanted; record the commits in the sync
note, because the merge base does not move.

## 4. Checklist

- [ ] `git log $BASE..upstream/neo` reviewed; every commit classified (merge /
      translate / skip) in the sync note.
- [ ] Runtime conflicts resolved without dropping Remi's Anima hooks
      ([remi-architecture-notes.md](../remi-architecture-notes.md) "Seams").
- [ ] No Gradio: the `rg` check above prints nothing; `requirements.txt` and
      `modules/launch_utils.py` install no Gradio.
- [ ] Upstream UI files are deleted again (`DU` conflicts and new UI files
      resolved with `git rm` after translating).
- [ ] Each `ui()` change is in `params()` with the hook's argument order;
      parity tests and `zh_CN` catalogs updated.
- [ ] New settings use `ui.*` widgets, have `zh_CN` labels, and are in the
      bootstrap `options` subset when the SPA reads them.
- [ ] New core params are in `remi/core/forms.py` with infotext keys;
      schema fixtures (`webui/tools/dump_schemas.py`) and the OpenAPI
      document (`tools/export_openapi.py`, then `pnpm gen:api`) regenerated.
- [ ] `/sdapi/v1` request and response models still match upstream's.
- [ ] `tools/check.sh` passes; the real-model smoke test passes (txt2img
      pixel parity with the previous baseline where the change should not
      alter images).
- [ ] `docs/upstream-sync-YYYY-MM-DD.md` written; README model-family list and
      [remi-architecture-notes.md](../remi-architecture-notes.md) updated when
      the change adds a family or a seam.
