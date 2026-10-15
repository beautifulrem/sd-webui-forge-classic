# Inventory C: built-in extensions and bundled scripts

Branch `remi-webui`, Gradio 4.40.0 + `gradio_rangeslider==0.0.8` (pinned in `modules/launch_utils.py:306`).
This file covers all 49 directories in `extensions-builtin/`, plus `scripts/` (7 files) and `modules/processing_scripts/` (6 files).
The goal is to decide which of them run unchanged on a Remi-owned Gradio-compatible package (UI-tree and event-graph recorder, a handler-executing server, and a React SPA renderer) and which need native work.

## 0. Method and how to read the numbers

* **Static scan.** A regex pass over every `.py`, `.js/.ts/.vue` and `.css` file. It counts `gr.<Component>(` calls, layout blocks, event-listener calls, `script_callbacks.on_*` registrations, FastAPI routes, `OptionInfo` and `Script` subclasses. Results are kept under `/tmp/extinv/scan.json`, which is not in the repo.
* **Dynamic build.** Each `scripts/*.py` was imported through the WebUI's own `modules.initialize.imports()`. Two missing third-party modules were stubbed: `huggingface_guess` and `k_diffusion`. `gguf` was side-loaded. Then every `scripts.Script` subclass ran `ui(is_img2img)` inside a fresh `gr.Blocks()` for both tabs, using the real Gradio 4.40 in `/tmp/remi-testenv`. Two numbers come from that run:
  * **blocks** = `len(demo.blocks)`: every component and layout created by `ui()`, including Gradio's implicit `Form` wrappers.
  * **fns** = `len(demo.fns)`: event listeners registered inside `ui()`.

  Listeners added later through `on_after_component`, `on_app_started` or `on_ui_tabs` are **not** counted there. They are listed separately.
* Where the dynamic build could not run, the number is a static estimate marked `~`. Reasons include: the panel needs foreign components that only exist in the full UI (ADetailer, Anima Resolutions); ControlNet needs registered preprocessors; a circular import in `backend.patcher.controlnet` (MultiDiffusion); the UI is an `on_ui_tabs` tab (Agent Scheduler, Prompt Workshop, JoyCaption).
* "per tab" means per txt2img or per img2img panel. AlwaysVisible scripts are built **twice**.

### Adaptation classes

The classes are judged against a **baseline compat contract**. The core UI needs this contract anyway, and it is listed in section 4.

| Class | Meaning | Count |
|---|---|---|
| **1** | Runs on the compat layer **unchanged**. It is either Python-only UI, or its JS uses only stable `elem_id`s plus the WebUI's global JS helpers (`gradioApp`, `onUiLoaded`, `selected_gallery_index`, `switch_to_*`), which the SPA keeps. | **32** |
| **2** | Needs **one native SPA widget or feature** not otherwise required by core (RangeSlider, Plot, HighlightedText). Or its JS relies on Gradio/Svelte DOM internals or Gradio's wire protocol, and must be replaced by a small native feature or a DOM-contract shim. | **13** |
| **3** | Needs a **substantial native rewrite**: its own frontend bundle, deep Gradio-internals surgery, or editor or canvas widgets. | **4** |

Class 3: `agent-scheduler`, `sd-webui-prompt-all-in-one-neo`, `sd-webui-tagcomplete-neo`, `sd_forge_controlnet`.
Class 2: `ADetailer-Neo`, `mobile`, `prompt-bracket-checker`, `repeat-generate-4NEO`, `sd-forge-prompt-workshop`, `sd-lock-img2img-steps`, `sd-webui-detail-daemon`, `sd-webui-joycaption-modified`, `sd-webui-language-diffusion`, `sd-webui-prompt-anchor`, `sd-webui-prompt-queue`, `sd_forge_lora`, `sd_forge_sparse`.
Everything else is class 1. Note: `sd_forge_ipadapter`, `sd_forge_controlllite` and the four `forge_preprocessor*` folders have no UI, but they only matter once ControlNet (class 3) is ported.

ADetailer is often assumed to be class 3, but the code does not support that. It has no canvas, no JS file and no `ForgeCanvas`. What it needs from the compat layer is: `gradio_rangeslider.RangeSlider` (`lib_adetailer/ui.py:8`, `:585`), a callable `gr.State` default (`ui.py:198`), and an extra listener bolted onto the foreign Generate button (`ui.py:207`). That makes it class 2.

---

## 1. Master table (49 extensions)

Column key:
* **UI entry**: S = `Script.ui` (AO = AlwaysVisible, sel = selectable script dropdown); T = `on_ui_tabs`; Set = settings options (count); AC = `on_after_component` / `after_component` injection; XN = extra-networks page; API = routes; BUI = `on_before_ui`; IP = `on_infotext_pasted`; CFG = `on_cfg_denoiser`.
* **blocks/fns**: per tab, as defined in section 0.
* **JS/CSS**: file count and total size.
* **Tests**: count of `tests/test_*.py`.

| # | Extension | Purpose | UI entry | blocks / fns per tab | JS / CSS | Routes | Special widgets | Tests | Class |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ADetailer-Neo | YOLO/mediapipe detect, then inpaint faces/hands after generation | S-AO (both tabs), Set 12, AC (Generate buttons), BUI (XYZ axes), API | ~ (41 inputs + ~25 layout) × `ad_max_models` (default 4) ≈ 265 / ≈ 170 (39 `.change` + 1 Generate `.click` + misc per model) | none | 3 (`/adetailer/v1/*`) | **RangeSlider**, InputAccordion, `gr.State(callable)` | 1 | **2** |
| 2 | agent-scheduler | Persistent task queue/history (SQLite), Enqueue button, AG Grid UI | S-AO (adds Enqueue row via AC), T "Agent Scheduler", Set 9, API, `on_image_saved`, `on_before_reload` | tab: 36 comps / 19 layouts / ~12 listeners; +3 comps & 2 listeners per gen tab | 14 files / 1.33 MB (`javascript/agent-scheduler.iife.js` 1,327,450 B, React+AG Grid+zustand) / 3 files 312 KB (`style.css` 310,120 B) | 34 (`/agent-scheduler/v1/*`, 13 deprecated aliases) + wraps `/file=` routes | Gallery ×2, File, HTML grid mount points, hidden Textboxes | 4 | **3** |
| 3 | anima-artist-scheduled-mixer | Per-step artist-tag blending schedule for Anima | S-AO, Set 1 (language), AC (Shift sliders by elem_id) | **828 / 147** (32 rows × 10 controls × base+hires) | none | 0 | `gr.on(... trigger_mode="always_last")` over about 660 triggers; inline `<style>` HTML | 7 | **1** (perf hotspot) |
| 4 | anima-lora-layer-weight | Per-block/module LoRA weights for Anima | S-AO, Set 1 | 50 / 3 | none | 0 | Tabs/TabItem | 3 | **1** |
| 5 | anima-lora-stage-scheduler | Stage-wise LoRA strength schedules plus templates | S-AO, Set 1, IP, CFG | 102 / 20 | none | 0 | `_js` clipboard writer (`scripts/anima_lora_stage_scheduler.py:309`), `.edit/.release/.input` | 4 (+`test_ui.py`) | **1** |
| 6 | extra-options-section | Exposes chosen settings inside txt2img/img2img | S-AO (section `extra_options`), Set 4 (import-time `options_section`) | 1 + N settings comps / 1 (`interface.load`) | none | 0 | **nested `gr.Blocks()` with `.load`** (`scripts/extra_options_section.py:35,67`) | 0 | **1** |
| 7 | forge_legacy_preprocessors | about 42 legacy ControlNet annotators (depth, openpose, lineart and so on; 667 py files, 115k LOC vendored) | none (registers into CN) | 0 / 0 | none | 0 | none | 0 (3 vendored `test_*` files are library code) | **1** (via CN) |
| 8 | forge_preprocessor_inpaint | Inpaint, inpaint_only, LaMa, NoobAI-XL inpaint preprocessors | none | 0 | none | 0 | none | 0 | **1** (via CN) |
| 9 | forge_preprocessor_reference | reference_only / adain / adain+attn | none | 0 | none | 0 | none | 0 | **1** (via CN) |
| 10 | forge_preprocessor_tile | tile_resample / colorfix / colorfix+sharp | none | 0 | none | 0 | none | 0 | **1** (via CN) |
| 11 | mobile | Moves Generate box above results on narrow screens | JS only | none | `javascript/mobile.js` 1,240 B | 0 | none | 0 | **2** (trivial) |
| 12 | prompt-bracket-checker | Marks token counter red on unbalanced brackets | JS only | none | `javascript/prompt-bracket-checker.js` 2,678 B | 0 | none | 0 | **2** (trivial) |
| 13 | repeat-generate-4NEO | Endless "repeat Generate" with stop button | JS only | none | `javascript/neo-repeat-generate.js` 19,003 B; `style.css` 1,308 B; 3 webp icons | 0 (reads `/prompt-queue/state`) | none | 0 | **2** (protocol rewrite) |
| 14 | sd-forge-anima-prompt-rescale | Anima prompt-embedding rescale | S-AO (section `accordions`) | 20-21 / 1 | none | 0 | none | 1 (`tests_rescale.py`) | **1** |
| 15 | sd-forge-anima-resolutions | Anima resolution presets/randomizer, prompt history, auto-fill of img2img size | S-AO (txt2img only, section `dimensions`), AC (10 tracked ids, global hook) | ~ 12 / ~6 in `ui()`, plus 2-3 listeners on foreign buttons (Generate, Send-to) | none | 0 | `.then` + `_js` using `selected_gallery_index()` (`scripts/anima_resolutions.py:760-785`) | 0 | **1** |
| 16 | sd-forge-dynamic-shift | Resolution-dependent flow shift | S-AO (section `sampler`), AC (sampler/scheduler), BUI (XYZ) | 19 / 5 (txt2img); 16 / 4 (img2img) | none | 0 | InputAccordion | 0 | **1** |
| 17 | sd-forge-negpip | NegPiP negative-weight prompting | S-AO with **no UI**, CFG | 0 / 0 | none | 0 | none | 5 | **1** |
| 18 | sd-forge-prompt-blacklist | Strips blacklisted tags from prompt | S-AO (section `prompt`), AC (prompt boxes) | 12 / 2 (outputs into foreign prompt textbox) | none | 0 | none | 1 | **1** |
| 19 | sd-forge-prompt-workshop | Anima tag-order prompt builder plus Booru grabber, hand-off to JoyCaption | T "Prompt Workshop" | static 74 comps (25 Textbox, 20 Button, 7 State, 1 Image) / 13 | `javascript/prompt_workshop.js` 14,469 B | 0 (outbound HTTP to boorus) | Image preview, `register_paste_params_button` to custom tab | 0 | **2** |
| 20 | sd-lock-img2img-steps | Lock img2img Steps against paste / Send-to | S-AO (img2img only, section `sampler`) | 2 / 0 | `javascript/lock_steps.js` 5,020 B | 0 | none | 0 | **2** |
| 21 | sd-postprocess-suite | 27-stage post-processing pipeline (LUT, grain, color) plus "Apply PP, send to img2img" | S-AO (section `accordions`), AC (`txt2img_gallery`) | **322 / 31** (27 InputAccordions, 64 Slider, 35 Number, 7 ColorPicker, 1 File) | `javascript/pp_send.js` 978 B | 0 | ColorPicker, File (.cube LUT), `js=` preprocessor on Gallery value | 0 | **1** (needs Gallery JS value contract) |
| 22 | sd-webui-detail-daemon | Detail Daemon sigma manipulation, 6 daemons | S-AO, Set 2, IP, CFG | **229 / 109** (12 Plot, 48 Slider, 18 Number, 6 Tab) | `style.css` 1,607 B | 0 | **gr.Plot (matplotlib)** ×12/tab, JS-only `_js="(x)=>x"` mirrors | 0 | **2** |
| 23 | sd-webui-joycaption-modified | JoyCaption VLM captioning tab, "Send to JoyCaption" in PNG Info | T "JoyCaption", AC (injects button into PNG-Info tab) | static 48 comps / 26 | `javascript/joycaption.js` 8,861 B | 0 | Image upload (`.upload/.clear`), `gr.Progress(track_tqdm=True)` ×3 | 0 | **2** |
| 24 | sd-webui-language-diffusion | Bundled UI translations (10 locales) plus flag icons on the localization dropdown | import-time `OptionInfo` patching of `localization` | 0 | 2 files 19 KB (`forge_language_flags.js` 16,158 B, `hints_i18n_patch.js` 2,908 B) / `style.css` 2,489 B; 10 JSON locales | 0 | Dropdown DOM decoration | 0 | **2** |
| 25 | sd-webui-prompt-all-in-one-neo | Vue 3 tag editor over prompt boxes: translate, history, favorites, presets, GPT | **No Python UI**; `on_app_started` (API), Set 1 | 0 | 36 files: built `javascript/main.entry.js` 707,648 B plus Vue sources / 48 files 813 KB (`style.css` 6,280 B loaded; `styles/*.min.css`) | **46** (`/physton_prompt/*`) | entire Vue app mounted into `gradioApp()` | 0 (7 manual scripts in `tests/`) | **3** |
| 26 | sd-webui-prompt-anchor | Second "anchor" prompt prepended at generation | S-AO (section `prompt`), Set 2 | 8 / 0 | `javascript/prompt_anchor.js` 10,010 B / `style.css` 3,742 B | 0 | none (JS reparents DOM) | 0 | **2** |
| 27 | sd-webui-prompt-queue | Client-driven prompt queue (fills prompt, clicks Generate) | T "Queue" (one `gr.HTML` root), API | 1 HTML / 0 | `javascript/prompt_queue.js` 17,646 B / `style.css` 5,819 B | 8 (`/prompt-queue/*`) | none | 0 | **2** |
| 28 | sd-webui-tagcomplete-neo | Booru tag autocomplete in prompt textareas (tags, LoRA, embeddings, wildcards, chants) | **No Python UI**; Set **60**, `on_app_started` (API), `on_model_loaded` | 0 | 15 files 140 KB (`tagAutocomplete.js` 77,077 B) | **16** (`/tacapi/v1/*`) + reads CSVs via Gradio `/file=` | none | 0 | **3** |
| 29 | sd_forge_anima_dave | DAVE attention patch for Anima | S-AO (`AnimaPatchPanel`) | 9 / 1 | none | 0 | InputAccordion | 2 | **1** |
| 30 | sd_forge_anima_freefuse | FreeFuse multi-subject LoRA fusion | S-AO | 42 / 1 | none | 0 | InputAccordion | 7 | **1** |
| 31 | sd_forge_anima_guidance | Anima guidance suite (NAG, SMC, DCW, skim, modulation) | S-AO, CFG | 105 / 1 | none | 0 | InputAccordion | 10 | **1** |
| 32 | sd_forge_anima_hires_guard | Hires-fix guard for Anima | S-AO (txt2img only) | 14 / 1 | none | 0 | InputAccordion | 3 | **1** |
| 33 | sd_forge_anima_presets | One-click preset across native and extension controls | S-AO (`after_component` collects native controls) | 7 / 2 (Apply outputs into other scripts' components) | none | 0 | none | 1 | **1** |
| 34 | sd_forge_anima_regional | Regional prompting for Anima | S-AO | 47 / 1 | none | 0 | InputAccordion | 5 | **1** |
| 35 | sd_forge_anima_xattn_boost | Cross-attention boost | S-AO (`AnimaPatchPanel`) | 9 / 1 | none | 0 | InputAccordion | 2 | **1** |
| 36 | sd_forge_compile | `torch.compile` of Anima blocks | S-AO (visibility gated) | 4 / 0 | none | 0 | none | 3 | **1** |
| 37 | sd_forge_controlllite | ControlNet-LLLite patchers (SD + Anima) | none (`add_supported_control_model`) | 0 | none | 0 | none | 0 | **1** (via CN) |
| 38 | sd_forge_controlnet | ControlNet integrated (units, preprocessors, editors) | S-AO (both tabs), Set 7, AC (22 A1111 ids plus a deferred `.render()`), IP, `on_before_reload`, API (with `--api`) | ~ 80 blocks / ~25 listeners per unit × `control_net_unit_count` (default 3), about 240 / 75 per tab, plus cross-wiring to 22 native components | 2 files 6 KB (`active_units.js` 4,305 B, `openpose_editor.js` 1,870 B) / `style.css` 2,261 B | 4 (`/controlnet/*`) | **ForgeCanvas ×4/unit** (input, generated, mask, scribble canvas), **RangeSlider**, `gr.State(UiControlNetUnit)`, HTML onclick links, hidden Textbox pose JSON | 0 | **3** |
| 39 | sd_forge_image_stitch | Multi-reference latent stitching | S-AO | 15 / 6 | `style.css` 123 B | 0 | **Gallery (interactive, `.select` with `SelectData`)**, Image (paste) | 0 | **1** |
| 40 | sd_forge_ipadapter | IP-Adapter / InstantID preprocessors and patcher | none | 0 | none | 0 | none | 0 | **1** (via CN) |
| 41 | sd_forge_lora | LoRA networks, extra-networks page, metadata editor | `before_ui` (`register_page`), import-time options (4), IP ×2, API (with `--api`) | editor: 7 comps / 4 listeners (rendered inside core extra-networks UI) | none | 2 (`/sdapi/v1/loras`, `/sdapi/v1/refresh-loras`) | **HighlightedText + `.select(SelectData)`**, TextArea | 0 | **2** |
| 42 | sd_forge_lora_schedule | LoRA strength schedule via `on_cfg_denoiser` | S-AO with no UI | 0 | none | 0 | none | 0 | **1** |
| 43 | sd_forge_multidiffusion | Tiled MultiDiffusion | S-AO | ~ 9 / 1 | none | 0 | ToolButton with `_js="currentImg2imgSourceResolution"` (`scripts/forge_multidiffusion.py:30-32`) | 0 | **1** |
| 44 | sd_forge_neveroom | Never-OOM UNet/VAE offload toggles | S-AO | 4 / 0 | none | 0 | none | 0 | **1** |
| 45 | sd_forge_pid | PiD integrated | S-AO | 15 / 1 | none | 0 | InputAccordion | 2 | **1** |
| 46 | sd_forge_radial | Radial sparse attention | S-AO (`ScriptBuiltinUI`, env-gated) | 15 / 1 | none | 0 | InputAccordion | 0 | **1** |
| 47 | sd_forge_sparse | Sparse attention with timestep range | S-AO | 14 / 1 | none | 0 | **RangeSlider** (`scripts/sparse.py:12,132`) | 0 | **2** |
| 48 | sd_forge_spectrum | Spectrum step forecasting (feature caching) | S-AO | 37 / 4 | none | 0 | presets via `gr.update`/`gr.skip` lists | 8 | **1** |
| 49 | soft-inpainting | Soft inpainting (A1111 port) | S-AO (img2img only, section `inpaint`) | 22 / 1 | none | 0 | InputAccordion, 9 Markdown | 0 | **1** |

**Totals.**
* Python LOC excluding `forge_legacy_preprocessors`: 67,988. Including it: 182,960.
* JS/TS/Vue: 78 files, 2.77 MB. About 75% of that is two prebuilt bundles: agent-scheduler at 1.33 MB and PAIO at 0.71 MB. Their TS/Vue sources add another 0.5 MB.
* CSS: 58 files, 1.17 MB.
* Custom HTTP routes: **113** (ADetailer 3, agent-scheduler 34, PAIO 46, prompt-queue 8, tagcomplete 16, ControlNet 4, LoRA 2).
* AlwaysVisible scripts with UI: 31. Two more AlwaysVisible scripts have no UI: negpip and lora_schedule.
* `on_ui_tabs` tabs: 4 (Agent Scheduler, Prompt Workshop, JoyCaption, Queue).
* Extensions with `tests/test_*.py`: 17 (68 test files).

---

## 2. Per-extension detail and concrete work

### Class 3

#### agent-scheduler (class 3)

* **Python UI.**
  * `scripts/task_scheduler.py:411-620` builds the tab. It has Pause/Resume/Reload/Clear/Export/Import/Search buttons, two `gr.Gallery` (`agent_scheduler_current_task_images`, `agent_scheduler_history_gallery`), a `gr.File` download, `gr.HTML` placeholders where the TS bundle mounts **AG Grid**, and hidden Textboxes (`agent_scheduler_history_selected_task/_image`).
  * Events with `_js`: `status.change(_js="agent_scheduler_status_filter_changed")` at `:579-581`; `save.click(_js="(x, y, z) => [x, y, selected_gallery_index()]")` at `:584-596`.
* **Gradio-internals surgery.** No compat layer will reproduce this generically. The script has to be rewritten against a Remi API.
  * `Script.after_component` (`:91-118`) walks `component.parent` up to `txt2img_generate_box` / `*_actions_column` / `*_toprow`. It **pops the last child** (`component.parent.children.pop()`) and `parent.add(self.enqueue_row)`, which re-parents a freshly built Row into the core layout.
  * `bind_enqueue_button` (`:145-203`) runs at `on_app_started`, i.e. after `Blocks` is closed. It:
    * reads `root.fns` and Gradio config dependencies through `agent_scheduler/helpers.py:70-140` (`get_default_config_dependencies`, `compare_components_with_ids`, `get_components_by_ids`, using `_id`, `targets`, `BlockContext.children`);
    * finds the Generate click dependency with the most inputs;
    * detects ControlNet's `gr.State` whose `value` is a `UiControlNetUnit`;
    * re-enters `with root:` to register new `.click`/`.change` listeners **after launch**. The click carries `_js="(...args) => submit_enqueue(...args)"`.
  * `on_app_started` (`:795-829`): with `queue_ui_placement == append_to_main` it does `with block: with block.children[1]: on_ui_tab()`. That builds a whole tab into a live, running Blocks.
  * `agent_scheduler/api.py:14` `from gradio.routes import App`.
  * `agent_scheduler/file_guard.py:16-20` wraps Gradio's `/file=`, `/file/` and `/gradio_api/file=` routes and uses `gradio.utils.abspath` (`:41`).
  * `task_runner.py:152,314` takes `gr.Request` and serialises `request.username`.
* **JS (`ui/src/extension/index.ts`, built into the IIFE).**
  * 33 `gradioApp()` queries.
  * Overrides `window.randomId` (`:296-343`, `:466`, `:474`) so the core `submit()` produces agent-scheduler task ids.
  * Defines `window.submit_enqueue[_img2img]` (`:365`, `:387`), which call core `create_submit_args`, `requestProgress` and `get_tab_index`.
  * Monkey-patches `window.modalSaveImage` (`:494-498`) and uses `MutationObserver` on tab elements (`:547-550`).
  * Builds AG Grid in `#agent_scheduler_*` containers (`:937`, `:1165`).
  * CSS: 310 KB of Tailwind and AG Grid theme, plus 3 rules on `.gradio-row`.
* **Work.** Replace the extension's frontend with a native SPA "Queue" module:
  * a React data grid (AG Grid can be kept, but as a first-class SPA dependency);
  * **an engine-level "enqueue current generate request" API**, so the Python side no longer reverse-engineers the Generate dependency;
  * a native "Enqueue" slot next to Generate (an SPA extension point, instead of `children.pop()`).

  Keep `/agent-scheduler/v1/*` (34 routes) as-is. Drop `file_guard` if the new server's static-file route already enforces an allow-list, or port it to the new route. The task replay path (`task_runner.py:580`, `wrap_gradio_call(txt2img|img2img)`) depends on the positional `fn_inputs` of the Generate dependency. Remi must expose an **ordered input-signature descriptor** for txt2img/img2img to replace this. 4 tests (`test_net`, `test_request_guard`, `test_signing`, `test_file_guard`) are FastAPI/unit tests and do not need Gradio (`test_file_guard` imports `gradio.utils`).

#### sd-webui-prompt-all-in-one-neo (class 3)

* **No Python UI.**
  * `scripts/on_app_started.py:69-450` registers 46 routes (`/physton_prompt/*`). They cover data store, history, favorites, translation (21 translator backends, plus `translators/server.py` at 300 KB), CSV, styles, extra-networks list, OpenAI generation, mbart50 and quality presets.
  * `guard_routes(app, ("/physton_prompt/",))` at `:493-502`.
  * 1 setting (`paio_neo_civitai_api_key`, `gr.Textbox type=password`).
* **JS.**
  * `javascript/main.entry.js` (708 KB, built from `src/` Vue 3 with axios, tippy, toastr and vue3-colorpicker).
  * `src/src/main.js` mounts `#physton-prompt-all-in-one` into `gradioApp()` on `onUiLoaded`.
  * `App.vue:211-300` targets `#txt2img_prompt`, `#txt2img_neg_prompt`, `#img2img_prompt`, `#img2img_neg_prompt`.
  * `App.vue:902-911` finds the `<textarea>`, inserts its editor **after `$prompt.parentElement.parentElement`**, and hides the original prompt (`:555`).
  * It writes back with `textarea.value = ...; dispatchEvent(new Event('input'))` (`:1119-1120`) and core `updateInput`.
  * Other couplings:
    * the hard-coded Svelte class `svelte-4xt1ch`, ×4;
    * `#tabs`, `#paste`, `.gradio-container`;
    * extra-network cards (`#txt2img_checkpoints_cards .card`);
    * tagcomplete DOM (`.autocompleteResults`, `.acListItem`);
    * `get_uiCurrentTabContent`, `switch_to_light/dark_theme`, `opts.sd_model_checkpoint`.
  * CSS: about 813 KB across 48 files. Only `style.css` (6 KB) is auto-loaded. `styles/main.min.css` (237 KB) and `animate.min.css` (80 KB) are loaded by the bundle through `/file=`, which needs the `--gradio-allowed-path` warning at `on_app_started.py:60-66`.
* **Work.** Re-host it as a native SPA "prompt editor" plugin. Two options:
  * (a) Port the Vue components to React. This is substantial: `phystonPrompt.vue` is 86 KB and `App.vue` 61 KB.
  * (b) Keep the Vue bundle but give it a **stable prompt-editor mount API**: `registerPromptDecorator(promptId, mountFn)` plus a typed `setPromptValue(id, text)`, instead of `parentElement.parentElement` and `svelte-*`.

  Either way, keep the 46 routes unchanged and serve its CSS through a whitelisted static route.

#### sd-webui-tagcomplete-neo (class 3)

* **Python** (`scripts/tag_autocomplete_helper.py`):
  * **60 settings options** (`:618-781`). They use `gr.Dropdown` with lambda choices plus `refresh=`, `gr.Slider`, `gr.Checkbox`, `gr.Textbox`, `gr.HTML` with `refresh=refresh_temp_files` (`:779`), and `.info()` / `.needs_reload_ui()`. They render through core `ui_settings`.
  * 16 routes `/tacapi/v1/*` (`:848-1169`), guarded at `:1175-1180`.
  * `on_model_loaded`.
  * Writes `tmp/tagAutocompletePath.txt`, which the JS reads.
* **JS** (15 files, 140 KB):
  * `_textAreas.js:9-115` attaches to `#txt2img_prompt textarea`, `#…_neg_prompt textarea`, the style editors, and also to **other extensions' fields**: ADetailer `[id^=script_txt2img_adetailer_ad_prompt] textarea`, MultiDiffusion `[id^=MD-t2i]…`, Prompt Workshop `#pw_* textarea` (15 ids), and third-party tabs.
  * Reads tag CSVs through **Gradio's `file=` route**: `_utils.js:44` `fetch(\`file=${filePath}\`)`, `tagAutocomplete.js:208-235,1901-1903`.
  * Uses `opts[...]` (about 25 `tac_*` keys), `onUiUpdate` and `updateInput` (×5) to push edits into Gradio state.
* **Work.**
  * Build a native SPA **autocomplete provider API** on prompt-like textareas: a component-level `autocomplete` capability, keyed by component kind or `elem_id` pattern.
  * Port `tagAutocomplete.js` to it, or keep it and guarantee the DOM contract: `#elem_id textarea` exists, `input` events propagate to SPA state, and `opts` / `onUiUpdate` exist.
  * The server must provide a `file=` compatible static route (with the allowed-path policy) or the JS must move to `/tacapi/v1/` file endpoints.
  * The 60 options need a robust settings renderer: refreshable choices, the `HTML` option type, `.info`.

#### sd_forge_controlnet (class 3)

* **Python UI** (`lib_controlnet/controlnet_ui/controlnet_ui_group.py`, 38 KB). Per unit:
  * `ForgeCanvas` for the input image (`:235`), the generated preview (`:241`) and the mask (`:268`), plus a scribble `canvas_editor.py:23`;
  * `RangeSlider` timestep range (`:405`) with 2 `gr.State`s;
  * `gr.State(self.default_unit)` holding a **dataclass `UiControlNetUnit`** (`:496`);
  * OpenPose editor: a hidden Textbox `cnet-pose-json`, plus `gr.HTML` links whose inline `onclick` runs JS (`openpose_editor.py:48-56`, `controlnet_ui_group.py:246-263`);
  * Radio/Dropdown/Slider/Checkbox; `.blur` on batch dir textboxes (`:851`).
* **Cross-UI wiring.**
  * `ControlNetUiGroup.on_after_component` (`:946-953`) records **22 native components by elem_id** (`a1111_context`, `:36-100`: width/height sliders, img2img tabs, inpaint upload tab, batch dirs, etc.) and wires callbacks once all are seen.
  * A class-level `global_batch_input_dir = gr.Textbox(...)` is created **outside any Blocks** at import (`:123`) and later `.render()`ed into the img2img batch tab (`:949`).
  * `on_before_reload` resets state. Agent-scheduler sniffs CN's `gr.State` type.
* **ForgeCanvas** (core, `modules_forge/forge_canvas/canvas.py:83-135`) is `gr.HTML` plus two `LogicalImage(gr.Textbox)`. The subclass overrides `preprocess`/`postprocess`/`get_block_name` and is driven by `canvas.js`.
* **JS.**
  * `active_units.js` walks `.tab-nav button`, `button.label-wrap span`, `.cnet-unit-enabled input`, `input[type=file]` and `#paste`/`#txt2img_tab` to show unit badges.
  * `openpose_editor.js` uses `.cnet-pose-json textarea`, `.cnet-download-pose a` and `updateInput`.
* **Work.**
  * A native SPA **ForgeCanvas** widget (also needed by core img2img/inpaint) with the LogicalImage value contract (base64 PNG fg/bg).
  * A native **RangeSlider**.
  * Support for `.render()` of a pre-created component and for components created outside a Blocks context.
  * `gr.State` holding arbitrary Python objects, which must stay server-side.
  * A replacement for `active_units.js` (unit-tab badges) and for the OpenPose upload/download links, as SPA features.
  * Ideally, a native "Control units" panel.

  The 4 routes (`lib_controlnet/api.py:42-74`) stay. The 6 preprocessor/patcher extensions (#7-10, #37, #40) need nothing beyond ControlNet working.

### Class 2

* **ADetailer-Neo.**
  * `RangeSlider` for the CN guidance start/end (`lib_adetailer/ui.py:585`).
  * The per-model `gr.State(lambda: state_init(w))` (`ui.py:198`) is updated by `.change` on every widget (`:200-203`, 39 attrs per model).
  * `target_button.click(fn=on_generate, inputs=all_inputs, outputs=state, queue=False)` attaches to the **core Generate button** obtained through `on_after_component` (`scripts/adetailer.py:807-815`, `ui.py:205-207`). This listener must fire, and its State write must land, **before** the generate fn reads the State. Gradio gets that right through `queue=False` ordering; Remi must keep it.
  * Infotext fields include callables (`ui.py:209-219`).
  * BUI registers XYZ axes (`:1074-1081`). 3 routes (`:1084-1095`). Option `ad_max_models` (needs reload).
  * Work: a native RangeSlider, plus ordered same-trigger listener execution.
* **mobile.** `javascript/mobile.js` moves `#{tab}_generate_box` into `#{tab}_results` when `#{tab}_results.offsetLeft === 0`. Work: make it SPA responsive layout (CSS/React) and drop the JS.
* **prompt-bracket-checker.** Uses `#{id} > label > textarea` (a Gradio Textbox DOM shape), `#{tab}_token_counter` and core `onEdit` (`javascript/prompt-bracket-checker.js:63-74`). Work: make it a native prompt-textbox validation feature, or keep the `> label > textarea` DOM shape.
* **repeat-generate-4NEO.**
  * Reads `window.gradio_config.components` / `.dependencies` (`:68`, `:84`) to find the Generate `fnIndex`.
  * Finds `session_hash` in `window.__gradio_client__` (`:110-126`).
  * **POSTs `./queue/join` with `fn_index`/`session_hash` directly** (`:230-242`).
  * Wraps `window.submit*` and `window.requestProgress` (`:396-418`).

  Work: replace it with a native "repeat generation" toggle that calls Remi's generate endpoint in a loop. Do not emulate the Gradio queue protocol for it.
* **sd-forge-prompt-workshop.**
  * JS reorders the tab button through `#tabs > .tab-nav button` (`javascript/prompt_workshop.js:45`, `:118`).
  * Writes `#txt2img_prompt textarea` and dispatches `input`.
  * `window.pw_trigger_jc_handoff` (`:299`) clicks a hidden paste button.
  * Python registers a `ParamBinding` to tab `joycaption` (`scripts/prompt_workshop.py:2812-2822`).
  * `_js` global function names (`:2784`, `:2801`).
  * 7 `gr.State`; `gr.Image` preview (`:2537`).

  Work: give the SPA tab-ordering metadata, a typed `setComponentValue(elem_id, value)` JS API, and Send-to bindings to custom tabs. Alternatively, keep the `.tab-nav button` DOM contract.
* **sd-lock-img2img-steps.** JS (`javascript/lock_steps.js:19-28,47-48,85-94`) watches `#img2img_steps input[type=number|range]`, restores the value after paste, and dispatches `input`/`change` to push it back into Gradio. Work: a native "lock" flag on the steps control, honoured by the paste/Send-to code path. Or the SPA must treat synthetic DOM `input` events as value changes.
* **sd-webui-detail-daemon.**
  * `gr.Plot` with matplotlib figures (`scripts/detail_daemon.py:96`, `:120`): 2 per daemon, 12 per tab.
  * JS-only mirror listeners `slider.release(None, slider, number, _js="(x) => x")` (`:132-139`).
  * `.release` updates plots. `isinstance(vis_arg, gr.components.Slider)` (`:143`).
  * CSS styles `.tab-nav button` inside `.detail-daemon-tab-group`.

  Work: a native **Plot** widget (render matplotlib to PNG/SVG server-side; the simplest Plot payload), support for `fn=None` client-only `js` listeners, and the `release` event.
* **sd-webui-joycaption-modified.**
  * Builds a tab with `gr.Image` (`scripts/joycaption.py:578`; `.upload` and `.clear` at `:646-651`) and `gr.Progress(track_tqdm=True)` (`:163`, `:215`, `:289`).
  * `on_after_component` **creates a `gr.Button` inside the still-open PNG-Info tab context** when it sees `pnginfo_image` and `html_info_pnginfo` (`:853-905`).
  * Attaches 2 more `.click` listeners on that button, one with `_js="joycaption_switch_to_tab"` (`:797-815`).
  * JS: `#tabs > .tab-nav button` tab switching and textarea writes.

  Work: tqdm progress streaming, a custom-tab switch API, and correct "current context" insertion semantics for components created inside `on_after_component`.
* **sd-webui-language-diffusion.**
  * Mutates the core `localization` `OptionInfo` (`scripts/language_diffusion_setup.py:79-175`): section `(None, …)`, patches `component_args`, pins to quicksettings, and re-scans localizations.
  * JS decorates Gradio's dropdown internals (`ul.options li.item`, `[role=option]`, `forge_language_flags.js:192-205`).
  * `hints_i18n_patch.js` patches core `localization.js` text replacement.

  Work: native i18n (SPA translation catalog from the 10 JSON files) plus icon support in the Dropdown option renderer.
* **sd-webui-prompt-anchor.**
  * JS moves `#prompt_anchor_group_{tab}` above `#{tab}_prompt` (`javascript/prompt_anchor.js:90-93`).
  * Persists to localStorage and dispatches `input`/`change` (`:82-84`).
  * Python is a plain AO script in section `prompt` (`scripts/prompt_anchor.py:143-151`).

  Work: an SPA "prompt section" slot ordering, so the script can declare placement above the prompt, plus native persisted-value support. Or keep the DOM and synthetic-event contract.
* **sd-webui-prompt-queue.**
  * Python: 8 routes (`scripts/prompt_queue.py:357-405`) and a tab that is a single `gr.HTML` root (`:412-420`).
  * JS renders the whole queue UI into `#prompt-queue-root`.
  * Injects `#{tab}_pq_add` next to `#{tab}_generate` (`javascript/prompt_queue.js:105-107`).
  * Adds a badge to `#tabs > .tab-nav > button` (`:158-185`).
  * **Drives generation by filling textareas (`updateInput` ×5) and `.click()`ing `#{tab}_generate`** (`:368-374`, interrupt at `:327`).

  Work: a native "add to queue" button slot, a programmatic `triggerGenerate(tab, overrides)` API, and tab badges. It can run unchanged if the SPA keeps `#{tab}_generate` clickable and `updateInput` semantics.
* **sd_forge_lora.**
  * Registers `ExtraNetworksPageLora` (`scripts/lora_script.py:52-54`); the cards come from core `ui_extra_networks` plus core JS.
  * The metadata editor uses `gr.HighlightedText` plus `.select(evt: gr.SelectData)` (`ui_edit_user_metadata.py:171`, `:184-194`) and `gr.TextArea`.
  * 4 import-time options (`:13-20`), 2 routes with `--api` (`:37-41`).

  Work: a native **HighlightedText** with a select event. The extra-networks card browser is a core SPA feature (see the core-frontend inventory).
* **sd_forge_sparse.** `RangeSlider(value=(0.15, 0.85))` (`scripts/sparse.py:132`). Work: a native RangeSlider (shared with ADetailer and ControlNet).

### Class 1: notes that still impose compat requirements

* **anima-artist-scheduled-mixer** (`scripts/anima_artist_scheduled_mixer.py`).
  * `MAX_ARTIST_ROWS = 32` (`:93`) times 10 controls times 2 (base/hires) gives **828 blocks / 147 listeners per tab**, the single largest panel (1,656 blocks across both tabs). This needs `visible=False` virtualisation in the SPA and efficient `gr.update` batches: `_resize_row_components` returns about 320 updates.
  * `Script.__init__` uses `self.on_after_component(callback, elem_id=...)` (`:1514-1523`).
  * `gr.on(triggers=[...about 660 triggers...], trigger_mode="always_last", api_name=False, show_api=False)` (`:1844-1848`), `.then` chains, and `.input/.edit/.blur/.submit`.
  * `gr.HTML("<style>…")` injection (`:1545`); a test asserts no `<script>`.
  * `tests/test_ui_wiring.py` introspects `demo.blocks`, `demo.config["dependencies"][…]["targets"]` and `trigger_mode`. **The compat package must expose a Gradio-shaped `config`.**
* **anima-lora-stage-scheduler.** `_js` clipboard (`:309`), 35 `gr.update`, `.edit`/`.release`. `tests/test_ui.py` builds the UI.
* **extra-options-section.** Nested `gr.Blocks()` inside `ui()` with `interface.load(...)` (`scripts/extra_options_section.py:35`, `:67`), and a conditional context manager (Accordion **or** Group). Uses `ui_settings.create_setting_component`.
* **sd-forge-anima-resolutions.**
  * Embedded in section `dimensions`. Builds against `txt2img_width`/`height` captured through a global `on_after_component` (`scripts/anima_resolutions.py:380-392`, `:828-848`); `ui()` raises if they are missing.
  * Adds listeners on core `txt2img_generate` and `txt2img_send_to_img2img/inpaint`.
  * A no-op `.click` followed by `.then(_js=…selected_gallery_index…)` (`:760-785`) relies on `.then` waiting for parallel same-trigger listeners.
  * Detects UI rebuild by object identity.
* **sd-forge-prompt-blacklist / sd_forge_anima_presets.** Outputs into components owned by core or other scripts (prompt boxes; native sampler/steps/cfg/refiner/hires plus every registered Anima preset control). Needs cross-scope outputs; `do_not_save_to_config` attribute on components.
* **sd-postprocess-suite.**
  * `js="pp_extract_and_pass"` receives the **txt2img Gallery value in JS** and slices it by `selected_gallery_index()` (`javascript/pp_send.js`).
  * Python then calls core `image_from_url_text` on it (`scripts/postprocess_suite.py:150-167`) and chains `.then(fn=None, js="switch_to_img2img")`.
  * Requires the Gallery JS-side value shape to match what `image_from_url_text` accepts.
* **sd_forge_image_stitch.** Interactive `gr.Gallery` value as `list[tuple[PIL, str]]`, `.select` with `SelectData.index`, `gr.Image` paste, `gr.skip()` (`scripts/image_stitch.py:40-90`).
* **sd-forge-dynamic-shift / ADetailer / detail-daemon.** Register XYZ-grid axes in `on_before_ui` by importing `scripts/xyz_grid.py`'s module object (`dynamic_shift.py:285`).
* **sd_forge_multidiffusion.** `_js="currentImg2imgSourceResolution"` is a core global (`scripts/forge_multidiffusion.py:30-32`).
* **Anima panels** (dave, xattn_boost, freefuse, guidance, hires_guard, regional, spectrum, compile, pid, neveroom, radial, soft-inpainting, prompt-rescale, layer-weight). These are plain Python: InputAccordion, Slider, Dropdown, Textbox, Markdown, `gr.update`/`gr.skip`, `infotext_fields` with `PasteField`. They also use shared core helpers with Gradio calls: `modules/anima_lora_ui.py:51-66` (`intro_block` = `gr.Markdown`), `modules/anima_feature_conflicts.py:375-440` (exclusive-feature wiring across scripts, `gr.update`), `modules/anima_presets.py:280-310` (preset control registry).
* **No-UI**: negpip, lora_schedule (AO, `ui()` returns nothing), controlllite, ipadapter, `forge_preprocessor_*`, `forge_legacy_preprocessors`.

---

## 3. scripts/ and modules/processing_scripts/

| File | Kind | Purpose | blocks / fns per tab | Special | Class |
|---|---|---|---|---|---|
| `scripts/xyz_grid.py` (925 L) | sel Script (both tabs) | X/Y/Z plot | 38 / 10 | multiselect Dropdown, ToolButton, `main_entry.on_refresh_checkpoint(callback, inputs, outputs)` (`:604-610`), `infotext_fields`/paste with lambdas (`:580-590`), `axis_options` mutated by 3 extensions | 1 |
| `scripts/prompt_matrix.py` (114 L) | sel Script | Prompt matrix | 11 / 0 | HTML, Radio | 1 |
| `scripts/prompts_from_file.py` (197 L) | sel Script | Batch prompts from text/file | 6 / 2 | **`gr.File(type="binary")`** + `.change` returning `gr.update(lines=…)` (`:118-121`) | 1 |
| `scripts/loopback.py` (129 L) | sel Script (img2img) | Loopback | 5 / 0 | none | 1 |
| `scripts/sd_upscale.py` (175 L) | sel Script (img2img) | SD upscale | ~7 / 0 (needs `shared.sd_upscalers`) | HTML, Radio over upscalers | 1 |
| `scripts/postprocessing_upscale.py` (217 L) | ScriptPostprocessing (Extras) | Upscale stage | ~20 / 4 | InputAccordion, Tabs with `Tab.select` used to set a hidden Number (`:62-63`), ToolButton, `create_refresh_button`, `show_progress="hidden"` | 1 |
| `scripts/degrid.py` (113 L) | ScriptPostprocessing (Extras) | DeGrid | ~3 / 1 | InputAccordion | 1 |
| `modules/processing_scripts/seed.py` (100 L) | ScriptBuiltinUI (section `seed`) | Seed / variation | 19 / 3 (+2 later) | ToolButton `_js='() => { setRandomSeed(...) }'` with `fn=None` (`:49-50`); `self.on_after_component(..., elem_id="generation_info_{tab}")` (`:63-64`) adds `reuse_seed.click(_js="(x, y) => [x, selected_gallery_index()]")` (`:100`) | 1 (core) |
| `modules/processing_scripts/sampler.py` (41 L) | ScriptBuiltinUI (section `sampler`) | Sampler/scheduler/steps | 5 / 0 | FormRow; elem_ids `{tab}_sampling/_scheduler/_steps` are hook targets for 5 extensions | 1 (core) |
| `modules/processing_scripts/refiner.py` (179 L) | ScriptBuiltinUI (section `accordions`, `opts.show_refiner`) | Refiner | 13 / 2 | InputAccordion, `infotext_fields` | 1 (core) |
| `modules/processing_scripts/rescale_cfg.py` (78 L) | ScriptBuiltinUI (section `cfg`, `opts.show_rescale_cfg`) | Rescale CFG | 2 / 0 | none | 1 (core) |
| `modules/processing_scripts/mahiro.py` (62 L) | ScriptBuiltinUI (section `cfg`, `opts.show_mahiro`) | MaHiRo | 2 / 0 | `PasteField(..., api=…)` | 1 (core) |
| `modules/processing_scripts/comments.py` (68 L) | AO, no UI | Strips `#` comments from prompts | 0 / 0 | none | 1 |

---

## 4. Requirements the extensions put on the compat layer (baseline contract)

Ordered by how many extensions break without each item.

1. **Component and layout constructors with the Gradio 4.40 signatures.** Extensions call these: Textbox/TextArea, Number, Slider, Checkbox, CheckboxGroup, Radio, Dropdown (multiselect, `allow_custom_value`), Button, HTML, Markdown, Image, Gallery, File, State, ColorPicker, Plot, HighlightedText; Row/Column/Group/Accordion/Tabs/Tab/TabItem/Blocks.
   * Kwargs that must be accepted (see the Gradio API-surface inventory): `elem_id`, `elem_classes`, `visible`, `interactive`, `scale`, `min_width`, `variant`, `info`, `precision`, `lines/max_lines`, `show_label`, `placeholder`, `type`, `height`.
   * Core subclasses that extensions depend on must keep working as subclasses: `ToolButton`, `FormRow/FormColumn/FormGroup/FormHTML/FormColorPicker`, `InputAccordion`/`InputAccordionImpl(gr.Checkbox)`, `DropdownMulti/Editable`, `ForgeCanvas`/`LogicalImage(gr.Textbox)`. The `LogicalImage` override of `preprocess/postprocess/get_block_name` must take effect.
   * Third-party: **`gradio_rangeslider.RangeSlider`** (ADetailer, ControlNet, Sparse) must be importable, e.g. as a shim package.
2. **Events.**
   * Listener-registration counts per tab measured in section 1: artist mixer 147, detail-daemon 109, postprocess 31.
   * `.click/.change/.input/.submit/.blur/.release/.select/.upload/.clear/.edit/.load/.then/.success`, `gr.on(triggers=…, trigger_mode=…)`.
   * Kwargs: `fn=None`, `_js`/`js` (a JS **string**: an arrow function or the name of a global, evaluated client-side with Gradio-shaped values), `queue`, `show_progress` (bool/"hidden"), `api_name`, `show_api`.
   * **Several listeners on one trigger run in registration order, and `queue=False` ones finish before queued ones read State.** ADetailer, Anima Resolutions, JoyCaption and agent-scheduler depend on this.
3. **Return values.** `gr.update(...)`, `gr.skip()`, tuple/list returns, and `gr.update` on layout blocks (visibility of Group/Row/Accordion).
4. **Event data and helpers.** `gr.SelectData`/`EventData` injection by annotation, `gr.Request` injection (`username`), and `gr.Progress(track_tqdm=True)`.
5. **State.** `gr.State` with callable defaults and arbitrary Python objects (dataclasses, dicts), kept per session server-side.
6. **Build-time semantics.**
   * Components created inside `on_after_component`/`after_component` callbacks land in the **currently open container** (JoyCaption).
   * `Script.on_after_component(cb, elem_id=…)`.
   * Components created outside Blocks and later `.render()`ed (ControlNet).
   * Nested `gr.Blocks()` rendered inline, with its own `.load` (extra-options-section).
   * `component.parent` / `children` / `elem_id` readable. Mutating them is only needed by agent-scheduler, which should be rewritten.
7. **Introspection.** `demo.blocks`, `demo.fns`, `demo.config["dependencies"][i]["targets"|"inputs"|"outputs"|"trigger_mode"]`, and `component._id`. Used by tests in 5+ extensions and by agent-scheduler.
8. **HTTP.** FastAPI `app` passed to `on_app_started(demo, app)`, with `app.get/post/put/delete` decorators (113 routes). The `/file=` static route with allowed-paths is required by tagcomplete (CSV), PAIO (CSS) and agent-scheduler's guard. `modules.extension_route_guard.guard_routes` must still apply.
9. **Global JS environment** (kept by the SPA): `gradioApp()`, `onUiLoaded`, `onUiUpdate`, `onAfterUiUpdate`, `onEdit`, `updateInput(el)`, `opts`, `selected_gallery_index()`, `switch_to_img2img/txt2img/…`, `get_tab_index`, `create_submit_args`, `requestProgress`, `randomId`, `submit/submit_img2img`, `currentImg2imgSourceResolution`, `setRandomSeed`, `modalSaveImage`, `window.localization`.
   * DOM contract used across 12 extensions: `#{elem_id}` on each component root; `#{elem_id} textarea` / `input[type=number|range|checkbox]` for value fields; `#tabs > .tab-nav > button` for top tabs; `#tab_{id}` panels.
   * A synthetic `input`/`change` DOM event on those fields must update SPA state.
10. **Settings.** `shared.opts.add_option(OptionInfo(default, label, component, component_args|callable, refresh=…, section=…, category_id=…).info().needs_reload_ui())`, with components Slider/Checkbox/Dropdown/Radio/Textbox(password)/HTML/ColorPicker. Total about 105 options from extensions: tagcomplete 60, ADetailer 12, agent-scheduler 9, ControlNet 7, extra-options 4, LoRA 4, detail-daemon 2, prompt-anchor 2, PAIO 1, Anima language options 3, plus language-diffusion's mutation of `localization`.

## 5. Native SPA widgets and features to build (beyond what core already needs)

| Widget / feature | Needed by | Notes |
|---|---|---|
| RangeSlider (2-thumb) | ADetailer, ControlNet, Sparse | Value `(lo, hi)` tuple; shim `gradio_rangeslider` package |
| Plot (static matplotlib) | Detail Daemon (12 per tab) | Server renders a PNG/SVG data URL; Gradio Plot payload `{type:"matplotlib", plot: base64}` |
| HighlightedText (+select) | LoRA metadata editor | Tag list with counts; select returns `(value, index)` |
| ForgeCanvas (image, mask, scribble) | ControlNet ×4/unit (core img2img too) | LogicalImage textbox contract |
| Prompt decorator / autocomplete API | tagcomplete, PAIO, prompt-anchor, bracket-checker | One extension point instead of 4 DOM hacks |
| Programmatic generate / enqueue API | agent-scheduler, prompt-queue, repeat-generate | Replaces `queue/join`, `randomId` hijack, `.click()` on Generate |
| Button/row slots next to Generate, above prompt, in PNG Info | agent-scheduler, prompt-queue, prompt-anchor, JoyCaption | Replaces `children.pop()` and `parent.add()`, DOM reparenting |
| Tab ordering, badges and switch API for custom tabs | Prompt Workshop, JoyCaption, prompt-queue | Replaces `.tab-nav button` manipulation |
| Native i18n catalog plus Dropdown option icons | language-diffusion (and core localization) | 10 locale JSONs |
| Responsive layout | mobile | CSS-only in SPA |
| Locked field / paste filter | lock-img2img-steps | Paste/Send-to honours a lock flag |
| Data grid | agent-scheduler | AG Grid as an SPA dependency |

## 6. Tests that depend on Gradio today

These test files import `gradio` directly or build UIs through `tests/extension_script_harness.py:94-104` (`build_ui` uses `gr.Blocks()`; `forge_infotext` uses `gr.skip()`). They must run against the compat package:

* anima-artist-scheduled-mixer: `conftest.py`, `test_infotext.py`, `test_runtime_dispatch.py`, `test_ui_wiring.py`. The last uses `demo.config["dependencies"]`, `targets`, `trigger_mode` and `_id`.
* anima-lora-layer-weight: `test_infotext.py` (`import gradio`).
* anima-lora-stage-scheduler: `test_infotext.py`, `test_ui.py`.
* sd_forge_anima_dave, sd_forge_anima_xattn_boost: `test_script.py`.
* sd_forge_anima_freefuse: `test_infotext.py`, `test_configure_sampling.py`.
* sd_forge_anima_guidance: `test_presets_registration.py`, `test_script_wiring.py`.
* sd_forge_anima_hires_guard, sd_forge_anima_regional: `test_infotext.py`, `test_presets_registration.py`; regional also has `test_feature_priority.py`.
* sd_forge_anima_presets: `test_anima_remi_presets.py:62` (`import gradio`).
* sd_forge_spectrum: `test_infotext.py:10`, `test_script.py`.
* sd-forge-negpip: `test_script.py`.
* agent-scheduler: `test_file_guard.py:50` (`from gradio import utils`).
* conftest-only harness users: sd_forge_compile, sd_forge_pid.

## 7. Suggested porting order

1. **Compat baseline** (section 4). This unlocks all 32 class-1 extensions and the 13 core scripts.
2. **RangeSlider, Plot and HighlightedText shims.** These unlock ADetailer, Sparse, Detail Daemon and LoRA.
3. **Generate/enqueue API plus UI slots plus tab API.** This unlocks prompt-queue, repeat-generate, prompt-anchor, Prompt Workshop and JoyCaption. lock-steps and mobile can follow as small native features.
4. **ForgeCanvas widget.** Core needs it anyway. ControlNet comes next, and with it the 6 preprocessor/patcher extensions.
5. **Prompt decorator / autocomplete API**, then tagcomplete, then PAIO (largest UI rewrite).
6. **agent-scheduler** native Queue module. It depends on step 3's enqueue API and an ordered generate-input signature.
