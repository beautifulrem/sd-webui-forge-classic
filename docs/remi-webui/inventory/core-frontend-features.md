# Remi WebUI inventory — Task B: Core frontend JS/CSS/HTML features

Scope: everything the **core** (non-extension) frontend does on top of stock Gradio 4.40:
`script.js`, `javascript/*.js` (31 files), `style.css`, `html/*`, `modules_forge/forge_canvas/*`,
the head/body injection in `modules/ui_gradio_extensions.py`, `localizations/`, and every Python-side
`_js=` / `js=` hook plus Python-generated inline `onclick=` handlers.
Branch `remi-webui`, read-only inventory (no source modified).

Line counts: `script.js` 227, `javascript/*.js` 5,956 total (largest: `extraNetworks.js` 801, `ui.js` 483,
`imageviewer.js` 276, `progressbar.js` 228, `localization.js` 204, `resizeHandle.js` 203),
`style.css` 1,865 lines / ~310 rule blocks, `forge_canvas/canvas.js` 857, `canvas.css` 161, `canvas.html` 58.

---

## 1. How the frontend gets loaded today

| Mechanism | Where | What |
|---|---|---|
| `javascript_html()` | `modules/ui_gradio_extensions.py:13-29` | Builds `<head>` additions in this order: (1) inline `window.localization = {...}` from `localization.localization_js(opts.localization)` (`modules/localization.py:26-37`); (2) `script.js`; (3) every `javascript/*.js` of core **and** all extensions via `scripts.list_scripts("javascript", ".js")` (classic scripts, global scope, alphabetical per dir, core first); (4) every `*.mjs` as `type="module"`; (5) `set_theme("<theme>")` if `--theme`. URLs are `file=<path>?<mtime>` (`webpath`, line 9-10). |
| `css_html()` | same file `:32-51` | `<link>` for every `style.css` (root `style.css` + 9 extensions' `style.css`, via `scripts.list_files_with_name`), then `data_path/user.css` if present, then inline `<style>html{background-color:...}` light/dark resolved from theme vars `background_fill_primary(_dark)`. |
| `reload_javascript()` | same file `:54-65` | Monkey-patches `gr.routes.templates.TemplateResponse`: JS + `<meta name="referrer" content="no-referrer">` injected before `</head>`, CSS injected before `</body>`. Called at `modules/ui.py:197` and on "Reload custom script bodies" (`modules/ui_settings.py:253`). |
| `canvas_head` | `modules_forge/forge_canvas/canvas.py:50-61` | `<link canvas.css>` + `<script canvas.js>` passed as `gr.Blocks(head=canvas_head)` at `modules/ui.py:208` (txt2img Blocks), `:483` (img2img Blocks), `:887` (root `demo`). |
| ForgeCanvas bootstrap | `canvas.py:138` | Each `ForgeCanvas(...)` registers `Context.root_block.load(None, js="async ()=>{new ForgeCanvas(uuid, ...)}")`. Instances: 4 in core (`ui.py:520,524,528,532`) + ControlNet (`controlnet_ui_group.py:235,241,268`, `canvas_editor.py:23`) per unit per tab. |
| Theme | `modules/shared_gradio_themes.py:44-71` | Gradio theme (Default or HF-hub theme JSON); adds custom vars `sd_webui_modal_lightbox_toolbar_opacity` / `_icon_opacity` consumed in CSS as `--sd-webui-modal-lightbox-*` (`style.css:852,939`). |
| Footer | `html/footer.html` via `ui.py:913-915` | FAQ / API / "Startup Profile" (`showProfile`) / "Reload UI" link + `versions_html()` (`ui.py:918-959`, includes `<a id="sd_checkpoint_hash">`). |
| Notification audio | `ui.py:909-910` | hidden `gr.Audio(elem_id="audio_notification")` if `notification.mp3` exists and `opts.notification_audio`. |
| Settings JSON bridge | `ui_settings.py:229` | hidden `gr.Textbox(elem_id="settings_json", value=opts.dumpjson())` — the **only** channel by which `opts` reaches JS (`ui.js:333-366`). |

**Implication for the SPA:** the SPA must (a) still serve and execute extension `javascript/*.js` / `*.mjs` and `style.css` files in the same order (extensions depend on the global API below), (b) provide `window.localization`, `window.opts`, `window.gradio_config`-like data, and (c) provide `head=` injection for Blocks.

---

## 2. Global JS runtime API defined by core (used by extensions)

Defined in `script.js` unless noted. Count = number of built-in extension `.js/.mjs` files (of 30) referencing the identifier (word match).

| Global | Defined at | Semantics | Ext files using |
|---|---|---|---|
| `gradioApp()` | script.js:1-11 | returns `<gradio-app>` shadowRoot or element (with `getElementById` shim) | 17 |
| `get_uiCurrentTab()` / `get_uiCurrentTabContent()` | script.js:16-25 | `#tabs > .tab-nav > button.selected`, `#tabs > .tabitem[id^=tab_]:not([style*="display: none"])` | 0 / 1 |
| `onUiUpdate(cb(mutations))` | :39 | every MutationObserver batch on gradioApp (childList, subtree) | 1 |
| `onAfterUiUpdate(cb)` | :51 | debounced 250 ms after mutations (:112-117) | 2 |
| `onUiLoaded(cb)` | :59 | fired once when `#txt2img_prompt` first appears (:123-126) | 13 |
| `onUiTabChange(cb)` | :67 | when selected top tab button changes (:130-134) | 0 |
| `onOptionsChanged(cb)` | :75 | after `opts` parsed and each time `#settings_json textarea.value` is set (ui.js:344-358) | 0 |
| `onOptionsAvailable(cb)` | :86 | once opts non-empty (immediate if already) | 0 |
| `executeCallbacks(queue,arg)` | :95 | try/catch runner | 0 |
| `uiElementIsVisible(el)`, `uiElementInSight(el)` | :209, :221 | visibility helpers (used by dragdrop paste) | 0 |
| `opts` (global object) | script.js:79; filled ui.js:342 | server options incl. `_comments_before/_after`, `_categories` (`modules/options.py:237-251`) | 5 |
| `localization` (window) | injected inline | translation dict, optional `rtl` | 2 |
| `updateInput(el)` | ui.js:415-419 | dispatch synthetic bubbling `input` so Gradio picks up a JS-side value change | 4 |
| `onEdit(id, el, ms, fn)` | ui.js:473-483 | debounced input listener | 1 |
| `requestProgress(id, container, gallery, atEnd, onProgress, timeout=40)` | progressbar.js:77 | progress bar + live preview polling | 2 |
| `randomId()` | progressbar.js:64 | `task(xxxxxxxxxxxxxxx)` | 2 |
| `request(url,data,ok,err)` (POST JSON) / `requestGet` | progressbar.js:7 / extraNetworks.js:656 | XHR helpers | 0 |
| `popup(el)`, `popupId(id)`, `closePopup()` | extraNetworks.js:546-582 | global modal `.global-popup` | 2 / 0 / 0 |
| `localSet/localGet/localRemove` | localStorage.js | safe localStorage | 1 |
| `selected_gallery_index()`, `all_gallery_buttons()`, `selected_gallery_button()`, `selected_gallery_index_id(id)`, `gallery_container_buttons` | ui.js:10-39 | current gallery selection from DOM | 1 |
| `get_tab_index(tabsId)` | ui.js:95 | index of selected button inside a `gr.Tabs` | 1 |
| `switch_to_*` | ui.js:58-93 | click top tabs / img2img mode tabs | 0 |
| `submit`, `submit_img2img`, `create_submit_args` | ui.js:118-198 | wrapped by extensions (e.g. agent-scheduler, prompt-queue) | 3 / 2 / 1 |
| `appendContextMenuOption`, `removeContextMenuOption`, `addContextMenuEventListener` | contextMenus.js:112-115 | right-click menu registry | 1 |
| `activePromptTextarea[tabname]` | extraNetworks.js:237 | last focused prompt per tab | 1 |
| `extraNetworksApplyFilter/Sort[tabname_full]`, `uiAfterScriptsCallbacks` | extraNetworks.js:235-236, 776 | — | 0 |
| `createVisualizationTable(data,cutoff,sort)` | profilerVisualization.js:35 | nested key table | 0 |
| `setupAllResizeHandles()`, `setupResizeHandle(el)` | resizeHandle.js:192-201 | — | 0 |
| `window.gradio_config` | (Gradio) | read by hints.js:84, localization.js:114, canvas.js:52/607 (`version.startsWith("4.")` gate!) | 1 |
| `ForgeCanvas`, `GradioTextAreaBind`, global consts `True`/`False` | canvas.js:1-857 | canvas widget; `True/False` (:856-857) let Python-formatted bools work in JS strings — **also relied on by `extraNetworks.js:85` (`== True`)** | 2 (`ForgeCanvas` Python-side) |
| `window.args_to_array` | ui.js:56 | compat alias of `Array.from` | — |

---

## 3. Per-file inventory

Legend — **Hooks**: functions called by name from Python `_js=`/`js=` or Python-generated HTML `onclick`. **Life**: lifecycle registrations. **DOM**: Gradio-structure dependencies (ids/classes/attributes produced by Gradio's Svelte components, not by our own `elem_id`s). **EP**: server endpoints.

### 3.1 `script.js` (227 lines)
- Features: core event bus (§2); global keyboard shortcuts (:139-204): **Ctrl/Cmd+Enter** = Generate (if running: Interrupt, then auto-regenerate when interrupt button hides unless `opts.ctrl_enter_interrupt`), **Alt+Enter** = Skip, **Esc** = Interrupt (unless global popup visible or lightbox focused).
- DOM: `gradio-app` shadowRoot, `#tabs > .tab-nav > button.selected`, `.tabitem[id^=tab_]` + inline `style="display: none"`, `button[id$=_generate|_interrupt|_skip]`, interrupt visibility read from `style.display === "block"`.
- Life: owns the MutationObserver driving all callbacks.

### 3.2 `javascript/ui.js` (483)
- Features: theme redirect `set_theme` (:3-8, `?__theme=`); gallery selection helpers (:10-54); tab switching (:58-93); submit wrappers that generate task id, show Interrupt/Skip, start progress, persist task id to localStorage, strip output args (`create_submit_args` :118-128 removes trailing 3-5 output values Gradio sends); Interrupt/Skip/Interrupting visibility (:130-142); restore-progress button (:144-148, :219-259); **paste "W x H" into width/height** (:265-281); **click-to-deselect style tokens** in multiselect (:286-298); merger progress (:310-317); `ask_for_style_name` prompt (:319-322, unused by core); `confirm_clear_prompt` (:324-331); settings JSON bridge + `opts` + change detection by redefining `textarea.value` setter (:333-366); checkpoint hash link to civitai in footer (:368-378); `restart_reload` "Reloading..." page that polls `/internal/ping` every 500 ms (:385-405); **confirm-before-leave** (`opts.confirm_leave`, :407-411); `updateInput` (:415); `selectCheckpoint(name)` (:421-426, via hidden `#change_checkpoint_text` + `#change_checkpoint`); `currentImg2imgSourceResolution` (:428-434, reads natural size of visible img/canvas in `#mode_img2img`); `updateImg2imgResizeToTextAfterChangingImage` (:436-445, unused by core); `setRandomSeed` (:447-454); `switchWidthHeight` (:456-468, unused by core — Python lambda used instead); `onEdit` (:473).
- Hooks: `submit`, `submit_txt2img_upscale`, `submit_img2img`, `submit_extras`, `restoreProgressTxt2img`, `restoreProgressImg2img`, `modelmerger`, `confirm_clear_prompt`, `switch_to_txt2img|img2img|sketch|inpaint|inpaint_sketch|extras`, `selected_gallery_index`, `selected_gallery_index_id`, `extract_image_from_gallery`, `currentImg2imgSourceResolution`, `setRandomSeed`, `restart_reload`, `selectCheckpoint` (onclick), `showSubmitInterruptingPlaceholder`.
- Life: `onUiLoaded` (:300-308), `onOptionsChanged` (:368).
- DOM: `.thumbnail-item.thumbnail-small`, `.selected`, `div[id$=_gallery].gradio-gallery .thumbnails`, `[style="display: block;"].tabitem`, `#tabs button` by index (0=txt2img,1=img2img,2=extras — **positional, breaks with `ui_tab_order`/hidden tabs**), `#mode_img2img button[no]`, `#<tab>_styles div.token / div.token-remove`, `#settings_json textarea`, `input[type=number]` inside sliders.
- EP: `GET ./internal/ping`.

### 3.3 `javascript/progressbar.js` (228)
- Features: progress bar (`.progressDiv > .progress`) inserted above `*_gallery_container`, % + ETA (`formatTime`), textinfo prefix, **progress in browser title** (`opts.show_progress_in_title`), hidden if `!opts.show_progressbar`; **live preview** images layered in `.livePreview` inside gallery (keeps ≤2 imgs for crossfade), polling period `opts.live_preview_refresh_period||500`; **screen wake lock** (`opts.prevent_screen_sleep_during_generation`); completion detection: `completed`, or active→inactive, or 40 s inactivity when not queued (restore uses timeout 0); falls back to `.gradio-video` when gallery hidden.
- Hooks: none directly (used by ui.js / extensions.js).
- Life: `onUiLoaded` (:48) records original title.
- EP: `POST ./internal/progress` `{id_task, live_preview:false}` and `{id_task, id_live_preview}` (`modules/progress.py:63-82`; response `active, queued, completed, progress, eta, live_preview, id_live_preview, textinfo`).

### 3.4 `javascript/imageviewer.js` (276)
- Features: full-screen **lightbox** `#lightboxModal` (built at DOMContentLoaded :196-276) with controls: zoom toggle (`modalImageFullscreen`), **tile preview** (background-repeat), save (only on txt2img/img2img; clicks `#save_txt2img|#save_img2img`), toggle live preview in modal (`opts.js_live_preview_in_modal_lightbox`), close; prev/next arrows; keys **S** save, **←/→** navigate, **Esc** close; **middle-click opens image in new tab**; left-click opens modal if `opts.js_modal_lightbox`; initially zoomed per `opts.js_modal_lightbox_initially_zoomed`; modal follows gallery selection / live preview (`updateOnBackgroundChange`).
- Life: `onAfterUiUpdate` (:186) attaches to `.gradio-gallery > button > button > img, .gradio-gallery > .livePreview`.
- DOM: Gradio gallery preview structure, `#tab_txt2img/#tab_img2img` style.display.

### 3.5 `javascript/imageviewerGamepad.js` (63)
- Features: **gamepad left stick** / **horizontal wheel (VR pointer)** navigate lightbox prev/next (`opts.js_modal_lightbox_gamepad`, repeat `opts.js_modal_lightbox_gamepad_repeat`).

### 3.6 `javascript/generationParams.js` (54)
- Features: clicking / arrow-keying in txt2img/img2img gallery, or closing the lightbox, clicks hidden `#<tab>_generation_info_button` → server re-renders infotext for the selected image.
- Life: `onUiLoaded` (:7). DOM: `#tabs div button.selected` innerText (=="txt2img"/"img2img", **localization-sensitive**).

### 3.7 `javascript/extraNetworks.js` (801)
- Features: Extra Networks (Lora / Checkpoints / Textual Inversion) panes under `#<tab>_extra_tabs`: moves per-page controls into the tab-nav bar (`.extra-networks-controls-div`); **search** (multi-word AND over `.search_terms, .description`; `.search_only` cards need ≥4 chars); **arch filter** by `data-sort-sdversion` vs `#forge_ui_preset` (see §9 bug); **sort** by path/name/date created/date modified, asc/desc (`data-sort-*` attrs); **tree view** toggle & directory tree (expand/collapse chevron, select dir → sets search to `data-path`); dirs buttons (`extraNetworksSearchButton`); refresh (clicks hidden `#<tab>_<page>_extra_refresh_internal`); **card click inserts/removes `<lora:name:w>` or TI text** into the active prompt (toggle semantics with separator `opts.extra_networks_add_text_separator`, positive/negative split, regexes :244-247); **compact prompt layout**: moves prompt/neg-prompt rows into the extra-network tab (`extraNetworksMovePromptToTab` :169-192); **save card preview** from gallery (`saveCardPreview`); **copy path** to clipboard; **metadata popup** (flattened JSON table via `createVisualizationTable`, or `<pre>`); **user-metadata editor** popup (fills hidden name textbox, clicks button, `popup(editor.page)`); **refresh single card** after edit; generic **popup** (`.global-popup`, close button, Esc closes); `requestGet`; `clickLoraRefresh` (re-filter on preset change); `toggleCss`.
- Hooks (Python `_js`): `extraNetworksUnrelatedTabSelected`, `extraNetworksTabSelected`, `applyExtraNetworkFilter`, `extraNetworksRefreshSingleCard`, `popupId`, `closePopup`, `clickLoraRefresh`. Hooks (HTML onclick): `cardClicked`, `saveCardPreview`, `selectCheckpoint` (ui.js), `extraNetworksSearchButton`, `extraNetworksTreeOnClick`, `extraNetworksControlSortOnClick`, `extraNetworksControlSortDirOnClick`, `extraNetworksControlTreeViewOnClick`, `extraNetworksControlRefreshOnClick`, `extraNetworksCopyCardPath`, `extraNetworksRequestMetadata`, `extraNetworksEditUserMetadata`.
- Life: `onUiLoaded` (:787) + custom "after scripts" observer waiting until `[id$=_extra_search]` count ≥ extra tab buttons − 2 (:776-801) → `setupExtraNetworks`.
- DOM: `#<tab>_extra_tabs > div.tab-nav`, `#<id> > label > textarea` (Gradio Textbox structure), `#<id>-button[aria-selected]`, `.main`.
- EP: `GET ./sd_extra_networks/metadata?page&item`, `GET ./sd_extra_networks/get-single-card?page&tabname&name` (`modules/ui_extra_networks.py:170-171`); card previews load from `./sd_extra_networks/thumb` (`:168`).

### 3.8 `javascript/edit-attention.js` (176)
- Features: **Ctrl/Cmd+↑/↓** in any prompt textarea (`[id*='_toprow'] [id*='_prompt'] textarea, .prompt textarea`): auto-selects enclosing `<…>`, `(…)`, `[…]` block or current word (delimiters `opts.keyedit_delimiters` + `opts.keyedit_delimiters_whitespace`), wraps as `(text:1.0)`, converts legacy `((x))`/`[x]` to explicit weight, step `opts.keyedit_precision_attention` (or `opts.keyedit_precision_extra` for `<lora:…:w>`), removes parens when weight returns to 1.
- Global `keydown` listener (:174).

### 3.9 `javascript/edit-order.js` (41)
- Features: **Alt+←/→** moves the comma-separated tag(s) under the selection left/right (`opts.keyedit_move`).

### 3.10 `javascript/history.js` (90)
- Features: custom **prompt undo/redo** (Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z), 600 ms snapshot debounce, 16-entry limit, for `txt2img_prompt`, `txt2img_neg_prompt`, `img2img_prompt`, `img2img_neg_prompt`, `hires_prompt`, `hires_neg_prompt`; enabled by `opts.undo_redo` (`onOptionsAvailable` :89).

### 3.11 `javascript/promptDebounce.js` (98)
- Features: **debounces prompt `input` events** sent to Gradio (`opts.prompt_debounce` ms) — captures and stops native input, re-dispatches after delay, flushes on blur and on any Ctrl/Alt/Meta keydown (so Ctrl+Enter sees latest text). Same 6 ids. `onOptionsAvailable` (:97).

### 3.12 `javascript/promptField.js` (24)
- Features: disable autocorrect/autocapitalize/autocomplete/spellcheck on 6 prompt textareas (`opts.no_spellcheck`).

### 3.13 `javascript/token-counters.js` (80)
- Features: token counter badge (`#<tab>_token_counter`, `#<tab>_negative_token_counter`) moved next to prompt; after 1 s idle triggers hidden `#<tab>_token_button` / `_negative_token_button` (server computes "n/75"); hidden if `opts.disable_token_counters`.
- Hooks: `update_txt2img_tokens`, `update_img2img_tokens` (via `ui_prompt_styles.py:123` string `update_<tab>_tokens`), `recalculate_prompts_txt2img`, `recalculate_prompts_img2img` (`infotext_utils.py:571`). `onUiLoaded` :74, `onOptionsChanged` :78.

### 3.14 `javascript/contextMenus.js` (175)
- Features: right-click (or 2-finger touch) **context menu** registry; built-ins on `#txt2img_generate|_interrupt` and `#img2img_generate|_interrupt`: **"Generate forever"** (500 ms poll re-clicks Generate when interrupt hidden) / **"Cancel generate forever"**. `onAfterUiUpdate(addContextMenuEventListener)` (:175). Menu styled from current tab computed style (`uiCurrentTab`).

### 3.15 `javascript/dragdrop.js` (128)
- Features: drop image **file or URL onto a prompt** → uploads to hidden `#<tab>_prompt_image` gr.File → server reads PNG info into prompt (`ui_toprow.py:96,103-108`); drop onto any Gradio image (`[data-testid="image"]`) replaces it (clicks clear button `.modify-upload button + button` then sets `input[type=file]`); **Ctrl+V paste image** goes to first empty visible image field (or last visible).
- DOM: `[data-testid="image"]`, `.modify-upload`, `.touch-none`, `input[type=file]`, placeholder contains "Prompt".

### 3.16 `javascript/imageDragAndDrop.js` (19)
- Features: on dragover of URL/HTML onto `#extras_image` / `#pnginfo_image`, auto-click "Remove Image" so the drop replaces (`opts.remove_image_on_hover`). DOM: `button[aria-label="Remove Image"]`.

### 3.17 `javascript/imageMaskFix.js` (43)
- Features: legacy Gradio sketch-canvas sizing fix for `#img2maskimg .touch-none canvas`. **Dead** with ForgeCanvas (`img2maskimg` is now ForgeCanvas HTML; no `.touch-none`). `onAfterUiUpdate` + window resize.

### 3.18 `javascript/aspectRatioOverlay.js` (117)
- Features: when dragging img2img width/height sliders, draws `#imageARPreview` rectangle over the current img2img source image (ForgeCanvas `div.forge-image-container img` for tabs 0-3; `#img_inpaint_base div[data-testid=image] img` for tab 4), auto-hides after 2 s.
- Life: `onAfterUiUpdate` (:84). DOM: `#img2img_width/height` input nesting (`parentElement.parentElement.parentElement.id`), `#tab_img2img` style.display.

### 3.19 `javascript/hires_fix.js` (19)
- Hook: `onCalcResolutionHires(enable,w,h,scale,rx,ry)` — greys out (`.inactive`) "Upscale by" vs "Resize width/height to" depending on which is used; hides `#txt2img_hires_fix_row2` if `opts.use_old_hires_fix_width_height`.

### 3.20 `javascript/inputAccordion.js` (88)
- Features: `InputAccordion` (accordion with checkbox in header = boolean input): injects visible checkbox into `.label-wrap span`, syncs open state ↔ hidden `gr.Checkbox` (`#<id>-checkbox`), moves `#<id>-extra` into header, mirrors disabled state, unlinks open/check after manual checkbox click.
- Hook: `inputAccordionChecked(id, checked)` (`ui_components.py:130`). `onUiLoaded` (:84). DOM: `.label-wrap.open` (Gradio Accordion internals). Used by core (Hires fix, Refiner-like accordions) and many extensions via `InputAccordion`.

### 3.21 `javascript/resizeHandle.js` (203)
- Features: **draggable column splitter** for every `.resize-handle-row` (`ResizeHandleRow`, `ui_components.py:29-40`; also extra-network tree pane `html/extra-networks-pane-tree.html`): converts flex row to grid `Lpx 16px 1fr`, mouse/touch drag, **double-click / double-tap resets**, min 320 px per column, collapses to flex below 2×320+64 px width, proportional rescale on window resize, `body.resizing` cursor.
- Hook: `setupAllResizeHandles` (`ui_extra_networks.py:774,784`). `onUiLoaded` (:203).

### 3.22 `javascript/settings.js` (83)
- Features: **settings search** (`#settings_search` moved into tab nav; 500 ms debounce; filters `#settings > div[id^=settings_] div[id^=column_settings_] > *` by text; non-empty search shows all pages, empty clicks `#settings_show_one_page`); "Show all pages" button (excludes defaults/sysinfo/actions/licenses); scroll to top on tab click; **category headings** (`.settings-category` spans inserted before nav buttons from `opts._categories`, localized).
- Life: `onUiLoaded` (:22), `onOptionsChanged` (:60). DOM: `#settings > div.tab-nav > button`, `#settings_search > label > input`.

### 3.23 `javascript/ui_settings_hints.js` (70)
- Features: renders `opts._comments_before/_after` HTML (links, `(info)` spans, `[info]` JS links from `OptionInfo.js()` `options.py:40-42`) next to each `#setting_<key>` label; `settingsHintsShowQuicksettings()` popup table of all setting names/labels.
- Hook (HTML onclick from `OptionInfo.js`): `settingsHintsShowQuicksettings` (`shared_options.py:457`). `onOptionsChanged` (:5). DOM: `.gradio-checkbox label span`, `.gradio-checkboxgroup/.gradio-radio span`.
- EP: `GET ./internal/quicksettings-hint` (`ui.py:973`).

### 3.24 `javascript/hints.js` (90)
- Features: **tooltips** — static English title map (28 entries: sampler, steps, CFG, seed, resize modes, inpaint fill modes, hires…) applied to SPAN/BUTTON/P text (localized), propagated to sibling inputs; plus `webui_tooltip` from `gradio_config.components[].props` → element `title`.
- Life: `onUiUpdate` (:66, 1 s batch), `onUiLoaded` (:83).

### 3.25 `javascript/localization.js` (204)
- Features: whole-page **text translation** via `window.localization` (text nodes, `title`, `placeholder`, `webui_tooltip`, Textbox placeholders); skip list for option values in model/style/upscaler dropdowns (:3-13), numbers, emoji, SCRIPT/STYLE/TEXTAREA; **RTL** support (enables `@media rtl` rules); `download_localization()` dumps all seen strings as JSON template.
- Hook: `download_localization` (`ui_settings.py:249`). Life: `onUiUpdate` on every added node (if localization non-empty).

### 3.26 `javascript/notification.js` (56)
- Features: when first gallery thumbnail src changes: play `#audio_notification` (volume `opts.notification_volume`), and if window unfocused show **browser Notification** "Generated N images" (minus `opts.return_grid`), click focuses tab; `#request_notifications` button requests permission.
- Life: `onAfterUiUpdate` (:7). DOM: `div[id^="tab_"] div[id$="_results"] .thumbnail-item > img`, `#audio_notification #waveform > div` **shadowRoot audio** (Gradio Audio internals).

### 3.27 `javascript/keepAlive.js` (26)
- Features: patches `requestAnimationFrame` to run via MessageChannel when tab hidden so Gradio keeps updating in background (`opts.keep_alive`).

### 3.28 `javascript/localStorage.js` (25) — `localSet/localGet/localRemove`.

### 3.29 `javascript/modelHelp.js` (38)
- Features: adds "?" link (wiki "Download Models") next to `#setting_sd_modules` (VAE/Text Encoder multiselect) label, visible only when preset not sd/xl and no module selected. `onUiLoaded` + `onAfterUiUpdate`. DOM: `div.wrap-inner`, `div.token`.

### 3.30 `javascript/populateLicense.js` (29)
- Hook: `populateLicense` (`ui_settings.py:241`, Licenses tab select) — fetches 14 upstream LICENSE files from raw.githubusercontent.com into `#<id>-license-content`.

### 3.31 `javascript/profilerVisualization.js` (165)
- Features: `createVisualizationTable` (hierarchical key/value or timing table with expandable rows & "others" bucket), `showProfile(path, cutoff)`.
- Hook (footer onclick): `showProfile('./internal/profile-startup')`. EP: `GET ./internal/profile-startup` (`ui.py:977`).

### 3.32 `javascript/extensions.js` (99)
- Hooks: `extensions_apply` (collect disabled/update checkboxes → JSON, then `restart_reload`), `extensions_check` (progress on `#extensions_installed_html`, sets statuses "Loading..."), `config_state_confirm_restore` (confirm dialog), HTML onclick/onchange: `install_extension_from_index(button,url)` (fills hidden `#extension_to_install`, clicks `#install_extension_button`), `toggle_all_extensions`, `toggle_extension`.

### 3.33 `modules_forge/forge_canvas/canvas.js` (857) + `canvas.html` + `canvas.css`
- Features (**img2img / sketch / inpaint / inpaint-sketch / ControlNet input+mask canvases**): upload (button, click-on-empty, drag&drop file, paste while hovered), remove, center (**R**), reset strokes (**Ctrl+X**), undo/redo (**Ctrl+Z / Ctrl+Y**, 16 states), maximize/minimize fullscreen (**F**), color picker (**E**), brush width/opacity/softness sliders with **W/A/S + wheel** adjustment, **Shift = eraser** (alpha 0 while held), wheel zoom around cursor (min 0.1), **right-drag pan**, brush indicator circle, soft brush (multi-pass strokes), contrast checkerboard scribbles for inpaint mask (`contrast_scribbles`), fixed color/width/alpha/softness modes (hides controls), `no_upload`, `no_scribbles`, consistent brush size across zoom (`opts.forge_canvas_consistent_brush`), plain background (`opts.forge_canvas_plain`, `_plain_color`), static toolbar (`opts.forge_canvas_toolbar_always`), height (`opts.forge_canvas_height`). Toolbar auto-hides when pointer leaves.
- Data binding: two hidden `LogicalImage` (gr.Textbox subclass, `canvas.py:83-121`) with `elem_id=<uuid>` and classes `logical_image_background` / `logical_image_foreground` carry PNG **data-URLs**; JS writes via `updateInput`-style dispatch and **polls every 100 ms** (`GradioTextAreaBind.listen`, :20-30) for server-side changes (e.g. "send to inpaint", copy between modes).
- Gating: `loadImage` returns early unless `gradio_config.version.startsWith("4.")` (:607-611) — SPA must provide a compatible `gradio_config.version` or remove this gate.
- Hook: `new ForgeCanvas(...)` via `root_block.load(js=...)` (`canvas.py:138`).

---

## 4. Every Python-side `_js=` / `js=` hook (core)

Notes: `_js` is translated to Gradio 4 `js` by `EventWrapper.__call__` (`modules/gradio_extensions.py:117-131`) and `Dependency.then` wrapper (`:163-176`). Semantics the new runtime must reproduce: the JS function runs **in the browser before** the Python `fn`, receives the current input values as positional args, and its return value (array, or single value) **replaces the inputs** sent to `fn`; with `fn=None` it is JS-only. String may be a function name (global lookup) or a function/arrow expression.

| # | Call site | Event | JS | What it does |
|---|---|---|---|---|
| 1 | `modules/ui.py:332-337` | `.release`/`.change` of each of 6 hires inputs (`enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y`, loop `:323-324`) | `onCalcResolutionHires` | grey-out hires size controls (js-only) |
| 2 | `modules/ui.py:378-388` | `toprow.prompt.submit`, `toprow.submit.click` (txt2img) | `submit` | task id → arg0, progress UI, localStorage, strip outputs |
| 3 | `modules/ui.py:396-407` | `output_panel.button_upscale.click` | `submit_txt2img_upscale` | as submit + arg2 = selected gallery index |
| 4 | `modules/ui.py:407` | `.then(select_gallery_image)` | `selected_gallery_index` | pass current index to select upscaled image |
| 5 | `modules/ui.py:411-413` | `restore_progress_button.click` (txt2img) | `restoreProgressTxt2img` | resume progress for stored task id; returns id |
| 6 | `modules/ui.py:573-576` | 12 copy buttons (4 img2img modes × 3 targets) | `switch_to_img2img` / `switch_to_sketch` / `switch_to_inpaint` / `switch_to_inpaint_sketch` | switch img2img mode tab (js-only, separate listener from copy fn) |
| 7 | `modules/ui.py:604-613` | `scale_by.change`, `img2img_update_resize_to` click | `currentImg2imgSourceResolution` | inject source w/h for "resize by" preview |
| 8 | `modules/ui.py:743-757` | `toprow.prompt.submit`, `toprow.submit.click` (img2img) | `submit_img2img` | as submit |
| 9 | `modules/ui.py:762-768` | `detect_image_size_btn.click` | `currentImg2imgSourceResolution` | auto-detect size → width/height |
| 10 | `modules/ui.py:770-772` | `restore_progress_button.click` (img2img) | `restoreProgressImg2img` | resume progress |
| 11 | `modules/ui_toprow.py:127` | `interrupt.click` (txt2img, img2img) | `function(){ showSubmitInterruptingPlaceholder("<id_part>"); }` | show "Interrupting..." |
| 12 | `modules/ui_toprow.py:149-155` | `clear_prompt_button.click` | `confirm_clear_prompt` | `confirm()` then clear both prompts |
| 13 | `modules/ui_prompt_styles.py:98-104` | style `delete.click` | inline `confirm("Delete style "+name+"?")` | confirm deletion |
| 14 | `modules/ui_prompt_styles.py:117-123` | `.then` after `materialize_styles` (toolbar apply button `ui_toprow.py:159` + dialog "materialize" `ui_prompt_styles.py:106`) | `function(){update_<tab>_tokens(); closePopup();}` | refresh token counters, close dialog |
| 15 | `modules/ui_common.py:211-219` | open-folder button | `(y, w) => [y, selected_gallery_index()]` | index injection |
| 16 | `modules/ui_common.py:231-237` | hidden `<tab>_generation_info_button.click` | `function(x, y, z){ return [x, y, selected_gallery_index()] }` | re-render infotext for selected image |
| 17 | `modules/ui_common.py:239-254` | save button | `(x, y, z, w) => [x, y, false, selected_gallery_index()]` | save selected image |
| 18 | `modules/ui_common.py:255-...` | save-zip button | `(x, y, z, w) => [x, y, true, selected_gallery_index()]` | save zip |
| 19 | `modules/ui_common.py:324-327` | `setup_dialog` show-button `.then` (styles editor) | `function(){ popupId('<dialog elem_id>'); }` | move Gradio group into global popup |
| 20 | `modules/ui_common.py:330` | dialog close button | `closePopup` | hide popup |
| 21 | `modules/infotext_utils.py:179-186` | send-to paste button (image part) | `extract_image_from_gallery` (or none) | pick selected gallery image |
| 22 | `modules/infotext_utils.py:218-225` | send-to paste button (infotext from image, `opts.send_image_info_not_ui`) | `extract_image_from_gallery` then `switch_to_<tabname>` | — |
| 23 | `modules/infotext_utils.py:228-233` | send-to (fields copy) `.then` | `switch_to_<tabname>` | tabname ∈ {txt2img, img2img, inpaint, extras} |
| 24 | `modules/infotext_utils.py:236` | send-to (no fields) | `switch_to_<tabname>` | — |
| 25 | `modules/infotext_utils.py:565-571` | prompt "paste" (↙) button `.then` | `recalculate_prompts_<tabname>` | refresh token counters |
| 26 | `modules/ui_extra_networks.py:761` | unrelated tab `.select` (Generation tab etc.) | `extraNetworksUnrelatedTabSelected('<tab>')` | restore prompt rows, hide controls |
| 27 | `modules/ui_extra_networks.py:764-765` | each extra-network page tab `.select` | `extraNetworksTabSelected(...); applyExtraNetworkFilter(...)` | move prompts, show controls, filter |
| 28 | `modules/ui_extra_networks.py:773-774` | hidden `_extra_refresh_internal.click` → `.then` ×2 | `applyExtraNetworkFilter('<tab>_<page>')`, `setupAllResizeHandles` | re-filter, re-attach splitters |
| 29 | `modules/ui_extra_networks.py:784` | `interface.load(pages_html).then` | `setupAllResizeHandles` | — |
| 30 | `modules/ui_extra_networks.py:824` | hidden `<tab>_save_preview.click` | `function(x, y, z){return [selected_gallery_index(), y, z]}` | save card preview from gallery |
| 31 | `modules/ui_extra_networks_user_metadata.py:69` | editor cancel | `closePopup` | — |
| 32 | `modules/ui_extra_networks_user_metadata.py:149` | editor save `.then` | `function(name){closePopup(); extraNetworksRefreshSingleCard(page, tab, name);}` | — |
| 33 | `modules/ui_extra_networks_user_metadata.py:191` | replace-preview button + `.then` | `selected_gallery_index_id('<tab>_gallery_container')`, then `extraNetworksRefreshSingleCard(...)` | — |
| 34 | `modules/processing_scripts/seed.py:49` / `:50` | 🎲 seed / subseed `.click` | `() => { setRandomSeed("<elem_id>"); }` | set -1 (js-only, `queue=False`) |
| 35 | `modules/processing_scripts/seed.py:100` | ♻ reuse seed | `(x, y) => [x, selected_gallery_index()]` | seed of selected image |
| 36 | `modules/ui_postprocessing.py:45-47` | extras submit | `submit_extras` | progress for extras |
| 37 | `modules/ui_extensions.py:591-595` | Apply & restart | `extensions_apply` | — |
| 38 | `modules/ui_extensions.py:598-602` | Check for updates | `extensions_check` | — |
| 39 | `modules/ui_extensions.py:683` | restore config state | `config_state_confirm_restore` | — |
| 40 | `modules/ui_settings.py:241` | Licenses tab `.select` | `populateLicense` | — |
| 41 | `modules/ui_settings.py:247` | Request notifications | `function(){}` (noop; listener in notification.js) | — |
| 42 | `modules/ui_settings.py:249` | Download localization template | `download_localization` | — |
| 43 | `modules/ui_settings.py:257-262` | Reload UI (`settings_restart_gradio`) | `restart_reload` | Reloading screen + ping |
| 44 | `modules/ui_components.py:130` | InputAccordion hidden checkbox `.change` | `function(checked){ inputAccordionChecked("<id>", checked); }` | sync accordion from server value |
| 45 | `modules/ui_checkpoint_merger.py:126-135` | merge `.then` | `modelmerger` | task id + progress |
| 46 | `modules_forge/main_entry.py:286-292` | `ui_forge_preset.change → .success → .then` | `clickLoraRefresh` | re-filter extra networks |
| 47 | `modules_forge/forge_canvas/canvas.py:138` | `root_block.load` per canvas | `async ()=>{new ForgeCanvas(...)}` | instantiate canvas |

**47 hook sites (table rows) total. Distinct global JS functions called from Python `_js`/`js` (by name or from inside an inline wrapper): 40** (`onCalcResolutionHires, submit, submit_txt2img_upscale, selected_gallery_index, restoreProgressTxt2img, switch_to_img2img, switch_to_sketch, switch_to_inpaint, switch_to_inpaint_sketch, currentImg2imgSourceResolution, submit_img2img, restoreProgressImg2img, showSubmitInterruptingPlaceholder, confirm_clear_prompt, update_txt2img_tokens, update_img2img_tokens, closePopup, popupId, extract_image_from_gallery, switch_to_txt2img, switch_to_extras, recalculate_prompts_txt2img, recalculate_prompts_img2img, extraNetworksUnrelatedTabSelected, extraNetworksTabSelected, applyExtraNetworkFilter, setupAllResizeHandles, extraNetworksRefreshSingleCard, selected_gallery_index_id, setRandomSeed, submit_extras, extensions_apply, extensions_check, config_state_confirm_restore, populateLicense, download_localization, restart_reload, inputAccordionChecked, modelmerger, clickLoraRefresh` — plus the `ForgeCanvas` constructor. Sites 11, 13-19, 26-28, 30, 32-35, 41, 44, 47 pass inline `function(...)`/arrow expressions rather than bare names, so the runtime must `eval`/`new Function` arbitrary JS source).

### 4.1 Python-generated inline HTML handlers (also name-bound)

| Source | Handler |
|---|---|
| `html/extra-networks-card.html:1` (`ui_extra_networks.py:268-280`) | `cardClicked('<tab>', <prompt js-str>, <neg js-str>, <allow_neg>)` |
| `ui_extra_networks_checkpoints.py:33` | `return selectCheckpoint(<name>)` (checkpoint card) |
| `ui_extra_networks.py:345` | `return saveCardPreview(event, '<tab>', '<local_preview>')` |
| `ui_extra_networks.py:543` | `extraNetworksSearchButton("<tab>", "<page>", event)` (dirs view) |
| `html/extra-networks-tree-button.html:3` | `extraNetworksTreeOnClick(event,'<tab>','<page>');{onclick_extra}` |
| `html/extra-networks-pane.html:11-48` | `extraNetworksControlSortOnClick` ×4, `extraNetworksControlSortDirOnClick`, `extraNetworksControlTreeViewOnClick`, `extraNetworksControlRefreshOnClick` |
| `html/extra-networks-copy-path-button.html` | `extraNetworksCopyCardPath(event)` |
| `html/extra-networks-metadata-button.html` | `extraNetworksRequestMetadata(event,'<page>')` |
| `html/extra-networks-edit-item-button.html` | `extraNetworksEditUserMetadata(event,'<tab>','<page>')` |
| `html/footer.html:6,8` | `showProfile('./internal/profile-startup')`; `gradioApp().getElementById('settings_restart_gradio').click()` |
| `modules/options.py:40-42` (`OptionInfo.js`) | `<a onclick='settingsHintsShowQuicksettings(); return false'>` (`shared_options.py:457`) |
| `modules/ui_extensions.py:148,182,518` | `toggle_all_extensions(event)`, `toggle_extension(event)`, `install_extension_from_index(this,'<url>')` |

---

## 5. Server endpoints called by core frontend

| Endpoint | Method | Caller | Defined |
|---|---|---|---|
| `./internal/progress` | POST JSON | progressbar.js:130,189 | `modules/progress.py:82` |
| `./internal/ping` | GET | ui.js:391 | `ui.py:975` |
| `./internal/quicksettings-hint` | GET | ui_settings_hints.js:51 | `ui.py:973` |
| `./internal/profile-startup` | GET | footer → profilerVisualization.js:160 | `ui.py:977` |
| `./sd_extra_networks/metadata` | GET | extraNetworks.js:701 | `ui_extra_networks.py:170` |
| `./sd_extra_networks/get-single-card` | GET | extraNetworks.js:746 | `ui_extra_networks.py:171` |
| `./sd_extra_networks/thumb` | GET (img src) | card HTML | `ui_extra_networks.py:168` |
| `file=<path>?<mtime>` | GET | all script/link tags, `notification.mp3`, canvas assets | Gradio file route (must be re-provided) |
| `webui-assets/css/sourcesanspro.css` | GET | style.css:3 `@import` | webui-assets repo mount |
| `raw.githubusercontent.com/...LICENSE` (14) | GET | populateLicense.js | external |
| `/internal/sysinfo(-download)`, `/internal/pending-tasks` | — | not called by core JS (links/API) | `ui.py:987-988`, `progress.py:81` |

Plus implicit Gradio channels the JS piggybacks on: clicking hidden Gradio buttons (`*_token_button`, `*_generation_info_button`, `change_checkpoint`, `img2img_update_resize_to`, `*_extra_refresh_internal`, `*_save_preview`, `install_extension_button`, `settings_show_one_page`, `settings_restart_gradio`, `save_txt2img/img2img`, user-metadata `_button`), and writing hidden textboxes (`change_checkpoint_text`, `*_preview_filename`, `extension_to_install`, metadata `_name`, LogicalImage pairs) + `updateInput`.

---

## 6. `opts` keys consumed by core JS (31)

`ctrl_enter_interrupt, confirm_leave, show_progressbar, show_progress_in_title, live_preview_refresh_period, prevent_screen_sleep_during_generation, js_modal_lightbox, js_modal_lightbox_initially_zoomed, js_live_preview_in_modal_lightbox, js_modal_lightbox_gamepad, js_modal_lightbox_gamepad_repeat, keyedit_precision_attention, keyedit_precision_extra, keyedit_delimiters, keyedit_delimiters_whitespace, keyedit_move, undo_redo, prompt_debounce, no_spellcheck, keep_alive, remove_image_on_hover, disable_token_counters, extra_networks_add_text_separator, lora_filter_disabled, use_old_hires_fix_width_height, notification_volume, return_grid, sd_checkpoint_hash, _comments_before, _comments_after, _categories`; Python-side baked into canvas: `forge_canvas_plain, forge_canvas_plain_color, forge_canvas_toolbar_always, forge_canvas_height, forge_canvas_consistent_brush`; card rendering: `extra_networks_card_height/width/text_scale`, `extra_networks_tree_view_default_width`.

The SPA needs a reactive `opts` store fed initially and after "Apply settings" / quicksettings changes (today: `settings_json` textarea set by Python outputs).

---

## 7. Gradio-internal DOM coupling (must disappear or be emulated)

`gradio-app` shadowRoot; `.tab-nav > button.selected`, `.tabitem` + inline `style.display`; `button[aria-selected]`, `#<id>-button` (tab buttons); `#<id> > label > textarea` (Textbox), `label > input` (search), `input[type=number|range]` within Slider (with 3-level parent nesting assumption in aspectRatioOverlay.js:95-100); Gallery: `.gradio-gallery`, `.thumbnails`, `.thumbnail-item.thumbnail-small`, `.selected`, `> button > button > img`; Image: `[data-testid="image"]`, `.modify-upload`, `.touch-none`, `button[aria-label="Remove Image"]`, `input[type=file]`; Dropdown multiselect: `div.token`, `div.token-remove`, `div.wrap-inner`, `ul.options li.item`; Accordion: `.label-wrap(.open)`; Audio: `#waveform > div` shadowRoot `<audio>`; `.block`, `.hide`, `.main`, `.gradio-container`; `window.gradio_config.components[].props.{elem_id,webui_tooltip,placeholder}`, `component-<id>` fallback ids, `gradio_config.version`. CSS adds `gradio-<blockname>` classes to every component (`gradio_extensions.py:17-25`) — `style.css`, extensions' CSS and `ui_settings_hints.js` depend on those (`.gradio-checkbox`, `.gradio-radio`, `.gradio-dropdown`, `.gradio-gallery`, `.gradio-video`, `.gradio-button` …).

**Recommendation:** the SPA should keep emitting `id=elem_id`, `gradio-<type>` + `elem_classes`, the Textbox `label > textarea` nesting, `.tabitem` / `.tab-nav button.selected` and Gallery `.thumbnail-item` structures as a **compatibility DOM contract** for the 30 extension JS files, even if core features are reimplemented natively.

---

## 8. `style.css` and template inventory

`style.css` (1,865 lines) sections: general Gradio fixes (:5-170, var overrides `--checkbox-label-gap`, `--block-background-fill`, `.hidden`, `.compact`, dropdown options z-index, overflow fixes); styled components (:171-231: tool buttons, `.gradio-button.tool`, `.inactive`); txt2img/img2img layout (:232-463: token counters, html-log/performance, generate box + interrupt/skip/interrupting states, compact prompt layout, ≥2500 px media query); settings (:464-714: `#quicksettings`, `#settings` two-column nav, `.settings-category`, `.settings-comment`, `#settings_search`); live preview/progress (:715-768: `.progressDiv`, `.livePreview` crossfade); global popup (:769-821: `.global-popup`, `.popup-table`, `.popup-metadata`); lightbox (:822-942 incl. `#imageARPreview`, theme-var opacities); context menu (:943-971); extensions tab tables (:972-1043); footer replacement (:1044-1062); extra networks cards/panes (:1063-1332); sticky results columns + resize handle (:1333-1371); extra-network tree view, search/sort/tree/refresh icons (data-URI SVG masks) (:1372-1865). Dark mode via `.dark` prefix (Gradio sets `.dark` on body when `?__theme=dark`).

`html/`: 11 templates — `extra-networks-{card, copy-path-button, edit-item-button, metadata-button, no-cards, pane, pane-dirs, pane-tree, tree-button}.html`, `footer.html`, `licenses.html` (+ `card-no-preview.jpg`, `ui.webp`). Server-side `str.format` templates producing HTML strings for `gr.HTML` — the SPA can either keep rendering them as trusted HTML (then onclick globals must exist) or replace extra networks with a native component fed by JSON.

---

## 9. Localization

- `localizations/` (root, `--localizations-dir`): **empty** (only placeholder file). Real data: `extensions-builtin/sd-webui-language-diffusion/localizations/*.json` — 10 locales (de_DE, es_ES, fr_FR, it_IT, ja_JP, ko_KR, pl_PL, pt_BR, ru_RU, zh_CN; zh_CN has 1,258 keys, no `rtl`). Merged per locale name by `modules/localization.py:9-37`, selected by `opts.localization` (`shared_options.py:456`, needs UI reload).
- Mechanism: exact full-string match of trimmed text nodes / `title` / `placeholder` → replacement, applied on load and on every DOM insertion. Keys include multi-line Markdown strings and labels with whitespace — the SPA should translate at render time (label/info/placeholder/choices/tooltip/Markdown/HTML text), honour the ignore list for model/style/upscaler option values, and support `rtl`.

---

## 10. FEATURE CHECKLIST for the native SPA

### A. App shell / runtime
- [ ] Top tabs (order `opts.ui_tab_order`, `opts.hidden_tabs`), tab change events, `?__theme=light|dark` + `--theme` + prefers-color-scheme background.
- [ ] Load & run extension `javascript/*.js` (classic, global) then `*.mjs` (module), inject all `style.css` + `user.css`, `head=` strings; provide compatibility globals (§2) and lifecycle hooks `onUiLoaded/onUiUpdate/onAfterUiUpdate/onUiTabChange/onOptionsChanged/onOptionsAvailable` with equivalent timing.
- [ ] Reactive `opts` (+`_comments_*`, `_categories`) and `window.localization`.
- [ ] `updateInput`-style bridge: DOM edits by extension JS on our inputs must propagate to state.
- [ ] Execute `js=` hooks: string name or expression, pre-`fn`, args = input values, return replaces inputs, js-only events, `load` hooks, `.then/.success` chains.
- [ ] Reload UI (Reloading… screen, poll `/internal/ping`), confirm-before-leave, keep-alive in background tab, footer (versions, API link, startup profile, checkpoint hash link to civitai).

### B. Prompt editing
- [ ] Ctrl/Cmd+↑/↓ attention weight editing (word/paren/`<lora>` auto-select, legacy paren conversion, precision opts).
- [ ] Alt+←/→ reorder comma-separated tags (`keyedit_move`).
- [ ] Custom undo/redo (16 states, 600 ms) on 6 prompt boxes (`undo_redo`).
- [ ] Prompt input debounce with flush on blur/modifier keys (`prompt_debounce`).
- [ ] Disable spellcheck/autocorrect (`no_spellcheck`).
- [ ] Token counters (pos/neg, "n/75", 1 s idle, include styles; `disable_token_counters`), refreshed after style apply / paste.
- [ ] Drop image file/URL onto prompt → read PNG info into prompt.
- [ ] Clear prompt with confirm; paste-params button; prompt placeholders with shortcut hints.
- [ ] Styles multiselect click-on-token-to-remove; style editor dialog (popup), delete confirm, apply/materialize.
- [ ] Prompt box styles (Default/Compact/Scrollable/Accordion) and compact layout moving prompts into extra-network tabs.

### C. Generation / progress
- [ ] Generate / Interrupt / Skip / Interrupting… state machine per tab (txt2img, img2img, extras, merger).
- [ ] Task ids `task(...)`, progress polling `/internal/progress` (%, ETA, textinfo), progress bar toggle, progress in document title, inactivity timeout.
- [ ] Live preview in gallery with crossfade (period opt), live preview in lightbox toggle.
- [ ] Restore progress after page reload (localStorage task id, restore button).
- [ ] Generate forever / cancel (context menu on Generate & Interrupt).
- [ ] Keyboard: Ctrl+Enter generate (or interrupt+regenerate / interrupt only per `ctrl_enter_interrupt`), Alt+Enter skip, Esc interrupt.
- [ ] Screen wake lock during generation.
- [ ] Hires-fix control greying (`onCalcResolutionHires`), hires resolution preview.
- [ ] Width/height: paste "W x H", swap, img2img detect size, "resize by" preview from source size, aspect-ratio overlay rectangle on img2img source while dragging sliders.
- [ ] Seed: random (🎲 → -1), reuse from selected image (♻).
- [ ] Hires upscale button selecting resulting image (`hires_button_gallery_insert`).

### D. Gallery / image viewer
- [ ] Gallery selection index available to handlers (save, save zip, open folder, send-to, reuse seed, upscale, save preview, replace preview).
- [ ] Infotext panel updates on selection change / arrow keys / lightbox close.
- [ ] Lightbox: click to open (`js_modal_lightbox`), initial zoom, zoom toggle, tile preview, save (S), prev/next (←/→, buttons), Esc close, middle-click open in new tab, follows live preview, icon/toolbar opacity theme vars.
- [ ] Gamepad stick / horizontal wheel navigation.
- [ ] Send to img2img / inpaint / extras (+ txt2img from PNG Info) with tab switch; copy between img2img modes with tab switch.
- [ ] Paste image from clipboard into first empty visible image field; drop-replace on images; remove-on-hover for extras/PNG Info.
- [ ] Notifications: sound (volume), browser notification when unfocused (count minus grid), permission request button.

### E. Extra networks
- [ ] Per-tab panes (txt2img/img2img) for Lora / Checkpoints / Textual Inversion (+ extension pages); controls in tab bar shown only for active page.
- [ ] Cards: preview img/video, name, description, search terms, copy-path, metadata (flattened JSON table popup), edit user metadata (popup editor, refresh single card), save/replace preview from gallery.
- [ ] Card click toggles insertion of prompt text into active (focused) prompt; positive+negative pairs; separator opt; checkpoint card selects model.
- [ ] Search (multi-term AND, `search_only` ≥4 chars), arch/preset filter (`lora_filter_disabled`), sort (path/name/created/modified, asc/desc), tree view toggle with folders (expand/select → filter), dirs buttons view, refresh, card size/text scale opts, resizable tree column.

### F. Settings
- [ ] Settings pages with left nav + category headings, search (500 ms) across all pages, Show all / single page, scroll to top on page change, comments before/after (HTML, links, `[info]` JS link → quicksettings list popup), apply, quicksettings bar, checkpoint/VAE/preset selectors, "?" help link for modules.
- [ ] Actions: reload UI, request notifications, download localization template, unload/list models, licenses tab (fetch 14 licenses), sysinfo.

### G. img2img / inpaint canvas (ForgeCanvas, also ControlNet)
- [ ] Upload (button/click/drag/paste-on-hover), remove, center (R), reset strokes (Ctrl+X), undo/redo (Ctrl+Z/Y, 16), maximize/minimize (F).
- [ ] Brush color (E), width/opacity/softness sliders, W/A/S + wheel adjust, Shift eraser, opacity 0 = eraser, soft brush, brush cursor indicator, consistent brush across zoom.
- [ ] Wheel zoom at cursor, right-drag pan, checkerboard/plain background, auto-hide/static toolbar, height opt, contrast (checker) mask scribbles, fixed color/width/alpha/softness variants, no_upload / no_scribbles variants.
- [ ] Two-way binding of background + foreground PNG data URLs with server (including server-pushed images from send-to / copy / ControlNet preprocessor output), preserving PNG info (`LogicalImage.infotext`).

### H. Hints / localization / tooltips
- [ ] Static tooltip map (28 labels) + per-component `tooltip=` (`webui_tooltip`), localized.
- [ ] Full localization by exact-string dictionary (labels, info, placeholders, choices except ignore list, Markdown/HTML text, tooltips), RTL, localization template export.

### I. Layout utilities
- [ ] Resizable split columns (`ResizeHandleRow`), double-click reset, min widths, mobile collapse, window-resize rescale.
- [ ] InputAccordion (checkbox in accordion header, linked/unlinked open state, disabled sync, `-extra` header slot, open state persisted via ui-config `input-accordion-open`).
- [ ] Global popup/modal (`popup`, `popupId`, `closePopup`, Esc) used by styles editor, metadata, profile, quicksettings hints, extension code.
- [ ] Context menu registry API for extensions.
- [ ] Sticky results column, sticky/compact generate box.

### J. Extensions tab
- [ ] Installed table with enable/update checkboxes (+ toggle all), Apply & restart, Check for updates (with progress), install from index/URL, config state restore with confirm.

---

## 11. Latent bugs / dead code found (decide parity vs. fix)

1. `extraNetworks.js:55-63` reads `#forge_ui_preset` radio `<input>`s, but `forge_ui_preset` is a **Dropdown** (`modules_forge/main_entry.py:69`) → `UIresult` always 3 ("all"); the architecture filter is effectively disabled. Also the mapping only knows SD1/SDXL/Flux.
2. `extraNetworks.js:85` compares `opts.lora_filter_disabled == True` — works only because `canvas.js:856` defines global `const True = true`.
3. `imageMaskFix.js` targets the old Gradio sketch DOM (`#img2maskimg .touch-none canvas`) — dead with ForgeCanvas.
4. `ui.js` `switch_to_*` / `switch_to_img2img_tab` use **positional** `#tabs` button indices 0/1/2 — wrong if `ui_tab_order` or `hidden_tabs` reorder/hide tabs; `generationParams.js:29` compares English tab text.
5. `ui.js` `ask_for_style_name`, `updateImg2imgResizeToTextAfterChangingImage`, `switchWidthHeight`, `progressbar.js` `rememberGallerySelection/getGallerySelectedIndex`, `extraNetworksTreeProcessFileClick`, `toggleCss` — unused by core (may be used by third-party extensions; keep as shims).
6. `canvas.js:607-611` silently refuses to load images unless `gradio_config.version` starts with "4." — must be satisfied or removed in the SPA.
7. `GradioTextAreaBind.listen` polls each canvas's 2 textareas every 100 ms (8+ intervals with ControlNet) — replace with state subscriptions.
8. `toggleCss` bug: `style.innerHTML == ""` (comparison, not assignment).
9. `extraNetworks.js:748` `page.replace(" ", "_")` replaces only first space.
