# Task D: Runtime mechanics the Remi Gradio-compat server must reproduce

Scope: everything that happens at **runtime** between the Python UI build, the event server, and the browser,
traced through the core of Forge Neo Remi (branch `remi-webui`, currently on `gradio==4.40.0`).
Reference implementation behaviour comes from `/tmp/remi-testenv/lib/python3.13/site-packages/gradio`
(4.40). The `gr4:` prefix marks a path in that package.

All line numbers refer to the repo at the time of writing. "Compat" means the Remi-owned
Gradio-compatible package, its server, and the React SPA.

---

## 0. Executive summary: what the compat runtime must provide

1. **The full Blocks build API runs in-process and without a browser.** Build the tree from nested sub-`Blocks`
   (`interface.render()`), `Context.root_block`, `render=False` plus a later `.render()`, manual `__enter__()` and `__exit__()`
   calls with no arguments, and mutation of component attributes after construction (`ui_loadsave`).
   The API-only mode (`--nowebui`) also calls `ui.create_ui()` headless (`modules/api/api.py:270-271`).
2. **The same event semantics as Gradio 4.40.** These are:
   - `fn` and `js` (with `_js` as an alias) on all events, `.then`, and `.success`.
   - The JS hook receives *input values followed by output values*. Its result is wrapped into an array under the exact rule in §4.3.
   - Special-argument injection of `gr.Request` by type annotation. The signature is followed through `functools.wraps`.
   - Missing positional parameters are filled with their defaults or with `None`.
   - Output count is tolerant: more returned values than outputs are ignored, and fewer is an error.
   - A single output is wrapped.
   - Updates use `{"__type__":"update"}`. `None` props are dropped. Session-level component copies are rebuilt from constructor args.
   - `gr.State` holds per-session values.
   - Generators stream, and `gr.Progress` is supported.
   - `gr.Info` and `gr.Warning` produce toasts.
   - `queue=False` is accepted, `trigger_mode` is honoured, and roughly 32 events run concurrently in a thread pool.
3. **Python-side value contracts for each component** (preprocess and postprocess). Subclasses such as `LogicalImage`
   override `preprocess` and `postprocess`, so the server must call the component's own methods.
   PIL images go through `gradio.processing_utils.save_pil_to_cache`, which webui monkeypatches
   (`modules/ui_tempdir.py:116-121`). The compat must call an overridable hook, or implement the same
   `already_saved_as` and PNG-info semantics itself.
4. **File transport.**
   - `/upload` puts files in the upload dir. `/file=<path>` serves files under an allowlist that combines `temp_file_sets`,
     `allowed_paths` (default `[data_path, canvas dir]`), the upload dir, and the cache.
   - Relative `file=javascript/...` URLs are used in the injected `<head>`.
   - FileData JSON is `{path,url,size,orig_name,mime_type,is_stream,meta}`.
5. **The server's own non-Gradio routes are kept as-is.** These are FastAPI routes: `/internal/progress`,
   `/internal/pending-tasks`, `/internal/ping`, `/internal/quicksettings-hint`, `/internal/sysinfo[-download]`,
   `/internal/profile-startup`, `/webui-assets`, `/sd_extra_networks/{thumb,cover-images,metadata,get-single-card}`,
   and `/sdapi/v1/*`.
6. **Lifecycle.**
   - `launch()` returns `(app, local_url, share_url)` and does not block.
   - Middleware can be edited after launch.
   - `app_started_callback(demo, app)` may register **new events after launch** (agent-scheduler does this with `with root:`).
   - `demo.close()` followed by a rebuild and relaunch in the same process handles "Reload UI".
   - Model work runs on the process main thread (`main_thread.loop()`).
7. **DOM contracts relied on by the bundled JS** are listed in §17. Examples: hidden components still exist in the DOM,
   a textarea sits under `#elem_id`, `gradio-<type>` classes are present, `#tabs > .tab-nav > button.selected`,
   `window.gradio_config.components[].props`, the gallery thumbnail classes, and the `settings_json` textarea setter interception.

---

## 1. Startup and launch lifecycle

### 1.1 Process entry

| Step | Location | Notes |
|---|---|---|
| `launch.py:main()` → `launch_utils.start()` | `modules/launch_utils.py:543-559` | Imports `webui`. It calls `webui.api_only()` if `--nowebui` is set (`:552-553`), otherwise `webui.webui()` (`:555`). **Then the main thread runs `main_thread.loop()` forever (`:557-559`).** The UI server runs on a daemon thread. |
| Module import side effects | `webui.py:16-28` | Runs `initialize.shush()`, `initialize_forge()`, `initialize.imports()`, `check_versions()`, and `initialize.initialize()`. |
| `initialize.imports()` | `modules/initialize.py:20-38` | Sets `GRADIO_ANALYTICS_ENABLED=False` and runs `import gradio`, `shared_init.initialize()`, and `from modules import gradio_extensions, processing, ui`. **Importing `gradio_extensions` installs every monkeypatch listed in §15** (`modules/gradio_extensions.py:88-178`). |
| `initialize.initialize()` | `modules/initialize.py:54-78` | Sets up models, then calls `initialize_rest()`. |
| `initialize_rest()` | `modules/initialize.py:81-…` | Loads samplers, extensions, and the config state. Calls `scripts.load_scripts()` (`:119`). On reload, optionally `importlib.reload`s every `modules.ui*` module (`:121-124`). The compat package must survive this reload, and module-level patches must be idempotent. |
| `ui.py` import-time | `modules/ui.py:30-54` | Sets warning filters, patches `gradio.utils.version_check` and `get_local_ip_address` with no-ops (`:45-48`), and optionally connects ngrok (`:50-54`). The compat should expose `gradio.utils` with those attributes, or `ui.py` must be edited. |

### 1.2 `webui_worker()`: `webui.py:70-188`

This is a loop (`while 1`). Each iteration:

1. `ui_tempdir.cleanup_tmpdr()` runs if `opts.clean_temp_dir_at_start` is set (`:85-87`). It deletes `*.png` files under `opts.temp_dir`.
2. `script_callbacks.before_ui_callback()` (`:89`).
3. `shared.demo = ui.create_ui()` (`:92`). See §2.
4. `shared.demo.queue(default_concurrency_limit=32)` runs unless `--no-gradio-queue` is set (`:95-96`).
5. `demo.launch(...)` (`:109-127`) is called with these kwargs, which the compat must accept:
   `share`, `server_name`, `server_port`, `ssl_keyfile`, `ssl_certfile`, `ssl_verify`, `debug`,
   `auth` (a list of `(user, pass)` from `initialize_util.get_gradio_auth_creds()`), `inbrowser`,
   `prevent_thread_lock=True`, `allowed_paths=cmd_opts.gradio_allowed_path + [canvas_js_root_path]`,
   `app_kwargs={"docs_url": "/docs", "redoc_url": "/redoc", "exception_handlers": {Exception: _handle_exception}}`,
   and `root_path=f"/{subpath}"`.
   - **The call returns the tuple `(app, local_url, share_url)`**, where `app` is a FastAPI instance.
6. `app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != "CORSMiddleware"]` (`:135`).
   - Note: Gradio 4.40 adds `CustomCORSMiddleware` (gr4:`routes.py:253`), so this filter is a **no-op** today. The compat app should not add an open CORS middleware.
7. `initialize_util.setup_middleware(app)` (`modules/initialize_util.py:196-201`). It inserts GZip and CORS at `user_middleware[0]`, then calls `app.build_middleware_stack()` **after the server started**. The compat app must be a Starlette or FastAPI app whose middleware stack can be rebuilt on the fly.
8. `progress.setup_progress_api(app)` (`modules/progress.py:80-82`), `ui.setup_ui_api(app)` (`modules/ui.py:963-992`),
   `create_api(app)` when `--api` is set (`webui.py:42-47`), and `ui_extra_networks.add_pages_to_demo(app)` (`modules/ui_extra_networks.py:167-171`).
9. `script_callbacks.app_started_callback(shared.demo, app)` (`:151`), wrapped by `extension_route_guard.snapshot_routes` and `guard_new_routes` (`:150-154`). Callbacks receive the **Blocks object** and the app. See §16 for what extensions do with `demo`.
10. Server command loop: `shared.state.wait_for_server_command(timeout=5)` (`:159-169`).
    - On `"stop"`, it calls `shared.demo.close()` and breaks.
    - On `"restart"` ("Reload UI" button, `ui_settings.py:257-262`), it calls `demo.close()`, sleeps 0.5 s, then runs
      `app_reload_callback`, `script_unloaded_callback`, and `initialize.initialize_rest(reload_script_modules=True)`, and loops back to step 1.
    - So **the compat server must fully stop listening on the port and be re-launchable in the same process**, with fresh Blocks and fresh routes.
11. The JS `restart_reload()` (`javascript/ui.js:385-405`) polls `./internal/ping` until it answers, then reloads the page.

### 1.3 `api_only_worker()`: `webui.py:50-67`

This mode uses a plain `FastAPI(exception_handlers=...)`, `setup_middleware`, `Api(app, queue_lock)`, `before_ui_callback`,
and `app_started_callback(None, app)` (**demo is `None`**), then `api.launch(...)` (uvicorn).
`Api.__init__` calls `ui.create_ui()` when script runners are empty (`modules/api/api.py:270-271`).
So **the whole UI build must work headless, with no server and no `shared.demo`**.
`scripts_postproc.create_args_for_run` also builds a throwaway `gr.Blocks()` for API extras (`modules/scripts_postprocessing.py:210-213`).

### 1.4 HTML shell injection

- `reload_javascript()` (`modules/ui_gradio_extensions.py:54-66`) replaces `gr.routes.templates.TemplateResponse` with a wrapper that:
  - injects `javascript_html()` and `<meta name="referrer" content="no-referrer"/>` before `</head>`;
  - injects `css_html()` before `</body>`.
- `javascript_html()` (`:13-29`) produces, in order:
  1. an inline `<script>` with localization JSON;
  2. `script.js`;
  3. every `javascript/*.js` from the core and active extensions, in `scripts.list_scripts("javascript", ".js")` order;
  4. every `.mjs` as `type="module"`;
  5. an optional `set_theme(...)`.
- URLs are **relative** `file=<path relative to cwd>?<mtime>` (`webpath`, `:9-10`; `util.truncate_path`, `modules/util.py:54-61`).
- `css_html()` (`:32-51`) adds every `style.css` (core plus extensions), `user.css`, and a `<style>` built from theme vars `background_fill_primary[_dark]` (via `shared_gradio_themes.resolve_var`).
- `ForgeCanvas` adds `canvas_head` (canvas.css and canvas.js as `file=<abs path>?mtime`) through `gr.Blocks(head=canvas_head)` (`modules/ui.py:208, 483, 887`; `modules_forge/forge_canvas/canvas.py:50-61`).
- `reload_javascript()` is also called from the "Reload custom script bodies" button (`modules/ui_settings.py:251-255`).
- **Compat contract:** the SPA's `index.html` must contain the same injected head and body fragments. Rebuild them at
  `create_ui()` time and when the button is pressed. The `Blocks(head=...)` of the root demo must be emitted, and `title="Stable Diffusion"` is used.

### 1.5 Theme

`gr.Blocks(theme=shared.gradio_theme, ...)` (`modules/ui.py:887`).

- `shared.gradio_theme` is a `gr.themes.Base`, `Default`, or a hub theme. It is loaded via `ThemeClass.load` or `from_hub` (`modules/shared_gradio_themes.py:44-71`).
- The two attributes `sd_webui_modal_lightbox_*` are appended to the theme object.
- `resolve_var()` reads theme attributes recursively (`:74-…`).
- The compat must provide `gr.themes.{Base,Default,ThemeClass}` objects that expose CSS variable attributes. It may degrade to a static Remi theme, but `resolve_var` must not crash.

---

## 2. `create_ui()`: composition model (`modules/ui.py:193-930`)

### 2.1 Build order

| # | Code | Line |
|---|---|---|
| 1 | `reload_javascript()` | 197 |
| 2 | `parameters_copypaste.reset()` clears global paste registries | 199 |
| 3 | `main_entry.capture_callbacks()` resets the refresh-callback registry | 200 |
| 4 | `UiSettings().register_settings()` calls `ui_settings_callback` (extensions add OptionInfo) | 202-203 |
| 5 | `scripts.scripts_current = scripts_txt2img; initialize_scripts(False)` | 205-206 |
| 6 | `with gr.Blocks(analytics_enabled=False, head=canvas_head) as txt2img_interface:` | 208 |
| 7 | `scripts_current = scripts_img2img; initialize_scripts(True)`; img2img Blocks | 480-483 |
| 8 | `scripts_current = None` | 829 |
| 9 | extras Blocks (`ui_postprocessing.create_ui()`) | 831-832 |
| 10 | PNG Info Blocks | 834-860 |
| 11 | `UiCheckpointMerger()` creates its own Blocks in `__init__` | 862 |
| 12 | `UiLoadsave(cmd_opts.ui_config_file)`; snapshot `ui_settings` | 864-865 |
| 13 | `settings.create_ui(loadsave, dummy_component)` creates the settings Blocks | 867 |
| 14 | `interfaces` = 5 built-ins, then `script_callbacks.ui_tabs_callback()` (extension tabs: list of `(Blocks, label, id)`), then Settings, then Extensions | 869-881 |
| 15 | `shared.tab_names` filled | 883-885 |
| 16 | **Root** `with gr.Blocks(theme=..., analytics_enabled=False, title="Stable Diffusion", head=canvas_head) as demo:` | 887 |
| 17 | `settings.add_quicksettings()` (inside it, `main_entry.make_checkpoint_manager_ui()`) | 888, `ui_settings.py:296-303` |
| 18 | `parameters_copypaste.connect_paste_params_buttons()`: **all paste and send-to events are registered on the root demo** | 890 |
| 19 | `with gr.Tabs(elem_id="tabs")`, sorted by `opts.ui_tab_order` and skipping `opts.hidden_tabs`. `with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"): interface.render()`. `loadsave.add_block(interface, ifid)` runs for every id except extensions and settings | 892-903 |
| 20 | `loadsave.add_component("webui/Tabs@tabs", tabs)`; `loadsave.setup_ui()` | 905-907 |
| 21 | optional hidden `gr.Audio(value=notification.mp3)` | 909-910 |
| 22 | footer `gr.HTML` (with `gr.__version__` in `versions_html`, `:957`) | 912-914 |
| 23 | `settings.add_functionality(demo)` | 916 |
| 24 | `settings.text_settings.change(...)`, `demo.load(...)` for `image_cfg_scale` visibility | 918-920 |
| 25 | `modelmerger_ui.setup_ui(dummy_component=..., sd_model_checkpoint_component=main_entry.ui_checkpoint)` | 922 |
| 26 | `main_entry.forge_main_entry()` wires preset events. It uses `Context.root_block.load` | 924, `main_entry.py:236-296` |
| 27 | `loadsave.dump_defaults()` runs if the settings changed; `demo.ui_loadsave = loadsave` | 926-928 |

### 2.2 Required Blocks semantics

- **Sub-Blocks built standalone, then `render()`ed into the root.**
  - In gr4 (`blocks.py:1333-1382`), `render()` merges the sub-Blocks' `blocks` dict and `fns` (dependencies) into the root with an id offset,
    and moves `temp_file_sets`.
  - Events registered while a sub-Blocks was the root (including `interface.load(...)` and `Context.root_block.load(...)`) must
    therefore work after merge. Examples:
    - `ui_extra_networks.create_ui` calls `interface.load(fn=pages_html, outputs=ui.pages)` on the txt2img and img2img sub-Blocks (`modules/ui_extra_networks.py:784`);
    - `ForgeCanvas` uses `Context.root_block.load(None, js=...)` (`canvas.py:138`) inside the img2img sub-Blocks.
  - **Every `load` dependency, regardless of the Blocks it was registered on, must fire once per page load** (per session), in registration order.
- **Cross-Blocks component references.** For example, `dummy_component`, a hidden `gr.Textbox` created inside `txt2img_interface` (`ui.py:211`),
  is an input in img2img (`:701`), settings (`ui_settings.py:183-185`), and the model merger (`ui.py:922`). The compat must
  allow a component to be an input or output of events registered on another Blocks, as long as both end up in the same root.
- **`Context.root_block`** must be the Blocks currently being built.
  - `.load(...)` is used on it at `modules_forge/main_entry.py:81, 293` and `modules_forge/forge_canvas/canvas.py:138`.
  - `.blocks.values()` is used at `modules/anima_feature_conflicts.py:411-415`.
- **`render=False` plus a later `.render()`.** `Toprow.create_submit_box` builds the generate Row with `render=not is_compact` (`modules/ui_toprow.py:111`)
  and later calls `self.submit_box.render()` inside the results column (`:77`). Re-parenting must keep its events.
- **Manual context management with zero-argument `__exit__()`.** Examples:
  - `extra_tabs.__enter__()` and `extra_tabs.__exit__()` (`ui.py:214-215, 475, 486-487, 824`);
  - `current_tab.__exit__()` and `current_row.__exit__()` (`ui_settings.py:169-193`);
  - `CURRENT_ROW.__enter__()` and `__exit__()` (`ui_settings.py:54-59`);
  - `_container.__enter__()` and `__exit__()` (`ui_toprow.py:91-101`);
  - `ExitStack().enter_context(gr.Accordion(...))` (`ui.py:218-221`).
- **Orphan blocks**: `gr.Group()` is created but never entered (`ui_settings.py:173`).
- **`gr.Tab` and `gr.TabItem` are the same class**, with `id=` (string), `elem_id`, `elem_classes`, `label`, and a `.select` event. `gr.Tabs` has `elem_id` and a `selected` attribute (`ui_loadsave.py:141-142`).
- **Children introspection**: `block.children` and `block.parent` (`ui_loadsave.py:147-152`; agent-scheduler pops and re-adds children, §16).
- **Re-entering the root after launch**: `with root:` in an `app_started` callback adds events. In gr4, `Blocks.__exit__` regenerates `self.config` (`blocks.py:2073`).
  The compat must regenerate the client config so that **page loads after the change see the new dependencies**.

### 2.3 Tab and visibility policy

- `opts.ui_tab_order` sorts the tabs; `opts.hidden_tabs` skips them (they are never rendered) (`ui.py:893-898`).
- JS assumes `#tabs` nav buttons 0, 1, and 2 are txt2img, img2img, and extras (`javascript/ui.js:58-93`).

---

## 3. Generation wiring (txt2img / img2img / upscale / extras)

### 3.1 txt2img inputs (`modules/ui.py:341-368`)

There are 26 fixed inputs plus `custom_inputs` (`scripts_txt2img.setup_ui()`, `:316`).

| idx | component | type → Python value |
|---|---|---|
| 0 | `dummy_component` (hidden Textbox, `:211`) | str. **The JS hook replaces it with `id_task`.** |
| 1 | `toprow.prompt` Textbox | str |
| 2 | `toprow.negative_prompt` Textbox | str |
| 3 | `toprow.ui_styles.dropdown` Dropdown(multiselect) | list[str] |
| 4 | `batch_count` Slider | int or float (JSON number) |
| 5 | `batch_size` Slider | int or float |
| 6 | `cfg_scale` Slider | float |
| 7 | `distilled_cfg_scale` Slider | float |
| 8 | `height` Slider | int or float |
| 9 | `width` Slider | int or float |
| 10 | `enable_hr` InputAccordion (hidden Checkbox) | bool |
| 11 | `denoising_strength` Slider | float |
| 12 | `hr_scale` Slider | float |
| 13 | `hr_upscaler` Dropdown | str |
| 14 | `hr_second_pass_steps` Slider | int |
| 15 | `hr_resize_x` Slider | int |
| 16 | `hr_resize_y` Slider | int |
| 17 | `hr_checkpoint_name` Dropdown | str |
| 18 | `hr_additional_modules` Dropdown(multiselect) | list[str] |
| 19 | `hr_sampler_name` Dropdown | str |
| 20 | `hr_scheduler` Dropdown | str |
| 21 | `hr_prompt` Textbox | str |
| 22 | `hr_negative_prompt` Textbox | str |
| 23 | `hr_cfg` Slider | float |
| 24 | `hr_distilled_cfg` Slider | float |
| 25 | `override_settings` Dropdown(multiselect, hidden by default) | list[str] of `"Infotext key: value"` |
| 26… | `scripts_txt2img.inputs`: `[script Dropdown(type="index") → int index] + each script's controls in creation order` | per component |

- **Outputs** (`:370-376`): `[gallery, player (Video), generation_info (hidden Textbox), infotext (HTML), html_log (HTML)]`.
- **Registration** (`:378-387`):
  ```python
  txt2img_args = dict(fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, None, "", ""]),
                      _js="submit", inputs=txt2img_inputs, outputs=txt2img_outputs, show_progress=False)
  toprow.prompt.submit(**txt2img_args).then(fn=cleanup)
  toprow.submit.click(**txt2img_args).then(fn=cleanup)
  ```
  - There are two triggers: Textbox `.submit` (Enter key, see note) and Button `.click`.
  - `.then(fn=cleanup)` has no inputs or outputs. It always runs after the main event, even when that event failed.
- **Handler signature**: `txt2img(id_task: str, request: gr.Request, *args)` (`modules/txt2img.py:164-165`).
  - The `request` is **injected** at position 1 by annotation through three `functools.wraps` layers.
  - This was verified against gr4 `helpers.special_args`: inputs `['task(x)','prompt',…]` become `['task(x)', <Request>, 'prompt', …]`.
  - `request.username` is used for `p.user` (`txt2img.py:57`).
- **Positional unpacking**: `txt2img_create_processing(id_task, request, prompt, …, override_settings_texts, *args, force_enable_hr=False)` (`txt2img.py:18`).
  - `*args` is exactly `scripts_txt2img.inputs` values.
  - `p.script_args = args` is later sliced by `script.args_from` and `script.args_to` (§12).
- **Return value**: a 5-tuple `(gr.update(value=images, visible=True) | gr.update(value=None, visible=False), gr.update(value=video_path|None, visible=…), generation_info_js: str(JSON), info_html, comments_html)` (`txt2img.py:148-157`).

### 3.2 txt2img hires upscale ✨ (`modules/ui.py:389-407`)

- `inputs = [dummy_component, output_panel.gallery, dummy_component_number (hidden Number, :212), output_panel.generation_info] + txt2img_inputs[1:]`.
- The JS hook is `submit_txt2img_upscale` (`javascript/ui.js:172-176`). It calls `submit(...)`, then sets `res[2] = selected_gallery_index()` (the hidden Number becomes the gallery index).
- Handler: `txt2img_upscale(id_task, request, gallery, gallery_index, generation_info, *args)` (`txt2img.py:160-161`).
  - `gallery` is `list[tuple[PIL.Image, caption]]` (Gallery `type="pil"`), or `None` when empty.
  - It uses `gallery[i][0].size` and `image[0].filename`, and sets `image[0].already_saved_as = filename.rsplit("?",1)[0]` (`txt2img.py:114-118`).
- Return: a 4-tuple `(gr.update(value=new_gallery, selected_index=gallery_index), json, html, html)`. `new_gallery` mixes preprocessed `(PIL, caption)` tuples and new PIL images.
- Outputs: `[gallery, generation_info, infotext, html_log]`, with `extra_outputs=[None, "", ""]`.
- `.then(fn=select_gallery_image, js="selected_gallery_index", inputs=[dummy_component], outputs=[gallery])`. The JS returns a number, which is wrapped to `[n]` because there is a backend fn and exactly 1 input. The handler returns `gr.update(selected_index=...)`. **The Gallery must support a `selected_index` prop update.**

### 3.3 img2img (`modules/ui.py:700-758`)

There are 40 fixed inputs plus custom inputs:

```
0 dummy (→ id_task) | 1 img2img_selected_tab Number (0..5; set by tab.select lambdas :558-559) |
2 prompt | 3 neg | 4 styles(list) |
5 init_img.background | 6 sketch.background | 7 sketch.foreground | 8 init_img_with_mask.background | 9 .foreground |
10 inpaint_color_sketch.background | 11 .foreground            ← LogicalImage (Textbox subclass) → PIL RGBA or None
12 init_img_inpaint gr.Image(type="pil", image_mode default RGB) | 13 init_mask_inpaint gr.Image(type="pil", image_mode="RGBA")
14 mask_blur Slider | 15 mask_alpha Slider | 16 inpainting_fill Radio(type="index")→int | 17 batch_count | 18 batch_size |
19 cfg_scale | 20 distilled_cfg_scale | 21 image_cfg_scale | 22 denoising_strength | 23 selected_scale_tab Number(0/1) |
24 height | 25 width | 26 scale_by | 27 resize_mode Radio(type="index")→int | 28 inpaint_full_res Radio(type="index")→int |
29 inpaint_full_res_padding | 30 inpainting_mask_invert Radio(type="index") | 31-33 batch dirs Textbox |
34 override_settings list | 35 img2img_batch_use_png_info (InputAccordion bool) | 36 png_info_props CheckboxGroup list[str] |
37 png_info_dir | 38 img2img_batch_source_type hidden Textbox ("upload"/"from dir", set by tab.select :550-551) |
39 img2img_batch_upload gr.Files → list[NamedString] or None
```

- `_js="submit_img2img"` (`javascript/ui.js:178-198`).
- Outputs are the same 5 as txt2img.
- Handler: `img2img(id_task: str, request: gr.Request, mode: int, …)` (`modules/img2img.py:302-303`).
- `mode` drives `Image.alpha_composite(sketch, sketch_fg)` and `fg.getchannel("A")` (`img2img.py:191-215`). So **LogicalImage must deliver `PIL.Image` in RGBA mode**.

### 3.4 Extras (`modules/ui_postprocessing.py:8-59`)

- `inputs = [dummy, tab_index (gr.State(0)), extras_image (gr.Image type="pil", image_mode="RGBA"), image_batch (gr.Files), input_dir, output_dir, show_extras_results (Checkbox), extras_video_input (Textbox), *script_inputs]` (`:43`).
  - `script_inputs` are the postprocessing scripts' dict-valued controls (`scripts_postprocessing.py:128-138, 160-170`).
- `tab_index` is a **`gr.State` used as an input and updated by `tab.select(fn=lambda: n, outputs=[tab_index], queue=False)`** (`:38-41`).
- The handler is `wrap_gradio_gpu_call(postprocessing.run_postprocessing_webui, extra_outputs=[None, ""])`, called as `(id_task, *args)` with no Request (`postprocessing.py:172-176`).
- It returns `(list[PIL], info_html, "")` (`postprocessing.py:110`).
- **It does not use `main_thread`.** It runs on the event worker thread under `queue_lock` (§5).
- Gallery batch files: `img.name` is used (`postprocessing.py:29-30`), so the **Files preprocess must yield `NamedString` (a str subclass with `.name`)**.
- `extras_image.change(fn=scripts.scripts_postproc.image_changed, queue=False)` has no inputs and no outputs (`:59`).
- `run_postprocessing` handles `isinstance(image, dict)` → `image["composite"]` (`postprocessing.py:19-20`), a legacy ImageEditor value.

### 3.5 Other generation-adjacent events

- **Interrupt and skip** (`modules/ui_toprow.py:119-128`):
  - `skip.click(fn=shared.state.skip)` is a bound method with no inputs.
  - `interrupt.click(fn=interrupt_function, _js='function(){ showSubmitInterruptingPlaceholder("<id>"); }')`. The JS returns `undefined`, which becomes `[]`, and the fn takes 0 args.
  - These **must execute while a generation event is blocked** (§5.4). `gr.Info(...)` is called inside.
- **Restore progress** (`ui.py:411-422, 770-781`): `fn=progress.restore_progress`, with JS `restoreProgressTxt2img` returning a bare string. It is wrapped into `[id]` (1 input). See §21 #1 for the output-count quirk.
- **Token counters** (`ui.py:467-470, 816-819`): `toprow.token_button` is a hidden Button clicked by JS (`javascript/token-counters.js:53`), with `fn=wrap_queued_call(update_token_counter)`.
  - Inputs are `[prompt, steps (Slider of the Sampler script), styles]` and the output is an HTML `token_counter`.
- **Resolution preview**: `component.release` for Sliders and `component.change` for the InputAccordion (`ui.py:321-337`). There are two events per component, and the second has `fn=None, _js="onCalcResolutionHires"` with inputs and **no outputs** (the JS mutates the DOM).
- `cfg_scale.change(fn=use_cfg, outputs=[negative_prompt], queue=False)`. `use_cfg` returns `gr.skip()` or `gr.update(interactive=…)` (`ui.py:61-62, 247, 299, 642`). This is an **`interactive` prop update on a Textbox**.
- `res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height])` (`ui.py:409, 760`).
- `detect_image_size_btn.click(fn=lambda w,h: (w or gr.skip(), h or gr.skip()), _js="currentImg2imgSourceResolution", inputs=[dummy, dummy], outputs=[width,height])` (`ui.py:762-768`). The JS returns an array of 3 (`javascript/ui.js:428-434`), which is **more values than the inputs**. Gr4 truncates the third value (§21 #4).
- `scale_by.change(**on_change_args)` and `button_update_resize_to.click(**on_change_args)` both use `_js="currentImg2imgSourceResolution"`, inputs `[dummy, dummy, scale_by]`, and output `scale_by_html` (`ui.py:604-613`).
- `i.change(fn=updateWH, inputs=[i], outputs=[width, height], show_progress="hidden")` fires for each image source (`ui.py:621-624`). **The `.change` must fire when a LogicalImage textarea is changed by canvas JS** (input event) **and** when the server updates it (send-to).
- `copy_image_buttons`: two click events each, a Python `copyCanvas_img2img` with LogicalImage I/O and a JS-only `switch_to_<name>` (`ui.py:567-576`).
- Tab selects: `tab.select(fn=lambda tabnum=i: tabnum, outputs=[img2img_selected_tab])` (`ui.py:558-559`). **The default-arg value is used because the fn is called with no inputs.** gr4 `special_args` appends the default (verified: `lambda tabnum=3: tabnum` with `[]` becomes `[3]`).
- `elem.select(fn=lambda tab=i: select_img2img_tab(tab), outputs=[inpaint_controls (FormGroup), mask_alpha])` (`ui.py:692-696`). These are **visibility updates on a layout block**.

---

## 4. Event model contract (generic)

### 4.1 Registration API surface used by the core

- Event methods on components:

  | Method | Used by |
  |---|---|
  | `.click` | Button |
  | `.change` | most components, including State |
  | `.input` | Dropdown (`main_entry.py:85-87`) |
  | `.submit` | Textbox |
  | `.blur` | Textbox (`ui_settings.py:315-316`) |
  | `.release` | Slider |
  | `.select` | Tab/TabItem |
  | `.upload` | Image (extension joycaption) |

- The Blocks-level method is `.load`.
- **Signature:** `event(fn=None|callable, inputs=None|Component|list, outputs=None|Component|list, show_progress=bool|"full"|"minimal"|"hidden", queue=bool, js=str, _js=str, trigger_mode=…, concurrency_limit=…)`.
  - `fn` may be positional (`ui.py:409`; `main_entry.py:85`).
  - `_js` is rewritten to `js` (`modules/gradio_extensions.py:111-126, 163-175`).
  - `inputs` and `outputs` may be **a single component** (`ui_checkpoint_merger.py:58-59`) or `None`.
- **Return** is a `Dependency` with `.then(...)` and `.success(...)`, accepting the same kwargs **including `_js`** (patched `Dependency.then`, `gradio_extensions.py:163-175`).
  - `.then` runs regardless of the outcome; `.success` runs only if the previous step did not raise (`ui_settings.py:339-350`; `main_entry.py:287-292`).
  - Chains can be arbitrarily long (`main_entry._refreshed_callback`, `main_entry.py:383-391`).
- `fn` kinds:
  - plain functions;
  - lambdas with default args;
  - **bound methods** (`shared.state.skip`, `self.ui_view`);
  - **`functools.partial`** (`ui_checkpoint_merger.py:58`; `infotext_utils.py:220`). Note that gr4 cannot see annotations through `partial`; a `Request` param there would be filled with `None` (verified).
  - callable instances;
  - **generators** (§4.8);
  - `None` (JS-only).
- `show_progress=False` and `True` are passed as booleans (gr4 typing says Literal, but bools work). The compat should map `False` to "hidden" and `True` to "full".
- Duplicate outputs are allowed: `outputs=[seed, seed]` (`modules/processing_scripts/seed.py:100`), `outputs=[res.infotext, res.infotext]` (`ui_common.py:235`). The later value wins, or skip.
- Usage counts across `modules`, `modules_forge`, `extensions-builtin`, `webui.py`, and `scripts` (`*.py`, tests excluded):

  | Construct | Count |
  |---|---|
  | `.then(` | 34 |
  | `.select(` | 29 |
  | `.release(` | 13 |
  | `.input(` | 7 |
  | `.submit(` | 6 |
  | `.blur(` | 3 |
  | `.success(` | 2 |
  | `gr.on(` | 1 |
  | `trigger_mode=` | 1 |
  | `concurrency_limit=` | 1 |
  | `js=` | 68 |
  | `_js=` | 59 |
  | `gr.State(` | 16 |
  | `gr.Info(` | 4 |
  | `gr.Warning(` | 4 |
  | `gr.Progress` | 3 |

  `every=`, `cancels=`, `api_name=`, and `preprocess=False` have no occurrences.
- Core event-binding lines per file (lines matching `.(click|change|input|submit|blur|release|select|load|then|success|upload)(`):

  | File | Lines |
  |---|---|
  | `ui.py` | 38 |
  | `ui_extensions.py` | 17 |
  | `ui_settings.py` | 15 |
  | `main_entry.py` | 9 |
  | `ui_common.py` | 8 |
  | `infotext_utils.py` | 8 |
  | `ui_prompt_styles.py` | 8 |
  | `ui_toprow.py` | 6 |
  | `ui_postprocessing.py` | 6 |
  | `ui_checkpoint_merger.py` | 6 |
  | `ui_extra_networks.py` | 5 |
  | `ui_extra_networks_user_metadata.py` | 4 |
  | `processing_scripts/seed.py` | 4 |
  | `ui_extra_networks_checkpoints_user_metadata.py` | 3 |
  | `scripts.py` | 1 |
  | `ui_components.py` | 1 |
  | `canvas.py` | 1 |

### 4.2 Triggers that must fire

- `.change` fires on **both** user edits and programmatic value changes. Programmatic changes include:
  - server outputs (e.g. paste sets `script_list` and `select_script` runs, `scripts.py:784`; paste sets `override_settings` and the `.change` visibility runs, `ui.py:184-188`);
  - JS writing the DOM and dispatching `input` (`updateInput()`, `javascript/ui.js:409-413`; canvas `GradioTextAreaBind.set_value`, `canvas.js:8-18`; InputAccordion `updateInput(gradioCheckbox)`, `javascript/inputAccordion.js:62`).
- `.input` fires only on user edits (Dropdown `main_entry.py:85-87`).
- `.load` fires per page load, for root-registered and sub-Blocks-registered dependencies alike.
- **Callable `value=`** (e.g. `gr.Textbox(value=lambda: opts.dumpjson())`, `ui_settings.py:229`; `gr.State(lambda: state_init(w))` in ADetailer) is evaluated at construction **and** on every page load (gr4 `components/base.py:210-227`, `attach_load_event`).

### 4.3 JS hook semantics (verified in the gr4 bundle `templates/frontend/assets/Blocks-B7cNA3Pj.js`)

```js
// wo(js, backend_fn, n_inputs, n_outputs)
const wrap = backend_fn ? n_inputs === 1 : n_outputs === 1;
let result = await (JS)(...[...input_values, ...output_values]);
if (typeof result === "undefined") return [];
return (wrap && !Array.isArray(result)) ? [result] : result;
// then: backend_fn ? submit(result as inputs) : apply result to outputs
```

- **Arguments are the input values followed by the current output values.** `create_submit_args` (`javascript/ui.js:118-128`) depends on this. It strips trailing output values by testing `Array.isArray(res[len-5|len-4|len-3])` (the gallery value is always an array, gr4 `Gallery.postprocess(None)` gives `[]`).
- The JS return value **replaces the inputs** sent to Python, and its **types may differ from the component types**:
  - `save` uses `(x, y, z, w) => [x, y, false, selected_gallery_index()]`, so HTML inputs carry a bool and an int (`ui_common.py:239-253`);
  - `open_folder` uses `(y, w) => [y, selected_gallery_index()]`, so a Button input carries an int (`ui_common.py:211-219`).
  - The server must preprocess these values with the component's own preprocess. That is the identity for HTML and Button, so the compat **must not type-validate** them.
- A JS return can have **more elements than inputs**. Gr4 truncates to `len(inputs)` (preprocess loops over the inputs), and fewer raises ValueError. See §21 #4.
- JS-only events (`fn=None`) with outputs apply the result to the outputs. Those without outputs simply execute.
- JS strings come in several forms:
  - a bare global function name (`"submit"`, `"closePopup"`, `"setupAllResizeHandles"`);
  - `function(...) {...}`;
  - arrow functions;
  - `async ()=>{...}` (canvas load).
- The source is evaluated as an expression with `new AsyncFunction(...)`.

### 4.4 Server-side argument binding (gr4 `helpers.special_args`, `helpers.py:870-982`)

1. `inspect.signature(fn)` (it follows `__wrapped__`) and `typing.get_type_hints(fn)` (`functools.wraps` copies `__annotations__`).
2. For each positional parameter `i`:
   - default is a `gr.Progress` instance: insert that Progress at `i`;
   - annotated `gr.Request`: insert the Request at `i`;
   - annotated `gr.EventData` or a subclass (`gr.SelectData`): insert `type_hint(target, data)` (used by `extensions-builtin/sd_forge_lora/ui_edit_user_metadata.py:184`, `sd_forge_image_stitch/scripts/image_stitch.py:60`);
   - the parameter has a default and `len(inputs) <= i`: insert the default.
3. While `len(inputs) < n_positional`: append the default, or `None` with a warning.
4. Call `fn(*inputs)`. Extra inputs beyond the parameters raise a Python `TypeError`, unless the fn takes `*args`.

`gr.Request` must expose `.username` (auth), `.headers`, `.client`, `.query_params`, `.session_hash`, and `.request`. The core and agent-scheduler use `.username` (`txt2img.py:57`, `img2img.py:257`, `agent_scheduler/task_runner.py:161`).

### 4.5 Output binding (gr4 `blocks.py:1687-1765`, `600-622`)

1. A `dict` whose keys are all components is reordered into a list; missing entries become `skip()`.
2. If `len(outputs) == 1`, wrap the result: `[result]`.
3. `validate_outputs`: **fewer values than outputs raises ValueError; more values are silently ignored** (`blocks.py:1655-1685`).
4. For each output:
   - Stateful (`gr.State`): a non-update value is stored in the session state. Updates to a State are ignored.
   - `is_prop_update(v)` (a dict with `__type__ == "update"`):
     - `delete_none(v, skip_value=True)` drops props whose value is `None`, except `value`;
     - a **new per-session component instance** is built as `block.__class__(**{**session_block.constructor_args, **update, render=False})`, without `value`;
     - `value` (if present) goes through `postprocess`.
     - Consequence: `choices` get normalized to `(label, value)` tuples, and later **preprocess for this session uses the updated instance**. This matters for `Dropdown(type="index")` with refreshed choices.
     - Side effect: the patched `__init__` (§15) re-fires `before_component` and `after_component` callbacks on every prop update.
   - A `Block` instance returned as a value is treated as an update with its constructor args.
   - Otherwise `value = (session_block or block).postprocess(v)`, then `async_move_files_to_cache(postprocess=True)` (patched, §14).
5. `gr.skip()` is `{"__type__": "update"}` (gr4 `helpers.py:1017-1018`). `gr.update(**kw)` is `{**kw, "__type__": "update"}`.
   `infotext_utils.type_of_gr_update = type(gr.skip())`, which is `dict` (`infotext_utils.py:21`), and `api.pasted_value` checks `__type__ in ("generic_update", "update")` (`api/api.py:40-54`).
   **The compat must keep the dict representation**; a custom class would break both.
6. Prop updates used in the core:
   - all components: `visible`, `value`, `interactive`;
   - Dropdown, Radio, CheckboxGroup: `choices`;
   - Gallery: `selected_index`;
   - Slider (from OptionInfo `component_args` via `get_value_for_setting`, `ui_settings.py:17-24`): `minimum`, `maximum`, `step`;
   - layouts (Row, Group, Column, FormRow, FormGroup, Accordion): `visible`.
   - `elem_classes` and `label` may also appear via extensions.

### 4.6 Session state

- `gr.State(value)` deep-copies the value per session (gr4 `state_holder.py:81-101`). The value never goes to the client, and `.change` fires on update.
- Core uses:
  - `ui_postprocessing.py:10` (tab index);
  - `ui_settings.py:50, 61` (`sd_model_checkpoint` State and FormRow placeholders `gr.State(None)` inside `settings.components`, so they are inputs of "Apply settings");
  - `ui_extensions.py:682`.
- Extension uses:
  - ControlNet `gr.State(UiControlNetUnit)` (`controlnet_ui_group.py:413-414, 496`), a deep-copied dataclass;
  - ADetailer `gr.State(callable)`;
  - prompt-workshop (6+1);
  - image_stitch.
  - Agent-scheduler inspects `isinstance(output, gr.State) and type(output.value).__name__ == "UiControlNetUnit"` (`task_scheduler.py:164`).
- Per-session component instances (§4.5) are part of the session state as well.

### 4.7 Queue, concurrency, threading

- `demo.queue(default_concurrency_limit=32)` (`webui.py:96`). Sync handlers run in a worker thread pool, and many requests are served concurrently.
- `queue=False` events are still concurrent; they are just not queued.
- **The compat must run handlers off the event loop with at least 32 concurrent workers.** Reasons:
  - a generation request blocks its worker for minutes (in `queue_lock`, then in `main_thread.run_and_wait_result`);
  - `restore_progress` busy-waits (`progress.py:153-154`);
  - interrupt, skip, the token counter, and paste must keep working meanwhile;
  - `/internal/progress` polling must keep answering.
- `trigger_mode` defaults (gr4 `events.py:442`):
  - `"once"` for all events except `.change` and `.key_up`, which default to `"always_last"`;
  - with `"once"`, a second trigger is ignored while the first is pending;
  - one explicit `trigger_mode="always_last"` exists (`anima-artist-scheduled-mixer:1846`).
- Only one `concurrency_limit=` usage exists, in an extension.
- `--no-gradio-queue` disables the queue. With the queue disabled, generators are not supported in gr4; the compat can ignore that limitation.

### 4.8 Generators, progress, toasts, errors

- **Generators**: each `yield` is postprocessed and pushed to the client as an intermediate update (the gr4 `process_generating` message), and the last yield is final.
  The core has no generator *handlers*: `wrap_gradio_call_no_job` does `list(func(...))` (`call_queue.py:76`), so a generator passed to it would be fully consumed.
  Generator handlers exist in `extensions-builtin/sd-webui-joycaption-modified/scripts/joycaption.py` (`:163, 215, 289`), which yields status strings and tuples.
- **`gr.Progress(track_tqdm=True)`** is a default argument; it is injected and called as `progress(1.0, desc=...)` (`joycaption.py:163-175`). The compat needs a callable with `__call__(progress, desc=None, total=None, unit=...)`, `.tqdm(iterable)`, and optional tqdm tracking. A no-op is acceptable, but a progress bar is better.
- **`gr.Info(msg, duration=…)` and `gr.Warning(msg, duration=…)`**:
  - in the core: `util.py:187, 197`, `ui_toprow.py:122`, `ui_checkpoint_merger.py:48-55`, `scripts.py:800`, `ui_settings.py:336`;
  - they are called from inside handlers, including `queue=False` ones (`ui_settings._change_model`, `ui_settings.py:333-337` with `queue=False`). In gr4 those are lost because there is no queue channel.
  - **The compat should deliver them to the triggering client regardless of `queue`.** They are called from worker threads, so the compat must use thread-local or contextvar event context.
- **Exceptions**: unwrapped handlers raise and the client shows an error toast. Wrapped handlers convert exceptions to an HTML error in the last output (§5.2). `.success` chains stop on an exception.

---

## 5. `modules/call_queue.py`, `fifo_lock.py`, `modules_forge/main_thread.py`

### 5.1 `queue_lock` (`call_queue.py:10`)

- `queue_lock` is a `fifo_lock.FIFOLock()`, a strictly FIFO re-implementation of `threading.Lock` (`modules/fifo_lock.py`).
- It serializes all GPU work across UI and API (`webui.py:42-47` passes it to `Api`). The API also takes it (`api/api.py:480, 553, 589, 600, 746-754`).

### 5.2 Wrappers

- `wrap_queued_call(func)` (`:13-20`) runs the call under `queue_lock`. It is used by the token counters (`ui.py:467-470, 816-819`).
- `wrap_gradio_gpu_call(func, extra_outputs)` (`:23-48`):
  1. If `args[0]` is a `str` matching `task(...)`, it becomes `id_task` and `progress.add_task_to_queue(id_task)` registers it as pending.
  2. `with queue_lock:` it calls `shared.state.begin(job=id_task)`, `progress.start_task(id_task)`, `res = func(*args, **kwargs)`, and `progress.record_results(id_task, res)`. In `finally` it calls `progress.finish_task(id_task)`, then `shared.state.end()`.
  3. Everything is wrapped in `wrap_gradio_call(..., add_stats=True)`.
- `wrap_gradio_call(func, extra_outputs, add_stats)` (`:51-64`): in `finally`, it resets `shared.state.{skipped, interrupted, stopping_generation, job_count, job}`.
- `wrap_gradio_call_no_job(func, extra_outputs=None, add_stats=False)` (`:67-131`):
  - `res = list(func(...))`, so **handlers must return an iterable**, normally a tuple.
  - On an exception, it uses `main_thread.last_exception` if set (e.g. `"OOM"`), else `f"{type}: {e}"`, and builds `res = extra_outputs + ["<div class='error'>…</div>"]`. The default `extra_outputs` is `[None, ""]`.
  - It calls `devices.torch_gc()`.
  - When `add_stats` is set, it **appends the HTML "Time taken / VRAM / profile" to `res[-1]`** (`:126-127`). So the last output must be an HTML str.
  - It returns `tuple(res)`.
  - Because of `@wraps`, the Gradio signature inspection sees the innermost function (Request injection works).
- `extra_outputs` per call site:

  | Call site | `extra_outputs` |
  |---|---|
  | txt2img and img2img | `[None, None, "", ""]` (`ui.py:379, 744`) |
  | upscale | `[None, "", ""]` (`ui.py:397`) |
  | extras | `[None, ""]` (`ui_postprocessing.py:46`) |
  | settings apply | `[gr.skip()]` (`ui_settings.py:307`) |
  | extensions tab | `[gr.skip()]` (`ui_extensions.py:645-651`) |
  | model merger | **a lambda** (`ui_checkpoint_merger.py:133`), see §21 #3 |
  | save | default (`ui_common.py:240, 256`) |
  | pnginfo | default (`ui.py:857`) |

### 5.3 `main_thread` (`modules_forge/main_thread.py`)

- A single consumer loop runs on the **process main thread** (`loop()`, `:46-60`; started by `launch_utils.py:557-559`).
- `async_run(func, *args, **kwargs)` enqueues a `Task` (`:63-72`). `run_and_wait_result(func, ...)` blocks the calling worker thread on `condition` until the task finishes, then returns `task.result` (`:75-83`).
- `Task.work()` catches all exceptions:
  - it sets `last_exception = "OOM"` (when `is_oom`) or `f"{type}: {e}"`, and the result stays `None` (`:27-43`);
  - success resets `last_exception = None`;
  - consequently `wrap_gradio_call_no_job`'s `list(None)` raises TypeError, and the UI shows `main_thread.last_exception` (`call_queue.py:77-82`).
  - The `.then(fn=cleanup)` after generate unloads weights if `last_exception == "OOM"` (`ui.py:70-74`).
- Users:
  - txt2img, txt2img_upscale, and img2img dispatch through it (`txt2img.py:160-165`, `img2img.py:302-303`);
  - extras, API, and all other handlers do **not** (API txt2img runs `process_images` directly on the uvicorn worker under `queue_lock`, `api/api.py:480-499`).
- **Compat contract:** the event server must never require the main thread. The main thread is dedicated to `main_thread.loop()`, and the server must run on its own thread (as Gradio does via `prevent_thread_lock=True`).

### 5.4 Resulting concurrency picture

| Thread | Work |
|---|---|
| uvicorn and event loop | HTTP, SSE, `/internal/progress` (sync FastAPI route, so it runs in the threadpool) |
| worker pool (≥32) | each queued event; generate holds `queue_lock` while waiting on main_thread |
| main thread | model and sampling work for txt2img, img2img, and upscale |

Interrupt and skip (`shared.state.interrupt()`, `skip()`) run on another worker and flip flags read by the sampler.

---

## 6. `modules/progress.py`: task registry and live preview

### 6.1 Module state

`current_task`, `pending_tasks: OrderedDict[id → enqueue time]`, `finished_tasks` (last 16), and `recorded_results` (last 2, `(id, res)`) (`:17-21`).

### 6.2 Lifecycle

- `add_task_to_queue` (`:54-55`) runs from `wrap_gradio_gpu_call` before the lock and from the API (`api/api.py:478`).
- `start_task` (`:24-28`) runs once the lock is acquired.
- `record_results` (`:48-51`) stores the **raw handler return**: a tuple containing `gr.update` dicts and PIL images.
- `finish_task` (`:31-39`).
- `create_task_id(type)` (`:42-45`) is used by the API.
- The UI's JS id is `randomId()`: `"task(" + 3×5 base36 chars + ")"` (`javascript/progressbar.js:64-71`). The client generates it and puts it in input slot 0 through the `submit*` hooks.
  The id is also stored in `localStorage` (`txt2img_task_id` and `img2img_task_id`) for restoring.

### 6.3 Routes (`setup_progress_api`, `:80-82`)

- `GET /internal/pending-tasks` returns `{size, tasks}`.
- `POST /internal/progress` takes `ProgressRequest{id_task, id_live_preview=-1, live_preview=True}` and returns
  `ProgressResponse{active, queued, completed, progress, eta, live_preview, id_live_preview, textinfo}` (`:63-77`).
  - If not active: `textinfo` is "Waiting..." or "In queue: i/n" (`:96-102`).
  - `progress` is `job_no/job_count + (1/job_count)*sampling_step/sampling_steps`, clamped to 1 (`:104-114`).
  - `eta` comes from `shared.state.time_start` (`:116-118`).
  - **Live preview encoding** (`:123-147`) happens only if `opts.live_previews_enable` and the client's `id_live_preview` differs from `shared.state.id_live_preview`:
    - the format is `opts.live_previews_image_format` (png, jpeg, or webp), or `gif` when `image.is_animated` (`save_all=True, loop=0`);
    - PNG uses `optimize=True` if the longest side is ≤256, else `compress_level=1`;
    - other formats are converted to RGB;
    - the result is returned as a **`data:image/<fmt>;base64,...` URL**, with the new `id_live_preview`.
- The client polls every `opts.live_preview_refresh_period` ms (`progressbar.js:150-200`).
  - Two loops run: `funProgress` (`live_preview:false`) and `funLivePreview`.
  - The preview image is inserted into the gallery element in the DOM. **The compat SPA must not wipe externally inserted DOM nodes inside the gallery container**, or it must provide an equivalent React preview slot.
- `restore_progress(id_task)` (`:152-160`) spins with `time.sleep(0.1)` while the task is current or pending, then returns the recorded result or `(skip, skip, skip, html)`.

---

## 7. `modules/ui_common.py`: output panel, save, send-to, HTML info

### 7.1 `create_output_panel(tabname, outdir, toprow)` (`:169-292`)

- `gr.Gallery(label="Output", show_label=False, elem_id=f"{tab}_gallery", columns=4, preview=True, height=opts.gallery_height or None, interactive=False, type="pil", object_fit="contain")` (`:192`).
- `gr.Video(label="Output", elem_id=f"{tab}_player", height=…, interactive=False, autoplay=…, loop=…, include_audio=False, show_download_button=False, show_share_button=False, visible=False)` (`:193`).
- Tool buttons:
  - 📂 `open_folder` (`visible=not hide_ui_dir_config`), with JS `(y, w) => [y, selected_gallery_index()]`, inputs `[gallery, button]`, outputs `[]` (`:211-219`). The handler receives `images` as `[(PIL, caption)]`. Its `images[index]["name"]` lookup raises and falls back (`:176-182`), a legacy dict access.
  - 💾 save and 🗃️ save_zip (not on extras).
  - send-to buttons 🖼️, 🎨️, 📐 (`:202-206`).
  - ✨ upscale (txt2img only).
- `download_files = gr.File(None, file_count="multiple", interactive=False, visible=False)` (`:222`).
- For txt2img and img2img:
  - `infotext` is `gr.HTML(elem_id=f"html_info_{tab}", elem_classes="infotext")`;
  - `html_log` is `gr.HTML`;
  - `generation_info` is a hidden `gr.Textbox(elem_id=f"generation_info_{tab}")`, which carries the JSON from `processed.js()`.
  - For extras, `generation_info` is `gr.HTML(elem_id="html_info_x_extras")` (`:270-273`).
- `generation_info_button` is a hidden Button with JS `function(x, y, z){ return [x, y, selected_gallery_index()] }` and inputs `[generation_info, infotext, infotext]`. `update_generation_info(generation_info, html_info, img_index)` returns `(html, skip)` (`:229-237`, `:19-28`).
  - The JS clicks this button when the gallery selection changes; see the imageviewer and gallery listeners.
- Save: `fn=wrap_gradio_call_no_job(save_files)`, JS `(x,y,z,w) => [x, y, false|true, selected_gallery_index()]`, inputs `[generation_info, gallery, infotext, infotext]`, outputs `[download_files, html_log]` (`:239-268`).
  - `save_files(js_data: str, images: list[(PIL, caption)], do_make_zip: bool, index: int)` (`:63-156`) returns `(gr.update(value=[paths], visible=True), html)`.
  - **`File.postprocess` takes a list of absolute paths and stats them** for `size`. The files are in `opts.outdir_save`, which may be outside the allowed paths, so gr4 copies them to the cache (§14).
- After the `on_after_component` callbacks: `seed.py` hooks the `generation_info_<tab>` component to wire the reuse-seed buttons (`modules/processing_scripts/seed.py:65-66`, `:84-100`). This **requires `after_component` to fire synchronously at construction with `component.elem_id` set**.
- Send-to registration: `register_paste_params_button(ParamBinding(paste_button, tabname=paste_tabname, source_tabname=..., source_image_component=gallery, paste_field_names=scripts_*.paste_field_names))` (`:275-290`). The events are connected later at the root (§8).

### 7.2 Helpers

- `create_refresh_button(refresh_component(s), refresh_method, refreshed_args, elem_id)` (`:295-316`):
  - it **mutates the Python components directly** (`setattr(comp, k, v)`, e.g. `choices` as plain strings);
  - it returns `gr.update(**args)` (one per component).
  - **The compat must accept `choices` as raw strings or as tuples in both the attribute and the update.**
- `setup_dialog(button_show, dialog, button_close)` (`:319-330`) sets `dialog.visible=False`, wires show to `gr.update(visible=True)`, then JS `popupId(elem_id)`, and wires close to JS `closePopup`.
  - The popup JS moves the Group's DOM node into a global popup container, so **the SPA must tolerate DOM re-parenting of a rendered Group**.
- `plaintext_to_html(text, classname)` (`:31-34`).

---

## 8. `modules/infotext_utils.py`: paste fields, send-to, `connect_paste`

### 8.1 Registries (reset per `create_ui`, `:50-52`)

- `paste_fields[tabname] = {"init_img": Component|None, "fields": list[PasteField], "override_settings_component": Dropdown|None}` (`:136-151`).
  Registered entries:
  - txt2img (`ui.py:452`), with `init_img = None`;
  - img2img (`ui.py:801`), with `init_img = init_img.background`;
  - inpaint (`ui.py:802`), with `init_img = init_img_with_mask.background`;
  - extras (`ui_postprocessing.py:57`), with `init_img = extras_image` and `fields = None`.
- Backwards-compatibility globals are set: `modules.ui.txt2img_paste_fields` and `img2img_paste_fields` (`:146-151`).
- `PasteField(component, target: str | callable, api=None)` is a tuple subclass with `.component`, `.label`, `.function`, and `.api` (`:35-43`). Plain `(component, target)` tuples are converted (`:138-141`).
- `registered_param_bindings` holds the `ParamBinding`s (`:24-32, 158-159`).

### 8.2 `connect_paste_params_buttons()` runs in the root demo (`ui.py:890`) and calls `_connect_paste_params_buttons` per binding (`:162-236`)

1. **Image transfer** applies when both a source image component and a destination image exist (`:170-186`):
   - The source is a Gallery: `fn = send_image_and_dimensions` (when the destination has `Size-1` and `Size-2` and the tab is not inpaint) or `image_from_url_text`, with `js="extract_image_from_gallery"`.
     - The JS returns `[[gallery[idx]]]` (`javascript/ui.js:41-56`), a one-element gallery list, so Python receives `[(PIL, caption)]`.
   - The source is an Image (PNG Info): `fn = send_image_and_dimensions or (lambda x: x)`, a PIL passthrough.
   - Outputs are `[dest_image, dest_width, dest_height]` or `[dest_image]`. The destination is a **LogicalImage** (img2img and inpaint canvas background) or a `gr.Image` (extras), so a PIL output must postprocess into each.
   - `send_image_and_dimensions` (`:244-265`) converts to RGB and returns `(img, sRound(w), sRound(h))`, or `gr.skip()`.
2. **Text paste** happens when there is a source text component (`:188-189`): `connect_paste(...)`.
3. **Field forwarding between tabs** happens when `source_tabname` is set (`:191-233`). The fields are those in `paste_field_names = ["Prompt","Negative prompt","Steps","Face restoration"] + Seed? + CFG? + binding.paste_field_names`.
   - If the source is a Gallery and `opts.send_image_info_not_ui` is set: a `partial(read_infotext, paste_fields=[…])` handler with `js="extract_image_from_gallery"` reads the PNG info from the image and returns a list of updates. Then `.then(fn=None, _js=f"switch_to_{tab}")`.
   - Otherwise: `fn=lambda *x: x`, with inputs being the matching **source tab components** and outputs the destination components. **This is a direct component-to-component copy, so values must round-trip through preprocess and postprocess unchanged.**
4. Otherwise it registers only the JS tab switch (`:235-236`).

### 8.3 `connect_paste(button, paste_fields, input_comp, override_settings_component, tabname)` (`:536-571`)

- `paste_func(prompt)` reads `data_path/params.txt` when the prompt is empty, runs `parse_generation_parameters`, calls `infotext_pasted_callback`, and returns `_parse_info(output, key, params)` per field.
- `_parse_info` (`:75-102`):
  - returns a callable's return value or `params[key]`;
  - `None` becomes `gr.skip()`, and an update dict is passed through;
  - otherwise it casts with `type(output.value)`: bool (`"False"` becomes False), `int` becomes `float(v)`, and any other type is `valtype(v)`;
  - failures become a skip.
  - **The compat must keep `component.value` as the Python-typed initial value** (§13 table): Slider int or float, Number per precision, Checkbox bool, Dropdown multiselect list, Textbox str or `None`, layouts with no usable value.
- `override_settings` gets `paste_settings(params)`, which returns `gr.update(value=pairs, choices=pairs, visible=bool(pairs))` (`:556-564`). This is a prop update with choices on a multiselect Dropdown.
- The chain is `.then(fn=None, js=f"recalculate_prompts_{tabname}")` (`:566-571`).
- **Outputs include layouts** (`hr_sampler_container`, `hr_prompts_container`, script `group`s), duplicates, and the script `Dropdown(type="index")`, which receives `gr.update(value=title)`.

### 8.4 `image_from_url_text(filedata)` (`:105-133`)

It accepts four shapes:

1. a list (takes the first element);
2. a dict with `is_file` and `name` (Gradio 3 legacy). For this shape it checks `ui_tempdir.check_tmp_file(shared.demo, name)` and then calls `images.read(name)`;
3. a `(PIL, caption)` tuple (Gradio 4 Gallery);
4. a base64 `str`, decoded via `api.decode_base64_to_image`.

`check_tmp_file` relies on `shared.demo.temp_file_sets`.

### 8.5 Parsing and override

- `parse_generation_parameters` (`:356-466`) and `create_override_settings_dict` (`:472-501`) are pure Python.
- They are also used by the API (`api/api.py:380-443`), which reads `paste_fields[tab]["fields"]`, `field.component.value`, and `script_runner.inputs` identity. **The UI component objects must exist in API-only mode.**

---

## 9. `modules/ui_loadsave.py`: `ui-config.json`

- File: `cmd_opts.ui_config_file`. The JSON is flat: `{"<path>/<field>": value}`. It is read at construction (`:18-35`).
- `add_block(x, path)` walks children recursively. Keys are:
  - for `gr.Tabs` with an `elem_id`: `"{path}/Tabs@{elem_id}"`;
  - for a component with a label: `"{path}/{label}"`;
  - for a Button: `"{path}/{value}"` (`:144-156`).
  - The root path is the tab id (`ui.py:903`), giving keys like `txt2img/Width/value`. The top-level tabs key is `webui/Tabs@tabs/selected` (`ui.py:905`).
- `add_component(path, x)` (`:37-142`) **mutates attributes before the root config is serialized**:

  | Class (exact `type(x)` match, not isinstance) | Fields |
  |---|---|
  | Slider, Radio, Checkbox, Textbox, Number, Dropdown, ToolButton, Button (only if currently visible) | `visible` |
  | Slider | `value`, `minimum`, `maximum`, `step` |
  | Radio | `value`, validated against the choices |
  | Checkbox, Textbox, Number | `value` (Textbox: `str()`, Number: `float()`) |
  | Dropdown | `value`, validated (multiselect: all values in choices). `x.init_field(saved)` is called if present (`scripts.py:773-782` makes the selected script group visible) |
  | InputAccordionImpl | `accordion.visible`, the checkbox `value`, `accordion` `open` (adds `"input-accordion-open"` to `elem_classes` if True) (`:57-60, 123-129`) |
  | Tabs | `selected`, validated against the child TabItem ids or index |

  - Custom-script components are prefixed `customscript/<file>.py/…` (`:45-46`) via `control.custom_script_source`, which `scripts.create_script_ui_inner` sets (`scripts.py:702`).
  - Fields can be skipped: `do_not_save_to_config` (`ui.py:308, 654`; `ui_checkpoint_merger.py`), and `_internal_preset_param` skips `value` and `step` (`ui.py:65-67, 464-465, 814`).
  - **Compat requirement:** `type(x) in [...]` means compat classes must be the exact classes `gr.Slider` etc. (no proxy subclasses for core components), with mutable `.value`, `.visible`, `.minimum`, `.maximum`, `.step`, `.choices`, `.multiselect`, `.selected`, `.open`, and `.elem_classes`. The config served to the client must reflect the post-mutation attributes.
- `setup_ui()` (`:243-250`): `ui_defaults_view.click(fn=self.ui_view, inputs=list(component_mapping.values()), …)`. These inputs are **hundreds of components** from every tab (the value-tracked ones), so the server must handle very wide input lists.
  - `iter_changes` maps an int value to `choices[int]` when the component has choices (`:174-199`), to handle `type="index"` Radios and Dropdowns.
- `dump_defaults()` runs if anything new was added (`ui.py:926-927`).

---

## 10. `modules/ui_settings.py` and OptionInfo-to-component mapping

### 10.1 `create_setting_component(key, is_quicksettings)` (`:27-74`)

- `info = opts.data_labels[key]`. `args = info.component_args()` if callable, else a dict.
  These are **evaluated at build time and again on every page load** (`get_value_for_setting`, `:17-24`, inside `demo.load`).
- Component choice:

  | `info.component` / default type | Result |
  |---|---|
  | explicit `component` (a class or factory lambda, e.g. `OptionHTML` returns `gr.HTML(elem_classes=…, **kwargs)`, `modules/options.py:62-67`) | that component |
  | `str` default | `gr.Textbox` |
  | `int` default | `gr.Number` |
  | `bool` default | `gr.Checkbox` |
  | anything else | an exception |

- Special cases:
  - `comp == gr.State` gives `gr.State(value)`;
  - `comp == FormRow` toggles a global `CURRENT_ROW` (manual `__enter__` and `__exit__`) and returns `gr.State(None)` (`:49-61`).
- Every setting component gets `label=info.label`, `value=current`, `elem_id=f"setting_{key}"`, and `**args`.
- With `info.refresh`, the component is wrapped in a `FormRow` with a `create_refresh_button` (`:63-70`).
- Component kinds used in `modules/shared_options.py` (279 `OptionInfo` entries):

  | Component | Count |
  |---|---|
  | `gr.Slider` | 45 |
  | `gr.Radio` | 21 |
  | `gr.Dropdown` | 10 |
  | `gr.Number` | 10 |
  | `ui_components.DropdownMulti` | 10 |
  | `ui_components.FormColorPicker` | 7 |
  | `gr.Textbox` | 3 |
  | `gr.CheckboxGroup` | 2 |
  | `ui_components.DropdownEditable` | 1 |
  | `gr.State` | 1 (`sd_model_checkpoint`, `:222`) |

  The rest are implicit Textbox, Number, or Checkbox, plus `OptionHTML`, `OptionDiv`, and `OptionRow`.
  More OptionInfo come from 16 other files (`modules_forge/shared_options.py`, `modules_forge/presets.py`, `modules/shared_items.py`, `modules/script_callbacks.py`, processing scripts, and extensions).

### 10.2 `UiSettings.create_ui(loadsave, dummy_component)` (`:128-294`)

- Its own `gr.Blocks` contains `Apply settings` and `Reload UI` buttons, `result` HTML, and `gr.Tabs(elem_id="settings")` with one `TabItem` per section, built through **manual `__enter__` and `__exit__` calls**.
- `self.components` is aligned 1:1 with `opts.data_labels`. A quicksetting or a skipped section gets `dummy_component`.
- There are extra tabs (Defaults with loadsave UI, Sysinfo, Actions, Licenses):
  - `gr.File(type="binary")` gives bytes to `check_file` (`:203, 264-277`);
  - `license_tab.select(fn=None, _js="populateLicense")`;
  - `restart_gradio.click(fn=shared.state.request_restart, _js="restart_reload")`.
- `self.text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)` (`:229`). A callable value means a per-load refresh.
  **JS depends on this textarea.** `load_webui_settings()` reads it, then overrides the instance's `value` property setter to re-parse `opts` and fire `onOptionsChanged` (`javascript/ui.js:333-365`).
  The SPA must render the textarea and set its value through `textarea.value = …` on updates. See §17 for the React value-tracker caveat.

### 10.3 `add_quicksettings()` (`:296-303`)

- An optional `gr.Accordion`, then `gr.Row(elem_id="quicksettings")`.
- Inside: `main_entry.make_checkpoint_manager_ui()` (§11), then a setting component per quicksetting.

### 10.4 `add_functionality(demo)` (`:305-362`)

- `submit.click(fn=wrap_gradio_call_no_job(lambda *args: self.run_settings(*args), extra_outputs=[gr.skip()]), inputs=self.components, outputs=[text_settings, result])`.
  - The inputs are **all settings**, around 300 or more, including `dummy_component` and State placeholders.
  - `run_settings` asserts `opts.same_type(value, default)`. `typemap {int: float}` (`modules/options.py:95`) makes JSON ints and floats interchangeable.
  - The value types must follow Gradio: Number gives float or int per precision, Slider gives a JSON number, CheckboxGroup a list, ColorPicker a str, Dropdown multiselect a list.
- Quicksettings: Textbox gets `.submit` and `.blur`; components with a `release` method (Slider) get `.release`; everything else gets `.change`.
  The handler is `fn=lambda value, k=k: self.run_settings_single(value, key=k)` (default-arg binding, §4.4), with outputs `[component, text_settings]`.
- A hidden Textbox `change_checkpoint_text` and a hidden Button `change_checkpoint` exist.
  JS `selectCheckpoint()` sets the textarea, dispatches `input`, and clicks the button (`javascript/ui.js:420-426`).
  The event is `.click(_change_model, queue=False).success(main_entry.checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], queue=False)` (`:339-350`).
- `demo.load(fn=get_settings_values, outputs=[all setting components], queue=False)` (`:357-362`). Every page load pushes `gr.update(value=…, **component_args)` to every setting component.

---

## 11. `modules_forge/main_entry.py`: checkpoint manager and presets

- `make_checkpoint_manager_ui()` (`:60-87`) creates:
  - Dropdowns `forge_ui_preset`, `setting_sd_model_checkpoint` (initial `value=None, choices=None`), `setting_sd_modules` (multiselect), and `forge_ui_dtype`;
  - a refresh `ToolButton`.
- Events:
  - `refresh_button.click(refresh_model_list, outputs=[ckpt, vae], queue=False)` is the `_event_refresh` dependency;
  - `Context.root_block.load(fn=refresh_model_list, …)` runs on each page load;
  - `.input(checkpoint_change|modules_change|dtype_change, inputs=[x, preset], queue=False, show_progress=False)`. These handlers take extra defaulted params (`save=True, refresh=True`), filled per §4.4. They return a `bool` with no outputs.
- `forge_main_entry()` (`:236-296`) resolves 19 components by infotext label or api name from `paste_fields` (`get_a1111_ui_component`, `:229-233`). For example, `"sampler_name"` and `"scheduler"` match the `api=` names of the Sampler script.
  - `ui_forge_preset.change(on_preset_change, outputs=22 targets, queue=False).success(_load_presets, inputs=[ckpt, vae, dtype, preset], queue=False).then(js="clickLoraRefresh", fn=None, queue=False)`.
  - `Context.root_block.load(on_preset_change, inputs=[preset], outputs=…)`.
  - `on_preset_change` returns 22 `gr.update`s, including `gr.update(value=…, **d_args)` and `gr.update(**batch_args)` (`:305-358`).
  - **Load-time events with inputs** read the *initial* component values for the session.
- `on_refresh_checkpoint(callback, inputs, outputs)` collects callbacks before `forge_main_entry`. `_refreshed_callback()` chains them as `.then(...)` on `_event_refresh` (`:367-391`).
  It is used by `ui.py:287` for the hires checkpoint and modules dropdowns.

---

## 12. `modules/scripts.py`: script UI and argument plumbing

- `ScriptRunner.initialize_scripts(is_img2img)` (`:622-656`) instantiates every script class.
  - `show()` decides between `AlwaysVisible` (alwayson) and selectable.
  - It then builds `on_before_component_elem_id` and `on_after_component_elem_id` maps (`:658-676`).
- `prepare_ui()` sets `self.inputs = [None]` (`:751-752`). **Slot 0 is reserved for the script dropdown.**
- `setup_ui_for_section(section, scriptlist=None)` (`:733-749`) runs for each section in `ui.py` (`:248, 301, 319, 643, 647, 684`). Sections are `ordered_ui_categories()` from `shared_items.ui_reorder_categories()`, with the user order taken from `opts.ui_reorder_list`. For each matching alwayson script (sorted by `sorting_priority`):
  - `create_group=True`: `with gr.Group(visible=script.alwayson) as group: create_script_ui(script)` and `script.group = group`;
  - otherwise the UI goes directly into the current context (e.g. the Seed script, `create_group=False`).
- `create_script_ui(script)` (`:678-731`):
  - `args_from = args_to = len(self.inputs)`;
  - `controls = script.ui(is_img2img)` returns a **list or tuple** of components (Seed returns a tuple, `processing_scripts/seed.py:68`);
  - for each control it sets `control.custom_script_source = basename(filename)`;
  - it builds API metadata from `control.label`, `.value`, `.minimum`, `.maximum`, `.step`, and `.choices` (tuples are reduced to `x[0]`, `:711-713`) into `api_models.ScriptInfo`;
  - it appends `script.infotext_fields` and `script.paste_field_names` to the runner;
  - `self.inputs += controls` and `args_to = len(self.inputs)`.
- `setup_ui()` (`:754-808`):
  1. Alwayson scripts with `section=None` are set up first.
  2. Then `dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + titles, value="None", type="index")` becomes `self.inputs[0]`.
  3. Then the selectable scripts are set up (each in a `gr.Group(visible=False)`).
  4. `dropdown.init_field` is used by loadsave (§9).
  5. `dropdown.change(fn=select_script, inputs=[dropdown], outputs=[groups])` returns a visibility update per selectable script.
  6. `infotext_fields` gains `(dropdown, lambda x: gr.update(value=x.get("Script","None")))` and `(group, onload_script_visibility)` per selectable script (with `gr.Warning`).
  7. It returns `self.inputs`, which becomes `custom_inputs` in `ui.py`.
- `run(p, *args)` (`:810-826`) reads `script_index = args[0]` (the **int index** from `type="index"`; 0 means None) and slices `args[script.args_from:script.args_to]`.
  The hooks `process`, `before_process`, and so on slice `p.script_args` the same way.
- **Component lifecycle callbacks.** The patched Gradio `__init__` (§15) calls `scripts.scripts_current.before_component(self, **kwargs)` and `after_component(self, **kwargs)`, plus the global `script_callbacks.before_component_callback` and `after_component_callback`, for **every** component and layout block created while `scripts_current` is set.
  - `before_component` matches `kwargs.get("elem_id")`. **Only kwargs are passed, not positional args.**
  - `after_component` matches `component.elem_id` (`scripts.py:1000-1024`).
  - Scripts may create new components *inside* these callbacks (injecting UI next to core controls), which nests into the current render context.
  - **The compat must call these hooks at exactly the same moments, with the same kwargs, for components and BlockContexts (Row, Column, Group, Tabs, Accordion, …).**
- `set_named_arg` (`:1066-1093`) uses `control.elem_id`.
- `scripts_postprocessing` (`modules/scripts_postprocessing.py`): `ui()` returns a **dict** `name → component`. Each script UI is wrapped in `gr.Row()` (`:160-170`), and `run()` passes `process(pp, **{name: value})`.

---

## 13. `modules/ui_components.py`: custom component classes

| Class | Base | Behaviour the compat must mirror |
|---|---|---|
| `FormComponent` | mixin | `get_expected_parent()` returns `gr.components.Form`. Gradio auto-wraps consecutive form components in a `Form` container (`fill_expected_parents`). Also applied to `gr.Dropdown` itself (`:14`). Sets `webui_do_not_create_gradio_pyi_thank_you`. |
| `ToolButton` | `gr.Button, FormComponent` | `value` is the first positional argument, and `elem_classes=["tool", *…]` (`:17-26`). `get_block_name()` returns `"button"`, so the CSS class is `gradio-button`. |
| `ResizeHandleRow` | `gr.Row` | Adds `"resize-handle-row"` to `elem_classes` after init (`:29-40`). `resizeHandle.js` builds the drag handles from the DOM. |
| `FormRow`, `FormColumn`, `FormGroup`, `FormHTML`, `FormColorPicker` | the matching gr class plus `FormComponent` | Block names `row`, `column`, `group`, `html`, `colorpicker`. |
| `DropdownMulti` | `gr.Dropdown` | Forces `multiselect=True` (kwargs only). |
| `DropdownEditable` | `gr.Dropdown` | Forces `allow_custom_value=True`. |
| `InputAccordionImpl` | `gr.Checkbox` | See the `InputAccordion` row. |
| `InputAccordion(value, **kw)` | factory | `InputAccordionImpl(value=value, setup=True, **kw)` (`:102-170`). It creates a **hidden Checkbox** with `elem_id=f"{id}-checkbox"`, `visible=False`, and a `.change(fn=None, _js='function(checked){ inputAccordionChecked("<id>", checked); }', inputs=[self])`. It then creates `gr.Accordion(elem_id=id, label, elem_classes=['input-accordion'], open=value)`. `__enter__` and `__exit__` delegate to the accordion, while the context manager returns the Checkbox. `extra()` returns `gr.Column(elem_id=id+'-extra', elem_classes='input-accordion-extra', min_width=0)`. `get_block_name()` returns `"checkbox"`. Its value as an input is **bool**. `accordion.custom_script_source` is copied by loadsave. |

**`LogicalImage(gr.Textbox)`** (`modules_forge/forge_canvas/canvas.py:83-121`):

- The hidden textarea holds a `data:image/png;base64,...` string.
- `preprocess` turns a str into a PIL RGBA image (or numpy when `numpy=True`). It sets `image.info = self.infotext` (component-level, so **shared across sessions**).
  Non-str values and strings that are not PNG data URLs give `None`.
- `postprocess` turns PIL or ndarray into a data URL and stores `value.info` in `self.infotext`.
- `get_block_name()` returns `"textbox"`.
- `ForgeCanvas` (`:124-138`) creates `gr.HTML(canvas_html with uuid)` plus 2 `LogicalImage`s with **identical `elem_id=uuid`**, distinguished by `elem_classes` `logical_image_foreground` and `logical_image_background`, both `visible=False` (`DEBUG_MODE`).
- `Context.root_block.load(None, js="async ()=>{new ForgeCanvas(...)}")` initializes the canvas JS.
- canvas.js binds `document.querySelector('#<uuid>.<class> textarea')` and polls `textarea.value` every 100 ms (`canvas.js:1-31`).
- **Server → client:** setting the value must update the DOM `textarea.value`.
- **Client → server:** JS sets `textarea.value` and dispatches `input`, so the framework must read that into component state and fire `.change`.

### 13.1 Component value contracts (gr4.40 preprocess and postprocess)

| Component | Python input (`preprocess`) | Python output accepted (`postprocess`) → client | `.value` attribute after init |
|---|---|---|---|
| Textbox / TextArea | `None` or `str(payload)` | `None` or `str(v)` | str or `None`; `lines` may be a float (`lines=1.05`, `ui_extra_networks_user_metadata.py:52`) |
| Number | `None`; raises `Error` if below `minimum` or above `maximum`; rounded per `precision` (None: unchanged JSON int or float; 0: int) | same rounding | as given |
| Slider | payload unchanged (JSON number) | `None` becomes `minimum` | int or float as given (None becomes minimum) |
| Checkbox | bool or `None` | `bool(v)` | bool |
| Radio | `type="value"`: value; `type="index"`: index in the choices or `None` | value as-is | value |
| Dropdown | as Radio. Multiselect: list (index mode gives a list of indices). Requires a list in multiselect index mode | multiselect wraps a scalar into a list. Values outside the choices only warn (unless `allow_custom_value`) | value; `.choices` is a list of `(label, value)` tuples (gr4 normalizes `str` to `(str, value)`) |
| CheckboxGroup | list (index mode: indices) | `None` becomes `[]`, a scalar becomes `[scalar]` | list |
| ColorPicker | str `#rrggbb` | str | str |
| HTML | identity | identity | str |
| Button | identity (label str) | `str(v)` | str |
| State | the session value (never sent to the client) | stored in the session | deep copy |
| Image `type="pil"` | `PIL.Image.open(path)`, EXIF-transposed (tag 274), `.convert(image_mode)` (default `"RGB"`; `"RGBA"` where set), except GIF. `.info` is preserved (PNG text chunks needed by PNG Info and `read_info_from_image`). `.filename` is the upload path | ndarray, PIL, or path, saved via `save_pil_to_cache` (patched) → FileData | FileData dict or `None` |
| Image `type="numpy"` / `"filepath"` | ndarray / path (`format_image` calls `save_pil_to_cache(..., name=, format=)` with a `name` kwarg; the patched `save_pil_to_file` has no `name` param, see §21 #6) | as above | as above |
| Gallery `type="pil"` | `None` if empty, else `[(PIL.Image.open(path), caption)]`. `PIL.filename` is the server path, which webui reuses as `already_saved_as` (`txt2img.py:117`) | list of PIL, ndarray, str, Path, or `(img, caption)`; `None` gives `[]`. Each PIL goes through `save_pil_to_cache` → `GalleryData[{image: FileData, caption}]` | `[]` |
| File (`type="filepath"`, single) | `NamedString(path)` | path or list of paths → FileData or ListFiles (with `size` from `stat`) | |
| File `type="binary"` | `bytes` (`ui_toprow.py:96` prompt image drop, `ui_settings.py:203`) | | |
| Files / File(multiple) | `list[NamedString]` | list of paths | |
| Video | path (format conversion if `format` is set) | path or `(path, subtitle)` → `VideoData{video: FileData, subtitles}`; `include_audio=False` | |
| Audio | (notification, output only, `ui.py:909-910`) | path | |

Layout blocks (Row, Column, Group, Tabs, TabItem, Accordion, Form) carry `visible`, `elem_id`, `elem_classes`, `variant`, `scale`, `min_width`, `equal_height`, `open`, `selected`, `id`, `label`, `render`, `children`, and `parent`. They accept prop updates only.

---

## 14. Image and file transport (`modules/ui_tempdir.py`, gr4 file routes)

### 14.1 Output images (PIL → browser)

1. `images.save_image(...)` sets `image.already_saved_as = fullfn` (`modules/images.py:779`) for every saved sample.
2. Gallery and Image postprocess call `gradio.processing_utils.save_pil_to_cache`, which is **replaced by `ui_tempdir.save_pil_to_file`** (`ui_tempdir.py:26-49, 116-121`):
   - If `already_saved_as` exists on disk: register both `path` and `path?mtime` in `shared.demo.temp_file_sets[0]`, and return `"path?mtime"`, so **no copy is made**.
   - Otherwise, write a PNG (always `.png`, whatever the `format`) into `opts.temp_dir` or the Gradio cache dir. **String items of `pil_image.info` are copied as PNG text chunks.**
3. `async_move_files_to_cache` is replaced by the version in `ui_tempdir.py:52-111`:
   - it strips `?…` from `payload.path`;
   - if `check_tmp_file` passes, it keeps the path;
   - else, for inputs, it enforces "in the upload folder";
   - else it copies the file into the block cache via `block.async_move_resource_to_block_cache` (which registers it);
   - it then sets `url = "/file=<path>"` (or `/stream/`, or a proxy).
4. The browser loads `/file=<abs path>`. The gr4 route (`routes.py:524-590`) allows the file if it is:
   - created by the app (in `temp_file_sets`);
   - under `blocks.allowed_paths`, which by default is `--gradio-allowed-path` (default `[data_path]`, `modules/cmd_args.py:44`) plus `canvas_js_root_path` (`webui.py:120`);
   - in the upload dir;
   - in the cache folder;
   - or a static file.
   It denies `blocked_paths` and directories, supports `Range`, and is guarded by the login check.
5. `register_tmp_file`, `check_tmp_file`, and `on_tmpdir_changed` (`ui_tempdir.py:16-23, 126-132`) **directly read and write `shared.demo.temp_file_sets`**, a list of sets of absolute path strings.
   The compat Blocks must expose `temp_file_sets` (mutable, with set 0 existing) or the module must be ported.

### 14.2 Input images (browser → PIL)

- Gradio's client uploads a dropped file with `POST /upload` (multipart). It gets back server paths in the upload dir (`GRADIO_TEMP_DIR`, else `<tmp>/gradio`), then sends FileData `{path, url, orig_name, size, mime_type, is_stream, meta:{_type:"gradio.FileData"}}` as the input payload.
- Gallery values coming back from the client carry `path` and `url` of earlier outputs. Those must pass the patched check (registered temp file) before `PIL.Image.open`.
- The canvas does not upload: it sends base64 data URLs through the LogicalImage textarea.
- `ui_tempdir.is_gradio_temp_path` (`:150-160`) recognizes `opts.temp_dir`, `GRADIO_TEMP_DIR`, and `<tmp>/gradio`. It is used by the open-folder logic.

### 14.3 Other transports

| Kind | Mechanism |
|---|---|
| Live preview | base64 data URL in the `/internal/progress` JSON (§6) |
| Extra network thumbnails | `./sd_extra_networks/thumb?filename=<quoted>&mtime=<m>` (`ui_extra_networks.py:222-225`) via `fetch_file` (checks `allowed_dirs` and the extension whitelist, `:91-105`) |
| Embedded cover images | `./sd_extra_networks/cover-images?page=&item=` (`:108-130`) |
| API | base64 in JSON (`api/api.py:92-130`: `decode_base64_to_image` accepts a data URL or http(s) when allowed; `encode_pil_to_base64`) |
| JS and CSS assets | `file=<relative path>?mtime` (cwd-relative; must be inside the allowed paths, normally `data_path` = repo root) and `/webui-assets` (StaticFiles mount of `modules/web`, `ui.py:990-992`) |
| Downloads | `gr.File` outputs of saved paths (copied to the cache if outside the allowed paths) |

---

## 15. Gradio monkeypatches installed by webui (behaviour the compat must make native)

From `modules/gradio_extensions.py`:

| Patch | Line | Native behaviour required |
|---|---|---|
| `Component.__init__` → `IOComponent_init` | 28-45, 88 | Pops `tooltip` into `self.webui_tooltip`. Calls `scripts_current.before_component(self, **kwargs)`, then `script_callbacks.before_component_callback`, then the original init, then `add_classes_to_gradio_component` (prepends `gradio-<get_block_name()>`, appends `multiselect`), then `after_component_callback`, then `scripts_current.after_component`. |
| `BlockContext.__init__` | 60-75, 90 | The same hooks for layout blocks. |
| `Block.get_config` | 48-57, 89 | Adds `webui_tooltip` to the component props and drops `example_inputs`. |
| `Blocks.get_config_file` | 78-85, 91 | Neutralizes `example_inputs`. |
| `install_ui_tempdir_override()` | 94 | §14. |
| `component_meta.create_or_modify_pyi` and `updateable` | 97-108 | No `.pyi` generation; the `updateable` decorator becomes the identity. The compat should not generate files. |
| `repair(grclass)` for every class in `gr.components.__all__ + gr.layouts.__all__` | 129-159 | `source=` is mapped to `sources=[source]`. **Unknown kwargs are dropped with a `GradioDeprecationWarning`** (not an error). `tooltip` is kept. Every event attribute is wrapped in `EventWrapper` to accept `_js`. `cls.update = gr.update` (old `gr.X.update(...)` calls). |
| `gradio.events.Dependency` subclass | 163-177 | `.then(_js=...)`. |
| `gr.Box = gr.Group` | 178 | Alias. |

Also from other modules:

- `ui.py:45-48` patches `gradio.utils.version_check` and `get_local_ip_address`.
- `ui_gradio_extensions.py:69` stores `shared.GradioTemplateResponseOriginal`.
- `canvas.py:16-30` patches `create_or_modify_pyi`.
- `ui_components.py:14` patches `gr.Dropdown.get_expected_parent`.

The compat must accept that code assigns these attributes. No-op holders are enough.

---

## 16. Post-launch and app-level consumers of the Gradio object model

- `script_callbacks.app_started_callback(demo, app)` (`webui.py:151`). Extensions receive the Blocks.
  **agent-scheduler** (`extensions-builtin/agent-scheduler/scripts/task_scheduler.py:96-200`, `agent_scheduler/helpers.py:70-140`) is the strongest consumer:
  - at build time, in an `after_component` hook, it does `component.parent.children.pop()` and `parent.add(row)` to move its Enqueue row;
  - at app start it reads `generate._id`, `root.fns.values()` (`fn.inputs`, `fn.outputs`, `fn.targets`), and `root.default_config.get_config()["dependencies"]`;
  - then, **after launch**, it registers `with root: submit_button.click(fn=f(request: gr.Request, *args), _js="(...args) => submit_enqueue(...args)", inputs=[checkpoint_dropdown] + fn_block.inputs)`.
  - **The compat must expose `Block._id`, `parent`, `children`, `add()`, `Blocks.fns` (BlockFunction with `.inputs`, `.outputs`, `.targets`, `.fn`), `Blocks.default_config`, and accept event registration after launch, re-serializing the config.**
- `script_callbacks.ui_tabs_callback()` returns `list[(gr.Blocks, label, elem_id)]`, which are rendered as top-level tabs (`ui.py:877`).
- `shared.demo` is used by `ui_tempdir` (temp_file_sets) and `infotext_utils.image_from_url_text`.
- `demo.ui_loadsave` is set as an attribute (`ui.py:928`).

---

## 17. Browser-side contracts the React SPA must honour (the bundled JS depends on them)

Details belong to the JS inventory. This is the runtime-critical subset seen while tracing:

1. **Hidden components must be in the DOM.** Gradio 4 mounts `visible=False` components and hides them. JS clicks or writes to:
   - `#txt2img_token_button`, `#change_checkpoint`, `#change_checkpoint_text textarea`, `#settings_json textarea`;
   - `#<accordion>-checkbox input`, `#<uuid>.logical_image_* textarea`;
   - `#img2img_update_resize_to`, `#*_extra_refresh_internal`, `#*_edit_user_metadata_button`, `#*_save_preview`, `#install_extension_button`, `#settings_show_one_page`.
2. **Element structure.**
   - `#<elem_id>` sits on the component wrapper, which contains a native `textarea` or `input`.
   - The class list has `gradio-<blockname>` plus `elem_classes`.
   - Number inputs are `input[type=number]` (`javascript/ui.js:453-456`).
   - Tabs are `#tabs > .tab-nav > button.selected` and `.tabitem[id^=tab_]` with `style="display: block|none"` (`script.js:16-25`, `javascript/ui.js:10-21, 95-104`).
   - Gallery thumbnails are `.thumbnails > .thumbnail-item.thumbnail-small.selected` under `div[id$=_gallery].gradio-gallery` (`javascript/ui.js:10-39`).
   - Accordions have `.label-wrap` with an `open` class toggled on click (`javascript/inputAccordion.js:7-69`).
3. **Programmatic input.** `updateInput(el)` dispatches a bubbling `input` Event with an overridden `target` (`javascript/ui.js:409-413`). The SPA must listen on the native element and sync state from `el.value` or `el.checked`.
4. **`window.gradio_config.components[]`** with `{id, props: {elem_id, webui_tooltip, placeholder, label, …}}` (`javascript/hints.js:84-89`, `javascript/localization.js:108-130`), and the fallback id `component-<id>`.
5. **`gradio-app` element and shadow root**: `gradioApp()` uses `document.getElementsByTagName("gradio-app")[0]` if present (`script.js:1-11`).
6. **MutationObserver UI lifecycle.** `onUiLoaded` fires once `#txt2img_prompt` appears (`script.js:120-136`). The SPA must render the full tree, including all tabs' content, into the DOM at load, as Gradio does.
7. **`settings_json` textarea.** JS redefines the instance `value` property (`javascript/ui.js:345-361`).
   React's controlled `<textarea>` also installs a value tracker on the node, so the compat should write this element through the native setter path (`el.value = x` after mount) and keep it uncontrolled.
8. **JS hook functions are globals** defined by `javascript/*.js` and extension scripts (`submit`, `submit_img2img`, `submit_extras`, `submit_txt2img_upscale`, `extract_image_from_gallery`, `selected_gallery_index`, `switch_to_*`, `confirm_clear_prompt`, `recalculate_prompts_*`, `restart_reload`, `currentImg2imgSourceResolution`, `onCalcResolutionHires`, `inputAccordionChecked`, `populateLicense`, `closePopup`, `popupId`, `setupAllResizeHandles`, `extensions_apply`, `modelmerger`, `clickLoraRefresh`, `config_state_confirm_restore`, …). They must be evaluated after these scripts load.
9. **Externally inserted DOM** (live preview `div` in the gallery, progress bar `div` before the gallery container, popup re-parenting, Enqueue row moves) must survive React reconciliation.

---

## 18. `modules/ui_extra_networks*.py`: pages, HTML, routes

- `register_page(page)` adds to `extra_pages` and recomputes `allowed_dirs` (`ui_extra_networks.py:83-88`).
  Default pages are Textual Inversion and Checkpoints (`:692-697`). LoRA comes from `sd_forge_lora` (`scripts/lora_script.py:52`).
- `create_ui(interface, unrelated_tabs, tabname)` (`:732-786`), per page:
  - `gr.Tab(page.title, elem_id=f"{tab}_{name}", elem_classes=["extra-page"])` containing an empty `gr.Column(elem_id=…_prompts)`, which JS moves the prompt boxes into;
  - `gr.HTML(page.create_html(tab, empty=True), elem_id=f"{tab}_{name}_cards_html")`, with "Loading..." placeholder HTML;
  - the user-metadata editor UI (a hidden `gr.Group` popup, `ui_extra_networks_user_metadata.py:163-171`).
  - Then a hidden `Save preview` button and a hidden filename Textbox (`:757-758`).
- Events:
  - `tab.select(fn=None, _js=extraNetworksTabSelected/UnrelatedTabSelected …)` (`:760-765`);
  - a hidden refresh button: `.click(fn=refresh, outputs=ui.pages).then(fn=lambda: None, _js=applyExtraNetworkFilter…).then(fn=lambda: None, _js="setupAllResizeHandles")` (`:773-774`);
  - **`interface.load(fn=pages_html, outputs=ui.pages).then(fn=lambda: None, _js="setupAllResizeHandles")`** (`:784`). The full card HTML (possibly megabytes) is delivered as HTML values on the first load. The compat must handle big HTML string outputs, and should gzip them (GZip middleware exists).
- The card HTML is a string template (`html/extra-networks-*.html`) with inline `onclick` handlers (`cardClicked`, `selectCheckpoint`, `extraNetworksSearchButton`, `saveCardPreview`). **The HTML component must inject raw HTML (`innerHTML`) and allow inline handlers.** The cards call back via:
  - `/sd_extra_networks/get-single-card?page=&tabname=&name=`, returning `{"html": …}` (`:149-164`);
  - `/sd_extra_networks/metadata`;
  - hidden button clicks into Gradio events (`edit_user_metadata` button: `.click(put_values_into_components, inputs=[name textbox], outputs=[html, textbox, html, html, textarea]).then(lambda: gr.update(visible=True), outputs=[box])`, `ui_extra_networks_user_metadata.py:154-161`).
- `setup_ui(ui, gallery)` (`:796-827`):
  - `button_save_preview.click(fn=save_preview, _js="function(x, y, z){return [selected_gallery_index(), y, z]}", inputs=[filename_tb, gallery, filename_tb], outputs=[*ui.pages])`. A Textbox input carries an int index, and the gallery is preprocessed to PIL tuples.
  - Each editor's `setup_ui(gallery)` uses `selected_gallery_index_id('<tab>_gallery_container')`.
- Routes (`add_pages_to_demo`, `:167-171`): `GET /sd_extra_networks/thumb`, `/cover-images`, `/metadata`, and `/get-single-card`. They are plain FastAPI and stay as-is.

---

## 19. `modules/api/api.py`: Gradio dependencies

- No direct `import gradio`, but it imports `modules.ui` (`:26`), which imports gradio. So the compat package must import cleanly in API-only mode.
- `Api.__init__` calls `ui.create_ui()` if the script runners are empty (`:270-271`). This builds the UI headless, so the compat Blocks must not require a running server.
- `init_default_script_args` reads `script.api_info.args[i].value` (from `control.value`) (`:325-339`).
- `apply_infotext` reads `infotext_utils.paste_fields[tab]["fields"]`, `field.function`, `field.label`, `field.api`, `type(field.component.value)`, and `script_runner.inputs` identity (`:369-443`). `pasted_value` unwraps `{"__type__": "update", "value": …}` dicts (`:40-54`).
- Processing runs under `self.queue_lock` (the shared FIFOLock) **on the uvicorn worker thread, not `main_thread`** (`:480-499`, `553-571`).
- `progressapi` (`/sdapi/v1/progress`) reads `shared.state` (`:619-643`). Interrupt and skip call `shared.state.interrupt()` and `skip()`.
- Extras API calls `postprocessing.run_extras`, which calls `scripts_postproc.create_args_for_run` (which may build a `gr.Blocks()`) (`postprocessing.py:179-208`).

---

## 20. `modules/ui_postprocessing.py`, PNG Info, and the model merger (runtime notes)

- PNG Info (`ui.py:834-860`):
  - `gr.Image(type="pil", image_mode="RGBA", source="upload", height="50vh")`, wired with `image.change(fn=wrap_gradio_call_no_job(run_pnginfo), outputs=[html, generation_info (hidden Textbox), html2])`.
  - `run_pnginfo` returns `("", geninfo, info_html)` (`modules/extras.py:18-36`). The returned PIL **must keep its PNG text info**.
  - It has four send-to buttons (`create_buttons`, `infotext_utils.py:154-155`) bound with `source_text_component=generation_info` and `source_image_component=image`.
- Model merger (`modules/ui_checkpoint_merger.py:120-160`):
  - `.click(lambda: "", outputs=[result], queue=False).then(fn=wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [...]), js="modelmerger", inputs=[dummy, …], outputs=[…, sd_model_checkpoint_component, result])`.
  - `inputs=filename_forge` and `outputs=result_html` are single components (`:58-59`).
  - `gr.Info` and `gr.Warning` are used with `duration=5`.

---

## 21. Latent quirks and bugs found while tracing (decide: replicate or fix)

1. **`restore_progress` output mismatch.**
   - `recorded_results` stores the 5-tuple from txt2img and img2img (gallery, video, geninfo, info, comments), but the restore click has 4 outputs (`ui.py:414-420, 773-779`).
   - Gr4 ignores the extra value, so `generation_info` receives `video_arg`: `gr.update(value=None, visible=False)` hides and clears it. `infotext` receives the JSON, and `html_log` receives the info HTML.
   - The compat must at least tolerate more returns than outputs. Fixing the mapping is recommended.
2. **`create_submit_args` heuristic** (`javascript/ui.js:118-128`).
   - For `submit_txt2img_upscale` (4 outputs) and `submit_extras` (3 outputs), it first tests `Array.isArray(res[len-5])`, which is the last *input*.
   - If the last script control's value is an array (a multiselect Dropdown or CheckboxGroup), one real input is dropped.
   - The compat should preserve the "inputs + outputs" argument order exactly; any fix should be made in JS.
3. **Model merger `extra_outputs` is a lambda** (`ui_checkpoint_merger.py:133`). On an exception, `extra_outputs_array + [...]` raises `TypeError` inside the error handler (`call_queue.py:84-87`). The compat sees a raised exception, which becomes an error toast.
4. **`detect_image_size_btn`**: the JS returns 3 values for 2 inputs (`ui.py:762-768`, `javascript/ui.js:428-434`).
   - Gr4 tolerates this. `validate_inputs` raises only when there are *fewer* values ("ignore extra arguments", gr4 `blocks.py:1583-1612`).
   - `preprocess_data` iterates only over `block_fn.inputs` (`blocks.py:1630-1650`), so the extra JS values are **silently truncated**.
   - **The compat must accept more values than inputs and truncate them; fewer values must raise.** This also applies to the payload sent from `create_submit_args`.
5. **`LogicalImage.__init__` calls `self.image_to_base64`**, which does not exist (`canvas.py:92`). It only crashes when `initial_image` is not `None`; core never passes one.
6. **The patched `save_pil_to_file(pil_image, cache_dir=None, format="png")`** has no `name` parameter. gr4 `image_utils.format_image` (Image `type="filepath"`) calls `save_pil_to_cache(im, cache_dir=…, name=…, format=…)`, which would raise TypeError. Any extension using `gr.Image(type="filepath")` input could trip on this, so the compat hook should accept `**kwargs`.
7. **`LogicalImage.infotext` is per component, not per session.** PNG info pasted by one user leaks to another user's next preprocess.
8. **`open_folder` uses the legacy `images[index]["name"]`** (`ui_common.py:178`), which always raises with Gradio 4 tuples and is swallowed. As a result, "Sub" directory opening never works.
9. **`gr.Info` and `gr.Warning` inside `queue=False` handlers are dropped in gr4** (no event channel), e.g. `_change_model` (`ui_settings.py:336`).
10. **Per-session prop updates re-run component `__init__`**, which re-fires `before_component` and `after_component` callbacks (§4.5). Extensions with side effects in those callbacks may misbehave at runtime. The compat may decide not to replicate this.
11. **The CORS filter in `webui.py:135` is a no-op on gr4.40** because the class is named `CustomCORSMiddleware`.

---

## 22. Checklist for the compat implementation (derived)

- [ ] `import gradio as gr` provides:
  - classes: `Blocks`, `Row`, `Column`, `Group`, `Box`, `Tabs`, `Tab`, `TabItem`, `Accordion`, `Textbox`, `TextArea`, `Number`, `Slider`, `Checkbox`, `CheckboxGroup`, `Radio`, `Dropdown`, `ColorPicker`, `HTML`, `Markdown`, `Button`, `Image`, `Gallery`, `File`, `Files`, `Video`, `Audio`, `State`, `Code`, plus the extension-used ones;
  - functions and objects: `update`, `skip`, `Info`, `Warning`, `Error`, `Request`, `EventData`, `SelectData`, `Progress`, `on`;
  - modules: `themes`, `components` (`Component`, `Form`, `__all__`), `layouts.__all__`, `blocks` (`Block`, `BlockContext`, `Blocks`), `context.Context.root_block`, `events.Dependency`, `utils` (`version_check`, `get_local_ip_address`, `abspath`, `NamedString`), `processing_utils.save_pil_to_cache` and `async_move_files_to_cache` (hookable), `routes.templates.TemplateResponse` (hookable), `component_meta`, `data_classes.FileData`, `__version__`.
- [ ] Constructor-kwarg tolerance as in `repair()`: drop unknown kwargs with a warning, map `source` → `sources`, keep `tooltip`, use `.update = gr.update`, and record `constructor_args`.
- [ ] Before and after component hooks with kwargs. `gradio-<block_name>` classes. `get_block_name()` is overridable.
- [ ] Mutable attributes reflected in the served config. Exact class identity for core components (`type(x) == gr.Slider`).
- [ ] Event registration surface per §4.1; JS semantics per §4.3; argument binding per §4.4; output binding per §4.5; State per §4.6; threading per §4.7; generators, Progress, and toasts per §4.8.
- [ ] Sub-Blocks `render()` merges, `.load` from any Blocks, callable `value` reload, `Context.root_block`, `render=False` and a later `.render()`, manual zero-argument `__enter__` and `__exit__`, `parent`, `children`, `add()`, `_id`, `fns`, `default_config`, and post-launch registration.
- [ ] `queue(default_concurrency_limit)`. `launch(**webui kwargs)` returns `(app, url, share_url)`, does not block, supports auth with `Request.username`, and supports `root_path`. Also `close()` and relaunch.
- [ ] Routes: index (with head and body injection and the Blocks `head`), `config` (or the SPA bootstrap JSON, exposed as `window.gradio_config`), `upload`, `file=` (allowlist via `temp_file_sets`, `allowed_paths`, upload, cache; Range support), event submit and stream (SSE or WebSocket), login, and heartbeat.
- [ ] Component data contracts per §13.1, including `NamedString`, Gallery `(PIL, caption)` tuples with `.filename`, PIL `.info` preservation, `save_pil_to_cache` semantics (`already_saved_as`, PNG text), and the FileData JSON shape.
- [ ] DOM contracts per §17.
