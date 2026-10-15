# Writing extensions for the Remi WebUI

Remi keeps the A1111 / Forge extension model for everything that happens
during generation (`scripts.Script` hooks, `script_callbacks`, settings,
extra networks, `/sdapi` routes), and replaces the Gradio user interface with
a **declarative parameter model**: a script describes its inputs once, in
`params()`, and that description drives the web interface, the HTTP API,
infotext paste, presets, queued jobs and validation. There is no Gradio, no
server-rendered HTML and no JavaScript injected into the page.

Read this together with [architecture.md](architecture.md) (layers and API)
and [schema-format.md](schema-format.md) (the JSON the frontend renders).
Working examples are the built-in extensions in `extensions-builtin/`; this
guide points at the simplest ones.

Contents:

1. [Layout of an extension](#1-layout-of-an-extension)
2. [Scripts and their hooks](#2-scripts-and-their-hooks)
3. [`params()`: inputs](#3-params-inputs)
4. [Layout nodes](#4-layout-nodes)
5. [Conditions](#5-conditions)
6. [Actions](#6-actions)
7. [Infotext, paste, presets and exclusive groups](#7-infotext-paste-presets-and-exclusive-groups)
8. [Extras (postprocessing) operations](#8-extras-postprocessing-operations)
9. [Pages (the former `on_ui_tabs`)](#9-pages-the-former-on_ui_tabs)
10. [Settings](#10-settings)
11. [Custom widgets (frontend)](#11-custom-widgets-frontend)
12. [Localization](#12-localization)
13. [Testing](#13-testing)
14. [Porting an A1111 / Forge extension](#14-porting-an-a1111--forge-extension)
15. [What is not supported, and what to do instead](#15-what-is-not-supported-and-what-to-do-instead)

## 1. Layout of an extension

Unchanged from A1111: a folder in `extensions/` (user) or
`extensions-builtin/` (shipped with Remi).

```
extensions/prompt-suffix/
├── scripts/prompt_suffix.py     # scripts.Script subclasses, callbacks, pages
├── lib_prompt_suffix/           # optional package (the extension folder is on sys.path)
├── localizations/zh_CN.json     # translations of your UI strings (English text → translation)
├── tests/test_prompt_suffix.py  # schema / processing tests (pytest)
├── install.py                   # optional: pip installs, run by the launcher
└── preload.py                   # optional: preload(parser) adds command-line flags
```

`javascript/` and `style.css` folders are **not loaded** any more (see
[§15](#15-what-is-not-supported-and-what-to-do-instead)). Any module that
imports `gradio` fails to import: Gradio is not installed.

## 2. Scripts and their hooks

```python
from modules import scripts
from remi import ui


class PromptSuffix(scripts.Script):
    sorting_priority = 50          # position among the scripts of its slot (ascending)
    section = None                 # slot of the generation form to render in (see below)

    def title(self):
        return "Prompt Suffix"     # also the source of the script id: "prompt_suffix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible   # or True: selectable in the "Script" dropdown; False: hidden

    def params(self, is_img2img):  # replaces ui(); one layout node, or None
        return ui.Section(
            self.title(),
            enable=ui.Toggle("enabled", False, label=self.title(), infotext="Suffix enabled"),
            children=[
                ui.Text("suffix", "", lines=2, prompt=True, label="Suffix", infotext="Suffix"),
                ui.Row(
                    ui.Choice("position", ["end", "start"], "end", style="radio", label="Position", infotext="Suffix position"),
                    ui.Slider("weight", 0.0, 2.0, 1.0, 0.05, label="Weight", infotext="Suffix weight", enabled_if={"param": "suffix", "ne": ""}),
                ),
                ui.Button("Use the prompt", "copy_prompt", "tool"),
            ],
        )

    @ui.action("copy_prompt")
    def copy_prompt(self, ctx, values):
        prompt = ctx.core.get("prompt", "")
        if not prompt:
            return ui.toast("The prompt is empty", "warning")
        return {"suffix": {"value": prompt}}

    def process(self, p, enabled, suffix, position, weight):
        if not enabled or not suffix:
            return
        text = suffix if weight == 1 else f"({suffix}:{weight:g})"
        p.all_prompts = [f"{text}, {prompt}" if position == "start" else f"{prompt}, {text}" for prompt in p.all_prompts]
        p.extra_generation_params.update({"Suffix enabled": True, "Suffix": suffix, "Suffix position": position, "Suffix weight": weight})
```

**Arguments are positional, in declaration order.** The runner builds a
`remi.ui.ScriptLayout` from `params()` (depth first, in the order the inputs
are declared, including the section's `enable` toggle) and calls every hook
with those values as positional arguments, exactly as A1111 passed the values
of the controls `ui()` returned. When the processing order must differ from
the visual order, set `param_order = ("steps", "sampler_name", ...)` on the
class (the Sampler script does).

**Every processing hook is unchanged** (`modules/scripts.py`): `setup`,
`before_process`, `after_model_load` (Remi: after checkpoint/VAE overrides
are loaded), `process`, `before_process_batch`,
`after_extra_networks_activate`, `process_before_every_sampling`,
`process_batch`, `post_sample`, `on_mask_blend`, `postprocess_batch`,
`postprocess_batch_list`, `postprocess_image`, `postprocess_maskoverlay`,
`postprocess_image_after_composite`, `postprocess`, `before_hr`, `cleanup`
(Remi: always runs, from `process_images`' `finally`) and, for selectable
scripts, `run(p, *args)`. So are the `script_callbacks` that do not build UI:
`on_app_started` (FastAPI routes), `on_model_loaded`, `on_cfg_denoiser` /
`on_cfg_denoised` / `on_cfg_after_cfg`, `on_extra_noise`,
`on_before_image_saved` / `on_image_saved`, `on_image_grid`,
`on_infotext_pasted`, `on_script_unloaded`, `on_before_reload`,
`on_list_optimizers`, `on_list_unets`, `on_before_token_counter`,
`on_ui_settings` (settings, [§10](#10-settings)) and `on_before_ui`.

**Script id.** Each script gets a stable owner id, the slug of its title
(`"Prompt Suffix"` → `prompt_suffix`; a clash gets `_2`; `core` and
`settings` are reserved). The API and other scripts address your params as
`prompt_suffix.<name>`.

**Where it renders.** `section` names a slot of the generation form:
`prompt`, `image`, `inpaint`, `sampler`, `dimensions`, `cfg`, `denoising`,
`seed`, `batch`, `accordions`, `checkboxes`, `override_settings`, or a name
of your own (it becomes a category of its own, `modules/shared_items.py`
`ui_reorder_categories`). Without a section the script goes to the scripts
list under the form. The user's category order (Settings → "Parameter order
for txt2img / img2img") decides the layout. Selectable scripts render only while chosen in the
"Script" dropdown.

**Failures are contained.** A script whose `params()` raises is disabled
(no slots, no hooks) and the reason is logged and listed in the bootstrap
`not_ported` list (`GET /api/v2/app`). A script that still declares only
A1111's `ui()` is disabled the same way: it has no parameters to receive.

## 3. `params()`: inputs

Every input has a `name` (`[A-Za-z_][A-Za-z0-9_]*`, unique within the
script), which is its param id in the API, drafts, presets and queued jobs.
Labels are English text; the frontend translates them
([§12](#12-localization)).

| class | value the script receives | example |
|---|---|---|
| `ui.Toggle(name, default=False, locks=None)` | `bool` | `ui.Toggle("enabled", False, label="Enable")` |
| `ui.Slider(name, minimum, maximum, default, step, integer=None)` | `int` when bounds, step and default are ints, else `float`; clamped | `ui.Slider("steps", 1, 150, 20, 1)` |
| `ui.Number(name, default, minimum=None, maximum=None, step=None, precision=None)` | `int`/`float`; out of bounds is rejected | `ui.Number("seed_offset", 0, precision=0)` |
| `ui.Text(name, default="", lines=1, max_lines=None, placeholder=None, prompt=False)` | `str`; `prompt=True` gives the prompt editor (weights, LoRA, completion) | `ui.Text("negative", "", lines=3, prompt=True)` |
| `ui.Choice(name, choices, default=None, style="dropdown" \| "radio", allow_custom=False, type="value")` | the choice value, or its index with `type="index"` | `ui.Choice("mode", ["fast", "best"], "fast", style="radio")` |
| `ui.MultiChoice(name, choices, default=(), style="dropdown" \| "checkboxes", allow_custom=False, type="value")` | `list` of values (or indices) | `ui.MultiChoice("blocks", ["in", "mid", "out"], ["mid"], style="checkboxes")` |
| `ui.Seed(name, default=-1)` | `int` (−1 = random); dice and recycle buttons | `ui.Seed("noise_seed")` |
| `ui.Color(name, default="#000000")` | `"#rrggbb"` | `ui.Color("tint", "#ffcc00")` |
| `ui.Image(name, kind="image" \| "mask" \| "sketch", source=None, mode=None)` | PIL image (converted to `mode`) or `None` | `ui.Image("reference")`, `ui.Image("mask", "mask", source="reference")` |
| `ui.ImageList(name, widget=None, props={})` | `list` of PIL images | `ui.ImageList("refs", widget="image-list")` |
| `ui.File(name, accept=None)` | filesystem path of the upload, or `None` | `ui.File("lut", accept=".cube")` |
| `ui.Json(name, default=None, validate=None)` | any JSON value (`validate(value)` may normalize or raise `ValueError`) | `ui.Json("rows", [])` |
| `ui.Custom(name, widget, default=None, props={})` | any JSON value, edited by a frontend widget ([§11](#11-custom-widgets-frontend)) | `ui.Custom("curve", "sigma-curve", None, props={...})` |
| `ui.RangeSlider(name, minimum, maximum, default=None, step=0.01, integer=None)` | `[low, high]` | `ui.RangeSlider("percent", 0.0, 1.0, (0.2, 0.8), 0.01)` |
| `ui.State(name, default)` | any JSON value; never rendered, always submitted | `ui.State("version", 2)` |

`choices` is a list of values or `(label, value)` pairs, or a **callable**
returning one; callables (also as `default` or a widget's `props`) are
evaluated once per request, so lists of models, samplers or files stay
current without a refresh button. A `None` default of a `Choice` is its first
choice.

Keyword arguments every input takes:

| keyword | meaning |
|---|---|
| `label`, `info` | English label (default: from the name; `""` hides it) and help text under the control |
| `visible_if`, `enabled_if` | [conditions](#5-conditions) evaluated in the browser |
| `on_change="action"` | run that [action](#6-actions) after any change of the value (user edit, paste, preset, reset) |
| `on_input="action"` | run it only after the user's own edit (Gradio's `.input`): for derived values an infotext paste already restores |
| `infotext="Key"` | infotext key the value is written under and pasted from ([§7](#7-infotext-paste-presets-and-exclusive-groups)) |
| `paste=callable` | `params -> value` for a derived paste (several keys, a renamed key, legacy formats) |
| `preset="field"` | Anima preset field this input belongs to (Remi's Anima presets) |
| `exclusive=(group, member)` | `Toggle` only: one-of-N group, resolved by the server |
| `width` | inside a `Row`: `"auto"`, `"fill"` or a 1–12 column span |
| `hidden=True` | submitted, never rendered (e.g. a choice bound to a tab strip) |
| `persist=False` | not kept in browser drafts: every page load starts from the default |
| `submit=False` | held by the form for actions only; jobs never carry it, processing gets the default |

`Toggle(locks="owner.param")` protects another param from programmatic
writes (paste, "Send to", reused jobs, presets, action patches) while the
toggle is on; `sd-lock-img2img-steps` is the whole extension:
`ui.Toggle("lock", False, label=..., locks="sampler.steps")`.

**Validation.** The server checks every value of every job before it is
queued (and of every action call): sliders are clamped, numbers out of bounds,
unknown choices, malformed colors and non-finite numbers are rejected with a
per-field error the form shows. Your hooks only ever see valid values of the
declared types (schema-format.md §3).

**Images.** An `Image` the user painted over another one (`kind="mask"` or
`"sketch"` with `source="<image param>"`) renders as a layer of one canvas
editor; masks arrive alpha-coded (255 = masked; `remi.ui.nodes.mask_channel`
turns any mask into an `L` image), sketches as RGBA. A `mask` without
`source` is an uploaded mask image.

## 4. Layout nodes

| node | meaning |
|---|---|
| `ui.Section(title, *children, enable=None, open=False, exclusive=None)` | collapsible panel; with `enable=ui.Toggle(...)` the toggle sits in the header (the former `InputAccordion`) and is the first argument |
| `ui.Group(title=None, *children, open=True)` | visual group; collapsible when titled |
| `ui.Row(*children)`, `ui.Column(*children, width=None)` | horizontal flow (12-column grid, wraps on narrow screens) and vertical stack |
| `ui.Tabs(ui.Tab(title, *children, value=...), ..., bind=None)` | tab strip; `bind="mode"` makes the selected tab the value of the `Choice` named `mode` (each `Tab` gives its `value`) |
| `ui.Markdown(text)` | CommonMark, no raw HTML; translated as one string |
| `ui.Html(text)` | trusted server HTML, sanitized in the browser, never translated |
| `ui.Separator()` | horizontal rule |
| `ui.Button(label, action, variant="secondary")` | calls the owner's action; variants `primary`, `secondary`, `ghost`, `tool`, `danger` |
| `ui.Output(name, kind="text" \| "html" \| "image" \| "json", label=None)` | output-only value, filled by an action's `__outputs__` |
| `ui.Gallery(name, label=None)` | output-only image list, filled by `__outputs__` |

Containers take children positionally and/or as `children=[...]`; `None`
children are skipped, so `ui.Row(a, b if is_img2img else None)` works.

```python
def params(self, is_img2img):
    return ui.Section(
        "Detail Booster",
        enable=ui.Toggle("enabled", False, label="Detail Booster"),
        children=[
            ui.Choice("mode", ["Simple", "Advanced"], "Simple", hidden=True),
            ui.Tabs(
                ui.Tab("Simple", ui.Slider("amount", 0.0, 1.0, 0.3, 0.01), value="Simple"),
                ui.Tab("Advanced", ui.Row(ui.Slider("start", 0.0, 1.0, 0.2, 0.01), ui.Slider("end", 0.0, 1.0, 0.8, 0.01)), value="Advanced"),
                bind="mode",
            ),
            ui.Image("reference", label="Reference", visible_if={"param": "mode", "eq": "Advanced"}) if is_img2img else None,
        ],
    )
```

## 5. Conditions

`visible_if` and `enabled_if` are small JSON predicates the browser evaluates
on every change; no round trip, no event wiring.

```python
{"param": "enabled"}                              # truthy
{"param": "mode", "eq": "Advanced"}               # also "ne"
{"param": "mode", "in": ["A", "B"]}
{"param": "core.cfg_scale", "gt": 1}              # "gt", "gte", "lt", "lte" (numbers)
{"all": [cond, ...]}, {"any": [cond, ...]}, {"not": cond}, True, False
```

Helpers build the same dicts: `ui.param("mode", eq="Advanced")`,
`ui.all_of(...)`, `ui.any_of(...)`, `ui.not_(...)`. A bare name is a param of
the same owner; `"core.<name>"` reads the generation form (`core.width`,
`core.enable_hr`, `core.mode` in img2img, …) and `"<script_id>.<name>"`
another script (`sampler.sampler_name`). Equality follows Python (`True ==
1`); a missing param is `undefined` (falsy). Conditions are checked when the
layout is built (unknown operators and non-numeric `gt` operands raise).

A condition that depends on something the browser does not know (a file on
disk, the loaded model) is an [action](#6-actions) patch (`visible`,
`enabled`) instead.

## 6. Actions

An action is a server function a `Button`, `on_change` or `on_input` calls.
It receives the owner's current values and returns a **patch**:

```python
@ui.action("refresh_models")          # @ui.action("name", gpu=True): queue behind generation
def refresh_models(self, ctx: ui.ActionContext, values: dict) -> dict:
    models = list_models()             # your code
    return {
        "model": {"choices": models, "value": values["model"] if values["model"] in models else models[0]},
        "strength": {"enabled": bool(models)},
        "core.width": {"value": 1024},                 # another owner's param
        **ui.toast(ui.message("Found {{n}} models", n=len(models)), "success"),
    }
```

- `values` are the owner's current values as the browser holds them (plain
  JSON, not yet validated: upload ids, not images). Validate what you use:
  `self.layout.coerce(values, ui.Context(), unknown="ignore")` in a script,
  `self.values(ctx, values)` in a page (which also resolves uploads to PIL
  images and paths). `ctx.core` is the generation form, `ctx.scripts` the
  other scripts' values by id, `ctx.tab` the form (`txt2img`, `img2img`,
  `extras`, `page:<id>`), `ctx.user` the logged-in user.
- A patch entry is `{param: {"value"?, "choices"?, "visible"?, "enabled"?,
  "label"?, "info"?, "minimum"?, "maximum"?, "step"?}}`; `"owner.param"`
  patches another owner (`"core.width"`). Special keys: `__toast__` (use
  `ui.toast(message, level)`, levels `info`, `success`, `warning`,
  `error`), `__outputs__` (`{output name: value}` for `Output` / `Gallery`
  nodes) and `__settings__` (settings the action already changed on the
  server). `ui.update(value=..., visible=...)` builds one entry.
- Texts shown to the user are translated by the client: return English
  catalog keys, and `ui.message("Saved {{name}}", name=...)` for texts with
  variable parts (the template is the catalog key, placeholders are filled
  after translation).
- Actions run on a worker thread outside the generation lock. `gpu=True`
  runs it behind the running generation, holding the lock (needed when it
  touches the model or the GPU, e.g. ControlNet's preprocessor preview).
- An exception becomes an error answer (500, shown as an error toast);
  raise `remi.ui.ValidationError([ui.ParamError(message, param=..., owner=...)])`
  for per-field messages (422). Prefer returning a warning toast for
  expected problems.

Built-in examples: `sd_forge_multidiffusion` (`detect_size`: reads the
img2img source image size from `ctx.core`), `sd_forge_spectrum` (preset
save / load / delete), `sd-forge-anima-resolutions` (choices refreshed from
history).

## 7. Infotext, paste, presets and exclusive groups

- **Writing infotext** is unchanged: put your values in
  `p.extra_generation_params` during processing.
- **Reading it back** is declarative: `infotext="Key"` on an input makes
  "Paste parameters", PNG Info's "Send to", reused queue jobs,
  `POST /api/v2/infotext/parse` and `/sdapi`'s `infotext` field restore it.
  The raw text is converted by the input's kind (a toggle reads `"True"`,
  a range slider `"0.2, 0.8"`, …); values that do not fit are skipped and
  reported, not fatal.
- `paste=lambda params: ...` computes the value from the whole parsed
  infotext (`params` is the dict of `Key: value` strings): for values spread
  over several keys, renamed keys or older formats. Return `None` to leave
  the param alone.
- `script_callbacks.on_infotext_pasted(callback)` still runs before the
  values are read, so an extension can rewrite `params` in place (legacy
  key migrations).
- **Exclusive groups:** `ui.Toggle("enable", ..., exclusive=("spatial",
  "regional"))` puts the toggle in a one-of-N group; switching one member on
  switches the others off (in the browser, through the server's `__conflicts__`
  action, and for API jobs by the declared priority). Remi's groups and their
  priority are in `modules/anima_feature_conflicts.EXCLUSIVE_GROUPS`.
- **Anima presets:** `preset="dave.strength"` binds an input to a field of
  Remi's Anima preset table (`modules/anima_presets.py`); applying a preset
  writes it. `preset=(field, value)` gives a localized choice value. Fields
  must exist in the table (a test checks).

## 8. Extras (postprocessing) operations

`scripts_postprocessing.ScriptPostprocessing` subclasses get `params(self)`
(no `is_img2img`) and receive their values as **keyword arguments**, so their
param order is free:

```python
from modules import scripts_postprocessing
from remi import ui


class ScriptPostprocessingDegrid(scripts_postprocessing.ScriptPostprocessing):
    name = "DeGrid"
    order = 2000

    def params(self):
        return ui.Section(
            "DeGrid",
            enable=ui.Toggle("degrid_enabled", True, label="DeGrid"),
            open=True,
            children=[ui.Slider("degrid_threshold", 0.0, 1.0, 0.0, 0.05, label="Threshold", info="0.0 for Auto")],
        )

    def process(self, pp, degrid_enabled: bool = True, degrid_threshold: float = 0.0):
        ...
```

(`scripts/degrid.py`; `scripts/postprocessing_upscale.py` shows tabs bound to
a hidden choice and actions.)

## 9. Pages (the former `on_ui_tabs`)

A page is a top-level view in the navigation, rendered from a layout like a
script, with its own actions. Register it when your script module is
imported:

```python
from pathlib import Path

from remi import pages, ui


class FolderIndex(pages.Page):
    def params(self):
        return ui.Column(
            ui.Text("folder", "", label="Folder", placeholder="/path/to/images"),
            ui.Image("preview", label="Preview image", on_change="describe"),
            ui.Custom("index", "task-runner", None, props={"action": "index", "label": "Index folder"}, label="", persist=False, submit=False),
            ui.Output("status", kind="text", label="Status"),
        )

    @ui.action("describe")
    def describe(self, ctx, values):
        image = self.values(ctx, values)["preview"]  # uploads resolved to PIL images
        text = ui.message("{{w}} × {{h}} pixels", w=image.width, h=image.height) if image else ""
        return {"__outputs__": {"status": text}}

    @ui.action("index")
    def index(self, ctx, values):
        folder = Path(self.values(ctx, values, uploads=False)["folder"])

        def work(task):  # runs on a thread; one task per page, name and user at a time
            files = sorted(folder.glob("*.png"))
            for number, path in enumerate(files, 1):
                task.check()  # raises once the user pressed Cancel
                task.progress(number / len(files), "Reading {{name}}", name=path.name)
            return {"__outputs__": {"status": ui.message("Indexed {{n}} images", n=len(files))}, **ui.toast("Done", "success")}

        return self.start_task(ctx, "index", work)  # gpu=True: behind generation, holding the lock


pages.register(FolderIndex("folder-index", "Folder Index", "database", send_to=[pages.SendTarget("preview", "Folder Index")]))
```

- `Page(id, title, icon=None, *, params=None, actions=None, widget=None,
  send_to=(), on_load=None)`. The id is `[a-z0-9][a-z0-9_-]*`; the owner id is
  `page:<id>` (`POST /api/v2/actions/page:folder-index/index`). Icons are
  names from the SPA's curated set of Lucide icons (`webui/src/app/icons.ts`:
  `camera`, `database`, `image`, `wand`, `wrench`, …; unknown names show a puzzle
  piece).
- Instead of subclassing, pass `params=` (a node or a callable building one
  per request) and `actions=` (an object or module whose `@ui.action`
  functions become the page's).
- `on_load="status"` runs that action each time the page is shown.
- `send_to=[pages.SendTarget(param, label)]` offers "Send to <label>" for
  images in PNG Info (the image lands in the `Image` input `param`, then its
  `on_change` runs).
- Long work runs as a **task** (`start_task`): the action answers at once,
  `task.progress(fraction, text, **params)` and `task.publish(**outputs)`
  report while it runs, the `task-runner` widget shows progress and Cancel and
  applies the work's returned patch at the end, even after the user left and
  came back. `Page.busy(*names)` tells whether a task is running.
- Shared widgets for pages: `task-runner`, `send-button` (send a value to the
  txt2img/img2img prompt, another page's image input, or open a URL) and
  `secret-text` (a write-only API key field; declare it `persist=False,
  submit=False`).
- The page appears in the navigation, follows the "UI Tab Order" and "Hide UI
  Tabs" settings, and disappears when the extension is disabled ("Reload UI").

Examples: `sd-webui-joycaption-modified` (tasks on the GPU, send targets,
send buttons) and `sd-forge-prompt-workshop`.

A page with `widget="name"` is a dedicated frontend view instead
(`webui/src/pages/custom/<name>/index.tsx`); see frontend.md "Adding a page".

## 10. Settings

Settings keep A1111's API. The `component` is a `remi.ui` input class and
`component_args` are its keyword arguments in the familiar Gradio style:

```python
from modules import script_callbacks, shared
from remi import ui


def on_ui_settings():
    section = ("prompt_suffix", "Prompt Suffix")
    shared.opts.add_option("prompt_suffix_default", shared.OptionInfo("", "Default suffix", ui.Text, {"lines": 2}, section=section))
    shared.opts.add_option(
        "prompt_suffix_mode",
        shared.OptionInfo("end", "Where to add the suffix", ui.Choice, {"choices": ["end", "start"], "style": "radio"}, section=section).info("start puts it before the prompt"),
    )
    shared.opts.add_option("prompt_suffix_strength", shared.OptionInfo(1.0, "Default weight", ui.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.05}, section=section).needs_reload_ui())


script_callbacks.on_ui_settings(on_ui_settings)
```

- Widgets: `ui.Toggle` (default for `bool`), `ui.Slider`, `ui.Number`,
  `ui.Text` (default for `str`), `ui.Choice` (`style="radio"`,
  `allow_custom=True`), `ui.MultiChoice` (`style="checkboxes"`), `ui.Color`,
  `ui.Json`, `ui.State`, `ui.Html`. `component_args` may be a callable
  (choices computed when the settings page loads).
- `refresh=callable` adds a refresh button (`POST
  /api/v2/actions/settings/refresh:<key>`) that re-reads the choices.
- `.info(text)`, `.link(label, url)`, `.html(markup)`,
  `.needs_restart()`, `.needs_reload_ui()` are kept and rendered as notes,
  links, help and badges. `onchange=` runs on the server after a change.
  `restrict_api=True` keeps `/sdapi/v1/options` from changing it.
- `shared.options_section((id, label), {...})` with
  `shared.options_templates.update(...)` at import time also still works
  (`extensions-builtin/sd_forge_lora/scripts/lora_script.py`).
- A setting that needs to be visible in the generation form belongs in the
  "Settings in UI" lists (`extra_options_txt2img` / `extra_options_img2img`),
  handled by the built-in `extra-options-section` extension.

The settings schema is exported by `remi/ui/settings.py`
(schema-format.md §6). Gradio component classes are recognized by name for
code that still passes them, but import nothing: prefer the `ui.*` classes.

## 11. Custom widgets (frontend)

When no input kind fits (a curve editor, a timeline, a block-weight grid),
declare `ui.Custom(name, "<widget>", default, props={...})` and implement the
widget in the SPA:

```
webui/src/widgets/<widget>/index.tsx      # default export: the component
webui/src/widgets/<widget>/locales/zh-CN.json
webui/src/widgets/<widget>/index.test.tsx
```

```tsx
import { useTranslation } from "react-i18next"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"

const NS = registerWidgetCatalogs("weight-curve", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { points?: number }

export default function WeightCurve({ value, onChange, props, disabled, id, label }: WidgetProps<number[] | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const points = value ?? Array.from({ length: props.points ?? 8 }, () => 1)
  // Put `id` on the primary focusable element: the field label points at it.
  return <div id={id} role="group" aria-label={label || tw("Weights")}>{/* … call onChange(next) to commit … */}</div>
}
```

- The directory name is the widget name; `import.meta.glob` discovers it at
  build time and loads it lazily the first time a node uses it. There is no
  list to edit.
- The field chrome (label, modified dot, reset, info text, validation
  message) is drawn around the widget. `onChange` commits a value (and runs
  the param's `on_change`). The value must be JSON; the server validates it
  as `Json` (give `ui.Custom` a `validate=` for structure checks).
- A widget can also edit another kind through its `widget` hint (today
  `ui.ImageList(name, widget="image-list")`) and can summarize its values in a
  collapsed section's header (`summary.ts`, see `widgets/controlnet-unit`).
- An unknown widget name renders the JSON editor, so values are never lost.
- **Widgets are part of the SPA build.** A third-party extension cannot
  inject one at runtime: contribute it to `webui/src/widgets` (a pull request
  to Remi, or a local patch). The launcher rebuilds `webui/dist` on the next
  start because the build stamp covers `webui/src`. Prefer the existing
  widgets: `range-slider`, `image-list`, `text-file`, `task-runner`,
  `send-button`, `secret-text`, `sigma-curve`, `stage-timeline`,
  `block-weights`, `block-weight-grid`, `region-boxes`, `xyz-values`, …
  (`webui/src/widgets/`).

Frontend conventions (design system, i18n, tests, accessibility) are in
[frontend.md](frontend.md#adding-a-widget).

## 12. Localization

- Every string your schema shows (labels, info texts, section titles,
  Markdown, choice labels, toast and message templates) is English source
  text and the **translation key**. Translations live in
  `<extension>/localizations/<locale>.json` (`{"English text": "译文"}`),
  for example `localizations/zh_CN.json`.
- The server merges, per locale, the core catalog
  (`localizations/<locale>.json`) with every enabled extension's catalog and
  serves the result at `GET /api/v2/i18n/<locale>`; the SPA translates with
  it and switches languages without a reload. Missing keys show the English
  text.
- Catalogs are merged into one flat table, so the same English text must
  have the same translation everywhere: do not re-translate strings core
  already translates (`tests/remi_port_harness.assert_localized` fails on
  contradictions).
- `ui.Html` text is never translated; choice **values** are never
  translated (only their labels). Use `(label, value)` pairs when the value
  is a key your code reads.
- Infotext keys are not translated (they are data).

## 13. Testing

Each extension keeps its tests in `<extension>/tests/` and they run on their
own (`python -m pytest extensions/<ext>/tests`; `tools/check.sh` runs every
built-in suite). `tests/remi_port_harness.py` gives Gradio-free helpers that
import a script the way Forge does and build its layout as the runner does.

```python
# extensions/prompt-suffix/tests/test_prompt_suffix.py
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("remi_port_harness", ROOT.parents[1] / "tests" / "remi_port_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)
harness.prepare_imports()
SCRIPT = ROOT / "scripts" / "prompt_suffix.py"
module = harness.load_script(SCRIPT, "prompt_suffix_under_test")


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_follow_the_processing_signature(tab):
    layout = harness.script_layout(module.PromptSuffix(), tab)
    hook = inspect.signature(module.PromptSuffix.process)
    assert layout.names == list(hook.parameters)[2:]  # after self, p
    assert layout.values_to_args({}) == [False, "", "end", 1.0]


def test_infotext_round_trip():
    layout = harness.script_layout(module.PromptSuffix())
    values, problems = layout.paste({"Suffix enabled": "True", "Suffix": "masterpiece", "Suffix position": "start", "Suffix weight": "0.5"})
    assert problems == [] and values == {"enabled": True, "suffix": "masterpiece", "position": "start", "weight": 0.5}


def test_schema_is_json_and_localized():
    schemas = [harness.script_layout(module.PromptSuffix(), tab).schema() for tab in ("txt2img", "img2img")]
    json.dumps(schemas, allow_nan=False)
    harness.assert_localized(ROOT, schemas, extra=["Prompt Suffix", "The prompt is empty"])


def test_imports_no_gradio():
    harness.assert_gradio_free([SCRIPT])
```

What to test:

- **Argument parity:** `layout.names` against the hook signature, and
  `values_to_args({})` against the defaults. This is the contract `/sdapi`
  clients, saved queue jobs and imported Agent Scheduler tasks rely on.
- **Paste round trip:** what `process` writes to `extra_generation_params`
  pastes back to the same values (`layout.paste(params)`).
- **Actions:** call them directly (`ui.call_action(script, "copy_prompt",
  ui.ActionContext(core={"prompt": "x"}), {})`) and check the patch.
- **Localization** and **no Gradio** as above.
- **Ports from `ui()`:** record the old `ui()` once with the pre-Remi code
  (the harness's recording CLI, run where Gradio is installed) into
  `tests/fixtures/legacy_ui_<tab>.json`, then
  `harness.assert_matches_snapshot(layout, snapshot, changes={index: {...}})`
  fails on any difference you did not list (kind, label, default, bounds,
  choices, infotext key, preset field). Built-in examples:
  `extensions-builtin/sd_forge_anima_dave/tests/test_params.py`,
  `extensions-builtin/sd_forge_controlnet/tests/`.

Processing tests that need the WebUI's options and script runners without a
model use `remi.testing.webui_state()`; server-level tests use
`remi.server.testing.server_app()` (a running app with a stub executor).

## 14. Porting an A1111 / Forge extension

The rule Remi's own ports followed (architecture.md "Built-in extension
porting rule"): replace `ui()` with `params()` **declaring the same inputs in
the same order**, keep the processing code, move tabs to pages, replace
JavaScript by declarations, actions or widgets, and add the parity test.

### Component mapping

| Gradio / Forge | Remi |
|---|---|
| `gr.Checkbox` | `ui.Toggle` |
| `gr.Slider` | `ui.Slider` (an `int` slider when bounds, step and default are ints) |
| `gr.Number(precision=0)` | `ui.Number(..., precision=0)` |
| `gr.Textbox(lines=n)` | `ui.Text(..., lines=n)`; prompt-like boxes `prompt=True` |
| `gr.Textbox(visible=False)` used as a hidden value | `ui.State`, or an input with `hidden=True` |
| `gr.Dropdown(choices)` | `ui.Choice` |
| `gr.Dropdown(multiselect=True)` | `ui.MultiChoice` |
| `gr.Dropdown(allow_custom_value=True)` | `ui.Choice(..., allow_custom=True)` |
| `gr.Dropdown(type="index")`, `gr.Radio(type="index")` | `ui.Choice(..., type="index")` (the wire value stays the choice) |
| `gr.Radio` | `ui.Choice(..., style="radio")` |
| `gr.CheckboxGroup` | `ui.MultiChoice(..., style="checkboxes")` |
| `gr.ColorPicker` | `ui.Color` |
| `gr.Image(type="pil")` | `ui.Image` (`mode="RGBA"` etc. converts) |
| `gr.Image(tool="sketch")`, `ForgeCanvas` | `ui.Image(...)` plus `ui.Image(name, "sketch" or "mask", source=...)` layers |
| `gr.Gallery` as an input | `ui.ImageList(name, widget="image-list")` |
| `gr.Gallery` as an output | `ui.Gallery(name)` filled by `__outputs__` |
| `gr.File`, `gr.UploadButton` | `ui.File(name, accept=...)` (or the `text-file` widget for a text box filled from a file) |
| `gr.State(value)` | `ui.State(name, value)` (JSON only: no objects, no callables) |
| `gr.JSON`, `gr.Code`, `gr.HighlightedText` | `ui.Json` / `ui.Output(kind="json")`, `ui.Text`, a widget |
| `gr.HTML` | `ui.Html` (static) or `ui.Output(kind="html")` (from an action) |
| `gr.Markdown` | `ui.Markdown` |
| `gr.Button`, `ToolButton` | `ui.Button(label, action, variant)` (`variant="tool"` for icon-like buttons) |
| `gr.Accordion` | `ui.Section(title, open=...)` |
| `InputAccordion` | `ui.Section(title, enable=ui.Toggle(...))` |
| `gr.Row`, `gr.Column`, `gr.Group`, `FormRow`, `FormGroup` | `ui.Row`, `ui.Column`, `ui.Group` |
| `gr.Tabs` / `gr.Tab` / `gr.TabItem` | `ui.Tabs(ui.Tab(...))`; bind it to a hidden `Choice` when the selected tab is an argument |
| `gradio_rangeslider.RangeSlider` | `ui.RangeSlider` |
| `gr.Plot` | a widget (`sigma-curve` replaced Detail Daemon's matplotlib plots) |
| `gr.Progress` | a page task's `task.progress()` |
| seed row (`ui_common.create_seed_inputs`) | `ui.Seed` |

### Event mapping and common patterns

| A1111 pattern | Remi |
|---|---|
| `x.change(lambda v: gr.update(visible=v == "a"), x, y)` | `visible_if={"param": "x", "eq": "a"}` on `y` (no server call) |
| `gr.update(interactive=False)` | `enabled_if`, or an action patch `{"enabled": False}` |
| `x.change(fn, inputs, outputs)` with server logic | `on_change="fn"` plus `@ui.action("fn")` returning a patch |
| `x.input(...)` | `on_input="fn"` |
| `button.click(fn, inputs, outputs)` | `ui.Button("Label", "fn")` plus `@ui.action("fn")` |
| refresh buttons (`create_refresh_button`, 🔄) | callable `choices` (re-evaluated per request), or a button whose action returns `{"param": {"choices": [...]}}` |
| `.then(...)` chains | do it in one action, or return several entries in the patch |
| `gr.Info` / `gr.Warning` | `ui.toast(message, "info" \| "warning")` |
| outputs written into another script's or the core form's controls | patch keys `"core.<param>"` / `"<script_id>.<param>"` |
| `self.infotext_fields = [(control, "Key")]`, `paste_field_names` | `infotext="Key"` on the input; `paste=` for derived values |
| `on_before_component` / `on_after_component` / `elem_id` hooks to reach a core control (`txt2img_width`, `img2img_steps`, …) | address it by reference: conditions `"core.width"`, patches `"core.width"`, reads `ctx.core["width"]`, locks `Toggle(locks="sampler.steps")`; the sampler fields live in the `sampler` script, the seed fields in `seed` |
| `script_callbacks.on_ui_tabs` | `pages.register(Page(...))` ([§9](#9-pages-the-former-on_ui_tabs)) |
| `on_ui_settings` with `gr.Slider` etc. | the same callback with `ui.Slider` etc. ([§10](#10-settings)) |
| `_js="..."` / `js=` client code (read the gallery, the canvas size, …) | an action reading server state (`ctx.core`, the upload index), or a widget |
| custom routes (`on_app_started(lambda _, app: app.add_api_route(...))`) | unchanged; the routes get the security guard ([§15](#15-what-is-not-supported-and-what-to-do-instead)) |
| reading `shared.opts` in `ui()` to choose the layout | read it in `params()` (built per schema request); mark the setting `.needs_reload_ui()` when the layout depends on it |

Checklist for a port:

1. Snapshot the old `ui()` (see [§13](#13-testing)) before changing it.
2. Write `params()` with the same inputs in the same order (`param_order`
   when the visual order must differ). Inputs the old UI kept outside the
   argument list go **after** the legacy arguments (always-on scripts), so
   positional API clients keep working.
3. Move infotext keys and paste functions onto the inputs; delete
   `infotext_fields` / `paste_field_names`.
4. Turn `.change` visibility into `visible_if` / `enabled_if`, and every
   server callback into an action.
5. Move `on_ui_tabs` to a page; delete `javascript/` and `style.css`.
6. Add `localizations/zh_CN.json` (or your locale) for every string.
7. Add the parity, paste, localization and no-Gradio tests.

## 15. What is not supported, and what to do instead

| not supported | why | instead |
|---|---|---|
| Gradio (`import gradio`, `ui()`, `gr.*` components, `gr.Blocks` tabs) | Gradio is not installed; the UI is a separate React app | `params()`, actions, pages |
| `javascript/*.js` and `style.css` of extensions, `on_before_component` / `on_after_component`, DOM ids (`elem_id`, `gradioApp().querySelector`) | there is no server-rendered DOM to patch, and the SPA's DOM is not an API (it changes between releases and is virtualized) | conditions and patches for behaviour, actions for logic, widgets for new controls, pages for whole views |
| arbitrary HTML/JS in the page (`gr.HTML` with `<script>`, inline handlers) | the SPA sanitizes all HTML (XSS protection) | `ui.Html` for static markup, a widget for interactive parts |
| `/file=` URLs for files on disk | files are served by opaque id only | store the image (`remi.pages.file_store(ctx).add_upload_bytes(data, name)`) and return its id in `__outputs__`; an `image` output or `Gallery` also accepts `data:` URLs |
| `gr.State` holding Python objects | values are JSON (they travel through the API, drafts and the queue database) | keep objects on the server (module state keyed by user / id) and pass ids |
| `user.js` | no script injection | `user.css` in the data folder is still applied |

**Extension HTTP routes** added in `on_app_started` keep working. They are
wrapped by `modules/extension_route_guard.py`: cross-site browser writes and
DNS-rebinding hosts are always refused. When the WebUI has logins (`--auth`)
and is reachable beyond the machine (`--listen`, a non-loopback
`--server-name`, `--share`, `--ngrok`), the routes also need the WebUI login
or `--api-auth` HTTP Basic credentials; under `--nowebui` they need
`--api-auth` credentials whenever it is set. A route can read the logged-in
user through `app.auth_dependency`.

**Third-party extensions that still use Gradio** fail to load cleanly: a
script file that imports Gradio is skipped, and a script with only `ui()` is
disabled; both are listed in the bootstrap `not_ported` list with the reason,
and the rest of the WebUI keeps working.
