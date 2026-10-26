# `remi.ui` schema format (v1)

Status: binding contract between the Python parameter model (`remi/ui`) and
the SPA renderer (`webui/src/ui`). Produced by `remi.ui.to_schema(node, ctx)`
and the tab/settings exporters described below. Keep this file in sync with
`remi/ui/schema.py` (typed shapes: `remi/ui/types.py`);
`tests/test_remi_ui_schema.py` snapshots the output
(`tests/fixtures/remi_ui_schema_snapshot.json`, one node of every kind).

All strings meant for people (`label`, `info`, `hint`, `title`, `text`,
choice `label`s and `choice_hints`, `notes`) are **English source strings**. The SPA translates them
through the locale catalog (key = the English text) and must fall back to the
English text when a key is missing. `text` of `markdown` nodes is translated
as one key; so is `html` text (key = the exact markup, translations keep it).

Every object below is plain JSON (no `NaN`, no `Infinity`). Keys that are
absent mean "default"; the renderer must not depend on optional keys being
present.

## 1. Nodes

Every node has `"type"`. A node is either an **input** (it has a `"name"` and
holds a value) or a **layout/display** node.

### 1.1 Common keys of every node

| key | type | meaning |
|---|---|---|
| `type` | string | node kind, see below |
| `visible_if` | Condition | node (and its subtree) is rendered only when true |
| `enabled_if` | Condition | inputs in the subtree are interactive only when true |

### 1.2 Common keys of every input

| key | type | meaning |
|---|---|---|
| `name` | string | param id, unique within its owner (`[A-Za-z_][A-Za-z0-9_]*`) |
| `label` | string | always present; `""` means "render without a label" |
| `info` | string? | help text under the control |
| `hint` | string? | tooltip on the label (A1111's `tooltip=` and the core hint map of `hints.js`); also the control's accessible description |
| `default` | JSON | the value a fresh form starts with (already resolved; never a function) |
| `on_change` | string? | action name to call (`POST /actions/{owner}/{action}`) after the user changes the value; also after a paste / "Send to" / bulk reset that changes it |
| `on_input` | string? | action name to call only after the user edits this control (Gradio's `.input`): never after pastes, presets, resets of several params or action patches. For derived updates a pasted infotext already records (e.g. a template's timing) |
| `width` | `"auto"` \| `"fill"` \| 1–12? | sizing hint inside a `row` (12-column grid); absent = `"auto"` |
| `hidden` | `true`? | keep the value (it is submitted) but do not render the control |
| `persist` | `false`? | never kept in drafts: every page load starts from `default` (e.g. a copy of a setting); a schema reload in the same session keeps the value like any other |
| `submit` | `false`? | held by the form for actions only: never sent with a job (so neither reused jobs nor drafts bring it back); processing gets `default` (also what a tool input, `arg=False` on the server, exports) |
| `infotext` | string? | infotext key this param is written to / read from (informational; parsing is server-side) |
| `preset` | string? | Anima preset field this param belongs to (informational) |
| `exclusive` | `{"group": str, "member": str}`? | `toggle` only: one-of-N group membership; server action `__conflicts__` resolves locks |
| `inactive_if` | Condition? | while true the control is shown dimmed (and announced as not in use) but stays editable: its value has no effect with the current values (e.g. Hires. fix "Upscale by" while "Resize width to" is set). Unlike `enabled_if` it never blocks input |

### 1.3 Input kinds

Wire value = what the client keeps in its form state and submits.

| `type` | extra keys | wire value |
|---|---|---|
| `toggle` | `locks`? (`"owner.param"`, or a param of the same owner) | `boolean`. While `true`, the locked param ignores programmatic writes (pasted parameters, "Send to", reused jobs, presets, action patch values); user edits and resets still apply |
| `slider` | `minimum`, `maximum`, `step` (numbers), `integer` (bool) | number (integer when `integer`) |
| `number` | `minimum`?, `maximum`?, `step`?, `precision`? (int), `integer` (bool) | number |
| `text` | `lines` (int ≥ 1), `max_lines`?, `placeholder`?, `prompt` (bool) | string. `prompt: true` = prompt editor (weights, LoRA, tag completion) |
| `choice` | `choices`, `style` (`"dropdown"`\|`"radio"`), `allow_custom` (bool), `choice_hints`? (`{choice label: tooltip}`) | the chosen choice `value` (string or number), or any string when `allow_custom` |
| `multichoice` | `choices`, `style` (`"dropdown"`\|`"checkboxes"`), `allow_custom` (bool), `choice_hints`? | array of choice values |
| `seed` | – | integer: `-1` (random) or `0` … `2^63-1`. Renderer adds dice (set -1) and recycle (last seed) buttons. Values above `2^53` lose precision in JavaScript numbers |
| `color` | – | string `#rrggbb` (or `#rgb`, `#rrggbbaa`) |
| `image` | `kind` (`"image"`\|`"mask"`\|`"sketch"`), `source`? (name of the `image` param this layer is painted over), `mode`? (`"RGB"`\|`"RGBA"`\|`"L"`) | upload id string (`POST /api/v2/uploads`) or `null` |
| `image_list` | `widget`?, `props`? | array of upload ids. With `widget`, the widget registered under that name edits the list (it receives `props`); the value keeps upload semantics (never restored from drafts) |
| `file` | `accept`? (string, e.g. `".safetensors,.pt"`) | upload id string or `null` |
| `json` | – | any JSON |
| `custom` | `widget` (string), `props` (object) | any JSON, owned by the widget registered as `widget` in `webui/src/widgets` |
| `state` | – | any JSON; never rendered, always submitted |

`integer` of a slider is `true` when its bounds, step and default were
declared as integers (`Slider(0.0, 2.0, 1.0, 1.0)` is a float slider).

A two-ended range (Gradio's RangeSlider) is `ui.RangeSlider(name, minimum,
maximum, default, step)`: a `custom` node with `widget: "range-slider"` and
`props: {minimum, maximum, step}`, wire value and processing argument
`[low, high]`. The server clamps both ends to the bounds and orders them;
integers when the bounds, step and default are (as `slider`). Infotext
pastes `"a, b"`, `"(a, b)"` or `"[a, b]"`.

`choices` is an array of `{"value": string|number, "label": string}`. The
label is translated, the value is submitted. Choice lists that depend on the
server state (samplers, checkpoints, …) are resolved when the schema is
produced; actions can replace them with a `choices` patch. Whether the script
receives the value or its index (`type="index"`) is server-side only: the
wire value is always the choice value.

**Image layers.** A `kind: "sketch"` or `kind: "mask"` input with a `source`
is a layer painted over that image on one canvas (same size as the source);
a `kind: "mask"` input **without** `source` is an uploaded mask image (render
an upload field with a mask preview). Every image is uploaded as a PNG.

- A **sketch** layer is RGBA colour: transparent where nothing was painted.
- A **mask** is alpha-coded: alpha 255 = masked (inpainted), 0 = kept; its
  colour is ignored. The canvas uploads the painted strokes on a transparent
  layer. A mask image without transparency (no alpha channel, or alpha 255
  everywhere, e.g. an uploaded black-and-white mask) is read by luminance
  instead: white = masked.

The img2img form has two mask params: `mask` (painted over `init_img`, mode
`"Inpaint"`) and `mask_upload` (an uploaded mask, mode `"Inpaint upload"`);
"Inpaint sketch" takes its mask from the `sketch` layer's alpha.

### 1.4 Layout and display nodes

| `type` | keys | meaning |
|---|---|---|
| `section` | `title`, `open` (bool), `enable`? (a `toggle` input node), `children` | collapsible panel. With `enable` it is the former *InputAccordion*: the toggle sits in the header and is an input of the owner (it is the first param of the subtree) |
| `group` | `title`?, `open` (bool), `children` | visual group; collapsible when it has a `title` |
| `row` | `children` | horizontal flow (wraps on narrow screens) |
| `column` | `width`?, `children` | vertical stack |
| `tabs` | `bind`?, `children` (only `tab` nodes) | tab strip. With `bind` the selected tab is the param named by `bind` (a `choice` of the same owner): selecting a tab sets the param to the tab's `value`, and the param's value selects the tab |
| `tab` | `title`, `value`?, `children` | one tab; `value` is the bound choice value |
| `markdown` | `text` | CommonMark, rendered without raw HTML |
| `html` | `text` | trusted server HTML (sanitize anyway) |
| `separator` | – | horizontal rule |
| `button` | `label`, `action`, `variant` (`"primary"`\|`"secondary"`\|`"ghost"`\|`"tool"`\|`"danger"`: destructive, e.g. delete) | calls the owner's action with the owner's current values |
| `gallery` | `name`, `label`? | output-only image list (filled by `__outputs__` of an action or by job results) |
| `output` | `name`, `kind` (`"text"`\|`"html"`\|`"image"`\|`"json"`), `label`? | output-only value |
| `slot` | `section` | placeholder in a **core** form: render here the scripts whose `section` equals this value (see §4) |

A node's `children` is always an array (possibly empty). Input names are
unique per owner across the whole tree, including inside `tabs`.

## 2. Conditions

```jsonc
true | false                                     // constant
{"param": "mode"}                                // truthy
{"param": "mode", "eq": "Inpaint"}               // ==
{"param": "mode", "ne": "Inpaint"}               // !=
{"param": "mode", "in": ["Inpaint", "Inpaint sketch"]}
{"param": "cfg_scale", "gt": 1}                  // also "gte", "lt", "lte"
{"all": [cond, ...]}  {"any": [cond, ...]}  {"not": cond}
```

`param` addresses a param of the same owner by name, another owner's param as
`"<script_id>.<name>"`, the generation form as `"core.<name>"`. A reference to a
param that does not exist evaluates as `undefined` (falsy; `eq`/`in` false,
`ne` true). Truthiness: `false`, `null`, `0`, `""`, `[]` are falsy. One
operator per leaf.

`eq`, `ne` and `in` compare like Python's `==` on JSON values (what
`remi/ui/conditions.py` evaluates and `webui/src/ui/conditions.ts` mirrors):
`true == 1` and `false == 0`, a string never equals a number (`"1" != 1`),
arrays and objects compare by content. `gt`, `gte`, `lt` and `lte` take a
finite number (the server rejects other operands when the layout is
declared); a value that is not a number (or boolean) never satisfies them.

## 3. Values, validation and errors

The client submits named values per owner: `{"core": {...}, "scripts":
{"<script_id>": {"<name>": value}}}`. Missing names take their defaults.
The server coerces each value by kind and rejects bad ones:

- `toggle`: booleans; also the numbers `0`/`1` and the strings `"true"`/`"false"`
  (any case, surrounding spaces ignored). Nothing else (`"yes"`, `"on"`,
  `""`, … are rejected).
- `slider`/`number`: finite numbers or numeric strings; `NaN`/`Infinity` are
  rejected. Sliders are **clamped** to `[minimum, maximum]`; a `number`
  outside its bounds is **rejected**. Integer kinds are rounded;
  `number.precision` rounds.
- `seed`: whole numbers (integers, integral floats like `5.0`, numeric
  strings) from `-1` to `2^63-1`; `1.7` or `-2` are rejected.
- `choice`: must be one of the choice values unless `allow_custom` (numeric
  strings are accepted for numeric choices); `multichoice` likewise per item.
- `text`: strings (numbers are stringified, `null` becomes `""`).
- `color`: `#rgb`, `#rrggbb` or `#rrggbbaa`.
- `image`/`file`: upload id strings or `null`; `image_list`: arrays of ids.
- `json`/`custom`/`state`: any JSON without `NaN`/`Infinity` anywhere in it.

`null` stands for "empty" where the kind has an empty value: `toggle` →
`false`, `text` → `""`, `multichoice`/`image_list` → `[]`; `image`, `file`,
`json`, `custom` and `state` keep `null`; a `choice` accepts it only when it
has no choices; numbers, seeds and colors reject it. A *missing* name always
means "the default".

Rules across params of the core form are checked after coercion: img2img
needs `init_img` in every mode but "Batch", `mask` in "Inpaint", `sketch` in
"Inpaint sketch", `mask_upload` in "Inpaint upload" and `batch_upload` for a
"Batch" from uploads; Extras needs `image` ("Single Image"), `image_batch`
("Batch Process"), `input_dir` ("Batch from Directory") or `video_input`
("Single Video"). The tab schema lists these rules as `core.required`
(§4): `[{"param", "message", "when"}]`, where `when` is a condition (§2) on
the core values under which `param` must not be empty (`null`, a blank
string or an empty list). Clients check them before submitting, mark the
fields with `message` (a catalog key, like every server message) and send
nothing; the server applies the same list.

A rejected request answers with

```json
{"errors": [{"owner": "core", "param": "width", "message": "expected a number, got 'abc'"}]}
```

listing every rejected value (also the selected script's when core values are
rejected). `owner` is `"core"`, a script id, `"settings"`, or `null` when the
body itself has the wrong shape (e.g. `"scripts"` is not an object); `param`
is `null` for errors about a whole owner (an unknown script id, an owner's
values that are not an object).

A script's values may also be a JSON array: its params in argument order
(`args`); more values than params is an error.

## 4. Tab schema (`GET /api/v2/schema/{tab}`)

```jsonc
{
  "version": 1,
  "tab": "txt2img",                       // "txt2img" | "img2img" | "extras"
  "core": {"owner": "core", "layout": Node, "defaults": {name: value},
           "required": [{"param": "init_img", "message": "an image is required", "when": Condition}, ...]},  // §3
  "scripts": [Script, ...],                // in render order
  "conflict_params": ["sampler.sampler_name", "core.enable_hr", ...]  // txt2img/img2img only
}
Script = {
  "id": "seed",                            // owner id: unique per tab, stable (slug of the title); never "core" or "settings"
  "title": "Seed",
  "kind": "alwayson" | "selectable",
  "section": "seed" | null,                // slot to render in; null/unknown -> slot "scripts"
  "order": 0,                              // sort key inside a slot (ascending)
  "visible_if": Condition?,                // selectable scripts: {"param": "core.script", "eq": title}
  "layout": Node | null,                   // null: the script has no params
  "defaults": {name: value},
  "args": [name, ...]                      // positional order of the processing arguments
}
```

The core form contains a `choice` param `script` (values `"None"` and every
selectable script's title) inside the `slot` `"scripts"` area; its
selectable scripts are rendered there, each shown only when selected.
`extras` scripts are the postprocessing operations (owner id = slug of the
operation name; their params are named by their `params()`).

`conflict_params` lists the `"owner.param"` references (core params as
`"core.<name>"`) bound to an Anima conflict role or an exclusive group: after
the user changes one of them the client calls the core `__conflicts__`
action (§5) with `"selected"` set to that reference and applies the returned
patch.

A script whose `params()` fails, or that only has A1111's Gradio `ui()`, is
left out of the tab (and its processing hooks do not run); the server logs
why and bootstrap `not_ported` lists it.

Core forms are built per request and follow the options: the categories
(`prompt`, `image`, `inpaint`, `sampler`, `accordions`, `checkboxes`,
`dimensions`, `cfg`, `denoising`, `seed`, `batch`, `override_settings`,
extension sections, `scripts`) come in the user's order, each with a `slot`.
The prompt category's group also opens with a `prompt_before` slot when a
script asks for it (Python `Script.slot = "prompt_before"`, e.g. Prompt
Anchor): its entry's `section` is then `"prompt_before"` and the client
renders it directly above the main prompt; the script's arguments stay in
the prompt category.
Choice params with `type="index"` (img2img `mode`, `resize_mode`, …) still
take their choice *value* on the wire.

The txt2img/img2img schema follows the current **UI preset** (setting
`forge_preset`): the core defaults (`width`, `height`, `cfg_scale`,
`distilled_cfg_scale`, Hires steps/CFG, `batch_size`), labels ("Shift",
"Frames"), slider bounds (`batch_size`) and visibility (`distilled_cfg_scale`
gets `"visible_if": false` when the preset has no distilled CFG/shift), and
the Sampler script's defaults (`steps`, `sampler_name`, `scheduler`).
Validation uses the same bounds and defaults. After `__forge_preset__` (§5)
the client applies the returned patch or re-fetches the schema.

## 5. Actions

`POST /api/v2/actions/{owner}/{action}` with

```jsonc
{
  "tab": "txt2img",
  "values": {...},                 // the owner's current values
  "core": {...},                   // the generation form's current values
  "scripts": {"<script_id>": {...}} // the other scripts' current values (needed by core.__conflicts__)
  // plus the action's own arguments, see below
}
```

returns a patch:

```jsonc
{
  "<param>": {"value"?, "choices"?, "visible"?, "enabled"?, "label"?, "info"?,
              "hint"?, "minimum"?, "maximum"?, "step"?},
  "<other_owner>.<param>": {...},          // patch another owner (e.g. "core.width")
  "__toast__": {"level": "info" | "success" | "warning" | "error", "message": str | {"text": str, "params"?: {...}}},  // a message (ui.message) is translated, then its {{name}} placeholders filled
  "__outputs__": {"<output name>": value},
  "__settings__": {"<setting>": value},    // settings the action already changed on the server
  "__refresh__": {"keys": ["<list>", ...]} // server lists the action rescanned ("models", "upscalers", ...)
}
```

Texts shown to the user are translated by the client: a toast `message` and
a string value of a `text` output when a catalog has them verbatim, and a
message object `{"text": str, "params"?: object}` (`remi.ui.message`) of a
`text` output, whose `text` is the English catalog key with `{{name}}`
placeholders filled from `params` after translation.

`visible`/`enabled` set in a patch override the static node state until the
next patch or schema reload; they are combined with `visible_if`/`enabled_if`
by logical AND. `__settings__` reports settings the server has stored: the
client updates its cached option values and must not write them back.
`__refresh__` names server-side lists the action rescanned; the server
announces them as an `options.changed` event (`keys`), so every client
refetches the schemas that offer them; clients ignore the key itself. Bad
action input answers with the §3 error body.

Built-in server actions with owner `core` (keys of their patches are relative
to the core owner: core params bare, others `"owner.param"`):

- `__conflicts__` — Anima conflict locks; body adds `"selected": "<owner>.<param>"`
  (core params as `"core.<name>"`; absent after a bulk change). Every
  conflicting control gets one entry: its resolved `value`, `enabled` (false
  while a rule locks it) and, where a rule narrows them, `choices`. Exclusive
  groups and pairwise rules are resolved together (a rule that switches a group
  member off frees the group). Values missing from `values`/`core`/`scripts`
  read as their defaults.
- `__preset__` — Anima preset; body adds `"label": str`.
- `__forge_preset__` — select a UI preset; body adds `"preset": str` (one of
  `sd`, `xl`, `flux`, …). Runs behind generation: the server switches the
  checkpoint, VAE/text encoders and low-bits dtype the preset remembers, saves
  the options, and returns the form patch (§4 preset defaults, labels, bounds,
  visibility; `"sampler.<name>"` for the Sampler script) plus `__settings__`
  (`forge_preset`, `sd_model_checkpoint`, `forge_additional_modules`,
  `forge_unet_storage_dtype`).
- `swap_dimensions`, `hr_resolution` — the size swap button and the Hires.
  fix "Upscaled resolution" output.

## 6. Settings schema (`GET /api/v2/options`)

```jsonc
{
  "version": 1,
  "categories": [
    {"id": "sd", "label": "Stable Diffusion",
     "sections": [{"id": "sd", "label": "Stable Diffusion", "items": [Node, ...]}]}
  ],
  "quicksettings": ["sd_model_checkpoint", ...],
  "values": {key: value},
  "defaults": {key: value}     // each stored setting's built-in default
}
```

Items are input nodes whose `name` is the setting key, plus `markdown`,
`html`, `separator` and `row` nodes (a `row` groups the inputs between two
former `OptionRow` markers). A help block between settings is a `markdown`
node (`OptionMarkdown`: text, emphasis, code and links, rendered as React
elements, never as HTML); that is what Remi's own settings
(`modules/shared_options.py`, `modules/shared_items.py`, the built-in
extensions) use. An `html` node comes only from a third-party extension's
`OptionHTML` (A1111 API) and is sanitised by the SPA;
`tests/test_remi_ui_schema.py` keeps the core settings free of it. Setting inputs carry these extra keys:

| key | meaning |
|---|---|
| `notes` | array of short plain-text help strings (each translated separately; never markup) |
| `links` | array of `{"label", "url"}` |
| `choice_hints` | (choice and multichoice settings) `{choice label: description}` from `OptionInfo.choice_info()`; the settings page lists them under the control and shows them as option tooltips |
| `default_order` | (ordering settings, e.g. `prioritized_callbacks_*`) the order used while the setting is empty, as plain names; the settings page shows it as a numbered list |
| `html` | trusted HTML help from the A1111 API (`OptionInfo.html()`, `comment_before` / `comment_after`) that third-party extensions use; Remi's own settings never set it |
| `needs_restart` / `needs_reload_ui` | changing it takes effect after a restart / page reload |
| `refresh` | `true`: the setting has a refresh action (`POST /actions/settings/refresh:<key>`) that returns a `choices` patch |
| `onchange` | `true`: the server runs a side effect when it changes |
| `restricted` | `true`: not changeable through the A1111 API (`/sdapi/v1/options`); `PATCH /api/v2/options` (the WebUI's own API) may change it |
| `locked` | why `PATCH /options` refuses changes (`--freeze-settings`, a frozen section or key, `--hide-ui-dir-config`, not stored); such a setting is also `"enabled_if": false` |

A setting that is not editable has `"enabled_if": false`; one that must not be
shown has `"hidden": true`. `default` is the setting's current value (it may
lie outside stale `choices`); the top-level `defaults` map holds the built-in
default of every stored setting (the settings page resets to it and filters
"changed" settings by it). Sections without an id (internal options) are
not exported; their values are in `values`.

## 7. Infotext parse (`POST /api/v2/infotext/parse`)

Request `{"text": str, "tab"?: "txt2img" | "img2img" | "extras" | null}`;
response

```jsonc
{
  "tab": "txt2img",                         // the tab given, or the detected one
  "core": {name: value},                    // core params the infotext sets (override_settings included)
  "scripts": {"<script_id>": {name: value}},
  "params": {key: value},                   // the parsed infotext (what PNG Info shows)
  "problems": [{"owner", "param", "message"}]  // values that did not fit their param (skipped)
}
```

Only params the infotext sets are present; the client merges them over the
current form. An empty `text` reads the last generation's infotext
(`params.txt`; nothing when the server runs with `--hide-ui-dir-config` or
`--no-prompt-history`). Without `tab` the server detects it: `img2img` when
the infotext has a key only the img2img form reads (`Mask blur`,
`Mask mode`, `Masked content`, `Inpaint area`, `Masked area padding`,
`Image CFG scale`), or a `Denoising strength` without Hires. fix keys
(`Hires upscale`, `Hires upscaler`, a non-zero `Hires resize-1`); `txt2img`
otherwise. Settings the infotext records but no param reads become the core
`override_settings` value (`["Key: value", ...]`).

## 8. Page schema (`GET /api/v2/pages/{id}`)

An extension page (former `on_ui_tabs`) is one owner, rendered like a script:

```jsonc
{
  "version": 1,
  "id": "toolbox",
  "title": "Toolbox",
  "icon": "wrench" | null,                 // lucide icon name
  "owner": "page:toolbox",                 // action owner (POST /actions/page:toolbox/{action})
  "widget": "agent-scheduler" | null,      // a dedicated frontend view; layout is then null
  "layout": Node | null,
  "defaults": {name: value},
  "on_load": "status",                     // optional: action run each time the page is shown
  "send_to": [{"kind": "image", "param": "image", "label": "JoyCaption"}]  // optional: what other views can send here
}
```

Page actions receive `{"values": {...}}` (no core form; `tab` is the form
id `page:<id>`) and return §5 patches; `__outputs__` fill the page's
`gallery`/`output` nodes. A `send_to` entry of kind `image` puts an image
(upload or output file id) into `param` (an `image` input), then runs its
`on_change`.

**Tasks.** An action that starts long work answers at once with
`{"__outputs__": {"task:<name>": Status}}`, `Status = {id, name, state
("queued"|"running"|"completed"|"failed"|"cancelled"), progress (0…1 or
null), text (string or message, §5), error, done, cancelling}`. The client
polls the page's `__task__` action with `{"task": name, "id": status.id}`:
running answers may carry partial outputs; the answer once the task has
ended carries the task's result patch (values, outputs, a toast; a failure
adds an error toast). `{"resume": true}` (a view mounting again) asks for
the status; it carries the result too while no view has received it (the
task ended while none followed it, e.g. during a reload). The
`__cancel__` action (`{"task": name}`) asks the task to stop.
