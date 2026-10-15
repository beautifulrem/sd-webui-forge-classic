# Ports: `scripts/`, Post Processing Suite, Detail Daemon, mobile, bracket checker

Status: ported 2026-10-15 (wave 3). Each port follows the architecture.md
"Built-in extension porting rule": `ui()` became `params()`, processing code
is unchanged, no Gradio import remains. Before porting, each `ui()` was
snapshotted (per input, in argument order: kind, label, info text — Gradio
`info=` or Forge `tooltip=` —, default, bounds, choices, infotext key, paste
function, preset field):

| port | snapshot | parity test |
|---|---|---|
| `scripts/` | `tests/fixtures/legacy_ui/<script>_<tab>.json` | `tests/test_scripts_xyz_grid.py`, `tests/test_scripts_ported.py` |
| Post Processing Suite | `extensions-builtin/sd-postprocess-suite/tests/fixtures/legacy_ui_<tab>.json` | `…/sd-postprocess-suite/tests/test_postprocess_suite_params.py` |
| Detail Daemon | `extensions-builtin/sd-webui-detail-daemon/tests/fixtures/legacy_ui_<tab>.json` | `…/sd-webui-detail-daemon/tests/test_detail_daemon_params.py` |

`tests/legacy_ui_contract.py` compares a layout with its snapshot; every
difference is listed in the test (`expected=`, `extra=`). Info texts are
compared when the snapshot records them (all of this group's do; older
snapshots without the key are not). It also checks that a
file is Gradio-free and collects the strings the zh-CN coverage tests check.

## UI-only params come last

Every rendered input is a param, and params are processing arguments. Inputs
the old UI kept outside the argument list (a preset picker, a preview) are
declared **after** the legacy arguments (`param_order`), so positional API
clients and the processing code see the same leading arguments; the
processing code ignores the extra ones. This is done for always-on scripts
only: `/sdapi` copies an always-on script's `args` index by index, but
replaces a selectable script's slice with the request's `script_args`, so a
selectable script must keep its argument count (Prompts from File keeps 4).

| script | trailing params | how processing ignores them |
|---|---|---|
| Post Processing Suite | txt2img `send` (`pp-send` widget), `preset` (both tabs) | `P.unflatten` reads the pipeline prefix; `_apply_mode` reads the img2img apply mode only on img2img |
| Detail Daemon | `daemon` (the bound tab), `curves` (`sigma-curve` widget) | `process` slices 12 values per daemon (`(12n + 2) // 12 == n`) |

## `scripts/`

- **X/Y/Z plot.** `x_type`/`y_type`/`z_type` are `type="index"` choices over
  the tab's axes (`current_axis_options` is now a property read from the
  module-level `axis_options` on every use, so axes that ADetailer, Detail
  Daemon and Dynamic Shift append keep working; their API —
  `axis_options`, `AxisOption`, `AxisOptionImg2Img`, `AxisOptionTxt2Img`,
  `apply_field`, `apply_override`, `boolean_choice`, `format_*`, `confirm_*`
  — is unchanged). The values dropdowns are the `xyz-values` widget
  (documented kind change `multichoice` → `custom`, value still a list of
  strings): one `axis_choices` action call returns every list axis' values
  (`{label: values}`; an axis whose `choices()` raises is logged and is
  `null`, the others still load), cached in the form's outputs until the
  schema reloads — a checkpoint refresh (`POST /models/refresh` →
  `options.changed`) reloads it, replacing the old `on_refresh_checkpoint`
  hook. The dropdowns of a form share one in-flight call; after a failure it
  is retried when the axis changes or the dropdown opens. As the old
  `select_axis` re-read `choices()` on every axis change, `axis_x|y|z` and
  `fill_x|y|z` also return the selected axis' fresh list (output
  `axis_choices/<label>`, which wins over the cached one), so lists that
  change without a schema reload (LoRA, upscalers) follow. Visibility of the
  text box, dropdown and "Fill values" button is a `visible_if` on the axis
  and the text mode. Conditions are static JSON, so the list axes are fixed
  when the layout is built (`Script.listed_axes`): axes must be added to
  `axis_options` before that (at import or `on_before_ui`, as ADetailer,
  Detail Daemon and Dynamic Shift do — the old `ui()` fixed its axis list
  at the same point). A list axis added later is still selectable but acts
  as a text axis, in the actions and in `run()`;
  `axis_x|y|z` (on change of an axis), `text_mode` (on change of
  `csv_mode`), `fill_x|y|z` and `swap_xy|yz|xz` are actions with the old
  handlers' semantics. Infotext keys `X/Y/Z Type`, `X/Y/Z Values` and the
  dropdown paste are kept.
- **Prompt matrix, Loopback, SD upscale.** Same arguments; `param_order`
  where the visual order differs. SD upscale's upscaler stays a `type="index"`
  choice (callable choices).
- **Prompts from File or Textbox.** The text box and the
  `gr.File(type="binary")` upload that filled it are one `text-file` widget
  (documented kind change `text` → `custom`, value still a string): it loads
  a file of any type, like the old upload, in the browser (nothing is
  uploaded; read as UTF-8, each line stripped as before) and auto-grows
  (`lines=2`, `max_lines=20`) instead of the `.change` that set `lines`. The argument count stays 4.
- **Extras: Upscale.** The hidden number Gradio's tab events set is the
  hidden choice `upscale_mode` (`0` scale by, `1` scale to) bound to the tab
  strip (`Tabs(bind=)`; documented kind change `number` → `choice`). Actions:
  `swap_size`, `refresh_upscalers`, `upscaler_scale` (with
  `set_scale_by_when_changing_upscaler`); "Upscaler 2" is enabled by
  `enabled_if` on its visibility. Extras operations receive keyword
  arguments, so their param order is free.
- **Extras: DeGrid.** Same params.

## Post Processing Suite

The panel is generated from `lib_postprocess.pipeline.STAGES`: one section per
category and stage, all closed (the SPA mounts a section's controls only when
it opens, so 141 inputs cost nothing until used), params named
`<stage>_<param>`. Documented kind change: the `.cube` LUT is a `File` upload
(it was a JSON param on the legacy path); uploads with the `.cube` extension
are accepted as text (`remi/files.py` `TEXT_EXTENSIONS`). Preset buttons are
actions (`preset_load|save|delete|refresh`; the `preset` combobox takes an
existing name or a new one); a saved preset keeps a copy of its LUT under
`presets/luts/`, and loading re-uploads it; deleting or overwriting the
preset removes that copy (only files inside `presets/luts/`). "Apply PP →
Send to img2img" is the `pp-send` widget: it calls `apply_send` with the file
id of the form's own result selected on the stage (txt2img's, as the old
button read the txt2img gallery; the newest txt2img result when the
selection is another tab's) as `file_id` (the action's own argument), which
runs the pipeline on the server and answers the processed copy's upload id
(`__outputs__.sent`); the widget then sends it to the img2img (or, with
`target: "inpaint"`, inpaint) source image like any "Send to", and reports
success only when the image was delivered. `pp_send.js` is deleted.

## Detail Daemon

`gr.Plot` (matplotlib, 12 figures per tab re-rendered on every slider release)
→ the `sigma-curve` widget, which computes `make_schedule` in the browser
(`webui/src/widgets/sigma-curve/schedule.ts`; Python's round-half-to-even and
numpy's `linspace` reproduced). `schedule.fixture.json` is the extension's own
output for 16 inputs (written by `tests/schedule_cases.py`); both test suites
check against it. The curve's opacity is the old figures'
`0.1 + (1 - fade) * 0.7`. The daemon tabs are bound to the hidden choice `daemon`, so
the widget's thumbnails select a tab. The JS-only slider ↔ number mirrors are
gone: Amount, Start Offset and End Offset are the unbounded number params
they always were. `style.css` is deleted; the settings use `ui.Slider` /
`ui.Toggle`. Infotext paste (current, multi-daemon and the old `DD_*` keys) is
kept, including the old parser's numbering of entries by position.

## Deleted: `mobile`, `prompt-bracket-checker`

Both were JavaScript-only and are core SPA features now:

- **mobile** moved the Generate box above the results on narrow screens. The
  generate page's phone layout puts the stage and history first and keeps the
  run controls stuck to the bottom of the parameters
  (`webui/src/pages/generate/index.tsx`, `ParamsColumn sticky="bottom"`).
- **prompt-bracket-checker** turned the token counter red on unbalanced
  `()`, `[]`, `{}`. The prompt editor underlines unclosed / unmatched brackets
  (escapes respected) and announces a summary in a polite live region
  (`webui/src/features/prompt-editor/tokenize.ts` `bracketProblems`, tests in
  `tokenize.test.ts`).

## Widgets

| widget | value | props |
|---|---|---|
| `xyz-values` | `string[]` | `{axis, action}` |
| `text-file` | `string` | `{accept, lines, max_lines, placeholder}` |
| `sigma-curve` | unused | `{daemons: [{prefix, title, value}], tab, enable}` |
| `pp-send` | unused | `{action, target}` |

Widget chrome strings live with each widget (`widgets/<name>/locales/<locale>.json`,
namespace `widget:<name>`, translator `tw`; `widgets/_lib/i18n.ts`). Schema
strings are in `localizations/zh_CN.json` (`scripts/`) and each extension's
`localizations/zh_CN.json`; tests check that every label, info text, choice,
Markdown text and toast is covered. The SPA's own catalog
(`webui/src/locales/zh-CN.json`) wins over `localizations/zh_CN.json`, so a
string both have must be translated the same way there (`Prompt`, `Clip skip`;
checked by `tests/test_scripts_ported.py`). Generic keys (`start`, `end`, `Width`, `Height`, `Refresh`) are kept:
catalogs are keyed by the English text, and these translations are the ones
the SPA and the language pack use for the same words. The language pack
(`sd-webui-language-diffusion`) is merged after the core catalog and still
overrides some of these terms (`Upscaler`, `Schedule rho`).

The two extension suites import their helpers as `pp_suite_testlib` /
`detail_daemon_testlib` (not `conftest`) and have unique test module names,
so they run in one pytest session with each other and `tests/`.
