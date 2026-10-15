# Ports: Forge panels (compile, MultiDiffusion, Never OOM, PiD, Radial, Sparse, Spectrum, NegPiP, LoRA schedule, comments)

Status: ported 2026-10-15 (wave 4). Each `ui()` became `params()` under the
architecture.md "Built-in extension porting rule"; processing is unchanged
and no Gradio import remains. Snapshots:
`extensions-builtin/<ext>/tests/fixtures/legacy_ui_<tab>.json`, compared by
`tests/remi_port_harness.py` (`assert_matches_snapshot`; wiring the port adds
— exclusive locks, `visible_if`/`enabled_if` — must be listed as changes).

## Argument differences

- **Spectrum** gains a trailing 16th argument, `preset_name` (the Presets
  box's name), following "UI-only params come last"
  (`ports/scripts-postprocess-detail-daemon.md`). Processing ignores it;
  `script_arguments` pads 15-value lists (older Agent Scheduler tasks,
  drafts) and refuses longer ones. API clients sending positional
  `alwayson_scripts` args may omit it. It is stored in drafts and queued jobs.
  Drop it once core can mark an input as form-only.
- **Sparse** `percent` is one `[low, high]` argument (`ui.RangeSlider`), as
  Gradio's RangeSlider passed it.
- **PiD** switch carries `exclusive=("pid_hires", "pid")` on txt2img only
  (the former `wire_exclusive_components(("pid", "hires"))`).
- **NegPiP**, **LoRA schedule**: no panel (no `ui()`); the runner gives
  them an empty layout.

## UI notes

- MultiDiffusion's 📐 button is the `detect_size` action ("Use source image
  size", below the tile sliders): it reads the img2img `init_img` upload's
  size from the server's upload index and clamps it to the slider bounds
  (not snapped to the 64 px step, as before).
- Spectrum's Apply Preset coerces the stored values through the layout (a
  hand-edited `presets.json` entry that does not validate gives an error
  toast); Save / Delete hold a lock around the update and the file write.
  Delete uses the `danger` button variant.

## Localization

Extension zh-CN catalogs are merged into one flat catalog with core's and the
language pack's (`sd-webui-language-diffusion`), so an entry for the same
English text must use the same translation (checked by
`tests/test_remi_port_forge_panels.py`); texts core already translates are
left to core. Toast messages are not translated by the SPA yet: their
catalog entries ("Enter a preset name", "Add a source image to read its
size") are kept for when it is, and f-string toasts (`Preset "{name}" saved`)
will need parameterized keys.
