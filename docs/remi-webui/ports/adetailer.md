# Port: ADetailer-Neo

Status: ported 2026-10-15 (wave 4). `ui()` became `params()`
(`lib_adetailer/ui.py`); processing is unchanged. Snapshot:
`extensions-builtin/ADetailer-Neo/tests/fixtures/legacy_ui_<tab>.json`; parity,
schema, processing, paste, Refresh, settings, X/Y/Z, routes, Gradio-free and
zh-CN tests: `extensions-builtin/ADetailer-Neo/tests/test_adetailer_params.py`.
The tests use fake detector names, so they pass with an empty `models/adetailer`.

## Arguments

- Positional (`param_order`): `ad_enable`, `ad_skip_img2img`, every tab's
  `ALL_ARGS` fields (`ad_model`, …, `ad_model_2nd`, …), then one UI-only
  `ad_model_is_world<suffix>` flag per tab ("UI-only params come last",
  `ports/scripts-postprocess-detail-daemon.md`). `pack_args` rebuilds the
  legacy `[enable, skip, dict per tab]` and drops the flags; argument lists
  that already hold dicts (`/sdapi` `alwayson_scripts`) pass unchanged.
- `/sdapi/v1/script-info` lists `2 + 40 × tabs` scalar args instead of
  `[enable, skip, dict…]`.

## UI notes

- "Detector Classes" shows while the tab's flag is set; `ad_model`'s
  `on_change` action `ad_model_changed` (and Refresh) set it from
  `"-world" in model`, so a YOLO-World detector found by Refresh works without
  rebuilding the UI.
- Pasting an infotext with any `ADetailer model*` key enables ADetailer and
  exactly the tabs it lists; the other tabs are switched off.
- "Refresh models" rescans without downloading and swaps the detector mapping
  in one step (a running job never sees a partial list).

## Detector downloads

- Upstream downloads the 10 stock detectors (huggingface.co, github.com,
  storage.googleapis.com) while the script loads, so a fresh data folder
  held startup on the network. Here loading only scans
  (`lib_adetailer/detection/common.py` `get_models`, `download=False`): the
  stock detectors not in `models/adetailer` yet are listed anyway, and
  `download_model` fetches one the first time a unit runs with it
  (`AfterDetailerScript.get_ad_model`). Refresh rescans without downloading.
- `--ad-no-huggingface` (`preload.py`) turns these downloads off: only the
  files already in the model folders are listed, and a unit naming a missing
  stock detector fails with a message naming the flag.
- Tests: `test_scanning_never_downloads_and_lists_the_stock_models`,
  `test_a_stock_detector_is_downloaded_when_a_unit_first_uses_it`,
  `test_without_downloads_only_the_files_are_listed`,
  `test_loading_the_script_makes_no_download`
  (`extensions-builtin/ADetailer-Neo/tests/test_adetailer_params.py`) and the
  offline startup in `tests/test_remi_server_runtime.py` (no flag, any
  non-local connection fails it).

## Open core issue

- Reusing a queued `/sdapi` job whose ADetailer args are dicts puts the dicts
  into the form fields (`args_to_values` keeps values it cannot map) and the
  form then fails validation. Needs a core hook letting a script map its raw
  processing args back to form values (ADetailer would unpack the dicts into
  `ad_*<suffix>`).
