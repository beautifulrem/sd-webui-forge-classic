"""Each tab remembers the panel it last generated with (in the extension's
data folder): the panel's defaults start there, a reset forgets it."""

import json
from types import SimpleNamespace

from test_remi_params import layout_of
from test_runtime_dispatch import _run_args, mixer


def _payload(enable=True, artist="wlop"):
    """A remembered panel as a generation writes it."""
    args = mixer._MixerArgs.from_args(_run_args())
    args = mixer._MixerArgs(**{**vars(args), "enable": enable, "component_values": (True, artist, *args.component_values[2:])})
    return mixer._current_settings_payload(
        args.enable, "", mixer.APPLY_BASE, 2, 1, False, False, 4.0, 3.0, 0.7, mixer.OPT_BALANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, True, *args.component_values
    )


def test_each_tab_remembers_its_own_panel(data_dir, monkeypatch):
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", {})
    folder = data_dir / "extension-data" / mixer._DATA_NAME
    folder.mkdir(parents=True)
    # Earlier versions kept one file for both tabs: each tab starts from it
    (folder / "artist_mixer_current_settings.json").write_text(json.dumps(_payload(artist="shared artist")))
    for tab in mixer.SETTINGS_TABS:
        assert mixer._current_settings_data(tab)["base_rows"][0][1] == "shared artist"
        assert layout_of(tab).defaults()["base_rows"][0]["artist"] == "shared artist"

    mixer._save_current_settings_data("txt2img", _payload(artist="txt2img artist"))
    mixer._save_current_settings_data("img2img", _payload(enable=False, artist="img2img artist"))
    txt2img, img2img = (layout_of(tab).defaults() for tab in mixer.SETTINGS_TABS)
    assert txt2img["enable"] and txt2img["base_rows"][0]["artist"] == "txt2img artist" and len(txt2img["base_rows"]) == 2
    assert not img2img["enable"] and img2img["base_rows"][0]["artist"] == "img2img artist"

    # A reset (the panel's action) restores the defaults instead of seeding the old shared panel again
    from remi import ui

    script = mixer.Script()
    script.tabname = "txt2img"
    patch = ui.call_action(script, "reset_defaults", ui.ActionContext(tab="txt2img"), {})
    assert patch["enable"] == {"value": False} and patch["__toast__"]["message"] == mixer._t("defaults_restored")
    assert patch["base_rows"]["value"] == mixer._builtin_defaults()["base_rows"]
    assert mixer._current_settings_data("txt2img") is None
    assert layout_of("txt2img").defaults() == mixer._builtin_defaults()
    assert mixer._current_settings_data("img2img")["base_rows"][0][1] == "img2img artist"


def test_a_generation_remembers_the_panel_of_its_tab(monkeypatch):
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", {})
    script = mixer.Script()
    script.tabname = "img2img"
    p = SimpleNamespace(is_api=False, extra_generation_params={}, distilled_cfg_scale=4.5)
    values = {**layout_of("img2img").defaults(), "enable": True, "base_rows": [{"artist": "(base artist:-1)"}]}
    script.process_before_every_sampling(p, *layout_of("img2img").values_to_args(values))
    remembered = mixer._current_settings_data("img2img")
    # The Shift the panel showed is the run's own (the Gradio panel kept a copy of it)
    assert remembered["base_rows"][0][1] == "(base artist:-1)" and remembered["base_shift"] == 4.5 and remembered["base_count"] == 1
    assert mixer._current_settings_data("txt2img") is None
    assert layout_of("img2img").defaults()["enable"] is True


def test_a_generation_remembers_the_panel_once_from_the_ui(monkeypatch):
    from modules import shared

    saved = []
    monkeypatch.setattr(mixer, "_save_runtime_current_settings", lambda tab, args: saved.append(args.base_row_count))
    monkeypatch.setattr(shared, "state", SimpleNamespace(job_count=1, job_no=0))
    script = mixer.Script()
    script.tabname = "txt2img"

    def generate(p):
        saved.clear()
        script.process_before_every_sampling(p, *_run_args())
        return len(saved)

    run = SimpleNamespace(is_api=False, extra_generation_params={})
    assert generate(run) == 1
    assert generate(run) == 0  # the same run's next batch or Hires pass
    assert generate(SimpleNamespace(is_api=False, is_hr_pass=True, extra_generation_params={})) == 0
    assert generate(SimpleNamespace(is_api=True, extra_generation_params={})) == 0  # API args are not the panel

    # An X/Y/Z grid: one job per cell, each on a copy of p with its own args
    shared.state.job_count = 4
    assert generate(SimpleNamespace(is_api=False, extra_generation_params={})) == 1  # the first cell
    shared.state.job_no = 1
    assert generate(SimpleNamespace(is_api=False, extra_generation_params={})) == 0


def test_an_unchanged_panel_is_not_written_again(monkeypatch):
    writes = []
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", {})
    monkeypatch.setattr(mixer, "write_json_atomic", lambda path, data: writes.append(data))
    args = mixer._MixerArgs.from_args(_run_args())
    mixer._save_runtime_current_settings("txt2img", args)
    mixer._save_runtime_current_settings("txt2img", args)
    assert len(writes) == 1 and writes[0]["enable"]


def test_a_schema_build_reads_the_remembered_panel_once(monkeypatch):
    """Every param's default comes from one read of the remembered panel; a
    generation's save or a reset is seen by the next build."""
    reads = []
    real = mixer._initial_defaults
    monkeypatch.setattr(mixer, "_initial_defaults", lambda tab: reads.append(tab) or real(tab))
    layout = layout_of("txt2img")
    for _build in range(2):
        layout.defaults()
    assert reads == ["txt2img"]

    mixer._save_current_settings_data("txt2img", _payload(artist="saved artist"))
    assert layout.defaults()["base_rows"][0]["artist"] == "saved artist" and len(reads) == 2
    mixer._delete_current_settings_data("txt2img")
    assert layout.defaults() == mixer._builtin_defaults() and len(reads) == 3
