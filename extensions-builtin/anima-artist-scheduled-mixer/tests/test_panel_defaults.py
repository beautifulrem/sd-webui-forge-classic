"""The panel's defaults are the built-in ones, never the last generation's:
a job that leaves the mixer out (API, queue import, a reused job) must not
inherit someone's artists. The browser keeps the user's panel as a draft."""

import json
from types import SimpleNamespace

from test_remi_params import layout_of
from test_runtime_dispatch import mixer


def test_a_generation_does_not_change_the_defaults(data_dir):
    before = {tab: layout_of(tab).defaults() for tab in ("txt2img", "img2img")}
    assert before["txt2img"] == mixer._builtin_defaults() and not before["txt2img"]["enable"]
    for tab in ("txt2img", "img2img"):
        script = mixer.Script()
        script.tabname = tab
        p = SimpleNamespace(is_api=False, extra_generation_params={}, distilled_cfg_scale=4.5)
        values = {**layout_of(tab).defaults(), "enable": True, "base_rows": [{"artist": "(base artist:-1)"}]}
        script.process_before_every_sampling(p, *layout_of(tab).values_to_args(values))
        assert layout_of(tab).defaults() == before[tab]
    # nothing written to the data folder
    assert not list(data_dir.rglob("artist_mixer_current_settings*.json"))


def test_files_remembered_by_earlier_versions_are_ignored(data_dir):
    folder = data_dir / "extension-data" / mixer._DATA_NAME
    folder.mkdir(parents=True)
    payload = {"version": 1, "enable": True, "base_rows": [["1", "wlop"]]}
    for name in ("artist_mixer_current_settings.json", "artist_mixer_current_settings_txt2img.json"):
        (folder / name).write_text(json.dumps(payload))
    assert layout_of("txt2img").defaults() == mixer._builtin_defaults()


def test_reset_restores_the_built_in_defaults():
    from remi import ui

    script = mixer.Script()
    script.tabname = "txt2img"
    patch = ui.call_action(script, "reset_defaults", ui.ActionContext(tab="txt2img"), {})
    assert patch["enable"] == {"value": False} and patch["__toast__"]["message"] == mixer._t("defaults_restored")
    assert patch["base_rows"]["value"] == mixer._builtin_defaults()["base_rows"]


def test_a_schema_build_computes_the_defaults_once(monkeypatch):
    """Every param's default comes from one computation; a template edit is seen by the next build."""
    calls = []
    real = mixer._builtin_defaults
    monkeypatch.setattr(mixer, "_DEFAULTS_CACHE", {})
    monkeypatch.setattr(mixer, "_builtin_defaults", lambda: calls.append(1) or real())
    layout = layout_of("txt2img")
    for _build in range(2):
        layout.defaults()
    assert len(calls) == 1
