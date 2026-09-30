import json
from types import SimpleNamespace

import pytest

from test_runtime_dispatch import _run_args, _sampling_pass, harness, mixer, runtime
from test_ui_wiring import _build_ui, _element

# Shared with the other panels' suites, loaded by path (see its docstring)


def _through_infotext(params):
    """``params`` written as Forge writes the parameters line and parsed
    back as a pasted infotext is."""
    return harness.through_infotext(params)[1]


def _row(artist, weight=1.0, blocks="0-51", start=0.0, end=1.0, peak=0.5, curve=mixer.CURVE_HOLD, stage=mixer.PRESET_CUSTOM, auto=False, enabled=True):
    return [enabled, artist, weight, blocks, start, end, peak, curve, stage, auto]


def _panel_args(language="en", hires_independent=True, disable_hires_mixing=False):
    """Script args of a panel in ``language``'s labels: three base rows (one
    disabled, one on an auto-shifted stage) and two independent hires rows."""
    label = lambda group, key: mixer._option_label(group, key, language)
    yes, no = mixer._bool_label(True, language), mixer._bool_label(False, language)
    width = mixer.ROW_WIDTH
    base = [
        _row('artist "quoted", 米山舞', 0.35, "0,3,5-12", 0.1, 0.8, 0.3, label("curve", mixer.CURVE_TRIANGLE), label("stage", mixer.PRESET_CUSTOM), no, yes),
        _row("disabled artist", enabled=no),
        _row("(wlop:1.2)", -0.5, "10-18", curve=label("curve", mixer.CURVE_SMOOTH), stage=label("stage", mixer.PRESET_STYLE), auto=yes, enabled=yes),
    ]
    hires = [_row("hires one", 0.8, curve=label("curve", mixer.CURVE_BACK)), _row("hires two", 1.5, "-1", 0.2, 0.9, 0.6)]
    head = [
        True,
        len(base),
        len(hires),
        hires_independent,
        disable_hires_mixing,
        3.0,
        3.0,
        0.85,
        label("optimization", mixer.OPT_QUALITY),
        label("combine", mixer.COMBINE_CONCAT),
        label("fusion", mixer.FUSION_QUALITY_DELTA),
        True,
        False,
    ]
    fill = lambda rows: [value for row in rows for value in row] + [None] * (width * (mixer.MAX_ARTIST_ROWS - len(rows)))
    return head + fill(base) + fill(hires)


def _p(enable_hr=True):
    return SimpleNamespace(enable_hr=enable_hr, distilled_cfg_scale=5.0, hr_distilled_cfg=3.0)


def _recorded(args=None, p=None):
    return mixer.generation_params(p or _p(), mixer._MixerArgs.from_args(args or _panel_args()))


def _paste(monkeypatch, tmp_path, params, language="en"):
    """``(script, controls, updates)``: the updates pasting ``params`` into a
    panel built in ``language`` makes, by component id (None: left alone)."""
    monkeypatch.setattr(mixer, "_language", lambda: language)
    mixer._paste_plan.cache_clear()
    script, _demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    fields = harness.paste_fields(script)
    updates = {field.component._id: update for field, update in zip(fields, harness.ui_paste_updates(script, params), strict=True)}
    return script, controls, updates


def _pasted_args(controls, updates):
    """The script args of the panel after the paste."""
    return [updates[component._id]["value"] if updates.get(component._id) is not None else component.value for component in controls]


def test_a_generation_records_the_rows_and_controls_in_internal_values():
    params = _recorded(_panel_args("zh"))
    assert params == _recorded(_panel_args("en"))  # independent of the UI language
    options = json.loads(params[mixer.INFOTEXT_OPTIONS])
    assert options == {
        "strength": 0.85,
        "optimization": mixer.OPT_QUALITY,
        "combine": mixer.COMBINE_CONCAT,
        "fusion": mixer.FUSION_QUALITY_DELTA,
        "apply_uncond": True,
        "hires_independent": True,
        "disable_hires_mixing": False,
    }
    base = json.loads(params[mixer.INFOTEXT_BASE])
    # The enabled rows only; the stage row's timings are those of the base Shift
    style = list(mixer._auto_stage_values(mixer.PRESET_STYLE, 5.0))
    assert base == [
        ['artist "quoted", 米山舞', 0.35, "0,3,5-12", 0.1, 0.8, 0.3, mixer.CURVE_TRIANGLE, mixer.PRESET_CUSTOM, False],
        ["(wlop:1.2)", -0.5, "10-18", *[round(value, 4) for value in style], mixer.CURVE_SMOOTH, mixer.PRESET_STYLE, True],
    ]
    assert [row[0] for row in json.loads(params[mixer.INFOTEXT_HIRES])] == ["hires one", "hires two"]


@pytest.mark.parametrize(
    "p, args",
    [
        (_p(enable_hr=False), _panel_args()),  # no Hires pass (img2img, or Hires. fix off)
        (_p(), _panel_args(hires_independent=False)),  # the Hires pass inherits the base rows
        (_p(), _panel_args(disable_hires_mixing=True)),  # the Hires pass does not mix
    ],
)
def test_the_hires_rows_are_recorded_only_when_the_hires_pass_mixes_with_them(p, args):
    params = _recorded(args, p)
    assert mixer.INFOTEXT_HIRES not in params and mixer.INFOTEXT_BASE in params


def test_the_sampling_pass_writes_the_record_beside_the_summary(monkeypatch):
    p, _unet, _attention = _sampling_pass(monkeypatch)
    try:
        assert p.extra_generation_params["Anima Artist Mixer"].startswith("2 artists")
        expected = mixer.generation_params(p, mixer._MixerArgs.from_args(_run_args()))
        assert {key: p.extra_generation_params[key] for key in expected} == expected
        assert json.loads(expected[mixer.INFOTEXT_BASE])[0][0] == "(base artist:-1)"
    finally:
        runtime._cleanup_runtime()


@pytest.mark.parametrize("record_language, paste_language", [("en", "en"), ("zh", "zh"), ("en", "zh"), ("zh", "en")])
def test_pasting_the_infotext_restores_the_panel(monkeypatch, tmp_path, record_language, paste_language):
    params = _recorded(_panel_args(record_language))
    parsed = _through_infotext({"Steps": "20", "Distilled CFG Scale": "5.0", "Hires Distilled CFG Scale": "3.0", **params, "Seed": "1"})
    assert {key: parsed[key] for key in params} == params

    script, controls, updates = _paste(monkeypatch, tmp_path, parsed, paste_language)
    pasted = _pasted_args(controls, updates)
    # The pasted panel generates with the same rows and controls
    assert mixer.generation_params(_p(), mixer._MixerArgs.from_args(pasted)) == params
    args = mixer._MixerArgs.from_args(pasted)
    assert args.enable is True and args.base_row_count == 2 and args.hires_row_count == 2
    # The labels are the pasting UI's
    assert args.optimization == mixer._option_label("optimization", mixer.OPT_QUALITY, paste_language)
    assert args.row_values(False)[7] == mixer._option_label("curve", mixer.CURVE_TRIANGLE, paste_language)
    assert args.row_values(False)[0] == mixer._bool_label(True, paste_language)
    # Rows past the pasted ones are hidden
    base_updates = [updates[component._id] for component in controls[len(mixer.HEADER_ARGS) :][: mixer.MAX_ARTIST_ROWS * mixer.ROW_WIDTH]]
    assert base_updates[mixer.ROW_WIDTH]["visible"] and not base_updates[mixer.ROW_WIDTH * 2]["visible"]
    # The text-encoding cache is a preference, not part of the image
    assert args_component(controls, "enable_cache")._id not in updates


@pytest.mark.parametrize("record_language, api_language", [("en", "en"), ("zh", "zh"), ("en", "zh"), ("zh", "en")])
def test_the_api_replays_the_infotext_into_the_same_rows_and_options(monkeypatch, tmp_path, record_language, api_language):
    # The paste functions give Gradio 4 gr.update dicts; the API takes their values
    params = _recorded(_panel_args(record_language))
    parsed = _through_infotext({"Steps": "20", "Distilled CFG Scale": "5.0", "Hires Distilled CFG Scale": "3.0", **params, "Seed": "1"})
    monkeypatch.setattr(mixer, "_language", lambda: api_language)
    mixer._paste_plan.cache_clear()
    script, _demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    replayed = mixer._MixerArgs.from_args(harness.api_replay(script, controls, parsed))

    assert mixer.generation_params(_p(), replayed) == params
    assert replayed.enable is True and (replayed.base_row_count, replayed.hires_row_count) == (2, 2)
    assert (replayed.hires_independent, replayed.disable_hires_mixing, replayed.apply_uncond) == (True, False, True)
    assert replayed.optimization == mixer._option_label("optimization", mixer.OPT_QUALITY, api_language)


def args_component(controls, name):
    return controls[mixer.HEADER_ARGS.index(name)]


def test_an_infotext_without_hires_rows_leaves_the_hires_rows_alone(monkeypatch, tmp_path):
    params = _recorded(_panel_args(hires_independent=False))
    _script, controls, updates = _paste(monkeypatch, tmp_path, _through_infotext(params))
    hires = controls[len(mixer.HEADER_ARGS) + mixer.MAX_ARTIST_ROWS * mixer.ROW_WIDTH :]
    assert all(updates[component._id] is None for component in hires)
    assert updates[args_component(controls, "hires_row_count")._id] is None
    assert updates[args_component(controls, "hires_independent")._id]["value"] is False
    assert updates[args_component(controls, "base_row_count")._id]["value"] == 2


@pytest.mark.parametrize(
    "params",
    [
        # What earlier versions wrote: the summary alone
        {"Anima Artist Mixer": "2 artists, strength=0.70, output_avg/interpolate, preset=Balance, shift=3.00"},
        {"Anima Artist Mixer": "disabled: AND prompts are not supported"},
        {},  # a generation without the Mixer
        {mixer.INFOTEXT_OPTIONS: "{not json", mixer.INFOTEXT_BASE: '"a string"', mixer.INFOTEXT_HIRES: "[1, 2"},
    ],
)
def test_an_infotext_without_the_record_leaves_the_panel_alone(monkeypatch, tmp_path, params):
    _script, _controls, updates = _paste(monkeypatch, tmp_path, _through_infotext({"Steps": "20", **params}))
    assert all(update is None for update in updates.values())


def test_a_paste_saves_the_panel_like_an_edit(monkeypatch, tmp_path):
    script, demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    marker = _element(demo, script.elem_id("paste_marker"))
    assert marker not in controls  # not a script arg
    paste = {component._id: key for component, key in script.infotext_fields}[marker._id]
    # The pasted Artist/Blocks textboxes save on blur, which a paste does not
    # fire: the marker changes on every paste that restores the panel
    params = _through_infotext(_recorded())
    first, second = paste(params), paste(params)
    assert first["value"] and second["value"] != first["value"]
    assert paste({"Anima Artist Mixer": "disabled: AND prompts are not supported"}) is None
    autosave = [dep for dep in demo.config["dependencies"] if len(dep["targets"]) > mixer.MAX_ARTIST_ROWS * 20]
    assert len(autosave) == 1 and (marker._id, "change") in autosave[0]["targets"]


def test_send_to_carries_the_panel_by_its_infotext_keys(monkeypatch, tmp_path):
    # "Send to" copies the fields named in paste_field_names from one tab's
    # panel to the same fields of the other's; the paste functions are those keys
    script, controls, _updates = _paste(monkeypatch, tmp_path, {})
    assert script.paste_field_names == list(mixer.INFOTEXT_KEYS)
    named = [component._id for component in harness.send_to_components(script)]
    unrecorded = {args_component(controls, name)._id for name in ("runtime_base_shift", "runtime_hires_shift", "enable_cache")}
    assert named == [component._id for component in controls if component._id not in unrecorded]
