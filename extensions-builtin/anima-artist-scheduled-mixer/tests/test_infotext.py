"""A mixed run's infotext records the panel and pastes back into it: the SPA
paste, ``/infotext/parse`` and the API's infotext replay all read it through
the panel's ``remi.ui`` layout."""

import json
from types import SimpleNamespace

import pytest

from test_runtime_dispatch import _run_args, _sampling_pass, mixer, runtime
from test_remi_params import harness, layout_of


def _row(artist, weight=1.0, blocks="0-51", start=0.0, end=1.0, peak=0.5, curve=mixer.CURVE_HOLD, stage=mixer.PRESET_CUSTOM, auto=False, enabled=True):
    return [enabled, artist, weight, blocks, start, end, peak, curve, stage, auto]


def _panel(language="en", hires_independent=True, disable_hires_mixing=False):
    """Named values of a panel: three base rows (one disabled, one on an
    auto-shifted stage) and two independent hires rows. ``language``: the
    labels its options are given in (templates and API callers may use the
    Gradio panel's labels in either language)."""
    label = lambda group, key: mixer._option_label(group, key, language)
    yes, no = mixer._bool_label(True, language), mixer._bool_label(False, language)
    base = [
        _row('artist "quoted", 米山舞', 0.35, "0,3,5-12", 0.1, 0.8, 0.3, label("curve", mixer.CURVE_TRIANGLE), label("stage", mixer.PRESET_CUSTOM), no, yes),
        _row("disabled artist", enabled=no),
        _row("(wlop:1.2)", -0.5, "10-18", curve=label("curve", mixer.CURVE_SMOOTH), stage=label("stage", mixer.PRESET_STYLE), auto=yes, enabled=yes),
    ]
    hires = [_row("hires one", 0.8, curve=label("curve", mixer.CURVE_BACK)), _row("hires two", 1.5, "-1", 0.2, 0.9, 0.6)]
    return {
        "enable": True,
        "hires_independent": hires_independent,
        "disable_hires_mixing": disable_hires_mixing,
        "global_strength": 0.85,
        "optimization": mixer.OPT_QUALITY,
        "combine_mode": mixer.COMBINE_CONCAT,
        "fusion_mode": mixer.FUSION_QUALITY_DELTA,
        "apply_uncond": True,
        "enable_cache": False,
        "base_rows": mixer._table(base),
        "hires_rows": mixer._table(hires),
    }


def _p(enable_hr=True):
    return SimpleNamespace(enable_hr=enable_hr, distilled_cfg_scale=5.0, hr_distilled_cfg=3.0)


def _args(values=None):
    """The script args of a panel (the API may give the options as labels)."""
    values = values or _panel()
    return [values.get(name, default) for name, default in zip(mixer.PARAM_NAMES, layout_of().defaults().values(), strict=True)]


def _recorded(values=None, p=None):
    p = p or _p()
    return mixer.generation_params(p, mixer._MixerArgs.from_args(_args(values), p))


def test_a_generation_records_the_rows_and_controls_in_internal_values():
    params = _recorded(_panel("zh"))
    assert params == _recorded(_panel("en"))  # whatever labels the options came in
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
    "p, values",
    [
        (_p(enable_hr=False), _panel()),  # no Hires pass (img2img, or Hires. fix off)
        (_p(), _panel(hires_independent=False)),  # the Hires pass inherits the base rows
        (_p(), _panel(disable_hires_mixing=True)),  # the Hires pass does not mix
    ],
)
def test_the_hires_rows_are_recorded_only_when_the_hires_pass_mixes_with_them(p, values):
    params = _recorded(values, p)
    assert mixer.INFOTEXT_HIRES not in params and mixer.INFOTEXT_BASE in params


def test_the_sampling_pass_writes_the_record_beside_the_summary(monkeypatch):
    p, _unet, _attention = _sampling_pass(monkeypatch)
    try:
        assert p.extra_generation_params["Anima Artist Mixer"].startswith("2 artists")
        expected = mixer.generation_params(p, mixer._MixerArgs.from_args(_run_args(), p))
        assert {key: p.extra_generation_params[key] for key in expected} == expected
        assert json.loads(expected[mixer.INFOTEXT_BASE])[0][0] == "(base artist:-1)"
    finally:
        runtime._cleanup_runtime()


def _infotext(params):
    return {"Steps": "20", "Distilled CFG Scale": "5.0", "Hires Distilled CFG Scale": "3.0", **params, "Seed": "1"}


@pytest.mark.parametrize("record_language", ["en", "zh"])
@pytest.mark.parametrize("start", ["defaults", "edited"])
def test_pasting_the_infotext_restores_the_panel(record_language, start):
    """Into a fresh panel (the API's infotext replay) or one with other values (the SPA paste)."""

    params = _recorded(_panel(record_language))
    mixer._paste_plan.cache_clear()
    layout = layout_of()
    begin = None if start == "defaults" else {"enable_cache": False, "global_strength": 0.1, "hires_rows": mixer._table([_row("old")] * 5)}
    pasted = harness.replayed_args(layout, _infotext(params), begin)
    args = mixer._MixerArgs.from_args(pasted, _p())

    # The pasted panel generates with the same rows and controls
    assert mixer.generation_params(_p(), args) == params
    assert args.enable is True and args.base_row_count == 2 and args.hires_row_count == 2
    # The options are pasted as their internal values
    assert (args.optimization, args.combine_mode, args.fusion_mode) == (mixer.OPT_QUALITY, mixer.COMBINE_CONCAT, mixer.FUSION_QUALITY_DELTA)
    values = dict(zip(mixer.PARAM_NAMES, pasted))
    assert values["base_rows"][0]["curve"] == mixer.CURVE_TRIANGLE and values["base_rows"][0]["enabled"] is True
    # The text-encoding cache is a preference, not part of the image
    assert values["enable_cache"] is (start == "defaults")


def test_an_infotext_without_hires_rows_leaves_the_hires_rows_alone():
    params = _recorded(_panel(hires_independent=False))
    pasted = harness.pasted(layout_of(), _infotext(params))
    assert "hires_rows" not in pasted
    assert pasted["hires_independent"] is False and len(pasted["base_rows"]) == 2


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
def test_an_infotext_without_the_record_leaves_the_panel_alone(params):
    mixer._paste_plan.cache_clear()
    assert harness.pasted(layout_of(), {"Steps": "20", **params}) == {}


def test_the_record_is_keyed_by_the_params_that_restore_it():
    # The infotext keys name the params that carry the panel ("Send to" and
    # reuse read them); every other recorded control is pasted with them
    layout = layout_of()
    assert layout.infotext == {mixer.INFOTEXT_OPTIONS: "enable", mixer.INFOTEXT_BASE: "base_rows", mixer.INFOTEXT_HIRES: "hires_rows"}
    pasted = {name for name, node in layout.by_name.items() if node.paste is not None}
    assert pasted == set(mixer.COMPACT_HEADER_ARGS) - {"enable_cache"} | set(mixer.TABLE_ARGS)
