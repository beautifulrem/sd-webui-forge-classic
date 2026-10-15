import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the other ported panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

boost = harness.load_script(ROOT / "scripts" / "anima_xattn_boost.py", "anima_xattn_boost_under_test")


def test_written_infotext_restores_the_panel_params():
    args = (True, 2.0, 0.85, 0.5)
    written = _record(args)
    assert written == {"Anima xattn boost enabled": True, "Anima xattn boost strength": 2.0, "Anima xattn boost band": 0.85, "Anima xattn boost renorm": 0.5}
    harness.assert_infotext_restores_params(harness.script_layout(boost.AnimaXattnBoostScript()), args, written)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_apply_preset_turns_the_panel_off_at_its_defaults(tab):
    harness.assert_preset_fields(harness.script_layout(boost.AnimaXattnBoostScript(), tab), "xattn_boost")


def test_process_patches_the_pass_and_writes_infotext():
    p, unet = harness.make_process()
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, True, 2.0, 0.85, 0.5)
    patched = p.sd_model.forge_objects.unet
    assert patched is not unet and "model_function_wrapper" in patched.model_options
    assert "model_function_wrapper" not in unet.model_options
    assert p.extra_generation_params == boost.generation_params(2.0, 0.85, 0.5)


def test_disabled_unit_strength_and_non_anima_are_no_ops():
    for enable, strength, anima in ((False, 2.0, True), (True, 1.0, True), (True, 2.0, False)):
        p, unet = harness.make_process()
        p.sd_model.is_anima_engine = anima
        boost.AnimaXattnBoostScript().process_before_every_sampling(p, enable, strength, 0.85, 0.5)
        assert p.sd_model.forge_objects.unet is unet
        assert p.extra_generation_params == {}


def _record(args):
    """The infotext a pass with the script args ``args`` records."""
    p, _ = harness.make_process(28)
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, *args)
    return p.extra_generation_params


@pytest.mark.parametrize("enable", [True, False], ids=["every-control-changed", "off"])
def test_the_infotext_replays_into_the_same_script_args(enable):
    layout = harness.script_layout(boost.AnimaXattnBoostScript())
    defaults = layout.values_to_args({})
    args = [enable, *(True, 1.5, 0.9, 0.25)[1:]]
    assert all(value != default for value, default in zip(args[1:], defaults[1:]))
    # Off records nothing: the paste and the API leave the panel at its defaults
    expected = args if enable else defaults
    params = harness.assert_infotext_round_trip(layout, args, _record, expected)
    assert (params == {}) is not enable


def _bare_process():
    unet = SimpleNamespace(model_options={})
    sd_model = SimpleNamespace(is_anima_engine=True, forge_objects=SimpleNamespace(unet=unet))
    return SimpleNamespace(sd_model=sd_model, extra_generation_params={}), unet


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=repr)
@pytest.mark.parametrize("field", ["strength", "band", "renorm"])
def test_a_non_finite_setting_is_refused_before_the_pass_is_patched(field, value):
    # The API reads JSON NaN / Infinity as floats: nan == 1.0 is False, so a
    # NaN gain was installed and row_gains [nan, 1] blackened the image
    args = {"enable": True, "strength": 2.0, "band": 0.85, "renorm": 0.5, field: value}
    p, unet = _bare_process()
    with pytest.raises(ValueError, match=rf"^Anima Cross-Attention Boost refuses non-finite script arguments: {field}={value!r}$"):
        boost.AnimaXattnBoostScript().process_before_every_sampling(p, *args.values())
    assert p.sd_model.forge_objects.unet is unet
    assert p.extra_generation_params == {}


def test_settings_outside_the_sliders_are_applied_and_recorded_bounded(monkeypatch):
    states = []
    monkeypatch.setattr(boost, "apply_xattn_boost_patch", lambda unet, state: states.append(state) or unet)
    p, _ = _bare_process()
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, True, 5.0, 0.2, 1.5)
    assert (states[0].gain, states[0].band, states[0].renorm) == (3.0, 0.5, 1.0)
    assert p.extra_generation_params == boost.generation_params(3.0, 0.5, 1.0)
    # Below the slider a gain is the off position
    p, unet = _bare_process()
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, True, 0.5, 0.85, 0.5)
    assert p.sd_model.forge_objects.unet is unet and p.extra_generation_params == {}
