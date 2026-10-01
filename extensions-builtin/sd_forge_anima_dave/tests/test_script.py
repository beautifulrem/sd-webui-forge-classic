import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the Cross-Attention Boost suite, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

dave = harness.load_script(ROOT / "scripts" / "anima_dave.py", "anima_dave_under_test")


def test_written_infotext_restores_the_panel_controls():
    harness.assert_infotext_restores_controls(
        dave.AnimaDaveScript(),
        (True, 0.3, 0.1, "8-18"),
        {"Anima DAVE enabled": True, "Anima DAVE strength": 0.3, "Anima DAVE tau": 0.1, "Anima DAVE blocks": "8-18"},
    )


def test_apply_preset_turns_the_panel_off_at_its_defaults():
    harness.assert_registers_preset(dave.AnimaDaveScript(), "dave")


def test_process_patches_the_pass_and_writes_infotext():
    p, unet = harness.make_process(40)
    dave.AnimaDaveScript().process_before_every_sampling(p, True, 0.3, 0.1, "8-18")
    patched = p.sd_model.forge_objects.unet
    assert patched is not unet and "model_function_wrapper" in patched.model_options
    assert "model_function_wrapper" not in unet.model_options
    assert p.extra_generation_params == dave.generation_params(0.3, 0.1, "8-18")


def test_disabled_zero_strength_and_non_anima_are_no_ops():
    for enable, strength, anima in ((False, 0.3, True), (True, 0.0, True), (True, 0.3, False)):
        p, unet = harness.make_process(40)
        p.sd_model.is_anima_engine = anima
        dave.AnimaDaveScript().process_before_every_sampling(p, enable, strength, 0.1, "8-18")
        assert p.sd_model.forge_objects.unet is unet
        assert p.extra_generation_params == {}


def _record(args):
    """The infotext a pass with the script args ``args`` records."""
    p, _ = harness.make_process(40)
    dave.AnimaDaveScript().process_before_every_sampling(p, *args)
    return p.extra_generation_params


@pytest.mark.parametrize("enable", [True, False], ids=["every-control-changed", "off"])
def test_the_infotext_replays_into_the_same_script_args(enable):
    script = dave.AnimaDaveScript()
    controls = harness.build_ui(script)
    args = [enable, *(True, 0.15, 0.05, "0-5, 12")[1:]]
    assert all(value != control.value for value, control in zip(args[1:], controls[1:]))
    # Off records nothing: the paste and the API leave the panel at its defaults
    expected = args if enable else [control.value for control in controls]
    params = harness.assert_infotext_round_trip(script, controls, args, _record, expected)
    assert (params == {}) is not enable


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=repr)
@pytest.mark.parametrize("field", ["strength", "tau"])
def test_a_non_finite_setting_is_refused_before_the_pass_is_patched(field, value):
    # The API reads JSON NaN / Infinity as floats: a NaN tau with strength > 0
    # raised "cannot convert float NaN to integer" inside the model wrapper,
    # aborting the job mid-sampling
    args = {"enable": True, "strength": 0.3, "tau": 0.1, "blocks": "8-18", field: value}
    p, unet = harness.make_process(40)
    with pytest.raises(ValueError, match=rf"^Anima DAVE refuses non-finite script arguments: {field}={value!r}$"):
        dave.AnimaDaveScript().process_before_every_sampling(p, *args.values())
    assert p.sd_model.forge_objects.unet is unet
    assert p.extra_generation_params == {}


def test_settings_outside_the_sliders_are_applied_and_recorded_bounded():
    assert _record((True, 1.5, 3.0, "8-18")) == dave.generation_params(1.0, 1.0, "8-18")
    assert _record((True, 0.3, 0.0, "8-18")) == dave.generation_params(0.3, 0.01, "8-18")
    assert _record((True, -0.5, 0.1, "8-18")) == {}
