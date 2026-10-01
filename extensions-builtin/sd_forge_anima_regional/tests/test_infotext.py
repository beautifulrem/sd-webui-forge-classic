"""A regional run's infotext replays into the panel it was run from, through
the WebUI's paste and through the API's infotext replay."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the other panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

SETTINGS = [True, "0-9, 20-27", 0.1, 0.8, 0.05, 0.3]
# (region values, the values a replay gives them): an inactive region
# records only its switch, so the replay leaves its other controls at their defaults
REGION_ONE_OFF = [
    (False, "hidden, unused", 0.1, 0.2, 0.3, 0.4, 0.5),
    (True, "red hair, blue eyes", 0.4, 0.1, 0.6, 0.9, 0.75),
    (True, "a castle: at night", 0.2, 0.3, 0.4, 0.6, 1.5),
]
REGION_THREE_OFF = [
    (True, "red hair, blue eyes", 0.05, 0.1, 0.45, 0.9, 0.75),
    (True, "a castle: at night", 0.55, 0.0, 0.45, 1.0, 1.25),
    (False, "hidden, unused", 0.1, 0.2, 0.3, 0.4, 0.5),
]


@pytest.fixture
def regional(monkeypatch):
    module = harness.load_script(ROOT / "scripts" / "anima_regional.py", "anima_regional_infotext_under_test")
    # No text encoder or patch: the infotext is what the pass records
    monkeypatch.setattr(module, "_region", lambda p, text, *placement: (text, placement))
    monkeypatch.setattr(module, "apply_regional_patch", lambda unet, state: unet)
    return module


def _recorder(regional):
    def record(args):
        """The infotext a regional pass with the script args ``args`` records."""
        p, unet = harness.make_process()
        unet.model.predictor = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        regional.AnimaRegionalScript().process_before_every_sampling(p, *args)
        return p.extra_generation_params

    return record


@pytest.mark.parametrize("regions", [REGION_ONE_OFF, REGION_THREE_OFF], ids=["region-1-off", "region-3-off"])
def test_the_infotext_replays_into_the_same_script_args(regional, regions):
    script = regional.AnimaRegionalScript()
    controls = harness.build_ui(script)
    defaults = [control.value for control in controls]
    args = SETTINGS + [value for region in regions for value in region]
    expected = list(args)
    for index, region in enumerate(regions):
        if not region[0]:
            start = len(SETTINGS) + index * len(region)
            expected[start + 1 : start + len(region)] = defaults[start + 1 : start + len(region)]
    assert all(value != default for value, default in zip(SETTINGS, defaults))

    params = harness.assert_infotext_round_trip(script, controls, args, _recorder(regional), expected)
    off = 1 if regions is REGION_ONE_OFF else 3
    # The recorded "False" of the inactive region replays as False
    assert params[f"Anima region {off} active"] is False
    assert not any(key.startswith(f"Anima region {off} ") and not key.endswith(" active") for key in params)


def test_an_off_panel_records_nothing_and_replays_its_defaults(regional):
    script = regional.AnimaRegionalScript()
    controls = harness.build_ui(script)
    args = [False, *SETTINGS[1:]] + [value for region in REGION_ONE_OFF for value in region]
    harness.assert_infotext_round_trip(script, controls, args, _recorder(regional), [control.value for control in controls])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=repr)
@pytest.mark.parametrize("field", ["start", "feather", "base_preserve", "region2_x", "region1_strength"])
def test_a_non_finite_setting_is_refused_before_anything_is_installed(regional, field, value):
    # The API reads JSON NaN / Infinity as floats: min(max(nan, 0), 1) is nan,
    # a NaN blend; a panel that records it writes "nan", which replays as NaN
    from modules.anima_feature_conflicts import REGIONAL, is_feature_active
    from modules.anima_support import negpip_extra_prompts

    names = regional.SCRIPT_ARG_NAMES
    args = SETTINGS + [value for region in REGION_ONE_OFF for value in region]
    args[names.index(field)] = value
    p, unet = harness.make_process()
    unet.model.predictor = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
    script = regional.AnimaRegionalScript()
    for hook in (script.process, script.process_before_every_sampling):
        with pytest.raises(ValueError, match=rf"^Anima Regional Conditioning refuses non-finite script arguments: {field}={value!r}$"):
            hook(p, *args)
    assert p.sd_model.forge_objects.unet is unet and p.extra_generation_params == {}
    assert not is_feature_active(p, REGIONAL) and negpip_extra_prompts(p) == []
