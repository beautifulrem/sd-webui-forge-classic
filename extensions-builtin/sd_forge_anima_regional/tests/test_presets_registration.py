"""The preset's regional.* values are the panel's own defaults, and the
infotext a pass writes restores the panel."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture
def regional():
    return harness.load_script(ROOT / "scripts" / "anima_regional.py", "anima_regional_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults(regional):
    # Region prompts are user content: the preset leaves them alone
    for tab in ("txt2img", "img2img"):
        harness.assert_registers_preset(regional.AnimaRegionalScript(), "regional", tab, unowned_labels=("Regional prompt",))


SETTINGS = (True, "2-20", 0.1, 0.6, 0.05, 0.2)
REGIONS = (
    (True, "a cat", 0.0, 0.0, 0.5, 1.0, 1.0),
    (True, "a dog", 0.5, 0.0, 0.5, 1.0, 0.8),
    (True, "a bird", 0.25, 0.25, 0.5, 0.5, 1.5),
)


@pytest.fixture
def sampling(regional, monkeypatch):
    """Regional with prompt encoding and the UNet patch stubbed out."""

    monkeypatch.setattr(regional, "_encode_cached", lambda p, text: (torch.zeros(1, 2, 4), None))
    monkeypatch.setattr(regional, "apply_regional_patch", lambda unet, state: unet)
    original = harness.make_process

    def make_process(blocks=28):
        p, unet = original(blocks)
        p.firstpass_steps = 20  # the base pass's total steps, as setup_conds sets them
        unet.model.predictor = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        return p, unet

    monkeypatch.setattr(harness, "make_process", make_process)
    return regional


def test_the_hooks_name_the_settings_in_script_arg_order(regional):
    import inspect

    settings = [name for name, *_ in regional.SCRIPT_ARGS if not name.startswith("region")]
    for hook in (regional.AnimaRegionalScript.process, regional.AnimaRegionalScript.process_before_every_sampling):
        assert list(inspect.signature(hook).parameters)[2 : 2 + len(settings)] == settings  # after self, p


def test_written_infotext_restores_the_panel_controls(sampling):
    args = (*SETTINGS, *(value for region in REGIONS for value in region))
    harness.assert_infotext_restores_controls(sampling.AnimaRegionalScript(), args, dict(zip(sampling.INFOTEXT_KEYS, args)))


def test_an_inactive_region_writes_only_its_switch(sampling):
    script = sampling.AnimaRegionalScript()
    controls = harness.build_ui(script)
    fields = {key: component for component, key in script.infotext_fields}
    assert list(fields) == script.paste_field_names == list(sampling.INFOTEXT_KEYS)
    args = (*SETTINGS, *REGIONS[0], False, *REGIONS[1][1:], *REGIONS[2])
    p, _ = harness.make_process()

    script.process_before_every_sampling(p, *args)

    written = p.extra_generation_params
    assert written["Anima region 2 active"] is False
    assert not any(key.startswith("Anima region 2 ") and key != "Anima region 2 active" for key in written)
    assert len(written) == len(fields) - 6
    for key, value in written.items():
        index = next(number for number, control in enumerate(controls) if control is fields[key])
        assert value == args[index], key
