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
# Shared with the other ported panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture
def regional():
    return harness.load_script(ROOT / "scripts" / "anima_regional.py", "anima_regional_presets_under_test")


def test_apply_preset_turns_the_panel_off_at_its_defaults(regional):
    # Region prompts are user content and the region editor holds no setting: the preset leaves them alone
    unowned = ("region1_prompt", "region2_prompt", "region3_prompt", "region_editor")
    for tab in ("txt2img", "img2img"):
        harness.assert_preset_fields(harness.script_layout(regional.AnimaRegionalScript(), tab), "regional", unowned)


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


def test_written_infotext_restores_the_panel_params(sampling):
    args = (*SETTINGS, *(value for region in REGIONS for value in region))
    p, _ = harness.make_process()
    sampling.AnimaRegionalScript().process_before_every_sampling(p, *args)
    written = p.extra_generation_params
    assert written == dict(zip(sampling.INFOTEXT_KEYS, args))
    harness.assert_infotext_restores_params(harness.script_layout(sampling.AnimaRegionalScript()), args, written)


def test_an_inactive_region_writes_only_its_switch(sampling):
    script = sampling.AnimaRegionalScript()
    layout = harness.script_layout(script)
    keys = [node.infotext for node in layout.inputs if node.infotext]
    assert keys == list(sampling.INFOTEXT_KEYS)
    args = (*SETTINGS, *REGIONS[0], False, *REGIONS[1][1:], *REGIONS[2])
    p, _ = harness.make_process()

    script.process_before_every_sampling(p, *args)

    written = p.extra_generation_params
    assert written["Anima region 2 active"] is False
    assert not any(key.startswith("Anima region 2 ") and key != "Anima region 2 active" for key in written)
    assert len(written) == len(keys) - 6
    harness.assert_infotext_restores_params(layout, args, written)


@pytest.mark.parametrize(
    "args",
    [
        # One region too many used to become a fourth region, installed before the infotext raised
        (*SETTINGS, *(value for region in REGIONS for value in region), *REGIONS[0]),
        # A partial region used to raise only while the regions were built
        (*SETTINGS, *(value for region in REGIONS for value in region))[:-1],
    ],
    ids=["extra-region", "partial-region"],
)
def test_a_wrong_argument_count_installs_nothing(sampling, monkeypatch, args):
    from modules.anima_feature_conflicts import REGIONAL, is_feature_active

    built = []
    monkeypatch.setattr(sampling, "_region", lambda *region: built.append(region))
    p, unet = harness.make_process()
    count = len(sampling.SCRIPT_ARGS) + 1  # and the region editor
    for hook in (sampling.AnimaRegionalScript().process, sampling.AnimaRegionalScript().process_before_every_sampling):
        with pytest.raises(TypeError, match=f"Anima Regional Conditioning takes {count} script arguments, got {len(args)}"):
            hook(p, *args)
    assert built == [] and p.sd_model.forge_objects.unet is unet and p.extra_generation_params == {}
    assert not is_feature_active(p, REGIONAL)
