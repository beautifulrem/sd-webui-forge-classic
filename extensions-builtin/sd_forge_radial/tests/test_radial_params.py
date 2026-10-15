"""The ported panel: params() declares what the Gradio ui() did, in the
processing order (which differs from the panel's)."""

import json
from pathlib import Path

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_radial.py"


@pytest.fixture(scope="module")
def radial():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "radial_params_under_test")


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(radial, tab):
    layout = harness.script_layout(radial.RadialAttentionForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab))


def test_arguments_reach_the_patch_in_its_order(radial, monkeypatch):
    seen = {}

    def patch_radial_attn(model, *args):
        seen["args"] = args
        return model

    monkeypatch.setattr(radial.PatchRadialAttn, "patch_radial_attn", staticmethod(patch_radial_attn))
    script = radial.RadialAttentionForForge()
    args = harness.script_layout(script).values_to_args(
        {"enable": True, "dense_block": 3, "decay_factor": 0.5, "dense_timestep": 4, "last_dense_timestep": 5, "block_size": 64, "allow_compile": True}
    )
    unet = object()
    p = type("P", (), {})()
    p.sd_model = type("M", (), {})()
    p.sd_model.forge_objects = type("F", (), {"unet": unet})()
    script.process_before_every_sampling(p, *args)
    # patch_radial_attn(model, dense_block, dense_timestep, last_dense_timestep, block_size, decay_factor, allow_compile)
    assert seen["args"] == (3, 4, 5, 64, 0.5, True)


def test_schema_export(radial):
    schema = harness.script_layout(radial.RadialAttentionForForge()).schema()
    json.dumps(schema, allow_nan=False)
    assert schema["enable"]["name"] == "enable"
    block_size = schema["children"][2]["children"][0]
    assert block_size["style"] == "radio" and [choice["value"] for choice in block_size["choices"]] == [64, 128]


def test_zh_cn_covers_the_panel(radial):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(radial.RadialAttentionForForge(), tab).schema() for tab in harness.TABS])


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT])
