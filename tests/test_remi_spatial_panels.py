"""The ported spatial / attention Anima panels (Regional, FreeFuse, DAVE,
Cross-Attention Boost, Resolutions) in a script runner: schema export,
argument slots, the conflict locks that replaced their Gradio listeners,
preset bindings and infotext paste."""

import importlib.util
import json
from pathlib import Path

import pytest

from remi import ui
from remi.testing import webui_state

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("remi_panel_harness", REPO / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

PANELS = {
    "regional": ("sd_forge_anima_regional", "anima_regional.py", "AnimaRegionalScript"),
    "freefuse": ("sd_forge_anima_freefuse", "anima_freefuse.py", "AnimaFreeFuseScript"),
    "dave": ("sd_forge_anima_dave", "anima_dave.py", "AnimaDaveScript"),
    "boost": ("sd_forge_anima_xattn_boost", "anima_xattn_boost.py", "AnimaXattnBoostScript"),
    "resolutions": ("sd-forge-anima-resolutions", "anima_resolutions.py", "AnimaRandomizeResolutionScript"),
}
IDS = {
    "regional": "anima_regional_conditioning",
    "freefuse": "anima_freefuse_multi_lora",
    "dave": "anima_dave_diversity",
    "boost": "anima_cross_attention_boost",
    "resolutions": "anima_random_resolution",
}


@pytest.fixture(scope="module")
def webui():
    classes = []
    for key, (extension, file, name) in PANELS.items():
        module = harness.load_script(REPO / "extensions-builtin" / extension / "scripts" / file, f"spatial_panel_{key}")
        classes.append(getattr(module, name))
    with webui_state(script_classes=classes) as state:
        yield state


def _entries(runner):
    return {entry["id"]: entry for entry in runner.scripts_schema()}


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_the_panels_export_their_schema(webui, tab):
    runner = getattr(webui, tab)
    entries = _entries(runner)
    json.dumps(list(entries.values()), allow_nan=False)
    assert set(IDS.values()) <= set(entries)
    assert all(not entries[script_id]["legacy"] and entries[script_id]["kind"] == "alwayson" for script_id in IDS.values())
    assert entries[IDS["regional"]]["args"][-1] == "region_editor"
    assert entries[IDS["resolutions"]]["section"] == "dimensions"
    resolutions = entries[IDS["resolutions"]]["args"]
    assert resolutions == (["randomize", "randomize_highres", "presets", "history"] if tab == "txt2img" else ["fit_source"])
    assert not runner.disabled_scripts


def test_named_values_become_the_hooks_positional_arguments(webui):
    runner = webui.txt2img
    args = runner.script_args_from_values(None, {IDS["dave"]: {"enabled": True, "strength": 0.2}, IDS["resolutions"]: {"randomize_highres": True}})
    dave = runner.by_id[IDS["dave"]]
    assert args[dave.args_from : dave.args_to] == [True, 0.2, 0.1, "8-18"]
    resolutions = runner.by_id[IDS["resolutions"]]
    assert args[resolutions.args_from : resolutions.args_to] == [False, True, None, ""]


def _conflicts(tab, values, selected=None):
    from remi.core.conflicts import conflict_patch

    return conflict_patch(tab, values, selected)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_regional_and_freefuse_are_one_of_the_spatial_group(webui, tab):
    regional, freefuse = IDS["regional"], IDS["freefuse"]
    both = {regional: {"enable": True}, freefuse: {"enable": True}}
    patch = _conflicts(tab, both, f"{regional}.enable")
    assert patch[f"{regional}.enable"] == {"value": True, "enabled": True}
    assert patch[f"{freefuse}.enable"]["value"] is False and patch[f"{freefuse}.enable"]["enabled"] is False
    # Headless priority: FreeFuse first
    patch = _conflicts(tab, both)
    assert patch[f"{freefuse}.enable"]["value"] is True and patch[f"{regional}.enable"]["value"] is False


def test_freefuse_locks_the_refiner_switch(webui):
    freefuse = IDS["freefuse"]
    patch = _conflicts("txt2img", {freefuse: {"enable": True}, "refiner": {"enable": True, "checkpoint": "None"}}, f"{freefuse}.enable")
    assert patch[f"{freefuse}.enable"]["value"] is True
    assert patch["refiner.enable"] == {"value": False, "enabled": False}
    assert patch["refiner.checkpoint"]["enabled"] is False
    assert "Refiner cleared" in patch["__toast__"]["message"]
    assert f"{freefuse}.enable" in _conflict_params("txt2img")


def _conflict_params(tab):
    from remi.core.tabs import tab_schema

    return tab_schema(tab)["conflict_params"]


def test_the_anima_preset_reaches_every_panel(webui):
    from remi.core.presets import anima_preset_patch

    patch = anima_preset_patch("txt2img", "Safe base aesthetic")
    for key, value in {
        f"{IDS['regional']}.enable": False,
        f"{IDS['regional']}.region2_x": 0.5,
        f"{IDS['freefuse']}.collect_step": 3,
        f"{IDS['dave']}.blocks": "8-18",
        f"{IDS['boost']}.band": 0.85,
        f"{IDS['resolutions']}.randomize": False,
    }.items():
        assert patch[key]["value"] == value, key
    assert f"{IDS['regional']}.region1_prompt" not in patch


def test_infotext_paste_restores_the_panels(webui):
    values, problems = webui.txt2img.paste_values(
        {"Anima region 2 active": "False", "Anima region 1 prompt": "a red cat", "Anima FreeFuse collect step": "5", "Anima DAVE tau": "0.05", "Anima xattn boost renorm": "0.25"}
    )
    assert problems == []
    assert values[IDS["regional"]] == {"region1_prompt": "a red cat", "region2_active": False}
    assert values[IDS["freefuse"]] == {"collect_step": 5}
    assert values[IDS["dave"]] == {"tau": 0.05}
    assert values[IDS["boost"]] == {"renorm": 0.25}


def test_resolution_actions_run_through_the_action_api(webui):
    script = webui.img2img.by_id[IDS["resolutions"]]
    ctx = ui.ActionContext(tab="img2img", owner=IDS["resolutions"], extra={"source": {"width": 832, "height": 1216}})
    patch = ui.call_action(script, "fit_to_source", ctx, {"fit_source": True})
    assert (patch["core.width"]["value"], patch["core.height"]["value"]) == (1216, 1792)
