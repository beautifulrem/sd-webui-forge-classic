"""The whole-stack preset panel, loaded on its own like Forge loads it."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import remi_script_harness as harness

from remi import ui

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT.parent
SCRIPT_MODULE = harness.load_script(ROOT / "scripts" / "anima_remi_presets.py", "anima_remi_presets_under_test")
GUIDANCE = harness.load_script(BUILTIN / "sd_forge_anima_guidance" / "scripts" / "anima_guidance.py", "anima_guidance_for_presets_test")
HIRES_GUARD = harness.load_script(BUILTIN / "sd_forge_anima_hires_guard" / "scripts" / "anima_hires_guard.py", "anima_hires_guard_for_presets_test")
RESCALE = harness.load_script(BUILTIN / "sd-forge-anima-prompt-rescale" / "scripts" / "anima_prompt_rescale.py", "anima_prompt_rescale_for_presets_test")


def test_preset_warnings_are_visible():
    # Forge's root logger is at ERROR: without its own handler the preset
    # script's Turbo advice would never be shown.
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        assert SCRIPT_MODULE.logger.isEnabledFor(logging.WARNING)
        assert SCRIPT_MODULE.logger.handlers and not SCRIPT_MODULE.logger.propagate
    finally:
        root.setLevel(level)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params(tab):
    from modules.anima_presets import PRESET_DESCRIPTIONS, PRESETS, SAFE_BASE_AESTHETIC_LABEL

    layout = harness.script_layout(SCRIPT_MODULE.AnimaRemiPresetScript(), tab)
    # The legacy panel passed no argument; the preset choice is the one param (intentional:
    # the Apply action reads it) and binds no preset field itself
    assert harness.legacy_snapshot(ROOT, tab)["inputs"] == []
    assert harness.contract(layout) == [
        {"name": "preset", "type": "choice", "label": "Preset", "info": None, "default": SAFE_BASE_AESTHETIC_LABEL, "choices": list(PRESETS), "style": "dropdown", "infotext": None, "preset": None}
    ]
    schema = harness.schema_json(layout)
    assert schema["type"] == "section" and schema["title"] == "Anima Remi Presets" and schema["open"] is False
    assert {"type": "button", "label": "Apply preset", "action": "apply", "variant": "primary"} in schema["children"]
    # The description follows the chosen preset (the legacy .change handler)
    for label in PRESETS:
        shown = [node["text"] for node in schema["children"] if node["type"] == "markdown" and ui.evaluate(node.get("visible_if"), {"preset": label})]
        assert shown[0] == PRESET_DESCRIPTIONS[label] and len(shown) == 2


def test_process_takes_the_param():
    # The hook takes any args; off-Anima it does nothing
    from types import SimpleNamespace

    SCRIPT_MODULE.AnimaRemiPresetScript().process(SimpleNamespace(sd_model=None), SCRIPT_MODULE.SAFE_BASE_AESTHETIC_LABEL)


def test_owner_relative_patch():
    patch = {"cfg_scale": {"value": 1.0}, "sampler.steps": {"value": 10}, "me.preset": {"value": "x"}, "__toast__": {"level": "info", "message": "m"}}
    assert SCRIPT_MODULE.owner_relative(patch, "me") == {"core.cfg_scale": {"value": 1.0}, "sampler.steps": {"value": 10}, "preset": {"value": "x"}, "__toast__": {"level": "info", "message": "m"}}


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    classes = [SCRIPT_MODULE.AnimaRemiPresetScript, GUIDANCE.AnimaGuidanceScript, HIRES_GUARD.AnimaHiresGuardScript, RESCALE.AnimaPromptRescaleScript]
    with webui_state(script_classes=classes) as state:
        yield state


def _apply(webui, tab, label, core=None, scripts=None):
    runner = getattr(webui, tab)
    script = next(script for script in runner.scripts if isinstance(script, SCRIPT_MODULE.AnimaRemiPresetScript))
    ctx = ui.ActionContext(tab=tab, owner=script.script_id, core=core or {}, scripts=scripts or {})
    patch = ui.call_action(script, "apply", ctx, {"preset": label})
    return patch, script


def _bound(webui, tab):
    """``owner.param`` (core params as ``core.param``) -> preset field, over the tab's layouts."""

    from remi.core.conflicts import _layouts
    from remi.core.presets import preset_bindings

    return {f"{owner}.{name}": field for field, owner, name, _own in preset_bindings(_layouts(tab))}


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_apply_sets_every_bound_control_to_the_chosen_preset(webui, tab):
    from modules.anima_presets import PRESET_CONFLICT_CONTROLS, SAFE_BASE_AESTHETIC_PRESET, TURBO_NAG_LABEL, TURBO_NAG_OVERRIDES, resolved_preset

    patch, script = _apply(webui, tab, TURBO_NAG_LABEL)
    bound = _bound(webui, tab)
    controls = {key: entry for key, entry in patch.items() if not key.startswith("__")}
    # Native controls (core form, Sampler and Refiner scripts) and the extension panels alike
    assert set(controls) == set(bound)
    assert {"core.cfg_scale", "sampler.sampler_name", "sampler.steps", "refiner.enable", "anima_guidance_corrections.nag_enable"} <= set(controls)
    assert ("core.enable_hr" in controls, "anima_hires_guard.enabled" in controls) == ((True, True) if tab == "txt2img" else (False, False))
    values = resolved_preset(TURBO_NAG_LABEL)
    for key, field in bound.items():
        assert controls[key]["value"] == values[field], key
        # Exactly the conflict-locked controls are unlocked (Turbo NAG's Euler locks
        # none of them again); the others keep their state
        assert controls[key].get("enabled") is (True if field in PRESET_CONFLICT_CONTROLS else None), key
    assert controls["anima_guidance_corrections.nag_enable"]["value"] is True and controls["core.cfg_scale"]["value"] == 1.0
    assert set(TURBO_NAG_OVERRIDES) <= set(SAFE_BASE_AESTHETIC_PRESET)
    assert patch["__toast__"] == {"level": "success", "message": f"Applied: {TURBO_NAG_LABEL} ({len(bound)} controls)"}
    assert not any(key.split(".")[0] == script.script_id for key in controls)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_the_ported_panels_in_a_tab(webui, tab):
    """The group's panels in a real runner: the tab schema, named values to
    their positional slots, and the core conflict action over the real
    Sampler script."""

    import json

    from remi.core.conflicts import conflict_patch
    from remi.core.tabs import tab_schema

    runner = getattr(webui, tab)
    schema = json.loads(json.dumps(tab_schema(tab), allow_nan=False))
    entries = {entry["id"]: entry for entry in schema["scripts"]}
    expected = {"anima_guidance_corrections", "anima_remi_presets", "anima_prompt_schedule_rescaler"} | ({"anima_hires_guard"} if tab == "txt2img" else set())
    assert expected <= set(entries) and not any(entries[owner]["legacy"] for owner in expected)
    assert ("anima_hires_guard" in entries) is (tab == "txt2img")
    assert entries["anima_guidance_corrections"]["args"] == list(GUIDANCE.SCRIPT_ARG_NAMES)
    assert {"anima_guidance_corrections.guidance_mode", "anima_guidance_corrections.momentum_enable", "anima_guidance_corrections.dcw_enable"} <= set(schema["conflict_params"])

    guidance = runner.by_id["anima_guidance_corrections"]
    args = runner.script_args_from_values(None, {"anima_guidance_corrections": {"enable": True, "nag_enable": "true", "nag_scale": "3.5"}})
    named = vars(GUIDANCE.AnimaGuidanceScript._arguments(args[guidance.args_from : guidance.args_to]))
    assert (named["enable"], named["nag_enable"], named["nag_scale"], named["guidance_mode"]) == (True, True, 3.5, GUIDANCE.STANDARD_GUIDANCE)
    assert runner.script_values_from_args(args)[1]["anima_guidance_corrections"]["nag_scale"] == 3.5

    patch = conflict_patch(tab, {"sampler": {"sampler_name": "Heun"}, "anima_guidance_corrections": {"momentum_enable": True}}, "anima_guidance_corrections.momentum_enable")
    assert patch["anima_guidance_corrections.momentum_enable"] == {"value": False, "enabled": False}


def test_apply_falls_back_to_the_default_preset_for_unknown_labels(webui):
    from modules.anima_presets import SAFE_BASE_AESTHETIC_LABEL

    default, _ = _apply(webui, "txt2img", SAFE_BASE_AESTHETIC_LABEL)
    for label in ("", None, "Removed preset"):
        assert _apply(webui, "txt2img", label)[0] == default, label
    assert default["sampler.sampler_name"]["value"] == "ER SDE" and default["sampler.steps"]["value"] == 36 and default["core.cfg_scale"]["value"] == 4.5


def test_apply_resolves_the_conflict_locks_again(webui):
    """The legacy outputs fired the sampler/mode/Momentum/DCW ``.change``
    handlers, which locked the controls again for the new values: the patch
    unlocks only what the rules leave editable and restores the mode choices."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_LABEL
    from remi.core.actions import core_actions
    from remi.core.conflicts import conflict_patch

    guidance = "anima_guidance_corrections"
    # Momentum is not supported on the default preset's ER SDE sampler
    patch, _ = _apply(webui, "txt2img", SAFE_BASE_AESTHETIC_LABEL)
    assert patch["sampler.sampler_name"]["value"] == "ER SDE"
    assert patch[f"{guidance}.momentum_enable"] == {"value": False, "enabled": False}
    assert patch[f"{guidance}.dcw_enable"] == {"value": False, "enabled": True}

    # DCW on had narrowed the mode choices (no FDG); the preset turns DCW off
    narrowed = conflict_patch("txt2img", {"sampler": {"sampler_name": "Euler"}, guidance: {"dcw_enable": True}})
    full = [choice["value"] for choice in conflict_patch("txt2img", {"sampler": {"sampler_name": "Euler"}})[f"{guidance}.guidance_mode"]["choices"]]
    assert "FDG (experimental)" in full and "FDG (experimental)" not in [choice["value"] for choice in narrowed[f"{guidance}.guidance_mode"]["choices"]]
    scripts = {"sampler": {"sampler_name": "Euler"}, guidance: {"dcw_enable": True, "guidance_mode": "APG"}}
    patch, _ = _apply(webui, "txt2img", SAFE_BASE_AESTHETIC_LABEL, scripts=scripts)
    assert [choice["value"] for choice in patch[f"{guidance}.guidance_mode"]["choices"]] == full
    assert patch[f"{guidance}.dcw_enable"]["value"] is False

    # The core __preset__ action resolves the locks the same way
    ctx = ui.ActionContext(tab="txt2img", scripts=scripts, extra={"label": SAFE_BASE_AESTHETIC_LABEL})
    core_patch = ui.call_action(core_actions, "__preset__", ctx, {})
    assert core_patch[f"{guidance}.momentum_enable"] == {"value": False, "enabled": False}
    assert [choice["value"] for choice in core_patch[f"{guidance}.guidance_mode"]["choices"]] == full


def test_gradio_free():
    harness.assert_gradio_free(ROOT, ("scripts/anima_remi_presets.py",))


def test_zh_cn_catalog_covers_the_panel():
    harness.assert_localized(ROOT, [harness.script_layout(SCRIPT_MODULE.AnimaRemiPresetScript(), tab) for tab in ("txt2img", "img2img")])
