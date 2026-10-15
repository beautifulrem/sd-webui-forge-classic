"""The panel's params() against the legacy Gradio ui() it replaces, its
schema, and the declarative compatibility wiring (conditions, conflict roles)."""

from pathlib import Path

import pytest
import remi_script_harness as harness

from modules.anima_feature_conflicts import APG_GUIDANCE, CFG_ZERO_STAR_GUIDANCE, FDG_GUIDANCE, SMC_GUIDANCE, STANDARD_GUIDANCE
from remi import ui

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = harness.load_script(ROOT / "scripts" / "anima_guidance.py", "anima_guidance_params_under_test")


def _layout(tab="txt2img"):
    return harness.script_layout(SCRIPT.AnimaGuidanceScript(), tab)


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_match_the_legacy_ui(tab):
    # Order, kinds, labels, info, defaults, bounds, choices, infotext keys and
    # preset fields are those of the Gradio panel: nothing changed.
    harness.assert_matches_legacy(_layout(tab), ROOT, tab)


def test_positional_args_are_the_processing_arguments():
    layout = _layout()
    assert tuple(layout.names) == SCRIPT.SCRIPT_ARG_NAMES
    args = layout.values_to_args({})
    # What the hooks read the positional args as
    assert vars(SCRIPT.AnimaGuidanceScript._arguments(args)) == layout.defaults()


def test_schema_export():
    schema = harness.schema_json(_layout())
    assert schema["type"] == "section" and schema["title"] == "Anima Guidance & Corrections" and schema["open"] is False
    assert schema["enable"] == {"type": "toggle", "name": "enable", "label": "Anima Guidance & Corrections", "default": False, "infotext": "Anima guidance enabled", "preset": "guidance.enabled"}

    def inputs(node):
        if "name" in node and node["type"] not in ("gallery", "output"):
            yield node
        for child in ([node["enable"]] if "enable" in node else []) + node.get("children", []):
            yield from inputs(child)

    names = [node["name"] for node in inputs(schema)]
    assert sorted(names) == sorted(SCRIPT.SCRIPT_ARG_NAMES) and len(names) == len(set(names))


def _visible(layout, name_or_text, values):
    """Whether the node (input name or markdown text) and every container
    around it are visible for ``values`` (this owner's, plus ``owner.param``)."""

    def walk(node, conditions):
        own = [*conditions, node.visible_if]
        if getattr(node, "name", None) == name_or_text or getattr(node, "text", None) == name_or_text:
            return own
        for child in node.children:
            found = walk(child, own)
            if found is not None:
                return found
        return None

    found = walk(layout.root, [])
    assert found is not None, name_or_text
    current = {**layout.defaults(), **values}
    return all(ui.evaluate(condition, current) for condition in found)


@pytest.mark.parametrize(
    ("mode", "shown"),
    [
        (STANDARD_GUIDANCE, set()),
        (SMC_GUIDANCE, {"smc_alpha", "smc_lambda"}),
        (CFG_ZERO_STAR_GUIDANCE, {"cfg_zero_init_steps"}),
        (APG_GUIDANCE, {"apg_eta", "apg_norm_threshold", "apg_momentum"}),
        (FDG_GUIDANCE, {"fdg_high_scale"}),
    ],
)
def test_mode_settings_are_shown_with_their_mode(mode, shown):
    layout = _layout()
    mode_settings = {"smc_alpha", "smc_lambda", "cfg_zero_init_steps", "apg_eta", "apg_norm_threshold", "apg_momentum", "fdg_high_scale"}
    assert {name for name in mode_settings if _visible(layout, name, {"guidance_mode": mode})} == shown


@pytest.mark.parametrize(
    ("switch", "settings"),
    [
        ("nag_enable", ("nag_scale", "nag_tau", "nag_alpha", "nag_start", "nag_end")),
        ("momentum_enable", ("momentum_strength", "momentum_ema")),
        ("guidance_range_enable", ("guidance_range_start", "guidance_range_end")),
        ("skim_enable", ("skim_scale", "full_negative", "disable_flip_filter", "start_percent", "end_percent", "flip_percent")),
        ("dcw_enable", ("dcw_lambda", "dcw_schedule", "dcw_bands")),
        ("flow_unipc_thresholding", ("flow_unipc_dynamic_thresholding_ratio", "flow_unipc_sample_max_value")),
        ("modulation_enable", ("modulation_weight", "modulation_start", "modulation_end", "modulation_base", "modulation_positive", "modulation_negative", "adapter_mode", "clip_mode")),
    ],
)
def test_switch_settings_are_shown_while_it_is_on(switch, settings):
    layout = _layout()
    assert _visible(layout, switch, {switch: False})
    for name in settings:
        assert not _visible(layout, name, {switch: False}), name
        assert _visible(layout, name, {switch: True}), name


def test_local_paths_are_shown_for_local_files_only():
    layout = _layout()
    on = {"modulation_enable": True}
    for mode, path in (("adapter_mode", "adapter_path"), ("clip_mode", "clip_path")):
        assert not _visible(layout, path, {**on, mode: "Auto-download pinned"})
        assert _visible(layout, path, {**on, mode: SCRIPT.LOCAL_FILE})


@pytest.mark.parametrize(
    ("values", "note"),
    [
        ({}, None),
        ({"guidance_mode": FDG_GUIDANCE}, SCRIPT.COMPATIBILITY_FDG),
        ({"guidance_mode": FDG_GUIDANCE, "momentum_enable": True, "dcw_enable": True}, SCRIPT.COMPATIBILITY_FDG),
        ({"momentum_enable": True}, SCRIPT.COMPATIBILITY_MOMENTUM),
        ({"momentum_enable": True, "dcw_enable": True}, SCRIPT.COMPATIBILITY_MOMENTUM),
        ({"dcw_enable": True}, SCRIPT.COMPATIBILITY_DCW),
        ({"dcw_enable": True, "guidance_mode": APG_GUIDANCE}, SCRIPT.COMPATIBILITY_DCW),
    ],
)
def test_one_compatibility_note_like_the_legacy_status(values, note):
    # The legacy status line: FDG, else Momentum, else DCW (resolution messages are the conflicts toast)
    layout = _layout()
    notes = (SCRIPT.COMPATIBILITY_FDG, SCRIPT.COMPATIBILITY_MOMENTUM, SCRIPT.COMPATIBILITY_DCW)
    assert [text for text in notes if _visible(layout, text, values)] == ([note] if note else [])


def _conflicts(values, selected=None):
    from remi.core.conflicts import conflict_patch

    layout = _layout()
    sampler = ui.ScriptLayout(ui.Choice("sampler_name", ["Euler", "Heun", "Anima ER SDE CNS"], "Euler", preset="native.sampler"), owner="sampler")
    return conflict_patch("txt2img", values, selected, layouts={"sampler": sampler, layout.owner: layout}), layout.owner


def test_the_core_conflict_action_locks_the_panel():
    patch, owner = _conflicts({"sampler": {"sampler_name": "Heun"}, "anima_guidance_corrections": {"momentum_enable": True, "dcw_enable": True}})
    assert patch[f"{owner}.momentum_enable"] == {"value": False, "enabled": False}
    assert patch[f"{owner}.dcw_enable"] == {"value": False, "enabled": False}
    assert "Momentum disabled: unsupported sampler Heun" in patch["__toast__"]["message"]

    # The last switch wins: DCW turned on resets FDG to Standard; FDG is then not offered
    patch, owner = _conflicts({"anima_guidance_corrections": {"guidance_mode": FDG_GUIDANCE, "dcw_enable": True}}, f"{owner}.dcw_enable")
    assert patch[f"{owner}.guidance_mode"]["value"] == STANDARD_GUIDANCE
    assert FDG_GUIDANCE not in [choice["value"] for choice in patch[f"{owner}.guidance_mode"]["choices"]]
    assert patch[f"{owner}.dcw_enable"] == {"value": True, "enabled": True}


def test_standalone():
    harness.assert_standalone(ROOT, ("scripts/anima_guidance.py",))


def test_zh_cn_catalog_covers_the_panel():
    harness.assert_localized(ROOT, [_layout("txt2img"), _layout("img2img")])
