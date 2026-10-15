"""The ported panel (``params()``) against the Gradio ``ui()`` it replaces
(tests/fixtures/legacy_ui_<tab>.json, recorded before the port): the 640
row controls are two artist-mix-table values, which a run reads into the same
processing arguments. Also the schema, actions and the Gradio-free load."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from test_runtime_dispatch import _run_p, mixer, runtime

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "anima_artist_scheduled_mixer.py"
_spec = importlib.util.spec_from_file_location("remi_anima_lora_harness", ROOT.parents[1] / "tests" / "remi_anima_lora_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def layout_of(tab="txt2img"):
    return harness.script_layout(mixer.Script(), tab)


# Choices take the internal values (the Gradio panel's values were the labels
# of the UI language); combine / fusion bind their preset by that value.
CHANGES = {
    "combine_mode": {"default": mixer.COMBINE_OUTPUT_AVG, "choices": [mixer.COMBINE_OUTPUT_AVG, mixer.COMBINE_CONCAT], "infotext": None},
    "fusion_mode": {"default": mixer.FUSION_INTERPOLATE, "choices": [mixer.FUSION_INTERPOLATE, mixer.FUSION_CONCAT_WITH_BASE, mixer.FUSION_QUALITY_DELTA], "infotext": None},
    # One infotext key per param: the options key names the enable switch,
    # the others paste with it
    **{name: {"infotext": None} for name in ("hires_independent", "disable_hires_mixing", "global_strength", "optimization", "apply_uncond")},
}


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(tab):
    layout = layout_of(tab)
    assert layout.names == list(mixer.PARAM_NAMES)
    legacy = dict(zip(mixer.SCRIPT_ARG_NAMES, harness.legacy_snapshot(ROOT, tab)["inputs"], strict=True))
    ported = {entry["name"]: entry for entry in harness.describe(layout)}
    harness.assert_entries([ported[name] for name in mixer.COMPACT_HEADER_ARGS], [legacy[name] for name in mixer.COMPACT_HEADER_ARGS], CHANGES)

    # Each chain's 320 row controls are one table, starting on the rows the
    # Gradio panel showed (its row count) with their values
    for chain in mixer.ROW_CHAINS:
        table = ported[f"{chain}_rows"]
        count = legacy[f"{chain}_row_count"]
        assert table["kind"] == "custom" and table["widget"] == "artist-mix-table"
        assert table["infotext"] == count["infotext"] and table["paste"]
        rows = [[legacy[f"{chain}_{key}_{index}"]["default"] for key in mixer.ROW_KEYS] for index in range(count["default"])]
        assert mixer._rows_of_table(table["default"]) == [_internal(row) for row in rows]
        assert all(legacy[f"{chain}_{key}_{index}"].get("visible") is False for key in mixer.ROW_KEYS for index in range(count["default"], mixer.MAX_ARTIST_ROWS))
    # The Shift copies are not params: a run reads p's Shifts
    assert {legacy[name]["visible"] for name in ("runtime_base_shift", "runtime_hires_shift")} == {False}
    # Added: the template controls, which the run ignores
    assert [ported[name]["kind"] for name in mixer.TEMPLATE_ARGS] == ["choice", "choice", "text", "text"]


def _internal(row):
    """A row of the Gradio panel (labels, Yes/No) in internal values."""
    return mixer._normalize_rows([row], 1, display=False)[0]


def test_the_schema_exports_two_tables_instead_of_640_controls():
    schema = harness.schema_json(layout_of())
    text = json.dumps(schema, ensure_ascii=False)
    assert text.count('"widget": "artist-mix-table"') == 2 and len(text) < 20000
    assert schema["enable"]["exclusive"] == {"group": "anima_spatial_conditioning", "member": "artist"}
    table = _find(schema, "hires_rows")
    assert table["visible_if"] == {"param": "hires_independent"}
    assert table["props"]["shift"] == ["core.hr_distilled_cfg", "core.distilled_cfg_scale"] and table["props"]["max_rows"] == 32
    assert table["props"]["default_blocks"] == {"Performance": "10-18", "Balance": "0-51", "Quality": "0-51"}
    # A user's choice only (Gradio's .input): a pasted infotext brings its own Blocks
    optimization = _find(schema, "optimization")
    assert optimization["on_input"] == "optimization_changed" and "on_change" not in optimization


def test_presets_bind_the_global_controls_by_value():
    from remi import ui

    bindings = harness.preset_bindings(layout_of())
    assert bindings == {
        "artist.enabled": ("enable", False),
        "artist.global_strength": ("global_strength", 0.7),
        "artist.optimization": ("optimization", mixer.OPT_BALANCE),
        "artist.combine": ("combine_mode", mixer.COMBINE_OUTPUT_AVG),
        "artist.fusion": ("fusion_mode", mixer.FUSION_INTERPOLATE),
        "artist.apply_uncond": ("apply_uncond", False),
        "artist.cache": ("enable_cache", True),
    }
    layout = layout_of()
    for _field, (name, value) in bindings.items():
        assert layout.by_name[name].coerce(value, ui.Context()) == value


def test_a_table_value_is_validated():
    from remi import ui

    layout = layout_of()
    rows = layout.coerce({"base_rows": [[True, "a"], {"Artist": "b", "Auto Shift": "Yes"}]})["base_rows"]
    assert [row["artist"] for row in rows] == ["a", "b"] and rows[1]["auto_shift"] == "Yes"
    for bad in ("text", [1, 2], [{"artist": "x"}] * 33):
        with pytest.raises(ui.ValidationError):
            layout.coerce({"base_rows": bad})


def _legacy_args(values):
    """The Gradio panel's 653 arguments for the named ``values``."""
    head = {
        **values,
        "base_row_count": len(values["base_rows"]),
        "hires_row_count": len(values["hires_rows"]),
        "runtime_base_shift": 5.0,
        "runtime_hires_shift": 3.0,
    }
    rows = []
    for chain in mixer.ROW_CHAINS:
        table = mixer._rows_of_table(values[f"{chain}_rows"])
        rows += [value for index in range(mixer.MAX_ARTIST_ROWS) for value in (table[index] if index < len(table) else [None] * mixer.ROW_WIDTH)]
    return [head[name] for name in mixer.HEADER_ARGS] + rows


@pytest.mark.parametrize("hires", [True, False])
def test_the_tables_run_exactly_as_the_640_row_arguments(monkeypatch, hires):
    """The same panel given as params() arguments and as the Gradio panel's
    positional arguments: equal run arguments, rows, infotext and NegPiP
    prompts: the port processes exactly what the Gradio panel did."""
    from test_infotext import _args, _panel

    values = _panel(hires_independent=hires)
    p = _run_p(enable_hr=True, distilled_cfg_scale=5.0, hr_distilled_cfg=3.0)
    compact = mixer._MixerArgs.from_args(_args(values), p)
    legacy = mixer._MixerArgs.from_args(_legacy_args(values), p)
    assert compact == legacy
    for second_pass in (False, True):
        assert mixer._resolve_run_rows(p, compact, second_pass, report=False)[2:] == mixer._resolve_run_rows(p, legacy, second_pass, report=False)[2:]
    assert mixer.generation_params(p, compact) == mixer.generation_params(p, legacy)

    from modules.anima_support import NEGPIP_EXTRA_PROMPTS

    registered = []
    for args in (_args(values), _legacy_args(values)):
        run = _run_p(enable_hr=True, distilled_cfg_scale=5.0, hr_distilled_cfg=3.0)
        mixer.Script().process(run, *args)
        registered.append(getattr(run, NEGPIP_EXTRA_PROMPTS)[mixer.ARTIST_MIXER])
    assert registered[0] == registered[1] and registered[0][:2] == ['artist "quoted"', "米山舞"]


def test_the_template_controls_are_optional_trailing_arguments():
    from test_infotext import _args

    args = _args()
    assert mixer._MixerArgs.from_args(args[: -len(mixer.TEMPLATE_ARGS)]) == mixer._MixerArgs.from_args(args)
    for count in (5, len(args) - 1, len(args) + 1):
        with pytest.raises(TypeError, match=f"takes {len(mixer.PARAM_NAMES)} script arguments, got {count}"):
            mixer._MixerArgs.from_args((args * 2)[:count])


@pytest.mark.parametrize("tab", harness.TABS)
def test_the_gradio_layout_cut_to_the_api_slots_is_refused(tab):
    """The API gives the script its params() slots only: a request built for
    the Gradio panel arrives cut to them and must fail, not run misread."""
    legacy = [entry["default"] for entry in harness.legacy_snapshot(ROOT, tab)["inputs"]]
    with pytest.raises(TypeError, match=r"argument 9 \(base_rows\) is a list of artist rows, got str"):
        mixer._MixerArgs.from_args(legacy[: len(mixer.PARAM_NAMES)])


def test_a_template_deleted_from_the_other_tab_does_not_block_a_job(monkeypatch):
    """Templates are shared by both tabs; a form still showing a deleted one
    submits it. The run ignores the dropdown: the name reads as the first
    template, or None when there is none left."""
    monkeypatch.setattr(mixer, "_template_choices", lambda: [])
    assert layout_of().coerce({"template": "a"})["template"] is None
    monkeypatch.setattr(mixer, "_template_choices", lambda: ["b", "c"])
    layout = layout_of()
    assert layout.coerce({"template": "a"})["template"] == "b"
    assert layout.coerce({"template": "c"})["template"] == "c"


def test_choosing_an_optimization_preset_rewrites_the_blocks():
    from remi import ui

    values = layout_of().defaults()
    values["optimization"] = mixer.OPT_PERFORMANCE
    values["base_rows"][0].update(stage=mixer.PRESET_STYLE, auto_shift=True)
    ctx = ui.ActionContext(tab="txt2img", core={"distilled_cfg_scale": 5.0, "hr_distilled_cfg": 2.0})
    patch = ui.call_action(mixer.Script(), "optimization_changed", ctx, values)
    base, hires = patch["base_rows"]["value"], patch["hires_rows"]["value"]
    assert {row["blocks"] for row in base + hires} == {"10-18"}
    assert (base[0]["start"], base[0]["end"]) == tuple(round(value, 4) for value in mixer._auto_stage_values(mixer.PRESET_STYLE, 5.0)[:2])
    assert base[1]["start"] == 0.0  # a custom row keeps its window


def test_pasting_keeps_the_recorded_blocks():
    """The SPA paste runs the pasted params' on_change actions: none of them
    may put the pasted rows back on the optimization preset's Blocks."""
    from test_infotext import _infotext, _panel, _recorded

    params = _recorded(_panel())
    mixer._paste_plan.cache_clear()
    values = harness.spa_paste(mixer.Script(), layout_of(), _infotext(params), {"optimization": mixer.OPT_PERFORMANCE}, core={"distilled_cfg_scale": 5.0, "hr_distilled_cfg": 3.0})
    assert values["optimization"] == mixer.OPT_QUALITY
    assert [row["blocks"] for row in values["base_rows"]] == ["0,3,5-12", "10-18"]
    assert [row["blocks"] for row in values["hires_rows"]] == ["0-51", "-1"]


def test_the_extension_is_gradio_free():
    assert harness.gradio_imports(ROOT) == []
    harness.assert_loads_without_gradio(SCRIPT)


def test_the_zh_cn_catalog_covers_every_string():
    catalog = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf8"))
    # The action toasts too (the SPA translates them, placeholders and all)
    strings = _strings(harness.schema_json(layout_of())) | {mixer.Script().title()} | set(mixer.TEXT.values())
    missing = sorted(text for text in strings if text not in catalog and not re.search("[　-鿿]", text))
    assert missing == []


def _find(node, name):
    if isinstance(node, dict):
        if node.get("name") == name:
            return node
        for key in ("children", "enable"):
            found = _find(node.get(key), name)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find(item, name)
            if found is not None:
                return found
    return None


def _strings(node) -> set[str]:
    """Every translatable string of a schema tree (the widgets' props included)."""

    found: set[str] = set()
    if isinstance(node, dict):
        for key in ("label", "info", "title", "text", "placeholder"):
            if isinstance(node.get(key), str) and node[key] and node.get("type") != "html":
                found.add(node[key])
        for choice in node.get("choices", ()):
            if isinstance(choice, dict) and isinstance(choice.get("label"), str):
                found.add(choice["label"])
        for key in ("children", "enable", "props", "curves", "stages"):
            if key in node:
                found |= _strings(node[key])
    elif isinstance(node, list):
        for item in node:
            found |= _strings(item)
    return found
