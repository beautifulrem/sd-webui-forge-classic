"""The template editor keeps built-in templates and other custom ones intact,
and hand-edited templates (templates.json) are bounded before a run reads them.

Each test has its own template store (conftest.py)."""

import pytest

from lib_stage_scheduler import options, templates, ui


def _store(**entries):
    templates._TEMPLATES.edit(lambda data: data.update(entries))


def _stored():
    return templates._TEMPLATES.load()


def _status(updates):
    return updates[-1]


BUILTIN = options.PRESET_STYLE_ALLOW
MINE = {"auto": True, "stage": options.STAGE_STYLE}
OTHER = {"auto": False, "stage": options.STAGE_COMPOSITION, "start": 0.1, "end": 0.3}


@pytest.mark.parametrize(
    "handler, args, status",
    [
        (ui._rename_template_ui, ("mine", "other"), "rename_taken"),
        (ui._rename_template_ui, ("mine", BUILTIN), "builtin_name_taken"),
        (ui._rename_template_ui, (BUILTIN, "renamed"), "builtin_not_renamed"),
        (ui._rename_template_ui, ("mine", " "), "empty_new_name"),
        (ui._rename_template_ui, ("gone", "renamed"), "rename_missing"),
        (ui._save_template_ui, (BUILTIN, True, "mine", 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH), "builtin_template_kept"),
        (ui._save_template_ui, (" ", True, "mine", 0.0, 1.0, 0.5, 1.0, options.CURVE_SMOOTH), "empty_template_name"),
        (ui._save_preset_combo_template_ui, (BUILTIN, [options.PRESET_STYLE_ALLOW], 3.0), "builtin_preset_kept"),
        (ui._save_preset_combo_template_ui, ("combo", [], 3.0), "no_presets"),
        (ui._delete_template_ui, (BUILTIN,), "builtin_not_deleted"),
    ],
)
def test_refused_edits_leave_every_template_as_it_was(handler, args, status):
    _store(mine=MINE, other=OTHER)
    before = _stored()

    assert _status(handler(*args, "mine", BUILTIN)) == options._t(status)
    assert _stored() == before
    assert templates._template_config(BUILTIN) == options.BUILTIN_TEMPLATES[BUILTIN]


def test_rename_and_delete_change_only_their_own_template():
    _store(mine=MINE, other=OTHER)

    seeded = {name: value for name, value in _stored().items() if name not in ("mine", "other")}  # the bundled templates.json

    assert _status(ui._rename_template_ui("mine", "renamed", "mine", BUILTIN)) == options._t("template_renamed").format(old="mine", new="renamed")
    assert _stored() == {**seeded, "renamed": MINE, "other": OTHER}
    ui._delete_template_ui("renamed", "renamed", BUILTIN)
    assert _stored() == {**seeded, "other": OTHER}


def test_saving_bounds_the_panel_values_and_keeps_the_source_templates_parts():
    parts = [{"stage": options.STAGE_STYLE, "mode": options.PART_BAN}, {"stage": options.STAGE_COMPOSITION, "mode": options.PART_ALLOW, "strength": 9}]
    _store(source={"auto": False, "stage": options.STAGE_COMPOSITION, "parts": parts})

    ui._save_template_ui("copy", False, "source", -1, 2, 0.5, 9, "not a curve", "source", BUILTIN)

    saved = _stored()["copy"]
    assert (saved["start"], saved["end"], saved["peak"], saved["strength"], saved["curve"]) == (0.0, 1.0, 0.5, 4.0, options.CURVE_SMOOTH)
    assert saved["stage"] == options.STAGE_COMPOSITION
    assert saved["parts"] == templates._template_parts({"parts": parts})
    assert [part["strength"] for part in saved["parts"]] == [1.0, 4.0]


def test_a_hand_edited_template_is_bounded_or_ignored():
    _store(
        loud={"auto": False, "start": 0.9, "end": 0.1, "peak": 2, "strength": 100, "parts": [{"stage": options.STAGE_STYLE, "mode": "x"}, {"stage": options.STAGE_CHARACTER, "strength": 100}]},
        broken=["not", "an", "object"],
    )

    # The unknown mode's part is dropped, the other part's strength capped at 4x
    (part,) = templates._template_parts(templates._template_config("loud"))
    assert (part["stage"], part["mode"], part["strength"]) == (options.STAGE_CHARACTER, options.PART_ALLOW, 4.0)
    assert templates._template_config("broken") == {}
    auto, start, end, peak, strength, curve = ui._apply_template_ui("loud", 3.0)
    # start/end swapped back into order, peak inside them, strength at its 4x cap
    assert (start["value"], end["value"], peak["value"], strength["value"]) == (0.1, 0.9, 0.9, 4.0)
    assert auto["value"] is False and curve["value"] in options.CURVE_CHOICES


def test_per_lora_templates_take_every_separator_and_skip_unknown_templates(caplog):
    lines = [f"a=>{BUILTIN}", f"b＝{BUILTIN}", f"c->{BUILTIN}", f"d={BUILTIN}", "e=no such template", "# f=ignored", "no separator"]

    assert templates._parse_lora_template_overrides("\n".join(lines)) == dict.fromkeys("abcd", BUILTIN)
    assert "no such template" in caplog.text and "no separator" in caplog.text


def test_the_builder_offers_a_custom_templates_own_name_only():
    _store(mine=MINE)

    assert ui._open_template_builder_ui(BUILTIN, 3.0)[3]["value"] == ""
    assert ui._open_template_builder_ui("mine", 3.0)[3]["value"] == "mine"


def test_help_and_preview_describe_bans_and_fall_back_to_the_character_template():
    unknown = ui._template_help_ui("no such template", 3.0)["value"]
    assert unknown == ui._template_help_ui(options.PRESET_CHARACTER_ALLOW, 3.0)["value"].replace(options.PRESET_CHARACTER_ALLOW, "no such template")
    preview = ui._preset_combo_preview_text([options.PRESET_STYLE_BAN, options.PRESET_CHARACTER_ALLOW], 3.0)
    assert options._t("zero_in_window") in preview and options._t("ban_action") in preview
