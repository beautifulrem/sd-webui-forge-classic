"""The template editor keeps built-in templates and other custom ones intact,
and hand-edited templates (templates.json) are bounded before a run reads them.

Each test has its own template store (conftest.py)."""

import pytest

from lib_stage_scheduler import options, templates, ui


def _store(**entries):
    templates._TEMPLATES.edit(lambda data: data.update(entries))


def _stored():
    return templates._TEMPLATES.load()


BUILTIN = options.PRESET_STYLE_ALLOW
MINE = {"auto": True, "stage": options.STAGE_STYLE}
OTHER = {"auto": False, "stage": options.STAGE_COMPOSITION, "start": 0.1, "end": 0.3}


TIMING = {"auto": True, "start": 0.0, "end": 1.0, "peak": 0.5, "strength": 1.0, "curve": options.CURVE_SMOOTH}


@pytest.mark.parametrize(
    "handler, args, status",
    [
        (ui.rename_template, ("mine", "other"), "rename_taken"),
        (ui.rename_template, ("mine", BUILTIN), "builtin_name_taken"),
        (ui.rename_template, (BUILTIN, "renamed"), "builtin_not_renamed"),
        (ui.rename_template, ("mine", " "), "empty_new_name"),
        (ui.rename_template, ("gone", "renamed"), "rename_missing"),
        (ui.save_template, (BUILTIN, True, "mine", TIMING), "builtin_template_kept"),
        (ui.save_template, (" ", True, "mine", TIMING), "empty_template_name"),
        (ui.save_preset_combo_template, (BUILTIN, [options.PRESET_STYLE_ALLOW]), "builtin_preset_kept"),
        (ui.save_preset_combo_template, ("combo", []), "no_presets"),
        (ui.delete_template, (BUILTIN,), "builtin_not_deleted"),
    ],
)
def test_refused_edits_leave_every_template_as_it_was(handler, args, status):
    _store(mine=MINE, other=OTHER)
    before = _stored()

    edit = handler(*args)
    assert edit.status == options._t(status) and not edit.ok and edit.renamed is None
    assert _stored() == before
    assert templates._template_config(BUILTIN) == options.BUILTIN_TEMPLATES[BUILTIN]


def test_rename_and_delete_change_only_their_own_template():
    _store(mine=MINE, other=OTHER)

    seeded = {name: value for name, value in _stored().items() if name not in ("mine", "other")}  # the bundled templates.json

    edit = ui.rename_template("mine", "renamed")
    assert (edit.status, edit.ok, edit.renamed) == (options._msg("template_renamed", old="mine", new="renamed"), True, ("mine", "renamed"))
    assert _stored() == {**seeded, "renamed": MINE, "other": OTHER}
    ui.delete_template("renamed")
    assert _stored() == {**seeded, "other": OTHER}


def test_saving_bounds_the_panel_values_and_keeps_the_source_templates_parts():
    parts = [{"stage": options.STAGE_STYLE, "mode": options.PART_BAN}, {"stage": options.STAGE_COMPOSITION, "mode": options.PART_ALLOW, "strength": 9}]
    _store(source={"auto": False, "stage": options.STAGE_COMPOSITION, "parts": parts})

    ui.save_template("copy", False, "source", {"start": -1, "end": 2, "peak": 0.5, "strength": 9, "curve": "not a curve"})

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
    timing = ui.apply_template("loud", 3.0)
    # start/end swapped back into order, peak inside them, strength at its 4x cap
    assert (timing["start"], timing["end"], timing["peak"], timing["strength"]) == (0.1, 0.9, 0.9, 4.0)
    assert timing["auto"] is False and timing["curve"] in options.CURVE_CHOICES


def test_per_lora_templates_take_every_separator_and_skip_unknown_templates(caplog):
    lines = [f"a=>{BUILTIN}", f"b＝{BUILTIN}", f"c->{BUILTIN}", f"d={BUILTIN}", "e=no such template", "# f=ignored", "no separator"]

    assert templates._parse_lora_template_overrides("\n".join(lines)) == dict.fromkeys("abcd", BUILTIN)
    assert "no such template" in caplog.text and "no separator" in caplog.text


def test_the_timeline_lists_the_builders_presets_and_every_template():
    from lib_stage_scheduler import params

    _store(mine=MINE)
    props = params.timeline_props("hr")
    assert [preset["value"] for preset in props["builder"]] == list(options.PRESET_PARTS)
    assert all(preset["label"] == options.TEMPLATE_LABELS_EN[preset["value"]] for preset in props["builder"])
    assert list(props["templates"]) == templates._template_names() and "mine" in props["templates"]
    assert props["shift"] == ["core.hr_distilled_cfg", "core.distilled_cfg_scale"] and props["template_param"] == "hr_template"
    assert [curve["value"] for curve in props["curves"]] == options.CURVE_CHOICES
