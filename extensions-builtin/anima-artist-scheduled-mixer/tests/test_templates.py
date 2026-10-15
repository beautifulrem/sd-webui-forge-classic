"""The panel's template actions: save a chain, apply, rename, delete; an
unreadable template file is never overwritten."""

import pytest

from remi import ui
from test_remi_params import layout_of
from test_runtime_dispatch import mixer


def _values(artist="wlop", **changes):
    values = layout_of().defaults()
    for chain in mixer.ROW_CHAINS:
        values[f"{chain}_rows"] = mixer._table([[True, artist, 1.0, "0-51", 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]])
    return {**values, **changes}


def _call(action, values, core=None):
    script = mixer.Script()
    script.tabname = "txt2img"
    return ui.call_action(script, action, ui.ActionContext(tab="txt2img", core=core or {"distilled_cfg_scale": 3.0}), values)


def _choices(patch):
    return [choice["value"] for choice in patch["template"]["choices"]]


def test_the_template_choices_show_the_templates_as_saved():
    patch = _call("template_save_base", _values(template_name="mine"))
    assert _choices(patch) == ["mine"] and patch["template"]["value"] == "mine"
    assert patch["__toast__"] == {"level": "success", "message": mixer._msg("saved_base_template", name="mine")}
    assert patch["template_name"] == {"value": ""}

    patch = _call("template_save_base", _values(template_name="other"))
    assert _choices(patch) == ["mine", "other"] and patch["template"]["value"] == "other"

    patch = _call("template_rename", _values(template="mine", rename_to="renamed"))
    assert _choices(patch) == ["other", "renamed"] and patch["template"]["value"] == "renamed"
    assert patch["__toast__"]["message"] == mixer._msg("renamed_template", name="renamed")

    patch = _call("template_rename", _values(template="renamed", rename_to="other"))
    assert patch["template"]["value"] == "renamed" and patch["__toast__"] == {"level": "warning", "message": mixer._msg("template_exists", name="other")}

    patch = _call("template_delete", _values(template="renamed"))
    assert _choices(patch) == ["other"] and patch["template"]["value"] == "other"
    assert patch["__toast__"]["message"] == mixer._msg("deleted_template", name="renamed")

    patch = _call("template_delete", _values(template="renamed"))
    assert _choices(patch) == ["other"] and patch["__toast__"] == {"level": "warning", "message": mixer._t("no_template_deleted")}


def test_a_hires_template_needs_the_independent_hires_chain():
    patch = _call("template_save_hires", _values(template_name="late"))
    assert patch["__toast__"] == {"level": "warning", "message": mixer._t("hires_template_disabled")} and _choices(patch) == []
    patch = _call("template_save_hires", _values(template_name="late", hires_independent=True))
    assert _choices(patch) == ["late"] and mixer._template_data()["late"]["hires_independent"] is True


def test_a_saved_template_applies_its_rows():
    assert _call("template_apply", _values(template=None))["__toast__"]["level"] == "warning"  # no templates
    _call("template_save_base", _values("wlop", template_name="mine", global_strength=1.2))
    patch = _call("template_apply", _values("other", template="mine", template_target=mixer.APPLY_BASE), {"distilled_cfg_scale": 5.0})
    assert [row["artist"] for row in patch["base_rows"]["value"]] == ["wlop"]
    assert "hires_rows" not in patch and patch["global_strength"] == {"value": 1.2}
    assert patch["__toast__"]["message"] == mixer._msg("applied_template_Base", name="mine")


def test_an_unreadable_template_file_is_never_overwritten():
    path = mixer._TEMPLATES.path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mine": ', encoding="utf-8")

    patch = _call("template_save_base", _values(template_name="new"))

    assert patch["__toast__"]["message"]["text"] == mixer._t("template_file_unreadable") and patch["__toast__"]["message"]["params"]["error"]
    assert path.read_text(encoding="utf-8") == '{"mine": '


@pytest.mark.parametrize("target, changes_hires", [(mixer.APPLY_BASE, False), (mixer.APPLY_HIRES, True), (mixer.APPLY_BOTH, True)])
def test_a_template_sets_the_hires_switches_only_where_it_applies_to_hires(target, changes_hires):
    _call("template_save_base", _values(template_name="base only"))  # saved with hires_independent off
    applied = mixer._applied_template("base only", target, _values("current", hires_independent=True), 3.0, 3.0)
    if changes_hires:
        assert applied["hires_independent"] is False and applied["disable_hires_mixing"] is False
        assert [row["artist"] for row in applied["hires_rows"]] == ["wlop"]  # a base template fills the hires rows too
    else:
        assert "hires_independent" not in applied and "disable_hires_mixing" not in applied and "hires_rows" not in applied


def test_templates_saved_with_labels_or_list_rows_still_apply():
    # Older templates: options as Chinese labels, rows with the per-row Shift column
    mixer._TEMPLATES.edit(lambda data: data.update(old={"base_rows": [[True, "wlop", 1.0, "0-27", 0.1, 0.9, 0.5, "平滑", "画风", 3.0, True]], "optimization": "性能", "combine_mode": "token拼接"}) or (None, ""))
    applied = mixer._applied_template("old", mixer.APPLY_BASE, _values(), 3.0, 3.0)
    (row,) = applied["base_rows"]
    assert (row["artist"], row["curve"], row["stage"], row["auto_shift"]) == ("wlop", mixer.CURVE_SMOOTH, mixer.PRESET_STYLE, True)
    assert (applied["optimization"], applied["combine_mode"]) == (mixer.OPT_PERFORMANCE, mixer.COMBINE_CONCAT)
