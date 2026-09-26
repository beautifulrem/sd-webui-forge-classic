import pytest

from test_runtime_dispatch import mixer


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    return tmp_path


def _rows(artist):
    row = [True, artist, 1.0, "0-51", 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]
    return [value for _ in range(mixer.MAX_ARTIST_ROWS) for value in row]


def _save(name, artist="wlop"):
    return mixer._save_base_template_ui(name, 1, 3.0, False, 0.7, mixer.OPT_BALANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, True, *_rows(artist))


def test_the_template_dropdown_shows_the_templates_as_saved():
    dropdown, status = _save("mine")
    assert dropdown["choices"] == ["mine"] and dropdown["value"] == "mine"
    assert status == mixer._t("saved_base_template").format(name="mine")

    dropdown, _status = _save("other")
    assert dropdown["choices"] == ["mine", "other"] and dropdown["value"] == "other"

    dropdown, status = mixer._rename_template_ui("mine", "renamed")
    assert dropdown["choices"] == ["other", "renamed"] and dropdown["value"] == "renamed"
    assert status == mixer._t("renamed_template").format(name="renamed")

    dropdown, status = mixer._rename_template_ui("renamed", "other")
    assert dropdown["value"] == "renamed" and status == mixer._t("template_exists").format(name="other")

    dropdown, status = mixer._delete_template_ui("renamed")
    assert dropdown["choices"] == ["other"] and dropdown["value"] == "other"
    assert status == mixer._t("deleted_template").format(name="renamed")

    dropdown, status = mixer._delete_template_ui("renamed")
    assert dropdown["choices"] == ["other"] and status == mixer._t("no_template_deleted")


def test_a_saved_template_applies_its_rows():
    _save("mine", artist="wlop")
    updates = mixer._apply_template_ui("mine", mixer.APPLY_BASE, 1, 1, 3.0, 3.0, mixer.OPT_BALANCE, *_rows("other"), *_rows("other"))
    base_row_updates = updates[13 : 13 + mixer.MAX_ARTIST_ROWS * 10]
    assert updates[0]["value"] == 1  # the template's row count
    assert base_row_updates[1]["value"] == "wlop" and base_row_updates[1]["visible"]
    assert updates[12] == mixer._t("applied_template").format(name="mine", target=mixer._option_label("apply_target", mixer.APPLY_BASE))


def test_an_unreadable_template_file_is_never_overwritten():
    path = mixer._TEMPLATES.path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mine": ', encoding="utf-8")

    _dropdown, status = _save("new")

    assert status.startswith(mixer._t("template_file_unreadable").split("{")[0])
    assert path.read_text(encoding="utf-8") == '{"mine": '
