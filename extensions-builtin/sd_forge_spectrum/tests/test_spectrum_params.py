"""The ported panel: params() declares what the Gradio ui() did, plus the
Presets box's name (a trailing argument processing ignores; Gradio kept it
out of the script arguments). Apply / Save / Delete are actions."""

import json
import pathlib
import sys

import pytest
import remi_port_harness as harness

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.presets import DEFAULTS, INFOTEXT_KEYS, PresetManager  # noqa: E402

SCRIPT = ROOT / "scripts" / "spectrum.py"
PRESET_NAME = {15: {"kind": "choice", "label": "Preset Name", "default": "", "choices": [], "infotext": None, "paste": False, "preset": None}}
SETTINGS = [0.5, 3, 0.2, 3, 0.1, 4, 0.8, 2, 12, "SEA (auto-calibrated)", 0.3, 3.0, "Legacy / fastest", True]


def _load():
    PresetManager.presets = {}  # the import must not create presets.json
    return harness.load_script(SCRIPT, "spectrum_params_under_test")


spectrum = _load()


@pytest.fixture(autouse=True)
def presets(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(PresetManager, "presets", saved)
    monkeypatch.setattr("lib_spectrum.presets.PRESET_FILE", str(tmp_path / "presets.json"))
    return saved


def _call(action, values):
    from remi import ui

    script = spectrum.SpectrumForForge()
    return ui.call_action(script, action, ui.ActionContext(tab="txt2img", owner="spectrum_integrated"), values)


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(tab):
    layout = harness.script_layout(spectrum.SpectrumForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab), PRESET_NAME)


def test_arguments_are_the_script_argument_table():
    layout = harness.script_layout(spectrum.SpectrumForForge())
    assert layout.names == ["enable", *spectrum.PARAM_NAMES, "preset_name"]
    assert [layout.by_name[name].infotext for name in spectrum.PARAM_NAMES] == list(INFOTEXT_KEYS)
    assert layout.values_to_args({}) == [False, *DEFAULTS, ""]


def test_save_stores_the_settings_and_selects_the_name(presets):
    values = {"preset_name": " fast ", **dict(zip(spectrum.PARAM_NAMES, SETTINGS))}
    patch = _call("save_preset", values)
    assert presets == {"fast": SETTINGS}
    assert patch["preset_name"] == {"choices": [{"value": "fast", "label": "fast"}], "value": "fast"}
    assert patch["__toast__"]["level"] == "success"


def test_save_validates_the_settings_first(presets):
    from remi import ui

    with pytest.raises(ui.ValidationError):
        _call("save_preset", {"preset_name": "bad", "degree": "many"})
    # Out-of-range sliders are clamped, as on submit
    _call("save_preset", {"preset_name": "clamped", "degree": 99})
    assert presets["clamped"][1] == 8


def test_save_needs_a_name(presets):
    patch = _call("save_preset", {"preset_name": "  "})
    assert presets == {} and patch["__toast__"]["level"] == "error" and "preset_name" not in patch


def test_apply_sets_every_setting(presets):
    presets["mine"] = SETTINGS
    patch = _call("apply_preset", {"preset_name": "mine"})
    assert patch == {name: {"value": value} for name, value in zip(spectrum.PARAM_NAMES, SETTINGS)}
    assert "enable" not in patch  # presets store the settings, never the switch


def test_apply_pads_an_old_preset_and_reports_an_unknown_one(presets):
    presets["old"] = SETTINGS[:7]
    patch = _call("apply_preset", {"preset_name": "old"})
    assert [patch[name]["value"] for name in spectrum.PARAM_NAMES] == [*SETTINGS[:7], *DEFAULTS[7:]]
    assert set(_call("apply_preset", {"preset_name": "nope"})) == {"__toast__"}


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ([0.5, "x"], None),  # does not convert
        ({"degree": 3}, None),  # not a list
        ([*SETTINGS[:9], "Nightly", *SETTINGS[10:]], None),  # unknown schedule
        ([9.0, 99, *SETTINGS[2:]], [1.0, 8, *SETTINGS[2:]]),  # clamped, as on submit
    ],
)
def test_apply_checks_a_hand_edited_preset(presets, stored, expected):
    presets["edited"] = stored
    patch = _call("apply_preset", {"preset_name": "edited"})
    if expected is None:
        assert set(patch) == {"__toast__"} and patch["__toast__"]["level"] == "error"
    else:
        assert [patch[name]["value"] for name in spectrum.PARAM_NAMES] == expected


def test_concurrent_saves_keep_every_preset(presets, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    names = [f"p{index}" for index in range(32)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda name: PresetManager.save_preset(name, *SETTINGS), names))
    assert sorted(presets) == sorted(names)
    assert sorted(json.loads((tmp_path / "presets.json").read_text(encoding="utf-8"))) == sorted(names)


def test_delete_removes_the_preset_and_clears_the_name(presets):
    presets.update(a=SETTINGS, b=SETTINGS)
    patch = _call("delete_preset", {"preset_name": "a"})
    assert list(presets) == ["b"]
    assert patch["preset_name"] == {"choices": [{"value": "b", "label": "b"}], "value": ""}
    assert set(_call("delete_preset", {"preset_name": "a"})) == {"__toast__"}


def test_the_name_choices_follow_the_saved_presets(presets):
    from remi import ui

    presets["x"] = SETTINGS
    layout = harness.script_layout(spectrum.SpectrumForForge())
    node = layout.by_name["preset_name"]
    assert node.values(ui.Context()) == ["x"] and node.allow_custom
    assert layout.values_to_args({"preset_name": "a new name"})[-1] == "a new name"


def test_schema_export():
    schema = harness.script_layout(spectrum.SpectrumForForge()).schema()
    text = json.dumps(schema, allow_nan=False)
    groups = [child for child in schema["children"] if child["type"] == "group"]
    assert [group["title"] for group in groups] == ["Schedule & compatibility", "Presets"] and not any(group["open"] for group in groups)
    buttons = groups[1]["children"][1]["children"]
    assert [(button["label"], button["action"], button["variant"]) for button in buttons] == [("Apply Preset", "apply_preset", "secondary"), ("Save Preset", "save_preset", "primary"), ("Delete Preset", "delete_preset", "danger")]
    assert '"spec_enabled"' in text


def test_zh_cn_covers_the_panel():
    schemas = [harness.script_layout(spectrum.SpectrumForForge(), tab).schema() for tab in harness.TABS]
    harness.assert_localized(ROOT, schemas, extra=["Enter a preset name"])


def test_imports_without_gradio():
    harness.assert_gradio_free([SCRIPT])
