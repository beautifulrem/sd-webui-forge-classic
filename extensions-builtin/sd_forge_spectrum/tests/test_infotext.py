"""A Spectrum run's infotext turns Spectrum back on with the same settings,
through the UI paste ("Send to" pastes the same fields) and the API's infotext
replay."""

import ast
import importlib.util
import json
import pathlib
import re
import sys
import types
from types import SimpleNamespace

import gradio as gr
import pytest
import torch

sys.argv = sys.argv[:1]
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.presets import CONTROLS, DEFAULTS, ENABLE_KEY, INFOTEXT_KEYS, PresetManager

SCRIPT = ROOT / "scripts" / "spectrum.py"
INFOTEXT_UTILS = REPO / "modules" / "infotext_utils.py"
API = REPO / "modules" / "api" / "api.py"

# Every control off its default, so a control that does not round-trip shows.
SETTINGS = {
    "spec_w": 0.5,
    "spec_m": 3,
    "spec_lam": 0.2,
    "spec_window_size": 3,
    "spec_flex_window": 0.1,
    "spec_warmup_steps": 4,
    "spec_stop_caching_step": 0.8,
    "spec_tail_actual_steps": 2,
    "spec_history_size": 12,
    "spec_schedule": "SEA (auto-calibrated)",
    "spec_refresh_ratio": 0.3,
    "spec_sea_beta": 3.0,
    "spec_compat_policy": "Legacy / fastest",
    "spec_verbose": True,
}


# --- Local paste/replay helpers (swap for tests/extension_script_harness.py's shared ones) ---


def _local_infotext_helpers() -> dict:
    """``re_param_code``, ``quote``, ``unquote``, ``_parse_info`` and
    ``PasteField`` from modules/infotext_utils.py, whose import needs the full
    WebUI install."""

    tree = ast.parse(INFOTEXT_UTILS.read_text(encoding="utf8"))
    wanted = {"re_param_code", "quote", "unquote", "_parse_info", "PasteField"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in wanted)
    ]
    namespace = {"json": json, "gr": gr, "Any": object, "type_of_gr_update": type(gr.skip()), "errors": SimpleNamespace(report=print)}
    exec(compile(ast.Module(nodes, []), str(INFOTEXT_UTILS), "exec"), namespace)  # noqa: S102 - our own source file
    assert wanted <= set(namespace)
    return namespace


def _local_api_pasted_value():
    """``pasted_value`` from modules/api/api.py (same reason)."""

    tree = ast.parse(API.read_text(encoding="utf8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pasted_value")
    namespace = {}
    exec(compile(ast.Module([node], []), str(API), "exec"), namespace)  # noqa: S102 - our own source file
    return namespace["pasted_value"]


INFOTEXT = _local_infotext_helpers()
pasted_value = _local_api_pasted_value()


def _local_through_infotext(params: dict) -> dict:
    """``params`` written as processing.create_infotext writes them and read
    back as parse_generation_parameters reads them."""

    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {INFOTEXT['quote'](value)}" for key, value in params.items())
    return {key: INFOTEXT["unquote"](value) for key, value in re.findall(INFOTEXT["re_param_code"], line)}


def _local_ui_paste(script, inputs, params) -> list:
    """The script args after pasting ``params`` (connect_paste: _parse_info of
    each infotext field) over a panel at its defaults."""

    parsed = _local_through_infotext(params)
    updates = {field.component: INFOTEXT["_parse_info"](field.component, field.label, parsed) for field in script.infotext_fields}
    return [updates[c]["value"] if c in updates and "value" in updates[c] else c.value for c in inputs]


def _local_api_replay(script, inputs, params) -> list:
    """The script args modules/api/api.py apply_infotext runs ``params`` with:
    each script field's value converted by pasted_value to
    type(component.value); the other args at their defaults."""

    parsed = _local_through_infotext(params)
    args = [component.value for component in inputs]
    indexes = {component: index for index, component in enumerate(inputs)}
    for field in script.infotext_fields:
        if field.component not in indexes:
            continue
        value = field.function(parsed) if field.function else parsed.get(field.label)
        target_type = type(field.component.value)
        if value is None or target_type is type(None):
            continue
        value = pasted_value(value, target_type)
        if value is not None:
            args[indexes[field.component]] = value
    return args


# --- The script under test ---


def _load():
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = lambda component, target: (component, target)
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
    PresetManager.presets = {}  # the import must not create presets.json
    spec = importlib.util.spec_from_file_location("spectrum_script_infotext_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PasteField = INFOTEXT["PasteField"]  # the real one, with .component / .label
    return module


spectrum = _load()


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    monkeypatch.setattr(spectrum.shared, "opts", SimpleNamespace(skip_early_cond=0.0, s_min_uncond=0.0), raising=False)
    monkeypatch.setattr(spectrum.paths, "data_path", str(tmp_path))


@pytest.fixture
def panel():
    from modules.anima_presets import reset_preset_controls

    reset_preset_controls("txt2img")
    script = spectrum.SpectrumForForge()
    script.tabname = "txt2img"
    with gr.Blocks():
        inputs = script.ui(False)
    return script, inputs


def _process():
    sd_model = SimpleNamespace(forge_objects=SimpleNamespace(unet=_FakePatcher()))
    return SimpleNamespace(
        sd_model=sd_model,
        extra_generation_params={},
        is_hr_pass=False,
        steps=20,
        sampler_name="Euler",
        scheduler="Simple",
        cfg_scale=4.0,
        distilled_cfg_scale=3.0,
    )


class _FakePatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _record(enable, *args) -> dict:
    """The infotext parameters a generation with script args ``enable, *args`` records."""

    p = _process()
    script = spectrum.SpectrumForForge()
    script.process_before_every_sampling(p, enable, *args, x=torch.zeros(2, 16, 8, 8))
    script.cleanup(p)
    return p.extra_generation_params


def _settings_args():
    return [SETTINGS[key] for key in INFOTEXT_KEYS]


def test_a_run_records_the_enable_switch_with_its_settings():
    params = _record(True, *_settings_args())
    assert params[ENABLE_KEY] is True
    assert {key: params[key] for key in INFOTEXT_KEYS} == SETTINGS


def test_spectrum_off_records_nothing():
    assert _record(False, *_settings_args()) == {}


@pytest.mark.parametrize("replay", [_local_ui_paste, _local_api_replay], ids=["ui paste", "api replay"])
def test_a_recorded_run_pastes_back_enabled_with_the_same_settings(panel, replay):
    script, inputs = panel
    params = _record(True, *_settings_args())

    enable, *args = replay(script, inputs, params)

    assert enable is True
    assert dict(zip(INFOTEXT_KEYS, args)) == SETTINGS
    # Running the pasted args records the same infotext again.
    assert _record(enable, *args) == params


@pytest.mark.parametrize("replay", [_local_ui_paste, _local_api_replay], ids=["ui paste", "api replay"])
def test_infotext_without_the_enable_key_leaves_the_switch_alone(panel, replay):
    # Images from before the key: the settings paste, the switch keeps its state (default off).
    script, inputs = panel
    params = {key: SETTINGS[key] for key in INFOTEXT_KEYS}

    enable, *args = replay(script, inputs, params)

    assert enable is False
    assert dict(zip(INFOTEXT_KEYS, args)) == SETTINGS


def test_send_to_carries_the_enable_switch(panel):
    script, inputs = panel
    assert script.paste_field_names == [ENABLE_KEY, *INFOTEXT_KEYS]
    assert [field.component for field in script.infotext_fields] == inputs


def test_presets_store_the_settings_not_the_switch(panel, monkeypatch):
    _, inputs = panel
    assert ENABLE_KEY not in INFOTEXT_KEYS
    assert len(inputs) == 1 + len(CONTROLS)

    saved = {}
    monkeypatch.setattr(PresetManager, "presets", saved)
    monkeypatch.setattr("lib_spectrum.presets.PRESET_FILE", str(pathlib.Path(spectrum.paths.data_path) / "presets.json"))
    PresetManager.save_preset("mine", *_settings_args())
    assert saved["mine"] == _settings_args()
    assert [update["value"] for update in PresetManager.get_preset("mine")] == _settings_args()
    # Old presets are still padded to the settings only.
    monkeypatch.setitem(saved, "old", _settings_args()[:7])
    assert len(PresetManager.get_preset("old")) == len(DEFAULTS)
