"""A Spectrum run's infotext turns Spectrum back on with the same settings.
The UI paste, "Send to" and the API's infotext replay all read it through the
panel's layout (``ScriptLayout.paste``)."""

import pathlib
import sys
from types import SimpleNamespace

import pytest
import remi_port_harness as harness
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.presets import CONTROLS, DEFAULTS, ENABLE_KEY, INFOTEXT_KEYS, PresetManager  # noqa: E402

SCRIPT = ROOT / "scripts" / "spectrum.py"

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


def _load():
    PresetManager.presets = {}  # the import must not create presets.json
    return harness.load_script(SCRIPT, "spectrum_script_infotext_under_test")


spectrum = _load()


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture(autouse=True)
def _environment(webui, monkeypatch, tmp_path):
    monkeypatch.setitem(webui.opts.data, "skip_early_cond", 0.0)
    monkeypatch.setitem(webui.opts.data, "s_min_uncond", 0.0)
    monkeypatch.setattr(spectrum.paths, "data_path", str(tmp_path))


@pytest.fixture
def layout(webui):
    return harness.script_layout(spectrum.SpectrumForForge())


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


def _replay(layout, params: dict, start: dict | None = None) -> list:
    """The script args after pasting the infotext ``params`` (as written to,
    and read back from, a parameters line) over the form values ``start``."""

    _, parsed = _through_infotext(params)
    values, problems = layout.paste(parsed)
    assert problems == []
    return layout.values_to_args({**(start or {}), **values})


def _through_infotext(params: dict) -> tuple[str, dict]:
    """The parameters line a run with ``params`` writes (processing's
    ``create_infotext``) and what the infotext parser reads back from it."""

    from modules.infotext_utils import parse_generation_parameters, quote

    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {quote(value)}" for key, value in params.items() if value is not None)
    return line, parse_generation_parameters(f"a prompt\n{line}", [])


def test_presets_bind_every_setting_and_the_switch(layout):
    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET

    fields = [name for name in SAFE_BASE_AESTHETIC_PRESET if name.startswith("spectrum.")]
    assert list(layout.presets) == fields
    defaults = layout.defaults()
    for field, (param, _own) in layout.presets.items():
        assert SAFE_BASE_AESTHETIC_PRESET[field] == defaults[param], field
    assert layout.preset_values(SAFE_BASE_AESTHETIC_PRESET)["enable"] is False  # Apply preset turns the panel off


def test_a_run_records_the_enable_switch_with_its_settings():
    params = _record(True, *_settings_args())
    assert params[ENABLE_KEY] is True
    assert {key: params[key] for key in INFOTEXT_KEYS} == SETTINGS


def test_spectrum_off_records_nothing():
    assert _record(False, *_settings_args()) == {}


def test_a_recorded_run_pastes_back_enabled_with_the_same_settings(layout):
    params = _record(True, *_settings_args())

    enable, *args, preset_name = _replay(layout, params)

    assert enable is True and preset_name == ""
    assert dict(zip(INFOTEXT_KEYS, args)) == SETTINGS
    # Running the pasted args records the same infotext again.
    assert _record(enable, *args, preset_name) == params


@pytest.mark.parametrize("switch", [False, True])
def test_infotext_without_the_enable_key_leaves_the_switch_alone(layout, switch):
    # Images from before the key: the settings paste, the switch keeps its state.
    params = {key: SETTINGS[key] for key in INFOTEXT_KEYS}

    enable, *args, _ = _replay(layout, params, start={"enable": switch})

    assert enable is switch
    assert dict(zip(INFOTEXT_KEYS, args)) == SETTINGS


def test_every_infotext_key_reads_back_into_its_setting(layout):
    assert list(layout.infotext) == [ENABLE_KEY, *INFOTEXT_KEYS]
    assert [layout.infotext[key] for key in INFOTEXT_KEYS] == list(spectrum.PARAM_NAMES)


def test_presets_store_the_settings_not_the_switch(layout, monkeypatch):
    assert ENABLE_KEY not in INFOTEXT_KEYS
    assert len(layout.inputs) == 1 + len(CONTROLS) + 1  # the switch, the settings, the preset name

    saved = {}
    monkeypatch.setattr(PresetManager, "presets", saved)
    monkeypatch.setattr("lib_spectrum.presets.PRESET_FILE", str(pathlib.Path(spectrum.paths.data_path) / "presets.json"))
    PresetManager.save_preset("mine", *_settings_args())
    assert saved["mine"] == _settings_args()
    assert PresetManager.get_preset("mine") == _settings_args()
    # Old presets are still padded to the settings only.
    monkeypatch.setitem(saved, "old", _settings_args()[:7])
    assert len(PresetManager.get_preset("old")) == len(DEFAULTS)
