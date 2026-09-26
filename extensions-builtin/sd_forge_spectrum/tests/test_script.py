import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

sys.argv = sys.argv[:1]
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_spectrum.presets import DEFAULTS, INFOTEXT_KEYS, PresetManager

SCRIPT = ROOT / "scripts" / "spectrum.py"
SEA = "SEA (auto-calibrated)"


def _load():
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        # Its image helpers need the full WebUI install; the script only needs PasteField
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = tuple
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
    PresetManager.presets = {}  # the import must not create presets.json
    spec = importlib.util.spec_from_file_location("spectrum_script_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spectrum = _load()


class _FakePatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=torch.nn.Identity(), predictor=object())
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _process(**overrides):
    values = dict(
        is_hr_pass=False,
        steps=20,
        sampler_name="Euler",
        scheduler="Simple",
        cfg_scale=4.0,
        distilled_cfg_scale=3.0,
        hr_second_pass_steps=10,
        hr_sampler_name=None,
        hr_scheduler="Simple",
        hr_cfg=4.0,
        hr_distilled_cfg=3.0,
    )
    values.update(overrides)
    sd_model = SimpleNamespace(forge_objects=SimpleNamespace(unet=_FakePatcher()))
    return SimpleNamespace(sd_model=sd_model, extra_generation_params={}, **values)


def _args(**controls):
    values = dict(zip(INFOTEXT_KEYS, DEFAULTS))
    values.update(controls)
    return [values[key] for key in INFOTEXT_KEYS]


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    monkeypatch.setattr(spectrum.shared, "opts", SimpleNamespace(skip_early_cond=0.0, s_min_uncond=0.0), raising=False)
    monkeypatch.setattr(spectrum.paths, "data_path", str(tmp_path))


def _run(p, **controls):
    spectrum.SpectrumForForge().process_before_every_sampling(p, True, *_args(**controls), x=torch.zeros(2, 16, 8, 8))
    return p


def _sea_key(p=None, **controls):
    p = _run(p or _process(), spec_schedule=SEA, **controls)
    return p._spectrum_states[0].sea_calibration.key


def test_sea_key_changes_with_every_input_the_calibration_depends_on():
    base = _sea_key()
    assert _sea_key() == base
    variants = {
        "beta": _sea_key(spec_sea_beta=3.0),
        "scheduler": _sea_key(_process(scheduler="Beta")),
        "shift": _sea_key(_process(distilled_cfg_scale=5.0)),
        "sampler": _sea_key(_process(sampler_name="Anima Flow PC3")),
    }
    for name, key in variants.items():
        assert key != base, name


def test_hires_sea_key_uses_the_hires_scheduler_and_shift():
    def hires(**overrides):
        return _process(is_hr_pass=True, **overrides)

    base = _sea_key(hires())
    assert _sea_key(hires(hr_scheduler="Beta")) != base
    assert _sea_key(hires(hr_distilled_cfg=5.0)) != base
    # The base-pass values do not drive the hires pass.
    assert _sea_key(hires(scheduler="Beta", distilled_cfg_scale=5.0)) == base


def test_feature_history_is_raised_to_degree_plus_two():
    p = _run(_process(), spec_m=8, spec_history_size=3)
    assert p.extra_generation_params["spec_history_size"] == 10
    assert p.extra_generation_params["spec_m"] == 8

    p = _run(_process(), spec_m=2, spec_history_size=12)
    assert p.extra_generation_params["spec_history_size"] == 12


def test_cleanup_resets_and_drops_the_pass_states():
    resets = []
    p = SimpleNamespace(_spectrum_states=[SimpleNamespace(reset=lambda: resets.append(1))] * 2)
    spectrum.SpectrumForForge().cleanup(p)
    assert resets == [1, 1]
    assert p._spectrum_states == []

    fresh = SimpleNamespace()  # Spectrum never ran for this generation
    spectrum.SpectrumForForge().cleanup(fresh)
    assert fresh._spectrum_states == []


def test_cleanup_frees_a_real_pass_state():
    p = _run(_process())
    state = p._spectrum_states[0]
    state.forecasters[0] = object()
    spectrum.SpectrumForForge().cleanup(p)
    assert p._spectrum_states == []
    assert not state.forecasters


def test_old_seven_value_presets_are_padded_with_defaults(monkeypatch):
    old = [0.5, 3, 0.2, 3, 0.1, 4, 0.8]
    monkeypatch.setattr(PresetManager, "presets", {"old": old})
    updates = PresetManager.get_preset("old")
    assert len(updates) == len(INFOTEXT_KEYS)
    assert [update["value"] for update in updates] == [*old, *DEFAULTS[len(old) :]]


def test_upstream_short_argument_list_is_padded_with_defaults():
    # Upstream Forge Neo's API arity stops before Feature History and the SEA controls.
    short = _args(spec_m=3)[:8]
    p = _process()
    spectrum.SpectrumForForge().process_before_every_sampling(p, True, *short, x=torch.zeros(2, 16, 8, 8))
    assert len(p._spectrum_states) == 1
    assert p.extra_generation_params["spec_m"] == 3
    assert p.extra_generation_params["spec_history_size"] == DEFAULTS[INFOTEXT_KEYS.index("spec_history_size")]
    assert p.extra_generation_params["spec_schedule"] == "Window"
