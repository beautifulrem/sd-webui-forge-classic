"""process_before_every_sampling wiring, run against a fake UNet patcher."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from lib_anima_guidance.advanced import APGState, CFGZeroStarState
from lib_anima_guidance.dcw import DCWState
from lib_anima_guidance.smc import SMCCFGState
from modules import anima_guidance_state
from modules.anima_feature_conflicts import (
    APG_GUIDANCE,
    CFG_ZERO_STAR_GUIDANCE,
    FDG_GUIDANCE,
    SMC_GUIDANCE,
    STANDARD_GUIDANCE,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_guidance.py"


def _load_script():
    sys.argv = sys.argv[:1]
    # The real module pulls in the sampler stack (k_diffusion); the script
    # only needs PasteField at import time.
    stubbed = "modules.infotext_utils" not in sys.modules
    if stubbed:
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = lambda component, key: (component, key)
        sys.modules["modules.infotext_utils"] = stub
    try:
        spec = importlib.util.spec_from_file_location("anima_guidance_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if stubbed:
            sys.modules.pop("modules.infotext_utils", None)
    return module


SCRIPT_MODULE = _load_script()


class FakeUnet:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})
        self.model = types.SimpleNamespace(predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent))
        self.cfg1_disabled = False

    def clone(self):
        return FakeUnet(self.model_options)

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = fn
        self.cfg1_disabled = self.cfg1_disabled or disable_cfg1_optimization

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_pre_cfg_function"] = [*self.model_options.get("sampler_pre_cfg_function", []), fn]

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = [*self.model_options.get("sampler_post_cfg_function", []), fn]


class FakeProcess:
    def __init__(self, unet=None):
        self.sampler_name = "Euler"
        self.is_hr_pass = False
        self.extra_generation_params = {}
        self.sd_model = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet or FakeUnet()))


DEFAULTS = dict(
    enable=True,
    guidance_mode=STANDARD_GUIDANCE,
    smc_alpha=0.2,
    smc_lambda=5.0,
    skim_enable=False,
    skim_scale=2.5,
    full_negative=False,
    disable_flip_filter=False,
    start_percent=0.0,
    end_percent=1.0,
    flip_percent=0.0,
    dcw_enable=False,
    dcw_lambda=-0.015,
    dcw_schedule="one_minus_sigma",
    dcw_bands="LL",
    cns_strength=1.0,
    flow_unipc_solver_type="bh2",
    flow_unipc_disable_corrector_first=0,
    flow_unipc_thresholding=False,
    flow_unipc_dynamic_thresholding_ratio=0.995,
    flow_unipc_sample_max_value=1.0,
    flow_pc3_gamma=1.0,
    flow_pc3_tolerance=0.005,
    modulation_enable=False,
    modulation_weight=3.0,
    modulation_start=0,
    modulation_end=-1,
    modulation_base="",
    modulation_positive="",
    modulation_negative="",
    adapter_mode="Auto-download pinned",
    adapter_path="",
    clip_mode="Auto-download pinned",
    clip_path="",
)


@pytest.fixture(autouse=True)
def anima_model(monkeypatch):
    monkeypatch.setattr(SCRIPT_MODULE, "_is_anima", lambda p: True)


def _run(p, **overrides):
    SCRIPT_MODULE.AnimaGuidanceScript().process_before_every_sampling(p, **{**DEFAULTS, **overrides})
    return p.sd_model.forge_objects.unet


def _registered(p):
    return [type(state) for state in getattr(p, anima_guidance_state._STATEFUL_GUIDANCE)]


def test_skim_then_range_then_zero_init_outermost(monkeypatch):
    calls = []
    for name in (
        "make_cfg_zero_star_cfg_function",
        "make_skim_cfg_function",
        "make_guidance_range_cfg_function",
        "make_zero_init_cfg_function",
        "make_guidance_range_uncond_skip",
    ):
        real = getattr(SCRIPT_MODULE, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            result = _real(*args, **kwargs)
            calls.append((_name, args[0] if args else None, result))
            return result

        monkeypatch.setattr(SCRIPT_MODULE, name, spy)

    p = FakeProcess()
    original = p.sd_model.forge_objects.unet
    unet = _run(
        p,
        guidance_mode=CFG_ZERO_STAR_GUIDANCE,
        cfg_zero_init_steps=1,
        skim_enable=True,
        guidance_range_enable=True,
        guidance_range_start=0.5,  # sigma window [0.0, 0.5]
        guidance_range_end=1.0,
    )

    assert unet is not original
    names = [name for name, _, _ in calls]
    assert names == [
        "make_cfg_zero_star_cfg_function",
        "make_skim_cfg_function",
        "make_guidance_range_cfg_function",
        "make_zero_init_cfg_function",
        "make_guidance_range_uncond_skip",
    ]
    # Each wrapper wraps the previous one's result.
    for (_, _, inner), (_, wrapped, _) in zip(calls, calls[1:]):
        assert wrapped is inner
    installed = calls[3][2]
    assert unet.model_options["sampler_cfg_function"] is installed
    assert unet.model_options["sampler_pre_cfg_function"] == [calls[4][2]]
    assert not unet.cfg1_disabled

    # Zero-init sees the first sigma although it lies outside the CFG range.
    x = torch.ones(1, 1, 2, 2)
    args = {"input": x, "sigma": torch.tensor([1.0]), "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25, "cond_scale": 4.0}
    assert torch.equal(installed(args), torch.zeros_like(x))


def test_uncond_skip_is_bound_to_the_installed_hook():
    p = FakeProcess()
    unet = _run(p, skim_enable=True, guidance_range_enable=True, guidance_range_start=0.5, guidance_range_end=1.0)
    (pre_cfg,) = unet.model_options["sampler_pre_cfg_function"]
    options = unet.model_options

    def uncond_after(timestep, model_options):
        return pre_cfg(None, "cond", "uncond", None, torch.tensor([timestep]), model_options)[2]

    assert uncond_after(0.9, options) is None  # outside the window: skipped
    assert uncond_after(0.25, options) == "uncond"  # inside the window: kept
    # Another extension replaced the CFG hook afterwards: keep the pass.
    assert uncond_after(0.9, {**options, "sampler_cfg_function": lambda args: None}) == "uncond"


@pytest.mark.parametrize(
    ("overrides", "disables_cfg1", "states"),
    [
        (dict(guidance_mode=SMC_GUIDANCE), True, [SMCCFGState]),
        (dict(guidance_mode=FDG_GUIDANCE), True, []),
        (dict(guidance_mode=CFG_ZERO_STAR_GUIDANCE), False, [CFGZeroStarState]),
        (dict(guidance_mode=APG_GUIDANCE), False, [APGState]),
        (dict(skim_enable=True), False, []),
        (dict(guidance_range_enable=True), False, []),
    ],
)
def test_cfg1_optimization_is_disabled_only_for_smc_and_fdg(overrides, disables_cfg1, states):
    p = FakeProcess()
    unet = _run(p, **overrides)
    assert callable(unet.model_options["sampler_cfg_function"])
    assert unet.cfg1_disabled is disables_cfg1
    assert _registered(p) == states


def test_replaced_existing_hook_is_recorded_only_when_replaced():
    def other_cfg(args):
        return args["cond"]

    p = FakeProcess(FakeUnet({"sampler_cfg_function": other_cfg}))
    _run(p, guidance_mode=APG_GUIDANCE)
    assert p.extra_generation_params["Anima CFG replaced existing"] == "other_cfg"

    p = FakeProcess(FakeUnet({"sampler_cfg_function": other_cfg}))
    _run(p, skim_enable=True)  # Skim wraps the existing hook
    assert "Anima CFG replaced existing" not in p.extra_generation_params


@pytest.mark.parametrize("enable", [True, False])
def test_each_pass_starts_without_the_previous_pass_state(enable):
    p = FakeProcess()
    _run(p, guidance_mode=CFG_ZERO_STAR_GUIDANCE, dcw_enable=True)
    assert _registered(p) == [CFGZeroStarState, DCWState]
    assert isinstance(p._anima_dcw_state, DCWState)

    p.sd_model.forge_objects.unet = FakeUnet()
    _run(p, enable=enable)
    assert _registered(p) == []
    assert p._anima_dcw_state is None
    assert p._anima_nag_modifier is None
