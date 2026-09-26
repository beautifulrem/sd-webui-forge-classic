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
    def __init__(self, model_options=None, blocks=28):
        self.model_options = dict(model_options or {})
        self.model = types.SimpleNamespace(
            predictor=types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent),
            diffusion_model=types.SimpleNamespace(
                blocks=[None] * blocks, final_layer=types.SimpleNamespace(hidden_size=2048)
            ),
        )
        self.cfg1_disabled = False

    def clone(self):
        return FakeUnet(self.model_options, len(self.model.diffusion_model.blocks))

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

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


# Written to infotext for diagnosis only; pasting them back restores nothing.
DIAGNOSTIC_ONLY_INFOTEXT_KEYS = {"Anima CFG replaced existing"}


def test_every_infotext_key_written_is_restored_on_paste():
    # The keys are spelled twice: in the ``SCRIPT_ARGS`` table (paste targets)
    # and at each ``p.extra_generation_params[...] =`` write. A typo in either
    # silently breaks paste-restore, so compare the two spellings.
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fields = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SCRIPT_ARGS"
    )
    pasted = {
        row.elts[1].value
        for row in fields.elts
        if isinstance(row.elts[1], ast.Constant) and row.elts[1].value is not None
    }
    written = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "extra_generation_params"
        and isinstance(node.slice, ast.Constant)
    }

    assert len(written) > 40  # the collector found the writes
    assert written - DIAGNOSTIC_ONLY_INFOTEXT_KEYS <= pasted, sorted(written - DIAGNOSTIC_ONLY_INFOTEXT_KEYS - pasted)
    assert DIAGNOSTIC_ONLY_INFOTEXT_KEYS <= written
    assert not DIAGNOSTIC_ONLY_INFOTEXT_KEYS & pasted


def test_dcw_is_not_installed_on_a_multi_evaluation_sampler():
    p = FakeProcess()
    p.sampler_name = "Heun"
    _run(p, dcw_enable=True)
    assert p._anima_dcw_state is None
    assert _registered(p) == []
    assert "Anima DCW" not in p.extra_generation_params
    assert p.extra_generation_params["Anima conflict resolution"] == "DCW disabled: unsupported sampler Heun"


def test_dcw_is_not_installed_with_an_inpaint_mask():
    p = FakeProcess()
    p.mask = torch.ones(1, 1, 8, 8)
    _run(p, dcw_enable=True)
    assert p._anima_dcw_state is None
    assert _registered(p) == []
    assert "Anima DCW" not in p.extra_generation_params
    assert p.extra_generation_params["Anima conflict resolution"] == "DCW disabled: incompatible with an inpaint mask"


def test_dcw_drops_itself_when_the_denoiser_has_a_mask():
    # A mask the install-time check did not see: the pre-denoiser callback
    # gets a blended copy of the latent, so DCW must not pretend to run.
    p = FakeProcess()
    _run(p, dcw_enable=True)
    state = p._anima_dcw_state
    state.capture_denoised(torch.zeros(1, 4, 8, 8))
    state.current_sigma = 0.9
    x = torch.ones(1, 4, 8, 8)
    params = types.SimpleNamespace(denoiser=types.SimpleNamespace(p=p, mask=torch.ones(1, 1, 8, 8)), x=x, sigma=torch.tensor([0.8]))
    SCRIPT_MODULE._dcw_before_denoiser(params)
    assert torch.equal(x, torch.ones_like(x))
    assert p._anima_dcw_state is None
    assert not any(key.startswith("Anima DCW") for key in p.extra_generation_params)
    assert p.extra_generation_params["Anima conflict resolution"] == "DCW disabled: incompatible with an inpaint mask"


def test_positional_script_args_follow_the_table():
    import inspect

    names = SCRIPT_MODULE.SCRIPT_ARG_NAMES
    # ui() returns the locals named in the table, in its order.
    parameters = list(inspect.signature(SCRIPT_MODULE.AnimaGuidanceScript._process_before_every_sampling).parameters)
    assert tuple(parameters[2:]) == names

    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    ui = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "ui")
    assigned = {node.id for node in ast.walk(ui) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    assert set(names) <= assigned

    # An API call with the positional values of the first 34 args only (the
    # rest take their defaults) plus Forge's own keywords.
    required = [name for name, _, _, default in SCRIPT_MODULE.SCRIPT_ARGS if default is SCRIPT_MODULE._REQUIRED]
    assert tuple(required) == names[: len(required)]
    values = [{**DEFAULTS, "dcw_enable": True}[name] for name in required]
    p = FakeProcess()
    SCRIPT_MODULE.AnimaGuidanceScript().process_before_every_sampling(p, *values, x=None, noise=None, c=None, uc=None)
    assert _registered(p) == [DCWState]

    # All positional values, with a trailing one set: it reaches its parameter.
    values = [{**DEFAULTS, "apg_momentum": 0.25, "guidance_mode": APG_GUIDANCE}.get(name, default) for name, _, _, default in SCRIPT_MODULE.SCRIPT_ARGS]
    p = FakeProcess()
    SCRIPT_MODULE.AnimaGuidanceScript().process_before_every_sampling(p, *values, "extra positional")
    assert p.extra_generation_params["Anima APG momentum"] == 0.25


def test_cfg_zero_star_zero_init_steps_skip_the_model():
    p = FakeProcess()
    unet = _run(p, guidance_mode=CFG_ZERO_STAR_GUIDANCE, cfg_zero_init_steps=1)
    wrapper = unet.model_options["model_function_wrapper"]
    calls = []

    def model_function(x, t, **c):
        calls.append(float(t[0]))
        return x * 0.5

    x = torch.ones(2, 1, 2, 2)

    def run(sigma):
        return wrapper(model_function, {"input": x, "timestep": torch.tensor([sigma] * 2), "c": {}, "cond_or_uncond": [0, 1]})

    # Spectrum is forced to actual forwards on exactly those steps.
    from modules import spectrum_force

    reasons = unet.model_options["transformer_options"][spectrum_force.KEY]
    assert spectrum_force.is_forced(reasons, 1.0)
    assert torch.equal(run(1.0), x)  # percent_to_sigma(0) = 1.0: the pass starts from noise
    assert calls == []
    assert torch.equal(run(0.5), x * 0.5)
    assert calls == [0.5]
    assert not spectrum_force.is_forced(reasons, 0.5)


def _modulation_process(blocks=28):
    p = FakeProcess(FakeUnet(blocks=blocks))
    p.steps = 10
    p.prompts = ["a (red:1.2) [cat:dog:5]"]
    p.negative_prompts = ["(bad:1.4)"]
    return p


def test_modulation_rejects_an_unsupported_depth_before_downloading(monkeypatch):
    downloads = []
    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", lambda *args, **kwargs: downloads.append(args) or "unused")
    p = _modulation_process(blocks=30)
    _run(p, modulation_enable=True)
    assert downloads == []
    assert p.extra_generation_params["Anima modulation guidance"] == "disabled: RuntimeError"

    for blocks in (28, 40, 52):  # 2B, 2.9B and 3.8B pass the check
        p = _modulation_process(blocks=blocks)
        monkeypatch.setattr(SCRIPT_MODULE, "check_adapter_for_dit", lambda *args: None, raising=False)
        monkeypatch.setattr(SCRIPT_MODULE, "encode_clip_pooled", lambda *args, **kwargs: 1 / 0)
        _run(p, modulation_enable=True)
        assert len(downloads) == 2  # adapter and CLIP-L
        downloads.clear()


def test_modulation_encodes_plain_prompt_text(monkeypatch):
    encoded = []

    def encode(prompts, **kwargs):
        encoded.extend(prompts)
        raise RuntimeError("stop after encoding")

    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(SCRIPT_MODULE, "check_adapter_for_dit", lambda *args: None, raising=False)
    monkeypatch.setattr(SCRIPT_MODULE, "encode_clip_pooled", encode)
    p = _modulation_process()
    _run(p, modulation_enable=True, modulation_positive="(masterpiece:1.2), detailed", modulation_negative="[blurry]")
    assert encoded == ["a red cat", "bad", "masterpiece, detailed", "blurry"]


def test_preset_warnings_are_visible():
    # Forge's root logger is at ERROR: without its own handler the preset
    # script's Turbo advice would never be shown.
    import logging

    spec = importlib.util.spec_from_file_location("anima_remi_presets_under_test", SCRIPT.with_name("anima_remi_presets.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        assert module.logger.isEnabledFor(logging.WARNING)
        assert module.logger.handlers and not module.logger.propagate
        guidance = logging.getLogger("AnimaGuidance")
        assert guidance.isEnabledFor(logging.WARNING) and guidance.handlers
    finally:
        root.setLevel(level)
