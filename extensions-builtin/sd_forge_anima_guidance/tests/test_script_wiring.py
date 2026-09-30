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
from lib_anima_guidance.modulation import resolve_artifact as resolve_artifact_under_test
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

    def append_transformer_option(self, k, v):
        options = self.model_options.setdefault("transformer_options", {})
        options.setdefault(k, []).append(v)

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
        (dict(guidance_mode=SMC_GUIDANCE), False, [SMCCFGState]),
        (dict(guidance_mode=FDG_GUIDANCE), True, []),
        (dict(guidance_mode=CFG_ZERO_STAR_GUIDANCE), False, [CFGZeroStarState]),
        (dict(guidance_mode=APG_GUIDANCE), False, [APGState]),
        (dict(skim_enable=True), False, []),
        (dict(guidance_range_enable=True), False, []),
    ],
)
def test_cfg1_optimization_is_disabled_only_for_fdg(overrides, disables_cfg1, states):
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
    # Infotext is written through ``_record(p, control_name, value)``, which
    # takes the key from the ``SCRIPT_ARGS`` table (the paste targets). A
    # misspelled control name would raise only when its branch runs, so
    # check every call statically; direct writes are diagnostic only.
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    recorded = [
        node.args[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_record"
    ]
    assert all(isinstance(name, ast.Constant) for name in recorded)
    recorded = {name.value for name in recorded}
    written = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "extra_generation_params"
        and isinstance(node.slice, ast.Constant)
    }

    assert len(recorded) > 40  # the collector found the writes
    assert recorded <= set(SCRIPT_MODULE.INFOTEXT_KEYS), sorted(recorded - set(SCRIPT_MODULE.INFOTEXT_KEYS))
    assert written == DIAGNOSTIC_ONLY_INFOTEXT_KEYS
    assert not DIAGNOSTIC_ONLY_INFOTEXT_KEYS & set(SCRIPT_MODULE.INFOTEXT_KEYS.values())


class _FakeModulationPatch:
    def __init__(self, **kwargs):
        pass

    def apply(self, unet):
        return unet.clone(), 2, 20


def _modulation_works(monkeypatch):
    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(SCRIPT_MODULE, "check_adapter_for_dit", lambda *args: None)
    monkeypatch.setattr(SCRIPT_MODULE, "encode_clip_pooled", lambda prompts, **kwargs: torch.zeros(len(prompts), 4))
    monkeypatch.setattr(SCRIPT_MODULE, "prepare_modulation_vectors", lambda **kwargs: (None, None, None))
    monkeypatch.setattr(SCRIPT_MODULE, "ModulationPatch", _FakeModulationPatch)


def _modulation_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", fail)


# (sampler, existing CFG hook, setup, overrides) of representative runs.
INFOTEXT_CONFIGS = {
    "disabled": ("Euler", None, None, dict(enable=False, skim_enable=True, nag_enable=True)),
    "smc_alpha_zero": ("Euler", None, None, dict(guidance_mode=SMC_GUIDANCE, smc_alpha=0.0)),
    "standard_extras": (
        "Euler",
        None,
        None,
        dict(
            nag_enable=True, nag_scale=3.0, nag_tau=2.0, nag_alpha=0.25, nag_start=0.1, nag_end=0.6,
            momentum_enable=True, momentum_strength=0.75, momentum_ema=0.5,
            dcw_enable=True, dcw_lambda=0.02, dcw_schedule="sigma", dcw_bands="all",
            skim_enable=True, skim_scale=2.0, full_negative=True, disable_flip_filter=True,
            start_percent=0.1, end_percent=0.9, flip_percent=0.5,
            guidance_range_enable=True, guidance_range_start=0.2, guidance_range_end=0.8,
        ),
    ),
    "smc_unipc": (
        "Anima Flow UniPC2",
        None,
        None,
        dict(
            guidance_mode=SMC_GUIDANCE, smc_alpha=0.3, smc_lambda=4.0, skim_enable=True,
            flow_unipc_solver_type="bh1", flow_unipc_disable_corrector_first=2, flow_unipc_thresholding=True,
            flow_unipc_dynamic_thresholding_ratio=0.99, flow_unipc_sample_max_value=2.0,
        ),
    ),
    "fdg_pc3_replaced": (
        "Anima Flow PC3",
        "other_cfg",
        None,
        dict(guidance_mode=FDG_GUIDANCE, fdg_high_scale=15.0, flow_pc3_gamma=0.5, flow_pc3_tolerance=0.01),
    ),
    "cfg_zero_cns": (
        "Anima ER SDE CNS",
        None,
        None,
        dict(guidance_mode=CFG_ZERO_STAR_GUIDANCE, cfg_zero_init_steps=2, cns_strength=0.5, guidance_range_enable=True),
    ),
    "apg_dcw": (
        "Euler a",
        None,
        None,
        dict(guidance_mode=APG_GUIDANCE, apg_eta=0.3, apg_norm_threshold=5.0, apg_momentum=-0.25, dcw_enable=True),
    ),
    "conflicts": ("Anima ER SDE CNS", None, None, dict(guidance_mode=FDG_GUIDANCE, momentum_enable=True, dcw_enable=True)),
    "modulation": (
        "Euler",
        None,
        _modulation_works,
        dict(
            modulation_enable=True, modulation_weight=2.5, modulation_base="base", modulation_positive="good",
            modulation_negative="bad", nag_enable=True,
        ),
    ),
    "modulation_failed_nag_unbatched": ("Euler", None, _modulation_fails, dict(modulation_enable=True, nag_enable=True)),
}

# The exact extra_generation_params (keys, values and order) of each
# configuration, captured before the installers were split up.
INFOTEXT_SNAPSHOTS = {
    "disabled": [],
    "smc_alpha_zero": [("Anima guidance enabled", True)],
    "standard_extras": [
        ("Anima guidance enabled", True),
        ("Anima NAG", True),
        ("Anima NAG scale", 3.0),
        ("Anima NAG tau", 2.0),
        ("Anima NAG alpha", 0.25),
        ("Anima NAG start", 0.1),
        ("Anima NAG end", 0.6),
        ("Anima CFG active range", True),
        ("Anima CFG range start", 0.2),
        ("Anima CFG range end", 0.8),
        ("Anima Momentum Guidance", True),
        ("Anima Momentum strength", 0.75),
        ("Anima Momentum EMA", 0.5),
        ("Anima DCW", True),
        ("Anima DCW lambda", 0.02),
        ("Anima DCW schedule", "sigma"),
        ("Anima DCW bands", "all"),
        ("Anima Skimmed CFG", True),
        ("Anima skim CFG", 2.0),
        ("Anima skim full negative", True),
        ("Anima skim disable flip filter", True),
        ("Anima skim start", 0.1),
        ("Anima skim end", 0.9),
        ("Anima skim flip", 0.5),
    ],
    "smc_unipc": [
        ("Anima guidance enabled", True),
        ("Anima Flow UniPC solver type", "bh1"),
        ("Anima Flow UniPC disabled correctors", 2),
        ("Anima Flow UniPC thresholding", True),
        ("Anima Flow UniPC threshold percentile", 0.99),
        ("Anima Flow UniPC threshold maximum", 2.0),
        ("Anima CFG guidance mode", "SMC-CFG"),
        ("Anima SMC alpha", 0.3),
        ("Anima SMC lambda", 4.0),
        ("Anima Skimmed CFG", True),
        ("Anima skim CFG", 2.5),
    ],
    "fdg_pc3_replaced": [
        ("Anima guidance enabled", True),
        ("Anima Flow PC3 gamma", 0.5),
        ("Anima Flow PC3 tolerance", 0.01),
        ("Anima CFG guidance mode", "FDG (experimental)"),
        ("Anima FDG detail guidance", 15.0),
        ("Anima CFG replaced existing", "other_cfg"),
    ],
    "cfg_zero_cns": [
        ("Anima guidance enabled", True),
        ("Anima CNS strength", 0.5),
        ("Anima CFG guidance mode", "CFG-Zero*"),
        ("Anima CFG-Zero* zero-init steps", 2),
        ("Anima CFG active range", True),
        ("Anima CFG range start", 0.0),
        ("Anima CFG range end", 1.0),
    ],
    "apg_dcw": [
        ("Anima guidance enabled", True),
        ("Anima CFG guidance mode", "APG"),
        ("Anima APG eta", 0.3),
        ("Anima APG norm threshold", 5.0),
        ("Anima APG momentum", -0.25),
        ("Anima DCW", True),
        ("Anima DCW lambda", -0.015),
        ("Anima DCW schedule", "one_minus_sigma"),
        ("Anima DCW bands", "LL"),
    ],
    "conflicts": [
        ("Anima guidance enabled", True),
        (
            "Anima conflict resolution",
            "FDG disabled: incompatible with Anima ER SDE CNS; Momentum disabled: unsupported sampler Anima ER SDE CNS",
        ),
        ("Anima CNS strength", 1.0),
        ("Anima DCW", True),
        ("Anima DCW lambda", -0.015),
        ("Anima DCW schedule", "one_minus_sigma"),
        ("Anima DCW bands", "LL"),
    ],
    "modulation": [
        ("Anima guidance enabled", True),
        ("Anima modulation guidance", True),
        ("Anima modulation weight", 2.5),
        ("Anima modulation start block", 2),
        ("Anima modulation end block", 20),
        ("Anima modulation positive direction", "good"),
        ("Anima modulation negative direction", "bad"),
        ("Anima modulation adapter mode", "Auto-download pinned"),
        ("Anima modulation CLIP mode", "Auto-download pinned"),
        ("Anima modulation base prompt", "base"),
        ("Anima NAG", True),
        ("Anima NAG scale", 2.0),
        ("Anima NAG tau", 2.5),
        ("Anima NAG alpha", 0.5),
        ("Anima NAG start", 0.0),
        ("Anima NAG end", 0.5),
    ],
    "modulation_failed_nag_unbatched": [
        ("Anima guidance enabled", True),
        ("Anima modulation guidance", "disabled: OSError"),
        ("Anima NAG", "inactive: no negative context for a positive-only call"),
        ("Anima NAG scale", 2.0),
        ("Anima NAG tau", 2.5),
        ("Anima NAG alpha", 0.5),
        ("Anima NAG start", 0.0),
        ("Anima NAG end", 0.5),
    ],
}


@pytest.mark.parametrize("name", list(INFOTEXT_CONFIGS))
def test_infotext_snapshot(monkeypatch, name):
    sampler, existing, setup, overrides = INFOTEXT_CONFIGS[name]

    def other_cfg(args):
        return args["cond"]

    if setup is not None:
        setup(monkeypatch)
    p = _modulation_process()
    if existing is not None:
        p.sd_model.forge_objects.unet.model_options["sampler_cfg_function"] = other_cfg
    p.sampler_name = sampler
    _run(p, **overrides)
    if getattr(p, "_anima_nag_modifier", None) is not None and name.endswith("_nag_unbatched"):
        p._anima_nag_modifier.on_unbatched()
    # repr() also pins the value types (True vs 1, 2 vs 2.0) infotext prints.
    actual = [(key, repr(value)) for key, value in p.extra_generation_params.items()]
    assert actual == [(key, repr(value)) for key, value in INFOTEXT_SNAPSHOTS[name]]


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
    names = SCRIPT_MODULE.SCRIPT_ARG_NAMES
    # ui() returns the locals named in the table, in its order.
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
    # One positional value short: rejected, as a missing parameter was.
    with pytest.raises(TypeError, match="clip_path"):
        SCRIPT_MODULE.AnimaGuidanceScript().process_before_every_sampling(FakeProcess(), *values[:-1])

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


def test_guidance_warnings_are_visible():
    # Forge's root logger is at ERROR: without its own handler Guidance's
    # warnings would never be shown. (The preset panel's logger is tested in
    # sd_forge_anima_presets.)
    import logging

    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        guidance = logging.getLogger("AnimaGuidance")
        assert guidance.isEnabledFor(logging.WARNING) and guidance.handlers
    finally:
        root.setLevel(level)


def test_local_modulation_files_follow_the_servers_directory_policy(monkeypatch, tmp_path):
    seen = []

    def resolve(*args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop after resolving")

    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", resolve)
    monkeypatch.setattr(SCRIPT_MODULE.paths, "models_path", str(tmp_path))
    for hidden in (False, True):
        monkeypatch.setattr(SCRIPT_MODULE.shared.cmd_opts, "hide_ui_dir_config", hidden)
        _run(_modulation_process(), modulation_enable=True, adapter_mode="Local file", adapter_path="x.pt")
        assert seen[-1]["models_root"] == str(tmp_path)
        assert seen[-1]["local_files_allowed"] is not hidden

    # With the real resolver, --hide-ui-dir-config refuses even a file inside the models folder.
    (tmp_path / "x.pt").write_bytes(b"x")
    monkeypatch.setattr(SCRIPT_MODULE, "resolve_artifact", resolve_artifact_under_test)
    p = _modulation_process()
    _run(p, modulation_enable=True, adapter_mode="Local file", adapter_path="x.pt")
    assert p.extra_generation_params["Anima modulation guidance"] == "disabled: RuntimeError"



def _positional(**overrides):
    values = {**DEFAULTS, **overrides}
    return [values.get(name, default) for name, _, _, default in SCRIPT_MODULE.SCRIPT_ARGS]


@pytest.mark.parametrize(
    ("sampler", "hires", "hr_sampler", "overrides", "anima", "expected"),
    [
        ("Euler", False, None, {}, True, False),
        ("Anima ER SDE CNS", False, None, {}, True, True),
        ("Anima ER SDE CNS", False, None, dict(cns_strength=0.0), True, False),
        # Guidance off: the sampler runs at its default strength 1.
        ("Anima ER SDE CNS", False, None, dict(enable=False, cns_strength=0.0), True, True),
        ("Euler", True, "Anima ER SDE CNS", {}, True, True),
        ("Anima ER SDE CNS", True, None, {}, True, True),  # hires reuses the first pass's sampler
        ("Euler", False, "Anima ER SDE CNS", {}, True, False),  # no hires pass
        ("Anima ER SDE CNS", False, None, {}, False, False),
    ],
)
def test_process_prefetches_the_cns_gamma_table_before_sampling(monkeypatch, sampler, hires, hr_sampler, overrides, anima, expected):
    fetched = []
    monkeypatch.setattr(SCRIPT_MODULE.sd_samplers_anima_cns, "prefetch_gamma", fetched.append)
    monkeypatch.setattr(SCRIPT_MODULE, "_is_anima", lambda p: anima)
    p = FakeProcess()
    p.sampler_name = sampler
    p.enable_hr = hires
    p.hr_sampler_name = hr_sampler
    SCRIPT_MODULE.AnimaGuidanceScript().process(p, *_positional(**overrides))
    assert fetched == ([p] if expected else [])
