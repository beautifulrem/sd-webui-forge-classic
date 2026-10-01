import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

negpip = harness.load_script(ROOT / "scripts" / "negpip.py", "negpip_under_test")

from backend.nn.anima import SelfCrossAttention  # noqa: E402
from backend.nn.unet import CrossAttention  # noqa: E402
from backend.sampling import condition, sampling_function  # noqa: E402
from lib_negpip import anima as negpip_anima  # noqa: E402
from lib_negpip import sd as negpip_sd  # noqa: E402


class _Model:
    is_anima_engine = True

    def __init__(self, dit):
        self.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=dit)))
        self.text_processing_engine_anima = SimpleNamespace(tokenize_line=lambda line: [])

    def is_webui_legacy_model(self):
        return False

    def get_learned_conditioning(self, prompt):
        return [torch.ones(2, 2) for _ in prompt]


class _DiT(torch.nn.Module):
    def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
        return x


@pytest.fixture
def anima(monkeypatch):
    dit = _DiT()
    model = _Model(dit)
    monkeypatch.setattr(negpip_anima, "shared", SimpleNamespace(sd_model=model))  # the real one loads a checkpoint
    monkeypatch.setattr(negpip.NegPiP, "_patched", [False, False])
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)  # keep the global callback list clean
    originals = SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions
    yield model, dit, originals
    negpip_anima.patch_anima_negpip(negpip.NegPiP, unpatch=True)


def _process(model, prompt="a cat, (dog:-1)"):
    return SimpleNamespace(sd_model=model, prompts=[prompt], negative_prompts=[""], scripts=SimpleNamespace(scripts=[]), script_args=[], extra_generation_params={})


def _assert_restored(model, dit, originals):
    assert (SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions) == originals
    assert "get_learned_conditioning" not in model.__dict__ and "forward" not in dit.__dict__


@pytest.mark.parametrize("finish", ["cleanup", "postprocess"])
def test_anima_batch_hooks_the_model_and_finishing_unhooks_it(anima, finish):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _process(model)

    script.process_batch(p)
    assert script.active and script.is_anima
    assert negpip.NegPiP._patched == [False, True]
    assert SelfCrossAttention.forward is not originals[0]
    assert "get_learned_conditioning" in model.__dict__ and "forward" in dit.__dict__
    assert p.extra_generation_params == {"NegPiP": True}
    assert p.cached_c == [None] * 3 and p.cached_uc == [None] * 3  # conditioning is encoded again, through the hook

    getattr(script, finish)(p)
    assert not script.active
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)

    script.cleanup(p)  # cleanup after postprocess (or twice) is harmless
    _assert_restored(model, dit, originals)


def test_a_batch_without_negative_weights_leaves_the_previous_hooks_undone(anima):
    model, dit, originals = anima
    script = negpip.NegPiP()
    script.process_batch(_process(model))
    assert script.active

    # The next batch (e.g. a later job) has no negative weight: the old hooks go
    p = _process(model, "a cat, (dog:1.2)")
    script.process_batch(p)
    assert not script.active and p.extra_generation_params == {}
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)


class _HiresProcess(SimpleNamespace):
    """Records the Hires conditioning encoded by whichever model is loaded."""

    def calculate_hr_conds(self):
        if self.hr_c is not None:
            return
        self.hr_c = negpip_anima.shared.sd_model.get_learned_conditioning(["(dog:-1)"])
        self.encodes += 1


def _hires_process(model):
    p = _process(model)
    return _HiresProcess(**vars(p), hr_c="first model's", hr_uc="first model's", encodes=0)


def test_a_hires_checkpoint_is_hooked_and_its_conditioning_encoded_again(anima):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _hires_process(model)
    script.process_batch(p)
    p.cached_hr_c = ["stale", "stale", "stale"]

    # The Hires pass loaded another checkpoint (and encoded with it, unhooked)
    hires_dit = _DiT()
    hires_model = _Model(hires_dit)
    p.sd_model = negpip_anima.shared.sd_model = hires_model
    script.before_hr(p)

    assert script.active and script.is_hr
    assert negpip_anima.patched_engine() is hires_model
    assert "get_learned_conditioning" in hires_model.__dict__ and "forward" in hires_dit.__dict__
    assert "get_learned_conditioning" not in model.__dict__ and "forward" not in dit.__dict__
    assert p.encodes == 1 and "c_negpip_mask" in p.hr_c  # encoded again, through the hook
    assert p.cached_hr_c == [None] * 3

    script.cleanup(p)
    _assert_restored(hires_model, hires_dit, originals)


def test_the_same_model_in_hires_keeps_its_conditioning(anima):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _hires_process(model)
    script.process_batch(p)
    script.before_hr(p)
    assert script.active and p.encodes == 0 and p.hr_c == "first model's"
    assert negpip_anima.patched_engine() is model


def test_a_hires_checkpoint_that_is_not_anima_turns_negpip_off_visibly(anima):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _hires_process(model)
    script.process_batch(p)

    hires_model = _Model(_DiT())
    hires_model.is_anima_engine = False
    p.sd_model = negpip_anima.shared.sd_model = hires_model
    script.before_hr(p)

    assert not script.active and p.encodes == 0
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)
    assert "NegPiP off in Hires" in str(p.extra_generation_params)


# Forge's attention parser reads "(foo:- 1)" as the literal text "foo:- 1" at
# weight 1.1 (no space may follow the minus); "(foo:-1.)" is weight -1.
# A colon may sit in the text (Danbooru tags such as ":d", "re:zero"); a
# bracket after an escaped backslash and an escape ("\\\\\\(") is literal.
_NOT_NEGATIVE = ["a cat, (foo:- 1)", "a cat, (foo:-0)", "a cat, (foo:- .5)", r"a cat, \\\(x:-1)"]
_NEGATIVE = {
    "a cat, (foo:-1)": -1.0, "a cat, (foo:-0.5)": -0.5, "a cat, (foo:-.5)": -0.5, "a cat, (foo:-1.)": -1.0, "a cat, (foo : -2 )": -2.0,
    "a cat, (:d:-1)": -1.0, "a cat, (re:zero:-0.5)": -0.5,
}


@pytest.mark.parametrize("prompt", _NOT_NEGATIVE)
def test_anima_text_the_parser_reads_as_no_negative_weight_installs_nothing(anima, prompt):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _process(model, prompt)
    script.process_batch(p)
    assert not script.active and p.extra_generation_params == {} and p.prompts == [prompt]
    assert negpip.NegPiP._patched == [False, False]
    _assert_restored(model, dit, originals)


@pytest.mark.parametrize("prompt", list(_NEGATIVE))
def test_anima_negative_weights_install_the_hooks(anima, prompt):
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _process(model, prompt)
    script.process_batch(p)
    assert script.active and p.extra_generation_params == {"NegPiP": True}
    assert p.prompts == [prompt]  # Anima reads the weights from the encoder's multipliers


class _SDModel:
    is_sdxl = False

    def __init__(self, unet):
        self.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=unet)))
        self.text_processing_engine = SimpleNamespace(tokenize_line=lambda line: (None, len(line.split())))

    def is_webui_legacy_model(self):
        return True


@pytest.fixture
def sd(monkeypatch):
    model = _SDModel(torch.nn.Module())
    monkeypatch.setattr(negpip_sd, "shared", SimpleNamespace(sd_model=model))  # the real one loads a checkpoint
    monkeypatch.setattr(negpip.NegPiP, "_patched", [False, False])
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)
    script = negpip.NegPiP()
    encoded = []

    def cond_dealer(p, target):  # the text encoder is not loaded
        encoded.append(target)
        return torch.zeros(1, 2, 6), 2

    monkeypatch.setattr(script, "_cond_dealer", cond_dealer)
    yield model, script, encoded
    negpip_sd.patch_sd_negpip(None, negpip.NegPiP, unpatch=True)


def _sd_process(model, prompt):
    p = _process(model, prompt)
    p.steps, p.batch_size, p.sampler_name = 20, 1, "Euler"
    return p


@pytest.mark.parametrize("prompt", _NOT_NEGATIVE)
def test_sd_text_the_parser_reads_as_no_negative_weight_is_left_in_the_prompt(sd, prompt):
    # It used to be cut from the prompt, then float("- 1") raised
    model, script, encoded = sd
    p = _sd_process(model, prompt)
    script.process_batch(p)
    assert not script.active and p.extra_generation_params == {} and encoded == []
    assert p.prompts == [prompt]


@pytest.mark.parametrize(("prompt", "weight"), list(_NEGATIVE.items()))
def test_sd_negative_weights_are_moved_out_of_the_prompt_and_encoded(sd, prompt, weight):
    model, script, encoded = sd
    p = _sd_process(model, prompt)
    script.process_batch(p)
    assert script.active and p.extra_generation_params == {"NegPiP": True}
    assert p.prompts == ["a cat, "]
    text = prompt[prompt.index("(") + 1 : prompt.rindex(":")]
    assert encoded == [(text, weight)] and script.conds_all[0][0][2] == 2


# Tokens each negative target encodes to (text tokens + end token, as _cond_dealer counts them)
_TARGET_TOKENS = {"red fur": 2, "blue eyes": 3, "green": 1, "a dark night sky": 4, "ugly": 1, "bad hands": 2}


def test_an_sd_step_applies_every_negative_target_of_its_schedule_entry(monkeypatch):
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)
    script = negpip.NegPiP()
    script.active, script.is_anima, script.is_hr = True, False, False
    script.is_xl, script.batch_size, script.rev, script.c_len, script.uc_len = False, 1, True, 1, 1
    script.has_hr_p = script.has_hr_n = False
    # The text encoder is not loaded: each target encodes to its own random tokens
    monkeypatch.setattr(script, "_cond_dealer", lambda p, target: (torch.randn(1, _TARGET_TOKENS[target[0]], 6), _TARGET_TOKENS[target[0]]))
    steps = 20
    positive = "a cat, [(red fur:-1), (blue eyes:-1):(green:-1), (a dark night sky:-1):10]"
    script.conds_all = script._calc_conds(None, script._getScheduledNegPip([positive], steps))
    script.unconds_all = script._calc_conds(None, script._getScheduledNegPip(["(ugly:-1), (bad hands:-1)"], steps))

    torch.manual_seed(0)
    attention = CrossAttention(8, context_dim=6, heads=2, dim_head=4)
    attention.to_k.weight = attention.to_v.weight  # a negated value token then reads -key
    seen = []

    def recording_attention(q, k, v, heads, mask=None):
        negated = sum(torch.allclose(v[0, t], -k[0, t]) and not torch.allclose(v[0, t], k[0, t]) for t in range(k.shape[1]))
        seen.append((k.shape[1], negated))
        return q

    monkeypatch.setattr(negpip_sd, "optimized_attention", recording_attention)
    negpip_sd._hook_forward(script, attention, False)
    try:
        # The sampler reports steps 0..steps-2 (the step count lags the model call by one)
        for sampling_step in range(steps - 1):
            script.denoiser_callback(SimpleNamespace(sampling_step=sampling_step))
            seen.clear()
            with torch.inference_mode():
                attention(torch.randn(2, 5, 8), torch.randn(2, 77, 6))  # rev: uncond, cond
            cond_tokens = 2 + 3 if sampling_step + 2 <= 10 else 1 + 4
            assert seen == [(77 + cond_tokens, cond_tokens), (77 + 1 + 2, 1 + 2)], sampling_step
    finally:
        negpip_sd._hook_forward(script, attention, True)


def test_an_sd_step_with_one_negative_target_negates_exactly_its_tokens(monkeypatch):
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)
    script = negpip.NegPiP()
    script.active, script.is_anima, script.is_hr = True, False, False
    script.is_xl, script.batch_size, script.rev, script.c_len, script.uc_len = False, 1, True, 1, 1
    script.has_hr_p = script.has_hr_n = False
    target = torch.randn(1, 3, 6)
    monkeypatch.setattr(script, "_cond_dealer", lambda p, t: (target, 3))
    script.conds_all = script._calc_conds(None, script._getScheduledNegPip(["a cat, (blue eyes:-1)"], 20))
    script.unconds_all = script._calc_conds(None, script._getScheduledNegPip(["plain"], 20))
    script.denoiser_callback(SimpleNamespace(sampling_step=0))

    torch.manual_seed(0)
    attention = CrossAttention(8, context_dim=6, heads=2, dim_head=4)
    x, context = torch.randn(2, 5, 8), torch.randn(2, 77, 6)
    uncond, cond = context.chunk(2)
    xn, xp = x.chunk(2)
    # Plain attention over the prompt plus the target, its value tokens negated
    ctx = torch.cat([cond, target], 1)
    v = attention.to_v(ctx)
    v[:, -3:] = -v[:, -3:]
    expected_p = attention.to_out(negpip_sd.optimized_attention(attention.to_q(xp), attention.to_k(ctx), v, attention.heads))
    expected_n = attention(xn, uncond)

    negpip_sd._hook_forward(script, attention, False)
    try:
        with torch.inference_mode():
            out = attention(x, context)
    finally:
        negpip_sd._hook_forward(script, attention, True)
    assert torch.allclose(out, torch.cat([expected_n, expected_p]), atol=1e-6)


# A negative weight around a nested group: the parser weights the text (the
# Anima mask comes from the parser), but the SD path only moves innermost
# groups out of the prompt, so it has nothing to do there.
_NESTED_ONLY = "a cat, (red, (dog):-1)"


def test_anima_a_nested_negative_weight_installs_the_hooks(anima):
    # Whatever the other prompts of the job hold
    model, dit, originals = anima
    script = negpip.NegPiP()
    p = _process(model, _NESTED_ONLY)
    script.process_batch(p)
    assert script.active and p.extra_generation_params == {"NegPiP": True}
    assert p.prompts == [_NESTED_ONLY]


def test_sd_a_nested_negative_weight_alone_leaves_negpip_off(sd):
    model, script, encoded = sd
    p = _sd_process(model, _NESTED_ONLY)
    script.process_batch(p)
    assert not script.active and p.extra_generation_params == {} and encoded == []
    assert p.prompts == [_NESTED_ONLY]
    assert negpip.NegPiP._patched == [False, False]
