import functools
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import operations
from backend.nn.anima import Block, _fn
from lib_anima_xattn_boost.boost import BoostState, apply_xattn_boost_patch, renorm_scale, row_gains
from modules import spectrum_force

B, T, H, W, D, CTX = 2, 1, 3, 4, 32, 16


def _block(seed=0):
    generator = torch.Generator().manual_seed(seed)
    with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        block = Block(D, CTX, num_heads=2, adaln_lora_dim=8)
    block.load_state_dict({k: torch.randn(v.shape, generator=generator) * 0.2 for k, v in block.state_dict().items()})
    return block.eval()


def _inputs(seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, H, W, D, generator=generator) * 3.0
    emb = torch.randn(B, T, D, generator=generator)
    context = torch.randn(B, 5, CTX, generator=generator)
    lora = torch.randn(B, T, 3 * D, generator=generator) * 0.1
    return x, emb, context, lora


def _reference(block, x, emb, context, lora, gains, frac, options=None):
    """anima_lora 703b34b ``Block._forward`` with ``_xattn_gain`` (per row) and
    ``img`` renorm, on remi's Block weights (no RoPE). ``block.cross_attn``
    runs whatever override is installed on it, with ``options``."""

    silu = torch.nn.functional.silu(emb)

    def modulation(mlp):
        return [chunk[:, :, None, None, :] for chunk in (mlp[2](mlp[1](silu)) + lora).chunk(3, dim=-1)]

    shift_sa, scale_sa, gate_sa = modulation(block.adaln_modulation_self_attn)
    shift_ca, scale_ca, gate_ca = modulation(block.adaln_modulation_cross_attn)
    shift_mlp, scale_mlp, gate_mlp = modulation(block.adaln_modulation_mlp)
    shape = x.shape[1:4]
    result = block.self_attn(_fn(x, block.layer_norm_self_attn, scale_sa, shift_sa).flatten(1, 3), None).unflatten(1, shape)
    x = x + gate_sa * result
    result = block.cross_attn(_fn(x, block.layer_norm_cross_attn, scale_ca, shift_ca).flatten(1, 3), context, transformer_options=options or {}).unflatten(1, shape)
    gain = gains.view(-1, 1, 1, 1, 1)
    x_new = result * (gate_ca * gain) + x
    if frac > 0.0:
        plain = result * gate_ca + x
        norm_plain = plain.norm(dim=-1, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)
        norm_new = x_new.norm(dim=-1, keepdim=True).clamp_min(1e-6).mean(dim=(1, 2, 3), keepdim=True)
        x_new = x_new * (norm_plain / norm_new) ** frac
    return x_new + gate_mlp * block.mlp(_fn(x_new, block.layer_norm_mlp, scale_mlp, shift_mlp))


class _Patcher:
    def __init__(self, dit, options=None):
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        return _Patcher(self.model.diffusion_model, self.model_options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _dit(*blocks):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(blocks)
    return dit


def _args(sigma, markers=(0, 1), batch=B):
    options = {"cond_or_uncond": list(markers)}
    return {"input": torch.zeros(batch, 16, T, H, W), "timestep": torch.tensor([float(sigma)] * batch), "c": {"transformer_options": options}, "cond_or_uncond": list(markers)}


def _run(dit, x, emb, context, lora):
    def model_function(_input, _t, transformer_options=None, **c):
        h = x
        for block in dit.blocks:
            h = block(h, emb, context, adaln_lora_B_T_3D=lora, transformer_options=transformer_options, silu_emb_B_T_D=torch.nn.functional.silu(emb))
        return h

    return model_function


def _boosted(gain=2.0, renorm=0.5, band=0.85, sigma=0.9, markers=(0, 1)):
    block = _block()
    x, emb, context, lora = _inputs()
    wrapper = apply_xattn_boost_patch(_Patcher(_dit(block)), BoostState(gain=gain, band=band, renorm=renorm)).model_options["model_function_wrapper"]
    with torch.no_grad():
        out = wrapper(_run(_dit(block), x, emb, context, lora), _args(sigma, markers))
        plain = _run(_dit(block), x, emb, context, lora)(None, None)
    return block, (x, emb, context, lora), out, plain


@pytest.mark.parametrize("renorm", [0.0, 0.5, 1.0])
def test_matches_the_reference_block_on_cond_rows(renorm):
    block, (x, emb, context, lora), out, _ = _boosted(renorm=renorm)
    with torch.no_grad():
        expected = _reference(block, x, emb, context, lora, torch.tensor([2.0, 1.0]), renorm)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_unit_gain_is_identity():
    assert not BoostState(gain=1.0, band=0.85, renorm=0.5).active(1.0)
    _, _, out, plain = _boosted(gain=1.0)
    assert torch.equal(out, plain)


def test_uncond_rows_are_unchanged():
    _, _, out, plain = _boosted(renorm=0.5)
    assert torch.equal(out[1], plain[1])
    assert not torch.allclose(out[0], plain[0])
    # Markers order decides the rows, not the position
    _, _, out, plain = _boosted(markers=(1, 0))
    assert torch.equal(out[0], plain[0]) and not torch.allclose(out[1], plain[1])


def test_positive_only_call_boosts_every_row():
    _, _, out, plain = _boosted(markers=(0,))
    assert not torch.allclose(out[0], plain[0]) and not torch.allclose(out[1], plain[1])


def test_renorm_matches_the_mean_token_norm_per_image():
    generator = torch.Generator().manual_seed(3)
    plain = torch.randn(2, 1, 4, 5, 8, generator=generator)
    boosted = plain + torch.randn(2, 1, 4, 5, 8, generator=generator) * torch.tensor([1.0, 0.0]).view(-1, 1, 1, 1, 1)

    def mean_norm(t):
        return t.norm(dim=-1).mean(dim=(1, 2, 3))

    full = renorm_scale(plain, boosted, 1.0)
    torch.testing.assert_close(mean_norm(boosted * full), mean_norm(plain))
    # One shared factor per image; an unchanged image keeps exactly 1
    assert full.shape == (2, 1, 1, 1, 1) and full[1].item() == 1.0
    torch.testing.assert_close(renorm_scale(plain, boosted, 0.5), full.sqrt())
    assert torch.equal(renorm_scale(plain, boosted, 0.0), torch.ones_like(full))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_renorm_accumulates_half_precision_in_float32(dtype):
    generator = torch.Generator().manual_seed(4)
    plain = (torch.randn(2, 1, 4, 5, 64, generator=generator) * 30.0).to(dtype)
    boosted = (plain.float() * torch.tensor([1.5, 1.0]).view(-1, 1, 1, 1, 1)).to(dtype)
    dims = (1, 2, 3)
    # The float32 upcast-then-norm formula it replaces.
    expected = plain.float().norm(dim=-1, keepdim=True).mean(dim=dims, keepdim=True) / boosted.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).mean(dim=dims, keepdim=True)
    scale = renorm_scale(plain, boosted, 1.0)
    assert scale.dtype == torch.float32 and scale[1].item() == 1.0
    torch.testing.assert_close(scale, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(renorm_scale(plain, boosted, 0.5), expected.sqrt(), rtol=1e-6, atol=1e-6)


def test_row_gains():
    assert row_gains([0, 2], 4, 2.0).flatten().tolist() == [2.0, 1.0, 2.0, 1.0]


def test_row_gains_are_built_once_per_pass_in_the_compute_dtype(monkeypatch):
    from lib_anima_xattn_boost import boost

    built = []
    monkeypatch.setattr(boost, "row_gains", lambda *args: built.append(args) or row_gains(*args))
    dit = _stub_dit()
    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.0)).model_options["model_function_wrapper"]
    h = torch.ones(B, 3, 4, dtype=torch.bfloat16)
    for _ in range(3):
        outs = wrapper(lambda *a, **k: [block(h, None, None) for block in dit.blocks], _args(1.0))
    assert len(built) == 1  # not once per model call, nor recast per block
    assert built[0][-1] == torch.bfloat16
    assert all(out.dtype == torch.bfloat16 and out[:, 0, 0].tolist() == [2.0, 1.0] for out in outs)


class _Stub(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class _StubBlock(torch.nn.Module):
    """Runs its cross-attention; records which forwards are overridden then."""

    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = _Stub(), _Stub()
        self.seen = []

    def forward(self, x, *args, **kwargs):
        self.seen.append(sorted(name for name, module in (("self", self.self_attn), ("cross", self.cross_attn)) if "forward" in module.__dict__))
        return self.cross_attn(x)


def _stub_dit(count=2):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(_StubBlock() for _ in range(count))
    return dit


def _overridden(dit):
    return [sorted(name for name, module in (("block", b), ("self", b.self_attn), ("cross", b.cross_attn)) if "forward" in module.__dict__) for b in dit.blocks]


def test_forwards_are_overridden_only_inside_the_band_and_restored():
    dit = _stub_dit()
    seen = []

    def model_function(*args, **kwargs):
        seen.append(_overridden(dit))
        for block in dit.blocks:
            block(torch.zeros(B, 1, 1), None, None)

    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    wrapper(model_function, _args(1.0))
    wrapper(model_function, _args(0.85))
    wrapper(model_function, _args(0.84))
    # Block forwards for the model call; attention forwards for a block call
    assert seen[0] == seen[1] == [["block"]] * 2
    assert all(block.seen == [["cross", "self"], ["cross", "self"], []] for block in dit.blocks)
    assert seen[2] == [[], []]
    assert _overridden(dit) == [[], []]

    raw = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.0)).model_options["model_function_wrapper"]
    raw(model_function, _args(1.0))
    assert seen[3] == [["block"]] * 2
    assert all(block.seen[-1] == ["cross"] for block in dit.blocks)


def test_no_markers_or_uneven_batch_passes_through():
    dit = _stub_dit()
    seen = []
    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    wrapper(lambda *a, **k: seen.append(_overridden(dit)), _args(1.0, markers=()))
    wrapper(lambda *a, **k: seen.append(_overridden(dit)), _args(1.0, markers=(0, 1), batch=3))
    wrapper(lambda *a, **k: seen.append(_overridden(dit)), _args(1.0, markers=(1,)))
    assert seen == [[[], []]] * 3


def test_forwards_are_restored_when_the_model_call_raises():
    dit = _stub_dit()
    state = BoostState(gain=2.0, band=0.85, renorm=0.5)
    wrapper = apply_xattn_boost_patch(_Patcher(dit), state).model_options["model_function_wrapper"]

    def model_function(*args, **kwargs):
        assert "forward" in dit.blocks[0].__dict__
        dit.blocks[0](torch.zeros(B, 1, 1), None, None)
        raise RuntimeError("model call failed")

    with pytest.raises(RuntimeError, match="model call failed"):
        wrapper(model_function, _args(1.0))
    assert _overridden(dit) == [[], []]
    assert state.call is None and state.captured is None


def test_a_call_of_another_batch_is_left_alone():
    block = _block()
    x, emb, context, lora = _inputs()
    dit = _dit(block)
    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    with torch.no_grad():
        # the markers describe a batch of 2; a nested call runs a single row
        out = wrapper(_run(dit, x[:1], emb[:1], context[:1], lora[:1]), _args(1.0))
        plain = _run(dit, x[:1], emb[:1], context[:1], lora[:1])(None, None)
    assert torch.equal(out, plain)


def test_chains_previous_wrapper_and_propagates_its_flags():
    dit = _stub_dit()
    calls = []

    def previous(model_function, args):
        calls.append(True)
        return model_function(args["input"], args["timestep"], **args["c"])

    previous.__spectrum_cache_safe__ = True
    wrapper = apply_xattn_boost_patch(_Patcher(dit, {"model_function_wrapper": previous}), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    wrapper(lambda *a, **k: None, _args(1.0))
    wrapper(lambda *a, **k: None, _args(0.5))
    assert calls == [True, True]
    assert wrapper.__spectrum_cache_safe__ is True and wrapper.__resolution_independent__ is False

    alone = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    assert alone.__spectrum_cache_safe__ is True and alone.__resolution_independent__ is True


def test_spectrum_is_forced_inside_the_band_only():
    patched = apply_xattn_boost_patch(_Patcher(_stub_dit()), BoostState(gain=2.0, band=0.85, renorm=0.5))
    force = patched.model_options["transformer_options"][spectrum_force.KEY]
    assert spectrum_force.is_forced(force, 1.0) and spectrum_force.is_forced(force, 0.85)
    assert not spectrum_force.is_forced(force, 0.8)


@pytest.mark.parametrize("renorm", [0.0, 0.5])
def test_composes_with_regional_in_either_wrapper_order(renorm):
    # Regional's cross-attention override replaces the positive row with the
    # region context's output (full coverage, strength 1). The boost scales
    # the output the block adds, whichever wrapper installs its override
    # first; Regional patches first in script order, so its override lands on
    # top of a model-call boost override and used to leave the regions
    # unboosted.
    sys.path.insert(0, str(ROOT.parent / "sd_forge_anima_regional"))
    from lib_anima_regional.regional import Region, RegionalState, apply_regional_patch

    block = _block()
    x, emb, context, lora = _inputs()
    region_context = torch.randn(1, 5, CTX, generator=torch.Generator().manual_seed(5))

    def regional(model):
        state = RegionalState(regions=[Region(region_context, 0.0, 0.0, 1.0, 1.0, 1.0)], blocks={0}, feather=0.0, base_preserve=0.0, start_sigma=100.0, end_sigma=0.0)
        return apply_regional_patch(model, state)

    def boost(model):
        return apply_xattn_boost_patch(model, BoostState(gain=2.0, band=0.85, renorm=renorm))

    def run(wrap, model_function=None):
        dit = _dit(block)
        dit.patch_temporal = dit.patch_spatial = 1
        wrapper = wrap(_Patcher(dit)).model_options["model_function_wrapper"]
        with torch.no_grad():
            return wrapper(model_function or _run(dit, x, emb, context, lora), _args(0.9))

    options = {"cond_or_uncond": [0, 1]}
    expected = run(regional, lambda *a, **k: _reference(block, x, emb, context, lora, torch.tensor([2.0, 1.0]), renorm, options))
    regional_only = run(regional)
    for wrap in (lambda model: boost(regional(model)), lambda model: regional(boost(model))):
        out = run(wrap)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert torch.equal(out[1], regional_only[1]) and not torch.allclose(out[0], regional_only[0])


@functools.cache
def _artist_mixer():
    """The Artist Mixer script, loaded through the shared harness like its own suite does."""
    mixer_root = ROOT.parent / "anima-artist-scheduled-mixer"
    if str(mixer_root) not in sys.path:
        sys.path.insert(0, str(mixer_root))
    spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    import modules

    try:
        from modules import sd_models  # noqa: F401
    except ImportError:
        # The model loader needs the full WebUI install; the Mixer's wrapper does not use it
        modules.sd_models = sys.modules["modules.sd_models"] = types.ModuleType("modules.sd_models")
    return harness.load_script(mixer_root / "scripts" / "anima_artist_scheduled_mixer.py", "anima_artist_mixer_under_boost_test")


@pytest.mark.parametrize("renorm", [0.0, 0.5])
def test_composes_with_the_artist_mixer_in_either_wrapper_order(renorm):
    # The Mixer's cross-attention override, installed for each model call like
    # Regional's, blends an artist's output into the positive row (output
    # average, strength 0.8). The boost scales the blended output the block
    # adds, whichever wrapper installs its override first.
    mixer = _artist_mixer()
    block = _block()
    x, emb, context, lora = _inputs()
    artist_context = torch.randn(1, 5, CTX, generator=torch.Generator().manual_seed(5))
    states = []

    def artist_mixer(model):
        artist = mixer.ArtistRuntime("artist", 0.6, {0}, 0.0, 1.0, 0.5, mixer.CURVE_HOLD, artist_context)
        state = mixer.MixerState("run", True, 0.8, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, [artist])
        states.append(state)
        model = model.clone()
        mixer._install_model_wrapper(model, model.model.diffusion_model, state)
        return model

    def boost(model):
        return apply_xattn_boost_patch(model, BoostState(gain=2.0, band=0.85, renorm=renorm))

    def run(wrap, model_function=None):
        dit = _dit(block)
        wrapper = wrap(_Patcher(dit)).model_options["model_function_wrapper"]
        with torch.no_grad():
            out = wrapper(model_function or _run(dit, x, emb, context, lora), _args(0.9))
        assert all("forward" not in module.__dict__ for module in (block, block.self_attn, block.cross_attn))
        return out

    options = {"cond_or_uncond": [0, 1]}
    expected = run(artist_mixer, lambda *a, **k: _reference(block, x, emb, context, lora, torch.tensor([2.0, 1.0]), renorm, options))
    mixer_only = run(artist_mixer)
    with torch.no_grad():
        plain = _run(_dit(block), x, emb, context, lora)(None, None)
    assert torch.equal(mixer_only[1], plain[1]) and not torch.allclose(mixer_only[0], plain[0])  # the artist mixed in
    for wrap in (lambda model: boost(artist_mixer(model)), lambda model: artist_mixer(boost(model))):
        out = run(wrap)
        assert states[-1].active_calls == 1 and states[-1].enabled
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert torch.equal(out[1], mixer_only[1]) and not torch.allclose(out[0], mixer_only[0])


@pytest.mark.parametrize("renorm", [0.0, 0.5])
def test_composes_with_dave_in_either_wrapper_order(renorm):
    # Both replace the block forward for the whole model call. DAVE removes
    # part of the token mean of the boosted, norm-matched block output (the
    # default hook order, DAVE first, so the boost's wrapper runs first), also
    # when the boost patches first: its (s - 1) * x_new must not escape it.
    sys.path.insert(0, str(ROOT.parent / "sd_forge_anima_dave"))
    from lib_anima_dave.dave import DaveState, apply_dave_patch, attenuate_dc

    block = _block()
    x, emb, context, lora = _inputs()

    def dave(model):
        return apply_dave_patch(model, DaveState(blocks=(0,), strength=0.3, tau=0.1))

    def boost(model):
        return apply_xattn_boost_patch(model, BoostState(gain=2.0, band=0.85, renorm=renorm))

    def run(wrap):
        dit = _dit(block)
        wrapper = wrap(_Patcher(dit)).model_options["model_function_wrapper"]
        args = _args(1.0)
        args["c"]["transformer_options"]["sampling_sigmas"] = torch.linspace(1.0, 0.0, 11)
        with torch.no_grad():
            out = wrapper(_run(dit, x, emb, context, lora), args)
        assert all("forward" not in module.__dict__ for module in (block, block.self_attn, block.cross_attn))
        return out

    with torch.no_grad():
        expected = attenuate_dc(_reference(block, x, emb, context, lora, torch.tensor([2.0, 1.0]), renorm), 0.3)
    for wrap in (lambda model: boost(dave(model)), lambda model: dave(boost(model))):
        torch.testing.assert_close(run(wrap), expected, rtol=1e-5, atol=1e-5)


def _modulation_linears(block):
    return [block.adaln_modulation_self_attn[2], block.adaln_modulation_cross_attn[2]]


def _count_calls(modules):
    calls = [0] * len(modules)
    for i, module in enumerate(modules):
        module.register_forward_hook(lambda *_a, i=i: calls.__setitem__(i, calls[i] + 1))
    return calls


@pytest.mark.parametrize("renorm", [0.5, 1.0])
def test_captured_modulation_is_bit_identical_to_the_recompute(renorm, monkeypatch):
    from lib_anima_xattn_boost import boost

    block, _, captured, _ = _boosted(renorm=renorm)
    # A block that does not run its modulation Linears through the capture
    # gets the gates recomputed: the old path, bit for bit.
    monkeypatch.setattr(boost, "_modulation_linear", lambda *_a: None)
    _, _, recomputed, _ = _boosted(renorm=renorm)
    assert torch.equal(captured, recomputed)


def test_modulation_linears_run_once_per_boosted_block_call(monkeypatch):
    from lib_anima_xattn_boost import boost

    blocks = [_block(0), _block(1)]
    x, emb, context, lora = _inputs()
    dit = _dit(*blocks)
    counts = [_count_calls(_modulation_linears(block)) for block in blocks]
    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    with torch.no_grad():
        for _ in range(2):
            wrapper(_run(dit, x, emb, context, lora), _args(1.0))
    assert counts == [[2, 2], [2, 2]]  # two model calls, once each
    # The recompute fallback runs them twice per call (what the capture saves)
    monkeypatch.setattr(boost, "_modulation_linear", lambda *_a: None)
    with torch.no_grad():
        wrapper(_run(dit, x, emb, context, lora), _args(1.0))
    assert counts == [[4, 4], [4, 4]]


def _no_overrides(block):
    from modules.forward_override import overridden_modules

    modules = [block, block.self_attn, block.cross_attn, *_modulation_linears(block)]
    active = {id(module) for module in overridden_modules()}
    return all("forward" not in module.__dict__ and id(module) not in active for module in modules)


def test_modulation_captures_are_on_for_the_block_call_only():
    import gc
    import weakref

    block = _block()
    x, emb, context, lora = _inputs()
    dit = _dit(block)
    seen, outputs = [], []
    for linear in _modulation_linears(block):
        linear.register_forward_hook(lambda _m, _i, out: outputs.append(weakref.ref(out)))
    block.mlp.register_forward_hook(lambda *_a: seen.append(["forward" in m.__dict__ for m in _modulation_linears(block)]))
    state = BoostState(gain=2.0, band=0.85, renorm=0.5)
    wrapper = apply_xattn_boost_patch(_Patcher(dit), state).model_options["model_function_wrapper"]
    with torch.no_grad():
        out = wrapper(_run(dit, x, emb, context, lora), _args(1.0))
    assert seen == [[True, True]] and _no_overrides(block) and state.captured is None
    del out
    gc.collect()
    assert len(outputs) == 2 and all(ref() is None for ref in outputs)  # nothing kept alive

    # renorm 0 needs no gates: nothing captured
    raw = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.0)).model_options["model_function_wrapper"]
    with torch.no_grad():
        raw(_run(dit, x, emb, context, lora), _args(1.0))
    assert seen[-1] == [False, False] and _no_overrides(block)


def test_modulation_captures_are_removed_when_the_block_raises():
    block = _block()
    x, emb, context, lora = _inputs()
    dit = _dit(block)

    def fail(*_a):
        assert all("forward" in m.__dict__ for m in _modulation_linears(block))
        raise RuntimeError("block failed")

    block.mlp.register_forward_hook(fail)
    state = BoostState(gain=2.0, band=0.85, renorm=0.5)
    wrapper = apply_xattn_boost_patch(_Patcher(dit), state).model_options["model_function_wrapper"]
    with torch.no_grad(), pytest.raises(RuntimeError, match="block failed"):
        wrapper(_run(dit, x, emb, context, lora), _args(1.0))
    assert _no_overrides(block) and state.captured is None and state.call is None
