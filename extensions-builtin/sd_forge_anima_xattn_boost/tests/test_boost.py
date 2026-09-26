import pathlib
import sys
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


def _reference(block, x, emb, context, lora, gains, frac):
    """anima_lora 703b34b ``Block._forward`` with ``_xattn_gain`` (per row) and
    ``img`` renorm, on remi's Block weights (no RoPE)."""

    silu = torch.nn.functional.silu(emb)

    def modulation(mlp):
        return [chunk[:, :, None, None, :] for chunk in (mlp[2](mlp[1](silu)) + lora).chunk(3, dim=-1)]

    shift_sa, scale_sa, gate_sa = modulation(block.adaln_modulation_self_attn)
    shift_ca, scale_ca, gate_ca = modulation(block.adaln_modulation_cross_attn)
    shift_mlp, scale_mlp, gate_mlp = modulation(block.adaln_modulation_mlp)
    shape = x.shape[1:4]
    result = block.self_attn(_fn(x, block.layer_norm_self_attn, scale_sa, shift_sa).flatten(1, 3), None).unflatten(1, shape)
    x = x + gate_sa * result
    result = block.cross_attn(_fn(x, block.layer_norm_cross_attn, scale_ca, shift_ca).flatten(1, 3), context).unflatten(1, shape)
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


def test_row_gains():
    assert row_gains([0, 2], 4, 2.0).flatten().tolist() == [2.0, 1.0, 2.0, 1.0]


class _Stub(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _stub_dit(count=2):
    dit = torch.nn.Module()
    blocks = []
    for _ in range(count):
        block = torch.nn.Module()
        block.self_attn, block.cross_attn = _Stub(), _Stub()
        blocks.append(block)
    dit.blocks = torch.nn.ModuleList(blocks)
    return dit


def _overridden(dit):
    return [sorted(name for name, module in (("block", b), ("self", b.self_attn), ("cross", b.cross_attn)) if "forward" in module.__dict__) for b in dit.blocks]


def test_forwards_are_overridden_only_inside_the_band_and_restored():
    dit = _stub_dit()
    seen = []

    def model_function(*args, **kwargs):
        seen.append(_overridden(dit))

    wrapper = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.5)).model_options["model_function_wrapper"]
    wrapper(model_function, _args(1.0))
    wrapper(model_function, _args(0.85))
    wrapper(model_function, _args(0.84))
    assert seen[0] == seen[1] == [["block", "cross", "self"]] * 2
    assert seen[2] == [[], []]
    assert _overridden(dit) == [[], []]

    raw = apply_xattn_boost_patch(_Patcher(dit), BoostState(gain=2.0, band=0.85, renorm=0.0)).model_options["model_function_wrapper"]
    raw(model_function, _args(1.0))
    assert seen[3] == [["cross"]] * 2


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
        assert "forward" in dit.blocks[0].cross_attn.__dict__
        raise RuntimeError("model call failed")

    with pytest.raises(RuntimeError, match="model call failed"):
        wrapper(model_function, _args(1.0))
    assert _overridden(dit) == [[], []]
    assert state.gains is None and state.captured is None


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
