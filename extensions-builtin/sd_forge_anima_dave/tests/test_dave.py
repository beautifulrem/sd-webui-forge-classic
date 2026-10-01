import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_dave.dave import DaveState, apply_dave_patch, attenuate_dc, in_window, model_blocks, window_steps
from modules import spectrum_force


def test_attenuation_scales_the_token_mean_and_keeps_the_residual():
    x = torch.randn(2, 1, 4, 6, 8) + 3.0
    out = attenuate_dc(x, 0.3)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    torch.testing.assert_close(out.mean(dim=(1, 2, 3), keepdim=True), 0.7 * mean)
    # AC part unchanged
    torch.testing.assert_close(out - out.mean(dim=(1, 2, 3), keepdim=True), x - mean)
    # Per sample and per channel
    torch.testing.assert_close(out[1], attenuate_dc(x[1:], 0.3)[0])


def test_attenuation_keeps_dtype_and_zero_is_identity():
    x = torch.randn(1, 1, 3, 3, 4).to(torch.bfloat16)
    assert attenuate_dc(x, 0.5).dtype is torch.bfloat16
    y = torch.randn(1, 1, 3, 3, 4)
    torch.testing.assert_close(attenuate_dc(y, 0.0), y)


def test_bf16_attenuation_matches_the_fp32_computation():
    torch.manual_seed(0)
    x = (torch.randn(2, 1, 16, 16, 64) * 4.0 + 20.0).to(torch.bfloat16)
    shift = 0.3 * x.float().mean(dim=(1, 2, 3), keepdim=True)
    expected = x.float() - shift
    # One bf16 rounding (half an ulp, <= 2^-8 relative) of the shift and one of the result
    bound = 2**-8 * (expected.abs() + shift.abs())
    assert ((attenuate_dc(x, 0.3).float() - expected).abs() <= bound).all()


def test_reference_frames_are_neither_averaged_nor_edited():
    x = torch.randn(1, 3, 2, 2, 4)
    out = attenuate_dc(x, 1.0, frames=1)
    torch.testing.assert_close(out[:, 1:], x[:, 1:])
    torch.testing.assert_close(out[:, :1], x[:, :1] - x[:, :1].mean(dim=(1, 2, 3), keepdim=True))


def test_window_covers_only_the_leading_fraction():
    sigmas = torch.linspace(1.0, 0.0, 31)  # 30 steps
    assert window_steps(30, 0.1) == 3
    active = [in_window(sigmas[i : i + 1], sigmas, 0.1) for i in range(30)]
    assert active == [True] * 3 + [False] * 27
    # Always at least one step; tau 1 covers the whole pass
    assert window_steps(8, 0.01) == 1
    assert all(in_window(float(sigmas[i]), sigmas, 1.0) for i in range(30))
    # A tau that is not a finite number reads as the default: round(nan) raised mid-sampling
    for tau in (float("nan"), float("inf"), "nan"):
        assert window_steps(30, tau) == window_steps(30, 0.1)


def test_window_is_off_without_a_schedule_and_on_img2img_tails():
    assert not in_window(1.0, None, 0.1)
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    # img2img with denoise 0.5 starts at sigma 0.5: outside the first 10%
    assert not in_window(0.5, sigmas, 0.1)
    assert in_window(1.0, sigmas, 0.1)


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (28, tuple(range(8, 19))),
        (40, (12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26)),
        (52, (16, 17, 20, 21, 24, 25, 26, 29, 30, 33, 34)),
    ],
)
def test_2b_block_band_maps_onto_original_blocks(total, expected):
    from modules.anima_lora_support import ANIMA_BLOCK_MAPPINGS

    blocks = model_blocks("8-18", total)
    assert blocks == expected
    if total != 28:
        mapping = ANIMA_BLOCK_MAPPINGS[(28, total)]
        assert [mapping[index] for index in blocks] == list(range(8, 19))


def test_block_list_is_clipped_to_the_2b_layout():
    assert model_blocks("26-40", 28) == (26, 27)
    with pytest.raises(ValueError, match="empty"):
        model_blocks("30", 52)


class _Patcher:
    def __init__(self, dit, options=None):
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        return _Patcher(self.model.diffusion_model, self.model_options)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class _Block(torch.nn.Module):
    def forward(self, x, emb=None, context=None, **kwargs):
        return x + 1.0


def _dit(count=3):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Block() for _ in range(count)])
    dit.patch_temporal = 1
    return dit


SIGMAS = torch.linspace(1.0, 0.0, 11)  # 10 steps, tau 0.1 -> step 0 only


def _args(sigma):
    x = torch.randn(1, 4, 1, 2, 2)
    return {"input": x, "timestep": torch.tensor([float(sigma)]), "c": {"transformer_options": {"sampling_sigmas": SIGMAS}}}


def _run_blocks(dit):
    def model_function(x, t, **c):
        h = torch.randn(1, 1, 2, 2, 4) + 5.0
        overridden = ["forward" in block.__dict__ for block in dit.blocks]
        for block in dit.blocks:
            h = block(h)
        return h, overridden

    return model_function


def test_blocks_are_attenuated_only_inside_the_window_and_restored():
    dit = _dit()
    patched = apply_dave_patch(_Patcher(dit), DaveState(blocks=(1,), strength=0.5, tau=0.1))
    wrapper = patched.model_options["model_function_wrapper"]

    out, overridden = wrapper(_run_blocks(dit), _args(1.0))
    assert overridden == [False, True, False]
    out, overridden = wrapper(_run_blocks(dit), _args(float(SIGMAS[1])))
    assert overridden == [False, False, False]
    assert all("forward" not in block.__dict__ for block in dit.blocks)


def test_forwards_are_restored_when_the_model_call_raises():
    dit = _dit()
    state = DaveState(blocks=(0, 2), strength=0.3, tau=0.1)
    wrapper = apply_dave_patch(_Patcher(dit), state).model_options["model_function_wrapper"]

    def model_function(x, t, **c):
        assert "forward" in dit.blocks[0].__dict__
        raise RuntimeError("model call failed")

    with pytest.raises(RuntimeError, match="model call failed"):
        wrapper(model_function, _args(1.0))
    assert all("forward" not in block.__dict__ for block in dit.blocks)
    assert state.frames is None


def test_block_output_mean_is_attenuated():
    dit = _dit(1)
    wrapper = apply_dave_patch(_Patcher(dit), DaveState(blocks=(0,), strength=0.25, tau=0.1)).model_options["model_function_wrapper"]
    h = torch.randn(1, 1, 2, 2, 4) + 5.0

    def model_function(x, t, **c):
        return dit.blocks[0](h)

    out = wrapper(model_function, _args(1.0))
    torch.testing.assert_close(out, attenuate_dc(h + 1.0, 0.25))


def test_an_edit_models_reference_frame_passes_the_wrapper_untouched():
    # The call's input holds the generated frame only; Anima.forward appends
    # the reference latent's frame after it, so the blocks see two
    dit = _dit(1)
    wrapper = apply_dave_patch(_Patcher(dit), DaveState(blocks=(0,), strength=1.0, tau=0.1)).model_options["model_function_wrapper"]
    h = torch.randn(1, 2, 2, 2, 4) + 5.0

    def model_function(x, t, **c):
        return dit.blocks[0](h)

    out = wrapper(model_function, _args(1.0))
    torch.testing.assert_close(out[:, 1:], h[:, 1:] + 1.0)
    torch.testing.assert_close(out[:, :1], attenuate_dc(h[:, :1] + 1.0, 1.0))


def test_chains_previous_wrapper_and_propagates_its_flags():
    dit = _dit()
    calls = []

    def previous(model_function, args):
        calls.append(True)
        return model_function(args["input"], args["timestep"], **args["c"])

    previous.__spectrum_cache_safe__ = True
    patched = apply_dave_patch(_Patcher(dit, {"model_function_wrapper": previous}), DaveState(blocks=(0,), strength=0.3, tau=0.1))
    wrapper = patched.model_options["model_function_wrapper"]
    wrapper(_run_blocks(dit), _args(1.0))
    assert calls == [True]
    assert wrapper.__spectrum_cache_safe__ is True
    assert wrapper.__resolution_independent__ is False

    alone = apply_dave_patch(_Patcher(dit), DaveState(blocks=(0,), strength=0.3, tau=0.1)).model_options["model_function_wrapper"]
    assert alone.__spectrum_cache_safe__ is True and alone.__resolution_independent__ is True


def test_spectrum_is_forced_inside_the_window_only():
    dit = _dit()
    state = DaveState(blocks=(0,), strength=0.3, tau=0.1)
    patched = apply_dave_patch(_Patcher(dit), state)
    force = patched.model_options["transformer_options"][spectrum_force.KEY]
    # Before the first call has published the schedule: conservative
    assert spectrum_force.is_forced(force, 0.5)
    patched.model_options["model_function_wrapper"](_run_blocks(dit), _args(1.0))
    assert spectrum_force.is_forced(force, 1.0)
    assert not spectrum_force.is_forced(force, float(SIGMAS[3]))
