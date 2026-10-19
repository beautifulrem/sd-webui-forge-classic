"""Inpaint "Masked content: latent noise" keeps the init latent's shape.

Regression: for 5D latents (Anima/Wan, ``(B, C, 1, H, W)``) the noise was drawn as
``(B, C, H, W)`` and broadcast against the ``(1, C, 1, H, W)`` mask into a
``(1, C, C, H, W)`` latent -- a 16-frame "video" that grew the process to 24 GB on MPS
and stalled sampling.
"""

import pytest
import torch

from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


def _masks(channels, h, w, ndim):
    latmask = torch.zeros(channels, h, w)
    latmask[:, : h // 2, :] = 1.0  # top half is masked (regenerated)
    mask, nmask = 1.0 - latmask, latmask
    if ndim == 4:
        return mask[None], nmask[None]
    return mask[None, :, None], nmask[None, :, None]


@pytest.mark.parametrize("batch", [1, 2])
def test_5d_latent_noise_keeps_shape(batch):
    from modules import shared
    from modules.processing import create_random_tensors, fill_latent_noise

    c, h, w = 16, 8, 8
    init_latent = torch.randn(batch, c, 1, h, w, device=shared.device)
    mask, nmask = _masks(c, h, w, 5)
    mask, nmask = mask.to(shared.device), nmask.to(shared.device)
    seeds = [123 + i for i in range(batch)]

    filled = fill_latent_noise(init_latent, mask, nmask, seeds)

    assert filled.shape == init_latent.shape
    # unmasked half untouched, masked half is exactly the seeded per-image noise
    torch.testing.assert_close(filled[..., h // 2 :, :], init_latent[..., h // 2 :, :])
    noise = create_random_tensors((c, h, w), seeds).to(init_latent.device)
    torch.testing.assert_close(filled[:, :, 0, : h // 2, :], noise[:, :, : h // 2, :])


def test_4d_latent_noise_matches_legacy_formula():
    from modules import shared
    from modules.processing import create_random_tensors, fill_latent_noise

    c, h, w = 4, 8, 8
    init_latent = torch.randn(2, c, h, w, device=shared.device)
    mask, nmask = _masks(c, h, w, 4)
    mask, nmask = mask.to(shared.device), nmask.to(shared.device)
    seeds = [7, 8, 9]  # all_seeds may be longer than the batch

    filled = fill_latent_noise(init_latent, mask, nmask, seeds)

    legacy = init_latent * mask + create_random_tensors((c, h, w), seeds[0:2]).to(init_latent.device) * nmask
    assert filled.shape == init_latent.shape
    assert torch.equal(filled, legacy)


def test_img2img_init_uses_shape_preserving_fill():
    """The img2img init path routes latent noise through the shape-checked helper."""
    import inspect

    from modules.processing import StableDiffusionProcessingImg2Img

    src = inspect.getsource(StableDiffusionProcessingImg2Img.init)
    assert "fill_latent_noise(self.init_latent" in src
