import torch

from backend.nn.anima import apply_rotary_pos_emb
from backend.nn.vae import DiagonalGaussianDistribution


def test_anima_rotary_embedding_preserves_half_rotation_result():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.zeros((1, 1, 4))
    sin = torch.ones((1, 1, 4))

    result = apply_rotary_pos_emb(x, cos, sin)

    assert torch.equal(result, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))


def test_vae_gaussian_sample_keeps_seeded_affine_result():
    parameters = torch.tensor([[[[2.0]], [[0.0]]]])
    distribution = DiagonalGaussianDistribution(parameters)
    torch.manual_seed(123)

    result = distribution.sample()

    assert torch.allclose(
        result,
        torch.tensor([[[[1.8885329]]]]),
        atol=1e-6,
        rtol=0.0,
    )
