import torch

from backend.nn.anima import VideoRopePosition3DEmb


def test_rope_embedding_is_built_once_per_grid_and_matches():
    emb = VideoRopePosition3DEmb(head_dim=128, h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0)
    x = torch.zeros(2, 1, 6, 5, 8)

    with torch.inference_mode():
        first = emb(x, device=x.device)
        again = emb(x, device=x.device)
        other = emb(torch.zeros(1, 1, 4, 4, 8), device=x.device)

    assert again is first
    assert torch.equal(first, emb._build(1, 6, 5, x.device))
    assert first.shape == (30, 64, 2, 2) and other.shape == (16, 64, 2, 2)

    for size in range(5, 10):  # bounded
        with torch.inference_mode():
            emb(torch.zeros(1, 1, size, size, 8), device=x.device)
    assert len(emb._cache) <= 4
