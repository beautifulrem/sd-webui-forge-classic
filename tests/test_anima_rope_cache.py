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


def test_moving_or_casting_the_module_drops_cached_embeddings():
    """The cache is a plain dict, so ``.to()`` would not move it: offloading
    the model must not keep embeddings on the old device."""
    emb = VideoRopePosition3DEmb(head_dim=128, h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0)
    parent = torch.nn.Sequential(emb)
    x = torch.zeros(1, 1, 4, 4, 8)

    with torch.inference_mode():
        emb(x, device=x.device)
    assert emb._cache
    parent.to("cpu")  # recurses through _apply
    assert not emb._cache

    with torch.inference_mode():
        emb(x, device=x.device)
    parent.half()
    assert not emb._cache


def test_rope_embedding_traces_in_one_graph_under_compile():
    """Whole-model torch.compile must not break its graph at the cache key."""
    import torch._dynamo

    emb = VideoRopePosition3DEmb(head_dim=128, h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0)
    x = torch.zeros(1, 1, 4, 4, 8)
    torch._dynamo.reset()
    compiled = torch.compile(emb, backend="eager", fullgraph=True)

    with torch.no_grad():
        result = compiled(x, device=x.device)
    assert torch.equal(result, emb._build(1, 4, 4, x.device))
    assert not emb._cache  # the compiled graph rebuilds it; eager callers keep the cache
