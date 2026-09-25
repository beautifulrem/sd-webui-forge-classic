import torch

from lib_artist_mixer.contexts import artist_mask_like, joined_mask, with_negpip_mask


def test_artist_mask_is_aligned_to_the_artist_context_batch():
    context = torch.zeros(4, 2, 3)

    mask = artist_mask_like(torch.tensor([[[1.0], [-1.0]]]), context)

    assert mask.shape == (4, 2, 1)
    assert artist_mask_like(None, context) is None


def test_joined_mask_fills_parts_without_a_mask_with_ones():
    base = torch.zeros(2, 3, 4)
    artist = torch.zeros(2, 2, 4)
    artist_mask = torch.tensor([[[-1.0], [1.0]]]).expand(2, -1, -1)

    joined = joined_mask([(base, None), (artist, artist_mask)], 1)

    assert joined.shape == (2, 5, 1)
    assert joined[0, :, 0].tolist() == [1.0, 1.0, 1.0, -1.0, 1.0]
    assert joined_mask([(base, None), (artist, None)], 1) is None


def test_foreign_context_options_carry_their_own_mask():
    base_mask = torch.ones(1, 3, 1)
    options = with_negpip_mask({"negpip_mask": base_mask, "other": 1}, None)

    assert options == {"negpip_mask": base_mask, "negpip_context_mask": None, "other": 1}
