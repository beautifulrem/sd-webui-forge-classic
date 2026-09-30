import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "lib_anima_freefuse" / "masks.py"
SPEC = importlib.util.spec_from_file_location("anima_freefuse_masks", MODULE_PATH)
MASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MASKS)
fit_mask_batch = MASKS.fit_mask_batch
generate_masks = MASKS.generate_masks
morphological_clean = MASKS._morphological_clean


def _clean(image):
    height, width = image.shape
    return morphological_clean(image.reshape(1, -1), height, width).view(height, width)


def test_morphological_clean_keeps_a_clean_region_in_place():
    square = torch.zeros(16, 16)
    square[4:12, 4:12] = 1.0
    assert torch.equal(_clean(square), square)

    corner = torch.zeros(9, 7)  # non-square grid, region on the border
    corner[:5, 3:] = 1.0
    assert torch.equal(_clean(corner), corner)
    assert torch.equal(_clean(torch.ones(5, 6)), torch.ones(5, 6))


def test_morphological_clean_removes_specks_and_fills_pinholes():
    square = torch.zeros(16, 16)
    square[4:12, 4:12] = 1.0
    noisy = square.clone()
    noisy[1, 14] = 1.0  # isolated speck
    noisy[7, 7] = 0.0  # pinhole
    assert torch.equal(_clean(noisy), square)


def test_generate_masks_preserves_per_image_assignments():
    left = torch.tensor(
        [
            [9.0, 9.0, 1.0, 1.0],
            [1.0, 1.0, 9.0, 9.0],
        ]
    )
    right = torch.tensor(
        [
            [1.0, 1.0, 9.0, 9.0],
            [9.0, 9.0, 1.0, 1.0],
        ]
    )
    background = torch.zeros(2, 4)

    masks = generate_masks(
        {"left": left, "right": right},
        background,
        height=2,
        width=2,
        bg_scale=0.0,
        iterations=1,
        feather=0,
    )

    assert masks["left"].shape == (2, 2, 2)
    assert not torch.equal(masks["left"][0], masks["left"][1])
    assert torch.equal(masks["left"][0], masks["right"][1])


def test_feathered_masks_still_partition_the_image_up_to_its_edges():
    # Left/right halves of an 8x8 grid, every token foreground
    column = torch.arange(8.0).repeat(8, 1).reshape(1, 64)
    left, right = 8.0 - column, column
    for feather in (1, 2):
        masks = generate_masks(
            {"left": left, "right": right},
            torch.full((1, 64), -1e3),
            height=8,
            width=8,
            bg_scale=1.0,
            iterations=1,
            feather=feather,
        )
        total = masks["left"] + masks["right"]
        # Edges and corners included: feathering blends regions, not the frame
        assert torch.allclose(total, torch.ones_like(total))
        # Only the columns within ``feather`` of the inner boundary are blended
        blended = (masks["left"][0] > 0) & (masks["left"][0] < 1)
        assert blended.any(dim=0).nonzero().flatten().tolist() == list(range(4 - feather, 4 + feather))


def test_generate_masks_rejects_inconsistent_batches():
    try:
        generate_masks(
            {"one": torch.ones(2, 4), "two": torch.ones(1, 4)},
            torch.ones(2, 4),
            height=2,
            width=2,
            bg_scale=1.0,
            iterations=1,
            feather=0,
        )
    except ValueError as error:
        assert "inconsistent batch sizes" in str(error)
    else:
        raise AssertionError("inconsistent FreeFuse batches did not fail")


def test_mask_batch_repeats_by_cfg_branch_without_averaging():
    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    fitted = fit_mask_batch(masks, 4)

    assert torch.equal(fitted[:2], masks)
    assert torch.equal(fitted[2:], masks)


def test_mask_batch_rejects_non_integral_expansion():
    masks = torch.ones(2, 1, 2)
    try:
        fit_mask_batch(masks, 3)
    except RuntimeError as error:
        assert "does not match model batch" in str(error)
    else:
        raise AssertionError("invalid FreeFuse mask batch did not fail")
