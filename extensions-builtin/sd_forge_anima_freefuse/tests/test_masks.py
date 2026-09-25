import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "lib_anima_freefuse" / "masks.py"
SPEC = importlib.util.spec_from_file_location("anima_freefuse_masks", MODULE_PATH)
MASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MASKS)
fit_mask_batch = MASKS.fit_mask_batch
generate_masks = MASKS.generate_masks


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
