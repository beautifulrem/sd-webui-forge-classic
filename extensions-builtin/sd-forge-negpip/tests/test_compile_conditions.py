"""NegPiP's compile_conditions hook, fed through Forge's real batching helpers."""

import sys
from pathlib import Path

import pytest
import torch

sys.argv = sys.argv[:1]  # modules.shared parses sys.argv on import
EXTENSION_PATH = Path(__file__).parents[1]
for path in (str(EXTENSION_PATH.parents[1]), str(EXTENSION_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.sampling import condition, sampling_function  # noqa: E402
from lib_negpip import anima as negpip_anima  # noqa: E402


@pytest.fixture
def hooked():
    originals = condition.compile_conditions, sampling_function.compile_conditions
    negpip_anima._hook_compile_conditions(False)
    try:
        yield condition.compile_conditions
    finally:
        negpip_anima._hook_compile_conditions(True)
        condition.compile_conditions, sampling_function.compile_conditions = originals


def _negpip_cond(signs):
    """Conditioning as NegPiP's get_learned_conditioning hook returns it."""
    mask = torch.tensor(signs, dtype=torch.float32).reshape(1, -1, 1)
    features = torch.arange(1, len(signs) * 3 + 1, dtype=torch.float32).reshape(1, len(signs), 3)
    return {"crossattn": features * mask, "c_negpip_mask": mask}


def test_cond_and_uncond_of_different_lengths_run_as_one_batch_with_aligned_masks(hooked):
    (cond,) = hooked(_negpip_cond([1, -1, 1, 1]))
    (uncond,) = hooked(_negpip_cond([-1, 1]))
    x = torch.zeros(1, 4, 2, 2)
    timestep = torch.ones(1)
    first = sampling_function.get_area_and_mult(cond, x, timestep)
    second = sampling_function.get_area_and_mult(uncond, x, timestep)

    assert sampling_function.can_concat_cond(first, second)
    batched = sampling_function.cond_cat([first.conditioning, second.conditioning])
    context, mask = batched["c_crossattn"], batched["c_negpip_mask"]
    assert context.shape == (2, 4, 3) and mask.shape == (2, 4, 1)
    # The mask tiles with its context: each token's V sign still matches the
    # (sign-baked) token it describes, also in the repeated uncond row.
    assert torch.equal(torch.sign(context), mask.expand_as(context))
    assert mask[:, :, 0].tolist() == [[1, -1, 1, 1], [-1, 1, -1, 1]]


def test_conditioning_without_a_mask_carries_only_the_context(hooked):
    crossattn = torch.ones(1, 3, 4)
    (compiled,) = hooked({"crossattn": crossattn})
    assert compiled["cross_attn"] is crossattn
    assert set(compiled["model_conds"]) == {"c_crossattn"}
    assert hooked(None) is None


def test_other_conditioning_forms_use_forge_compile_conditions(hooked):
    # SDXL-style dicts carry a pooled vector; bare tensors are the unhooked
    # Anima form. Both keep Forge's own model conds.
    (pooled,) = hooked({"crossattn": torch.ones(1, 3, 4), "vector": torch.ones(1, 8)})
    assert set(pooled["model_conds"]) == {"c_crossattn", "y"}
    (plain,) = hooked(torch.ones(1, 3, 4))
    assert set(plain["model_conds"]) == {"c_crossattn"}


def test_unhook_leaves_a_later_override_in_place():
    original = condition.compile_conditions
    saved = sampling_function.compile_conditions

    def later(cond):
        return None

    negpip_anima._hook_compile_conditions(False)
    try:
        sampling_function.compile_conditions = later
        negpip_anima._hook_compile_conditions(True)
        assert condition.compile_conditions is original
        assert sampling_function.compile_conditions is later
        negpip_anima._hook_compile_conditions(True)  # already removed: no-op
        assert condition.compile_conditions is original
    finally:
        negpip_anima._COMPILE_HOOK = None
        condition.compile_conditions = original
        sampling_function.compile_conditions = saved
