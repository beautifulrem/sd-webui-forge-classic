import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.argv = sys.argv[:1]  # modules.shared parses sys.argv on import
EXTENSION_PATH = Path(__file__).parents[1]
for path in (str(EXTENSION_PATH.parents[1]), str(EXTENSION_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.nn.anima import SelfCrossAttention  # noqa: E402
from backend.sampling import condition, sampling_function  # noqa: E402
from lib_negpip import anima as negpip_anima  # noqa: E402
from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY  # noqa: E402


class _Engine:
    """Stands in for AnimaTextProcessingEngine: per-line T5 weight multipliers."""

    def __init__(self, multipliers):
        self.multipliers = multipliers

    def tokenize_line(self, line):
        return [SimpleNamespace(t5_multipliers=chunk) for chunk in self.multipliers[line]]


class _Prompt(list):
    is_negative_prompt = False


@pytest.fixture
def negatives_seen():
    saved = negpip_anima._NEGATIVES_SEEN[0]
    yield negpip_anima._NEGATIVES_SEEN
    negpip_anima._NEGATIVES_SEEN[0] = saved


@pytest.fixture
def class_hook():
    original = SelfCrossAttention.forward
    negpip_anima._hook_forwards(False)
    try:
        yield original
    finally:
        negpip_anima._hook_forwards(True)
        SelfCrossAttention.forward = original


def _attention():
    torch.manual_seed(0)
    attention = SelfCrossAttention(8, context_dim=6, n_heads=2, head_dim=4)
    captured = {}

    def compute_attention(q, k, v, transformer_options=None, mask=None):
        captured.update(k=k, v=v)
        return q

    attention.compute_attention = compute_attention
    return attention, captured


def test_mask_marks_negative_weights_and_fits_the_token_length():
    engine = _Engine({"a": [[1.0, -0.5], [2.0]], "long": [[-1.0, 1.0, -1.0, -1.0]], "plain": [[]]})

    mask = negpip_anima._build_negpip_mask(engine, "a", 5, torch.device("cpu"), torch.float32)
    assert mask.tolist() == [1.0, -1.0, 1.0, 1.0, 1.0]  # chunks joined, padded with 1
    assert negpip_anima._build_negpip_mask(engine, "long", 3, "cpu", torch.float32).tolist() == [-1.0, 1.0, -1.0]
    assert negpip_anima._build_negpip_mask(engine, "plain", 2, "cpu", torch.float16).tolist() == [1.0, 1.0]


def test_conditioning_hook_restores_signs_and_returns_the_v_mask(negatives_seen):
    class Model:
        def __init__(self):
            self.text_processing_engine_anima = _Engine({"(cat:-1)": [[1.0, -1.0]], "dog": [[]]})

        def get_learned_conditioning(self, prompt):
            # the encoder applied the negative weight to token 1 of the first line
            return [torch.tensor([[1.0, 1.0], [-2.0, -2.0], [3.0, 3.0]]), torch.full((3, 2), 4.0)]

    model = Model()
    negatives_seen[0] = False
    negpip_anima._hook_get_learned_conditioning(model, False)
    try:
        conds = model.get_learned_conditioning(_Prompt(["(cat:-1)", "dog"]))
    finally:
        negpip_anima._hook_get_learned_conditioning(model, True)

    assert set(conds) == {"crossattn", "c_negpip_mask"}
    assert conds["crossattn"][0].tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert torch.equal(conds["crossattn"][1], torch.full((3, 2), 4.0))
    assert conds["c_negpip_mask"].shape == (2, 3, 1)
    assert conds["c_negpip_mask"][:, :, 0].tolist() == [[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
    assert negatives_seen[0] is True
    assert "get_learned_conditioning" not in model.__dict__
    assert isinstance(model.get_learned_conditioning(_Prompt(["dog"])), list)


def test_dit_hook_passes_the_mask_only_once_a_negative_was_seen(negatives_seen):
    class DiT(torch.nn.Module):
        def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
            self.seen = kwargs["transformer_options"]
            return x

    dit, mask, options = DiT(), torch.ones(1, 3, 1), {"other": 1}
    negpip_anima._hook_dit_forward(dit, False)
    try:
        negatives_seen[0] = True
        dit.forward(torch.zeros(1), torch.zeros(1), torch.zeros(1, 3, 2), c_negpip_mask=mask, transformer_options=options)
        assert dit.seen[NEGPIP_MASK_KEY] is mask and dit.seen["other"] == 1
        assert options == {"other": 1}  # the caller's shared dict is not written to

        negatives_seen[0] = False
        dit.forward(torch.zeros(1), torch.zeros(1), torch.zeros(1, 3, 2), c_negpip_mask=mask, transformer_options=options)
        assert dit.seen[NEGPIP_MASK_KEY] is None
    finally:
        negpip_anima._hook_dit_forward(dit, True)
    assert "forward" not in dit.__dict__


def test_cross_attention_hook_negates_values_only(class_hook):
    attention, captured = _attention()
    x, context = torch.randn(2, 5, 8), torch.randn(2, 3, 6)
    mask = torch.tensor([[[1.0], [-1.0], [1.0]]])  # batch 1: tiled onto the CFG batch
    _, plain_k, plain_v = attention.compute_qkv(x, context)
    _, _, masked_v = attention.compute_qkv(x, context * mask)
    assert not torch.allclose(plain_v, masked_v)

    attention(x, context, None, {NEGPIP_MASK_KEY: mask})
    assert torch.allclose(captured["k"], plain_k)
    assert torch.allclose(captured["v"], masked_v)

    attention(x, context, None, {NEGPIP_MASK_KEY: mask, NEGPIP_CONTEXT_MASK_KEY: None})  # swapped-in context
    assert torch.allclose(captured["v"], plain_v)
    attention(x, context, None, {})
    assert torch.allclose(captured["v"], plain_v)


def test_unpatching_restores_the_original_attention_forward(class_hook):
    original = class_hook
    assert SelfCrossAttention.forward is not original

    negpip_anima._hook_forwards(True)
    assert SelfCrossAttention.__dict__["forward"] is original

    # Another patch wrapped ours: unpatching keeps its chain but ours passes through.
    negpip_anima._hook_forwards(False)
    ours = SelfCrossAttention.forward
    SelfCrossAttention.forward = lambda self, *args, **kwargs: ours(self, *args, **kwargs)
    negpip_anima._hook_forwards(True)
    attention, captured = _attention()
    x, context = torch.randn(1, 5, 8), torch.randn(1, 3, 6)
    attention(x, context, None, {NEGPIP_MASK_KEY: torch.tensor([[[-1.0], [-1.0], [-1.0]]])})
    assert torch.allclose(captured["v"], attention.compute_qkv(x, context)[2])


def test_patch_and_unpatch_round_trip(monkeypatch, negatives_seen):
    class Model:
        def get_learned_conditioning(self, prompt):
            return [torch.ones(2, 2) for _ in prompt]

    class DiT(torch.nn.Module):
        def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
            return x

    class NegPiP:
        _patched = [False, False]

    model, dit = Model(), DiT()
    model.text_processing_engine_anima = _Engine({"x": [[]]})
    model.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=dit)))
    monkeypatch.setattr(negpip_anima, "shared", SimpleNamespace(sd_model=model))  # the real one loads a checkpoint
    originals = SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions

    negpip_anima.patch_anima_negpip(NegPiP)
    try:
        hooked_attention = SelfCrossAttention.forward
        negpip_anima.patch_anima_negpip(NegPiP)  # already patched: no second layer
        assert NegPiP._patched[1] and SelfCrossAttention.forward is hooked_attention
        assert hooked_attention is not originals[0]
        assert "get_learned_conditioning" in model.__dict__ and "forward" in dit.__dict__
        assert sampling_function.compile_conditions is condition.compile_conditions is not originals[1]

        crossattn, mask = torch.ones(1, 2, 4), -torch.ones(1, 2, 1)
        (compiled,) = condition.compile_conditions({"crossattn": crossattn, "c_negpip_mask": mask})
        assert compiled["cross_attn"] is crossattn
        assert compiled["model_conds"]["c_negpip_mask"].cond is mask
    finally:
        negpip_anima.patch_anima_negpip(NegPiP, unpatch=True)

    assert not NegPiP._patched[1]
    assert (SelfCrossAttention.forward, condition.compile_conditions, sampling_function.compile_conditions) == originals
    assert "get_learned_conditioning" not in model.__dict__ and "forward" not in dit.__dict__
