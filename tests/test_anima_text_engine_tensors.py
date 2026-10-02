"""CPU tensor checks for the adapted upstream encoder, without checkpoints."""

import ast
import numbers
from pathlib import Path
from types import SimpleNamespace
from typing import Final, TypeAlias

import torch

from test_anima_text_engine_compat import Tokenizer, load_engine


ROOT = Path(__file__).resolve().parents[1]


class Encoder(torch.nn.Module):
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(151644, 2)
        self.calls = 0
        self.adapter_calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, *, embeds, **kwargs):
        self.calls += 1
        self.mask = attention_mask
        return embeds, None

    def preprocess_text_embeds(self, cond, ids):
        self.adapter_calls += 1
        return torch.ones((*ids.shape, cond.shape[-1]), device=cond.device)


def common_model():
    namespace = {
        "torch": torch, "numbers": numbers, "Final": Final, "TypeAlias": TypeAlias,
        "memory_management": SimpleNamespace(
            text_encoder_device=lambda: torch.device("cpu"),
            intermediate_device=lambda: torch.device("cpu"),
        ),
    }
    tree = ast.parse((ROOT / "backend/text_processing/_comfy.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "_comfy.py", "exec"), namespace)
    return namespace["SDClipModel"]


def make_clip(special_tokens):
    return common_model()(Encoder(), special_tokens=special_tokens, enable_attention_masks=True, return_attention_masks=True)


def test_qwen_left_and_right_padding_masks():
    clip = make_clip({"pad": 151643})
    embeds, mask, count, info = clip.process_tokens([[151643, 151643, 5, 6, 151643]], torch.device("cpu"))
    assert mask.tolist() == [[0, 0, 1, 1, 0]]
    assert count == [2]
    assert embeds.shape == (1, 5, 2)
    assert embeds.dtype == torch.float32
    assert info == []


def test_end_token_is_visible_but_following_tokens_are_masked():
    clip = make_clip({"pad": 0, "end": 5})
    _, mask, count, _ = clip.process_tokens([[2, 5, 3, 0]], torch.device("cpu"))
    assert mask.tolist() == [[1, 1, 0, 0]]
    assert count == [2]


def test_inserted_embeddings_extend_attention_mask():
    clip = make_clip({"pad": 151643})
    inserted = torch.full((2, 2), 42.0)
    embeds, mask, count, info = clip.process_tokens([[7, inserted, 8]], torch.device("cpu"))
    assert embeds.shape == (1, 4, 2)
    torch.testing.assert_close(embeds[0, 1:3], inserted)
    assert mask.tolist() == [[1, 1, 1, 1]]
    assert count == [4]
    assert info[0]["index"] == 1
    assert info[0]["size"] == 2


def make_anima():
    namespace, _ = load_engine()
    namespace["SDClipModel"] = common_model()
    namespace["torch"] = torch
    namespace["memory_management"] = SimpleNamespace(text_encoder_device=lambda: torch.device("cpu"))
    model = Encoder()
    return namespace["Qwen06Engine"](model, Tokenizer(), Tokenizer()), model


def test_anima_conditioning_weights_eos_padding_and_duplicate_cache():
    engine, model = make_anima()
    cond, duplicate = engine(["a (b:-2)", "a (b:-2)"])
    assert cond.shape == (1, 512, 2)
    torch.testing.assert_close(cond[0, :3], torch.tensor([[1., 1.], [-2., -2.], [1., 1.]]))
    assert torch.count_nonzero(cond[0, 3:]) == 0
    assert cond is duplicate
    assert model.calls == model.adapter_calls == 1


def test_anima_long_conditioning_is_not_truncated():
    engine, _ = make_anima()
    cond, = engine(["a" * 512])
    assert cond.shape == (1, 513, 2)


def test_empty_anima_prompt_produces_eos_and_padding():
    engine, model = make_anima()
    cond, = engine([""])
    assert cond.shape == (1, 512, 2)
    assert model.mask.tolist() == [[0]]
    torch.testing.assert_close(cond[0, 0], torch.ones(2))
    assert torch.count_nonzero(cond[0, 1:]) == 0
