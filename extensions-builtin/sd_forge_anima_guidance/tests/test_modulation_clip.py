"""CLIP-L loading and pooled encoding on the CPU, with a tiny CLIP text model."""

from __future__ import annotations

import json
import pathlib
import string
import sys

import pytest
import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib_anima_guidance import modulation

transformers = pytest.importorskip("transformers")

MAX_LENGTH = 8


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(modulation, "_CLIP_CACHE", {})
    monkeypatch.setattr(modulation, "_POOLED_CACHE", {})


def _config():
    return transformers.CLIPTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=MAX_LENGTH,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )


def _write_tokenizer(directory: pathlib.Path) -> None:
    tokens = ["<|startoftext|>", "<|endoftext|>"]
    for letter in string.ascii_lowercase:
        tokens += [letter, f"{letter}</w>"]
    directory.mkdir()
    (directory / "vocab.json").write_text(json.dumps({token: index for index, token in enumerate(tokens)}), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"model_max_length": MAX_LENGTH}), encoding="utf-8")


@pytest.fixture
def clip(tmp_path):
    """(weights path, config dir, tokenizer dir, fp32 reference model)."""

    config_dir = tmp_path / "config"
    _config().save_pretrained(config_dir)
    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer(tokenizer_dir)
    torch.manual_seed(0)
    reference = transformers.CLIPTextModel(_config()).eval()
    # Stored in fp16 like the pinned clip_l.safetensors (position_ids is not
    # a persistent buffer, so the file has none).
    state = {key: value.half().contiguous() for key, value in reference.state_dict().items()}
    assert not any(key.endswith("position_ids") for key in state)
    reference.load_state_dict({key: value.float() for key, value in state.items()})
    path = tmp_path / "clip_l.safetensors"
    save_file(state, str(path))
    return str(path), str(config_dir), str(tokenizer_dir), reference


def _encode(clip, prompts):
    path, config_dir, tokenizer_dir, _ = clip
    return modulation.encode_clip_pooled(prompts, clip_path=path, config_dir=config_dir, tokenizer_dir=tokenizer_dir, device="cpu")


def test_load_builds_on_meta_and_assigns_the_file_weights(clip):
    path, config_dir, tokenizer_dir, reference = clip
    model, tokenizer = modulation._load_clip(path, config_dir, tokenizer_dir)

    assert not any(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()])
    # assign=True keeps the file's tensors (and dtype) instead of copying
    # into freshly initialised fp32 parameters
    assert next(model.parameters()).dtype == torch.float16
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    position_ids = model.text_model.embeddings.position_ids
    assert torch.equal(position_ids, torch.arange(MAX_LENGTH).expand((1, -1)))
    assert tokenizer.model_max_length == MAX_LENGTH
    assert modulation._load_clip(path, config_dir, tokenizer_dir)[0] is model


def test_pooled_output_matches_the_reference_model(clip):
    *_, reference = clip
    pooled = _encode(clip, ["a cat", "dog"])

    tokenizer = transformers.CLIPTokenizer.from_pretrained(clip[2])
    tokens = tokenizer(["a cat", "dog"], padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    with torch.no_grad():
        expected = reference(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask).pooler_output
    assert pooled.dtype == torch.float32 and pooled.shape == (2, 16)
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_cpu_encoder_keeps_the_cached_model_in_fp32(clip, monkeypatch):
    path, config_dir, tokenizer_dir, _ = clip
    _encode(clip, ["a cat"])
    model, _ = modulation._load_clip(path, config_dir, tokenizer_dir)
    assert next(model.parameters()).dtype == torch.float32

    conversions = []
    original = torch.nn.Module.to

    def counting_to(self, *args, **kwargs):
        if self is model:
            before = next(self.parameters()).dtype
            result = original(self, *args, **kwargs)
            conversions.append(next(self.parameters()).dtype != before)
            return result
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", counting_to)
    _encode(clip, ["a dog"])  # a pooled-cache miss
    assert conversions and not any(conversions)


def test_pooled_cache_evicts_the_oldest_batch(clip, monkeypatch):
    monkeypatch.setattr(modulation, "_POOLED_CACHE_SIZE", 2)
    calls = []
    original = modulation._load_clip

    def counting_load(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(modulation, "_load_clip", counting_load)
    first = _encode(clip, ["a cat"])
    assert _encode(clip, ["a cat"]) is first and len(calls) == 1
    _encode(clip, ["b"])
    _encode(clip, ["c"])  # evicts "a cat"
    assert len(modulation._POOLED_CACHE) == 2
    assert _encode(clip, ["c"]) is not None and len(calls) == 3
    again = _encode(clip, ["a cat"])
    assert again is not first and len(calls) == 4
    assert torch.equal(again, first)


def test_unexpected_weights_are_rejected(clip, tmp_path):
    path, config_dir, tokenizer_dir, _ = clip
    from safetensors.torch import load_file

    state = load_file(path)
    state["text_model.extra.weight"] = torch.zeros(1)
    bad = tmp_path / "bad.safetensors"
    save_file(state, str(bad))
    with pytest.raises(RuntimeError, match="do not match the pinned architecture"):
        modulation._load_clip(str(bad), config_dir, tokenizer_dir)


def test_unsupported_meta_buffer_is_an_error(clip, monkeypatch):
    path, config_dir, tokenizer_dir, _ = clip

    original_init = transformers.CLIPTextModel.__init__

    def init_with_extra_buffer(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self.register_buffer("scratch", torch.zeros(2), persistent=False)

    # transformers' lazy module ignores a replaced attribute on import
    monkeypatch.setattr(transformers.CLIPTextModel, "__init__", init_with_extra_buffer)
    with pytest.raises(RuntimeError, match="unsupported meta buffer 'scratch'"):
        modulation._load_clip(path, config_dir, tokenizer_dir)
    assert modulation._CLIP_CACHE == {}
