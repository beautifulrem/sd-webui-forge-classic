"""The artist conditioning cache: a bounded LRU by entry count and bytes."""

import importlib.util
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_artist_scheduled_mixer.py"


def _load():
    import modules

    try:
        from modules import sd_models  # noqa: F401
    except ImportError:
        # The model loader needs the full WebUI install; the cache does not use it
        modules.sd_models = sys.modules["modules.sd_models"] = types.ModuleType("modules.sd_models")
    spec = importlib.util.spec_from_file_location("anima_artist_mixer_cache_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixer = _load()


@pytest.fixture
def encoder(monkeypatch):
    """Records encodes; each text encodes to a 4-byte-per-value tensor of its length."""
    calls = []

    def encode(p, texts):
        calls.append(tuple(texts))
        size = sum(len(text) for text in texts)
        return torch.zeros(size, dtype=torch.float32), (torch.ones(1, dtype=torch.float32) if "neg" in texts[0] else None)

    monkeypatch.setattr(mixer, "_COND_CACHE", OrderedDict())
    monkeypatch.setattr(mixer, "encode_extra_prompts", encode)
    monkeypatch.setattr(mixer, "conditioning_cache_key", lambda p: "model")
    return calls


P = SimpleNamespace()


def test_repeated_texts_are_encoded_once_and_uncached_calls_bypass(encoder):
    first = mixer._encode_text_batch(P, ["a"])
    assert mixer._encode_text_batch(P, ["a"]) is first
    assert encoder == [("a",)]
    mixer._encode_text_batch(P, ["b"], use_cache=False)
    assert encoder == [("a",), ("b",)]
    assert list(mixer._COND_CACHE) == [("model", ("a",))]


def test_entry_limit_evicts_the_least_recently_used(encoder, monkeypatch):
    monkeypatch.setattr(mixer, "_COND_CACHE_LIMIT", 2)
    mixer._encode_text_batch(P, ["a"])
    mixer._encode_text_batch(P, ["b"])
    mixer._encode_text_batch(P, ["a"])  # hit: "b" is now the oldest
    mixer._encode_text_batch(P, ["c"])
    assert [key[1] for key in mixer._COND_CACHE] == [("a",), ("c",)]
    mixer._encode_text_batch(P, ["b"])  # evicted: encoded again
    assert encoder.count(("b",)) == 2


def test_byte_limit_counts_masks_and_keeps_the_newest_entry(encoder, monkeypatch):
    # 4 bytes per character, plus 4 for a NegPiP mask.
    monkeypatch.setattr(mixer, "_COND_CACHE_BYTES", 40)
    mixer._encode_text_batch(P, ["aaaa"])  # 16 bytes
    mixer._encode_text_batch(P, ["neg"])  # 12 + 4 bytes
    assert len(mixer._COND_CACHE) == 2
    mixer._encode_text_batch(P, ["cc"])  # 8 bytes: 40 in total, still fits
    assert len(mixer._COND_CACHE) == 3
    mixer._encode_text_batch(P, ["d"])  # 44 bytes: the oldest goes
    assert [key[1] for key in mixer._COND_CACHE] == [("neg",), ("cc",), ("d",)]
    # A single entry over the budget is still kept (it is in use).
    mixer._encode_text_batch(P, ["x" * 20])
    assert [key[1] for key in mixer._COND_CACHE] == [("x" * 20,)]


def test_cache_key_separates_models(encoder, monkeypatch):
    mixer._encode_text_batch(P, ["a"])
    monkeypatch.setattr(mixer, "conditioning_cache_key", lambda p: "other model")
    mixer._encode_text_batch(P, ["a"])
    assert encoder == [("a",), ("a",)]
