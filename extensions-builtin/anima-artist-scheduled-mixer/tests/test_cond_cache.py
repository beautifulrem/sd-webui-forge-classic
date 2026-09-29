"""The artist conditioning cache: a bounded LRU by entry count and bytes."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from lib_artist_mixer import runtime


@pytest.fixture
def encoder(monkeypatch):
    """Records encodes; each text encodes to a 4-byte-per-value tensor of its length."""
    calls = []

    def encode(p, texts):
        calls.append(tuple(texts))
        size = sum(len(text) for text in texts)
        return torch.zeros(size, dtype=torch.float32), (torch.ones(1, dtype=torch.float32) if "neg" in texts[0] else None)

    monkeypatch.setattr(runtime, "_COND_CACHE", OrderedDict())
    monkeypatch.setattr(runtime, "encode_extra_prompts", encode)
    monkeypatch.setattr(runtime, "conditioning_cache_key", lambda p: "model")
    return calls


P = SimpleNamespace()


def test_repeated_texts_are_encoded_once_and_uncached_calls_bypass(encoder):
    first = runtime._encode_text_batch(P, ["a"])
    assert runtime._encode_text_batch(P, ["a"]) is first
    assert encoder == [("a",)]
    runtime._encode_text_batch(P, ["b"], use_cache=False)
    assert encoder == [("a",), ("b",)]
    assert list(runtime._COND_CACHE) == [("model", ("a",))]


def test_entry_limit_evicts_the_least_recently_used(encoder, monkeypatch):
    monkeypatch.setattr(runtime, "_COND_CACHE_LIMIT", 2)
    runtime._encode_text_batch(P, ["a"])
    runtime._encode_text_batch(P, ["b"])
    runtime._encode_text_batch(P, ["a"])  # hit: "b" is now the oldest
    runtime._encode_text_batch(P, ["c"])
    assert [key[1] for key in runtime._COND_CACHE] == [("a",), ("c",)]
    runtime._encode_text_batch(P, ["b"])  # evicted: encoded again
    assert encoder.count(("b",)) == 2


def test_byte_limit_counts_masks_and_keeps_the_newest_entry(encoder, monkeypatch):
    # 4 bytes per character, plus 4 for a NegPiP mask.
    monkeypatch.setattr(runtime, "_COND_CACHE_BYTES", 40)
    runtime._encode_text_batch(P, ["aaaa"])  # 16 bytes
    runtime._encode_text_batch(P, ["neg"])  # 12 + 4 bytes
    assert len(runtime._COND_CACHE) == 2
    runtime._encode_text_batch(P, ["cc"])  # 8 bytes: 40 in total, still fits
    assert len(runtime._COND_CACHE) == 3
    runtime._encode_text_batch(P, ["d"])  # 44 bytes: the oldest goes
    assert [key[1] for key in runtime._COND_CACHE] == [("neg",), ("cc",), ("d",)]
    # A single entry over the budget is still kept (it is in use).
    runtime._encode_text_batch(P, ["x" * 20])
    assert [key[1] for key in runtime._COND_CACHE] == [("x" * 20,)]


def test_cache_key_separates_models(encoder, monkeypatch):
    runtime._encode_text_batch(P, ["a"])
    monkeypatch.setattr(runtime, "conditioning_cache_key", lambda p: "other model")
    runtime._encode_text_batch(P, ["a"])
    assert encoder == [("a",), ("a",)]
