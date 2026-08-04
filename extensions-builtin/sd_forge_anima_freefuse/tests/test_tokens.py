import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "lib_anima_freefuse" / "tokens.py"
SPEC = importlib.util.spec_from_file_location("anima_freefuse_tokens", MODULE_PATH)
TOKENS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOKENS)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


def test_concepts_align_after_lora_tags_are_removed():
    positions, background = TOKENS.concept_token_positions(
        FakeTokenizer(),
        "alice <lora:alice:1> beside bob",
        {"one": "alice", "two": "bob"},
    )
    assert positions == {"one": list(range(5)), "two": [14, 15, 16]}
    assert background == [17]


def test_explicit_background_is_resolved():
    positions, background = TOKENS.concept_token_positions(
        FakeTokenizer(),
        "alice and bob in a park",
        {"one": "alice", "two": "bob"},
        "park",
    )
    assert positions["one"] == list(range(5))
    assert background == [19, 20, 21, 22]


def test_engine_token_ids_use_the_appended_eos_position():
    prompt = "alice and bob"
    prompt_ids = [ord(character) for character in prompt] + [1]
    _, background = TOKENS.concept_token_positions(
        FakeTokenizer(),
        prompt,
        {"one": "alice", "two": "bob"},
        prompt_ids=prompt_ids,
    )
    assert background == [len(prompt)]


def test_missing_or_overlapping_concepts_fail():
    try:
        TOKENS.concept_token_positions(
            FakeTokenizer(),
            "alice and bob",
            {"one": "alice", "two": "missing"},
        )
    except ValueError as error:
        assert "not found verbatim" in str(error)
    else:
        raise AssertionError("missing concept did not fail")

    try:
        TOKENS.concept_token_positions(
            FakeTokenizer(),
            "alice",
            {"one": "alice", "two": "alice"},
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("overlapping concept did not fail")
