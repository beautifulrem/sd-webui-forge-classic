"""Prompt/token alignment helpers for Anima's T5-side conditioning."""

from __future__ import annotations

import re

_EXTRA_NETWORK = re.compile(r"<(?:lora|lyco|hypernet):[^>]+>", re.IGNORECASE)


def clean_prompt(prompt: str) -> str:
    return _EXTRA_NETWORK.sub("", str(prompt))


def _find_subsequence(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    for start in range(0, len(sequence) - len(needle) + 1):
        if sequence[start : start + len(needle)] == needle:
            return list(range(start, start + len(needle)))
    return []


def concept_token_positions(
    tokenizer,
    prompt: str,
    concepts: dict[str, str],
    background: str = "",
    *,
    prompt_ids: list[int] | None = None,
):
    """Resolve concept phrases exactly against the T5 tokens used by Anima."""

    prompt = clean_prompt(prompt)
    if prompt_ids is None:
        prompt_ids = list(
            tokenizer([prompt], truncation=False, add_special_tokens=False)[
                "input_ids"
            ][0]
        )
        eos_position = len(prompt_ids)
    else:
        prompt_ids = list(prompt_ids)
        # AnimaTextProcessingEngine.tokenize_line appends T5 EOS.
        eos_position = max(0, len(prompt_ids) - 1)
    result = {}
    occupied = set()
    for name, phrase in concepts.items():
        phrase = str(phrase).strip()
        if not phrase:
            raise ValueError(f"FreeFuse concept phrase for {name!r} is empty")
        phrase_ids = list(
            tokenizer([phrase], truncation=False, add_special_tokens=False)[
                "input_ids"
            ][0]
        )
        positions = _find_subsequence(prompt_ids, phrase_ids)
        if not positions:
            raise ValueError(
                f"FreeFuse concept phrase {phrase!r} was not found verbatim in the positive prompt"
            )
        overlap = occupied.intersection(positions)
        if overlap:
            raise ValueError(
                f"FreeFuse concept phrase {phrase!r} overlaps another concept at token {min(overlap)}"
            )
        occupied.update(positions)
        result[name] = positions

    background = str(background).strip()
    if background:
        background_ids = list(
            tokenizer([background], truncation=False, add_special_tokens=False)[
                "input_ids"
            ][0]
        )
        bg_positions = _find_subsequence(prompt_ids, background_ids)
        if not bg_positions:
            raise ValueError(
                f"FreeFuse background phrase {background!r} was not found verbatim in the positive prompt"
            )
    else:
        # AnimaTextProcessingEngine appends T5 EOS immediately after these IDs.
        bg_positions = [eos_position]
    return result, bg_positions
