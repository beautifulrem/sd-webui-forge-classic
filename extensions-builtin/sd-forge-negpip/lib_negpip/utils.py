import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from lib_negpip import IS_NEO
from modules.anima_support import negpip_extra_prompts

# This runs on every prompt of every generation, so each part may match its
# input only one way; the search stays linear in the prompt length:
# - whitespace before the colon is part of the weighted text (separate ``\s*``
#   around the text backtracked cubically on unclosed weights);
# - the weight is ``1``, ``1.``, ``1.5`` or ``.5`` spelled as alternatives
#   (``\d*\.?\d+`` could split a digit run between ``\d*`` and ``\d+`` in
#   quadratically many ways);
# - a ``(`` escaped by one backslash starts no weight (the attention parser reads
#   ``\(`` as a literal); otherwise every ``\(`` of a long run started a scan
#   over the rest of it. ``\\(`` (escaped backslash, real bracket) still does.
#
# The weight is what Forge's attention parser (``re_attention`` in
# backend/text_processing/parsing.py, used by the SD and the Anima text
# engines) reads: ``:\s*([+-]?[.\d]+)\s*\)`` whose number ``float()`` accepts.
# So whitespace may sit around the colon and before the ``)``, but not between
# the minus and the number: ``(foo:- 1)`` is the text "foo:- 1" at weight 1.1.
# ``[.\d]+`` that float() accepts is exactly ``\d+(?:\.\d*)?|\.\d+``. A zero
# (``-0``, ``-0.0``) matches but is not negative; ``negative_weights`` drops it.
NEG_PATTERN = re.compile(
    r"(?<!(?<!\\)\\)\((?P<text>(?:[^\\(:)]|\\[()])+?):\s*"
    r"(?P<weight>-(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)


def reset_prompt_cache(p: "StableDiffusionProcessing"):
    c = 3 if IS_NEO else 2

    p.cached_c = [None] * c
    p.cached_uc = [None] * c
    if hasattr(p, "cached_hr_c"):
        p.cached_hr_c = [None] * c
        p.cached_hr_uc = [None] * c


def hr_dealer(p: "StableDiffusionProcessing") -> tuple[bool, bool]:
    return (
        bool(getattr(p, "hr_prompts", None)),
        bool(getattr(p, "hr_negative_prompts", None)),
    )


def negative_weights(prompt: str) -> list[tuple[str, str, float]]:
    """(matched text, weighted text, weight) of each weight below zero."""
    return [
        (match[0], match["text"], weight)
        for match in NEG_PATTERN.finditer(prompt)
        if (weight := float(match["weight"])) < 0.0
    ]


def has_negative(prompt: str) -> bool:
    return any(float(match["weight"]) < 0.0 for match in NEG_PATTERN.finditer(prompt))


def have_negative(prompts: list[str]) -> bool:
    return any(has_negative(p) for p in prompts)


def any_negative(p: "StableDiffusionProcessing") -> bool:
    return any(
        [
            have_negative(p.prompts),
            have_negative(p.negative_prompts),
            have_negative(getattr(p, "hr_prompts", None) or ""),
            have_negative(getattr(p, "hr_negative_prompts", None) or ""),
            # Prompts other scripts encode themselves (Regional regions,
            # Artist Mixer artists) need NegPiP's conditioning too.
            have_negative(negpip_extra_prompts(p)),
        ]
    )
