from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.text_processing.parsing import re_attention
from lib_negpip import IS_NEO
from modules.anima_support import negpip_extra_prompts


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
    """``(span, weighted text as written, weight)`` of each innermost
    ``(...:w)`` whose weight is below zero.

    The prompt is read with Forge's own attention tokens (``re_attention`` in
    backend/text_processing/parsing.py, used by the SD and the Anima text
    engines), so NegPiP takes exactly what the parser weights negatively: the
    text may hold ":" ("(:d:-1)", "(re:zero:-1)") and any "\\x" escape, a
    bracket a backslash escapes starts nothing ("\\(", but "\\\\(" is an
    escaped backslash and a real bracket), whitespace may sit around the
    colon and before ")" but not after the minus ("(foo:- 1)" is text at 1.1),
    and a zero ("-0") is not negative. A weight around a nested "(" is left
    to the parser, as before. One pass over the tokens, with a stack of the
    open round brackets: linear in the prompt length (it runs on every prompt
    of every generation)."""

    found = []
    opened = []  # [start offset, holds a nested "("] per open round bracket
    for token in re_attention.finditer(prompt):
        text, weight = token.group(0), token.group(1)
        if text == "(":
            if opened:
                opened[-1][1] = True
            opened.append([token.start(), False])
        elif weight is not None:
            if not opened:
                continue  # a weight with no "(": literal text to the parser
            start, nested = opened.pop()
            try:
                value = float(weight)
            except ValueError:  # "[.\d]+" float() refuses: the parser fails on it too
                continue
            inner = prompt[start + 1 : token.start()]
            if value < 0.0 and inner and not nested:  # "(:-1)" weights no text
                found.append((prompt[start : token.end()], inner, value))
        elif text == ")" and opened:
            opened.pop()
    return found


def has_negative(prompt: str) -> bool:
    return bool(negative_weights(prompt))


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
