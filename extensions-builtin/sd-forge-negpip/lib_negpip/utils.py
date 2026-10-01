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


def _negative_groups(prompt: str) -> list[tuple[str, str, float, bool]]:
    """``(span, weighted text as written, weight, innermost)`` of each
    ``(...:w)`` whose weight is below zero and whose text the parser reads as
    something.

    The prompt is read with Forge's own attention tokens (``re_attention`` in
    backend/text_processing/parsing.py, used by the SD and the Anima text
    engines) and the same rules ``parse_prompt_attention`` applies to them, so
    a group counts exactly when the parser weights text in it: the text may
    hold ":" ("(:d:-1)", "(re:zero:-1)") and any "\\x" escape, a bracket a
    backslash escapes starts nothing ("\\(", but "\\\\(" is an escaped
    backslash and a real bracket), whitespace may sit around the colon and
    before ")" but not after the minus ("(foo:- 1)" is text at 1.1), a zero
    ("-0") is not negative, and a group of no text ("(:-1)", "(\\:-1)", "( BREAK :-1)")
    weights nothing. One pass over the tokens with a stack of the open round
    brackets (each knows whether it holds a nested "(" and whether text was
    read inside it, handed to its parent when it closes): linear in the prompt
    length (it runs on every prompt of every generation)."""

    found = []
    opened = []  # [start offset, holds a nested "(", holds text] per open round bracket
    squares = 0  # open "[", so a "]" closes one or is literal text, as in the parser

    def close():
        _start, _nested, has_text = group = opened.pop()
        if opened and has_text:
            opened[-1][2] = True
        return group

    for token in re_attention.finditer(prompt):
        text, weight = token.group(0), token.group(1)
        if text.startswith("\\"):
            reads_text = len(text) > 1  # "\\(" is "(", a lone "\\" is nothing
        elif text == "(":
            if opened:
                opened[-1][1] = True
            opened.append([token.start(), False, False])
            continue
        elif text == "[":
            squares += 1
            continue
        elif weight is not None and opened:
            start, nested, has_text = close()
            try:
                value = float(weight)
            except ValueError:  # "[.\\d]+" float() refuses: the parser fails on it too
                continue
            if value < 0.0 and has_text:
                found.append((prompt[start : token.end()], prompt[start + 1 : token.start()], value, not nested))
            continue
        elif text == ")" and opened:
            close()
            continue
        elif text == "]" and squares:
            squares -= 1
            continue
        else:  # plain text, ":", or a bracket / weight closing nothing: literal
            # The parser splits it on re_break ("BREAK" between word boundaries)
            # and keeps the non-empty parts: none are left only from standalone
            # "BREAK" words. Checked by words, not with re_break, which
            # backtracks quadratically on a long whitespace run.
            words = text.split()
            reads_text = not words or any(word != "BREAK" for word in words)
        if reads_text and opened:
            opened[-1][2] = True
    return found


def negative_weights(prompt: str) -> list[tuple[str, str, float]]:
    """``(span, weighted text as written, weight)`` of each innermost negative
    group (see ``_negative_groups``): the targets the SD path moves out of the
    prompt and encodes. A weight around a nested "(" stays in the prompt."""

    return [(span, text, weight) for span, text, weight, innermost in _negative_groups(prompt) if innermost]


def has_negative(prompt: str) -> bool:
    """Whether the parser weights any text below zero, nested groups included
    ("(cat, (dog):-1)"): on Anima the hooks then go in, and the mask comes
    from the parser's own multipliers. It never misses one; where round groups
    nest it may also say yes when nested weights cancel ("((a:-1):-1)") or zero
    ("((a:-0.5):-0)") the sign, harmlessly: the mask is then all ones."""

    return bool(_negative_groups(prompt))


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
