"""Rescale A1111/Forge prompt-editing schedules across step counts and denoise.

Background
----------
A1111 / Forge prompt *editing* / *scheduling* uses ``[from:to:when]`` blocks
(also ``[to:when]`` and ``[from::when]``). ``when`` is parsed against the
generation's total step count: an integer ``when >= 1`` is an absolute step,
a float ``when < 1`` is a fraction of the steps.

How img2img actually counts steps (verified against Forge neo source)
---------------------------------------------------------------------
* The conditioning schedule is built against the **full Steps slider value**
  (``setup_conds`` passes ``p.steps``); img2img does not rebuild it against the
  executed count. So ``[a:b:15]`` always means schedule-step 15 of the slider
  total.
* img2img only executes the tail: ``t_enc = int(min(denoise, 0.999) * steps)``
  steps actually run (e.g. 17 for 35 steps @ 0.5).
* The CFG denoiser's step counter starts at **0** and increments per model call,
  with **no offset** for the denoise window. A switch fires when that 0-based
  counter reaches the schedule-step number.

Net effect: a switch authored at schedule-step 15 of a 35-step txt2img run fires
at executed-step 15 of the ~17-step img2img run -- about 88% through the pass
instead of the intended ~43% -- and any ``when`` past ``t_enc`` never fires.

What this does
--------------
It re-targets each ``when`` to the schedule-step that fires at the **same
fraction of the executed img2img pass** the switch sat at originally::

    f      = when_src / source_steps      (or when_src itself if it was a fraction)
    t_enc  = int(min(denoise, 0.999) * target_steps)   # steps img2img runs
    when_new (absolute) = round(f * t_enc)             # 0-based step that fires there
    when_new (fraction) = when_new / target_steps      # decimal that maps to it

So a switch 43% through a 35-step txt2img run is moved to fire 43% through the
~17 executed steps of a 0.5-denoise img2img refine: ``15`` becomes ``7``. With
``denoise == 1`` and equal step counts it is a no-op up to rounding.

The parsing is nesting-aware: it only rewrites the trailing numeric ``when`` of
genuine *scheduled* blocks, leaving alternation blocks (``[a|b|c]``), attention
weights (``(cat:1.3)``), and plain text untouched, at every nesting depth.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = [
    "executed_steps",
    "remap_when",
    "find_schedule_whens",
    "rescale_prompt_schedule",
]


def executed_steps(target_steps: int, denoise: float) -> int:
    """Number of steps img2img actually runs: ``int(min(denoise, 0.999) * steps)``.

    Matches Forge's ``setup_img2img_steps``. Always at least 1.
    """
    t = int(min(float(denoise), 0.999) * max(int(target_steps), 1))
    return max(1, t)


def remap_when(
    f: float,
    target_steps: int,
    denoise: float,
    *,
    executed_steps_override: Optional[int] = None,
) -> int:
    """Schedule-step that fires at fraction ``f`` of the executed img2img pass.

    Because the denoiser counts from 0 over the ``t_enc`` executed steps with no
    offset, the switch fires at the schedule-step equal to that counter value, so
    the step we want is ``round(f * t_enc)``, clamped into ``[1, t_enc]``.
    """
    f = max(0.0, min(1.0, float(f)))
    t_enc = (
        max(1, int(executed_steps_override))
        if executed_steps_override is not None
        else executed_steps(target_steps, denoise)
    )
    k = int(round(f * t_enc))
    return max(1, min(t_enc, k))


def _is_number(text: str) -> Optional[float]:
    """Return the float value of ``text`` if it is a bare (signed) number."""
    t = text.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def find_schedule_whens(prompt: str) -> List[Tuple[int, int, float]]:
    """Find every scheduled-``when`` numeric literal in ``prompt``.

    Returns a list of ``(start, end, value)`` spans (into ``prompt``) for the
    trailing number of each ``[...:when]`` scheduled block, at any nesting depth.
    Alternation blocks (containing a top-level ``|``) and attention weights in
    parentheses are skipped.
    """
    # Stack frame per open bracket: track this frame's own colon positions and
    # whether a top-level '|' (alternation) appeared in it. Parens are tracked
    # only for depth so their ':' (weights) and contents are ignored.
    spans: List[Tuple[int, int, float]] = []
    # Each '[' frame: {"colons": [positions], "alt": bool}
    bracket_stack: List[dict] = []
    paren_depth = 0
    i = 0
    n = len(prompt)
    escaped = False
    while i < n:
        ch = prompt[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_stack.append({"colons": [], "alt": False, "open": i})
        elif ch == "]":
            if bracket_stack:
                frame = bracket_stack.pop()
                colons = frame["colons"]
                if colons and not frame["alt"]:
                    last_colon = colons[-1]
                    num_text = prompt[last_colon + 1:i]
                    val = _is_number(num_text)
                    if val is not None:
                        # Span of the numeric text (preserving leading/trailing
                        # whitespace handling: rewrite exactly the slice).
                        spans.append((last_colon + 1, i, val))
        elif ch == ":":
            # A colon belongs to the innermost '[' frame only when we are not
            # nested inside parentheses relative to that frame's contents.
            if bracket_stack and paren_depth == 0:
                bracket_stack[-1]["colons"].append(i)
        elif ch == "|":
            if bracket_stack and paren_depth == 0:
                bracket_stack[-1]["alt"] = True
        i += 1
    spans.sort()
    return spans


def _format_when(step_k: int, as_fraction: bool, target_steps: int) -> str:
    """Format a remapped schedule-step ``step_k`` as schedule text.

    Absolute output emits ``step_k`` directly (an integer ``when`` maps to that
    schedule-step). Fractional output emits ``step_k / target_steps`` as a
    decimal, which the parser multiplies back up to the same schedule-step.
    """
    k = max(1, int(round(step_k)))
    if as_fraction:
        # A decimal `when` is parsed as (value * target_steps); choose the
        # decimal that lands on schedule-step k, clamped to stay a fraction.
        v = k / max(int(target_steps), 1)
        v = max(0.001, min(0.999, v))
        return ("%.4f" % v).rstrip("0").rstrip(".")
    return str(min(int(target_steps), k))


def rescale_prompt_schedule(
    prompt: str,
    source_steps: int,
    target_steps: int,
    denoise: float,
    *,
    as_fraction: bool = False,
    executed_steps_override: Optional[int] = None,
) -> str:
    """Rewrite the ``when`` values of every scheduled block in ``prompt``.

    Parameters
    ----------
    prompt:
        The prompt text containing ``[from:to:when]`` style schedules.
    source_steps:
        Step count the schedule was originally authored for (e.g. the txt2img
        step count). Used to interpret absolute integer ``when`` values as a
        fraction of the original run.
    target_steps:
        Step count the img2img pass will use. Only affects absolute-step output.
    denoise:
        img2img denoising strength in ``(0, 1]``. ``1`` (with equal step counts)
        is a no-op.
    as_fraction:
        When ``True`` emit the ``when`` as a decimal fraction of ``target_steps``
        (some people prefer reading fractions); when ``False`` (default) emit the
        absolute integer schedule-step. Both resolve to the same firing step.
    executed_steps_override:
        Optional exact executed-step count. Forge Neo's ``img2img_fix_steps``
        mode runs the requested slider count instead of multiplying it by
        denoise; callers can pass ``target_steps`` here for that mode.

    Returns the rewritten prompt. On any unexpected input it returns the prompt
    unchanged rather than raising.
    """
    try:
        source_steps = max(1, int(source_steps))
        target_steps = max(1, int(target_steps))
        denoise = max(1e-6, min(1.0, float(denoise)))
        spans = find_schedule_whens(prompt)
        if not spans:
            return prompt
        out = prompt
        # Rewrite right-to-left so earlier indices stay valid.
        for start, end, value in sorted(spans, reverse=True):
            # Resolve the authored value to a source fraction.
            if value < 1.0:
                f = value  # already a fraction of the source run
            else:
                f = value / source_steps
            f_new = remap_when(
                f,
                target_steps,
                denoise,
                executed_steps_override=executed_steps_override,
            )
            new_text = _format_when(f_new, as_fraction, target_steps)
            out = out[:start] + new_text + out[end:]
        return out
    except Exception:
        return prompt
