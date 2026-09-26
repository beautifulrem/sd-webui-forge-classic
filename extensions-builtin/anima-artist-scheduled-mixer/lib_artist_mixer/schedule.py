"""When and where an artist applies: sampling-window curves and the block /
inline-weight syntax of the Anima Artist Scheduled Mixer."""

from __future__ import annotations

import logging
import re

from modules.anima_lora_support import to_float, window_curve_factor

logger = logging.getLogger("anima_artist_scheduled_mixer")

CURVE_SMOOTH = "Smooth"
CURVE_HOLD = "Hold"
CURVE_TRIANGLE = "Triangle"
CURVE_FRONT = "Front loaded"
CURVE_BACK = "Back loaded"


def parse_inline_weight(text):
    value = str(text or "").strip()
    if not value:
        return "", 1.0
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    wrapped = re.fullmatch(rf"[\(\[](.+?)[：:]\s*{number}[\)\]]", value)
    if wrapped:
        return wrapped.group(1).strip(), to_float(wrapped.group(2), 1.0)
    if not (value.startswith("<") and value.endswith(">")):
        trailing = re.fullmatch(rf"(.+?)[：:]\s*{number}", value)
        if trailing:
            return trailing.group(1).strip(), to_float(trailing.group(2), 1.0)
    return value, 1.0


def parse_blocks(text, num_blocks):
    """Per-artist block field: empty means every block, negative indices count
    from the end, full-width commas are accepted and bad parts are skipped.
    A field that selects nothing (only bad or out-of-range parts) is the empty
    set, with a warning: that artist mixes into no block.
    (anima_support.parse_block_list is the strict panel-level variant.)"""
    source = str(text or "").replace("，", ",").replace(" ", "")
    if not source.strip(","):
        return set(range(num_blocks))
    result = set()
    for token in source.split(","):
        if not token:
            continue
        if "-" in token[1:]:
            dash = token.index("-", 1)
            try:
                start = int(token[:dash])
                end = int(token[dash + 1 :])
            except Exception:
                continue
            if start < 0:
                start += num_blocks
            if end < 0:
                end += num_blocks
            if start > end:
                start, end = end, start
            for idx in range(max(0, start), min(num_blocks - 1, end) + 1):
                result.add(idx)
        else:
            try:
                idx = int(token)
            except Exception:
                continue
            if idx < 0:
                idx += num_blocks
            if 0 <= idx < num_blocks:
                result.add(idx)
    if not result:
        logger.warning("Anima artist mixer: Blocks %r selects none of blocks 0-%d; that artist is skipped", str(text), num_blocks - 1)
    return result


_CURVE_SHAPES = {
    CURVE_HOLD: "hold",
    CURVE_TRIANGLE: "triangle",
    CURVE_FRONT: "front",
    CURVE_BACK: "back",
}


def curve_factor(progress, start, end, peak, curve):
    """Strength factor at sampling ``progress`` (0..1) of a window [start, end]
    peaking at ``peak``; ``curve`` is one of the CURVE_* keys (anything else
    is the smooth hump). ``progress`` None means full strength. Front / back
    loaded keep a quarter of the artist at their weak end."""
    if progress is None:
        return 1.0
    return window_curve_factor(
        progress, start, end, peak, _CURVE_SHAPES.get(curve, "smooth"), floor=0.25
    )
