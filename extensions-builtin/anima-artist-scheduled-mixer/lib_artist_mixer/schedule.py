"""When and where an artist applies: sampling-window curves and the block /
inline-weight syntax of the Anima Artist Scheduled Mixer."""

from __future__ import annotations

import re

CURVE_SMOOTH = "Smooth"
CURVE_HOLD = "Hold"
CURVE_TRIANGLE = "Triangle"
CURVE_FRONT = "Front loaded"
CURVE_BACK = "Back loaded"


def clamp(value, low, high):
    return max(low, min(high, value))


def to_float(value, fallback=0.0):
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


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
    (anima_support.parse_block_list is the strict panel-level variant.)"""
    source = str(text or "").replace("，", ",").replace(" ", "")
    if not source:
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
    return result or set(range(num_blocks))


def curve_factor(progress, start, end, peak, curve):
    """Strength factor at sampling ``progress`` (0..1) of a window [start, end]
    peaking at ``peak``; ``curve`` is one of the CURVE_* keys (anything else
    is the smooth hump). ``progress`` None means full strength."""
    if progress is None:
        return 1.0
    start = clamp(start, 0.0, 1.0)
    end = clamp(end, 0.0, 1.0)
    if end < start:
        start, end = end, start
    if progress < start or progress > end:
        return 0.0
    if end - start <= 1e-6:
        return 1.0
    local = (progress - start) / max(end - start, 1e-6)
    if curve == CURVE_HOLD:
        return 1.0
    if curve == CURVE_FRONT:
        return 1.0 - 0.75 * local
    if curve == CURVE_BACK:
        return 0.25 + 0.75 * local
    peak = clamp(peak, start, end)
    peak_local = (peak - start) / max(end - start, 1e-6)
    if curve == CURVE_TRIANGLE:
        if local <= peak_local:
            return local / max(peak_local, 1e-6)
        return (1.0 - local) / max(1.0 - peak_local, 1e-6)
    if local <= peak_local:
        raw = local / max(peak_local, 1e-6)
    else:
        raw = (1.0 - local) / max(1.0 - peak_local, 1e-6)
    raw = clamp(raw, 0.0, 1.0)
    return raw * raw * (3.0 - 2.0 * raw)
