"""Anima resolution helper (Forge Neo Remi).

Four features, all multiples of 64 so latent dimensions divide cleanly:

1. Resolution presets in the txt2img dimensions section: a standard list
   (W+H <= 2176) and a high-res list (W+H in [2560, 3072], at most 1856 per
   side). Picking one sets the generation form's width and height in the
   browser (the ``resolution-presets`` widget).
2. img2img size fit: when the source image changes (Send to img2img /
   inpaint, an upload), the ``source-size-fit`` widget runs the
   ``fit_to_source`` action, which sets the img2img width and height to the
   multiple-of-64 pair with W+H as close to 3072 as the aspect drift cap
   allows (:func:`upscale_target`). The source size is the one the image's
   infotext records (``Size``: the first pass, also when Randomize or Hires.
   fix changed it), else the image's own size.
3. Randomize toggles: each txt2img job rolls its size from the standard
   list, the high-res list, or both lists' combined pool (``before_process``).
4. Session prompt history: each txt2img job's positive prompt is kept in a
   session ring per user (100 entries, most recent first; without login all
   jobs share one user). Every job counts, queued and API jobs included, but
   a job is logged once: script cells (X/Y/Z, Prompt S/R, prompts from file)
   do not add their variants. Picking an entry puts it back into the prompt
   (``use_history``); Refresh reloads the list.

Designed for Anima (https://huggingface.co/circlestone-labs/Anima).
"""

from __future__ import annotations

import logging
import random
import re
import threading

from modules import scripts
from remi import ui

logger = logging.getLogger("AnimaResolutions")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Post-upscale W+H must land in this window. The auto-adjuster picks the
# multiple-of-64 target pair within the window with the HIGHEST total (i.e. as
# close to TARGET_TOTAL_HIGH / 3072 as possible), subject to the pair's
# aspect-ratio drift from the source staying within MAX_UPSCALE_DRIFT. The
# objective is resolution: get the img2img first-pass as big as the VAE-safe
# window allows, with the drift cap acting purely as a guardrail so the ratio
# is never thrown badly off in the process. Pair targets with lower denoise as
# usual.
TARGET_TOTAL_LOW = 2560
TARGET_TOTAL_HIGH = 3072

# Aspect-ratio drift guardrail for the img2img auto-upscale. Drift is the
# fractional change in W/H relative to the source (e.g. 0.03 = the target's
# aspect ratio differs from the source's by at most 3%). Among the in-window
# multiple-of-64 pairs, only those at or under this drift are eligible, and the
# largest-total eligible pair wins - so the size push toward 3072 can never
# squash the ratio by more than this. Raise it (e.g. 0.05) to grab the literal
# maximum total more aggressively at the cost of more ratio drift; lower it
# (e.g. 0.015) to stay more faithful to the source ratio at the cost of total.
# If NO in-window pair is within this cap (only happens for very extreme source
# ratios), the adjuster falls back to the single least-drift pair so it still
# emits a sane target.
MAX_UPSCALE_DRIFT = 0.03

# Hard per-side cap: neither target dimension may exceed this. A target that
# would need a side beyond this is rejected and the least-drift in-window
# multiple-of-64 pair under the cap is used instead.
MAX_SIDE_LIMIT = 1856

# All preset base dims and all computed upscale targets must be multiples
# of this. 64 is the SDXL latent bucket; off-multiple dims cause stretch
# distortion when the latent is decoded.
BUCKET = 64

PROMPT_HISTORY_MAX = 100
PROMPT_PREVIEW_CHARS = 160


# Standard preset list. All multiples of 64, W+H <= 2176, neither dimension
# below 640. Sorted by total pixel count, largest first. This is also the pool
# the Randomize toggle rolls from.
RESOLUTIONS = [
    (1088, 1088),  # 1.18 MP
    (1152, 1024),  # 1.18 MP
    (1024, 1152),  # 1.18 MP
    (1216, 960),  # 1.17 MP
    (960, 1216),  # 1.17 MP
    (1280, 896),  # 1.15 MP
    (896, 1280),  # 1.15 MP
    (1344, 832),  # 1.12 MP
    (832, 1344),  # 1.12 MP
    (1088, 1024),  # 1.11 MP
    (1024, 1088),  # 1.11 MP
    (1152, 960),  # 1.11 MP
    (960, 1152),  # 1.11 MP
    (1216, 896),  # 1.09 MP
    (896, 1216),  # 1.09 MP
    (1408, 768),  # 1.08 MP
    (768, 1408),  # 1.08 MP
    (1280, 832),  # 1.06 MP
    (832, 1280),  # 1.06 MP
    (1024, 1024),  # 1.05 MP
    (1088, 960),  # 1.04 MP
    (960, 1088),  # 1.04 MP
    (1472, 704),  # 1.04 MP
    (704, 1472),  # 1.04 MP
    (1344, 768),  # 1.03 MP
    (1152, 896),  # 1.03 MP
    (896, 1152),  # 1.03 MP
    (768, 1344),  # 1.03 MP
    (1216, 832),  # 1.01 MP
    (832, 1216),  # 1.01 MP
    (1408, 704),  # 0.99 MP
    (704, 1408),  # 0.99 MP
    (1536, 640),  # 0.98 MP
    (1280, 768),  # 0.98 MP
    (1024, 960),  # 0.98 MP
    (960, 1024),  # 0.98 MP
    (768, 1280),  # 0.98 MP
    (640, 1536),  # 0.98 MP
    (1088, 896),  # 0.97 MP
    (896, 1088),  # 0.97 MP
    (1152, 832),  # 0.96 MP
    (832, 1152),  # 0.96 MP
    (1344, 704),  # 0.95 MP
    (704, 1344),  # 0.95 MP
    (1472, 640),  # 0.94 MP
    (640, 1472),  # 0.94 MP
    (1216, 768),  # 0.93 MP
    (768, 1216),  # 0.93 MP
    (960, 960),  # 0.92 MP
    (1024, 896),  # 0.92 MP
    (896, 1024),  # 0.92 MP
    (1088, 832),  # 0.91 MP
    (832, 1088),  # 0.91 MP
    (1408, 640),  # 0.90 MP
    (1280, 704),  # 0.90 MP
    (704, 1280),  # 0.90 MP
    (640, 1408),  # 0.90 MP
    (1152, 768),  # 0.88 MP
    (768, 1152),  # 0.88 MP
    (1344, 640),  # 0.86 MP
    (960, 896),  # 0.86 MP
    (896, 960),  # 0.86 MP
    (640, 1344),  # 0.86 MP
    (1216, 704),  # 0.86 MP
    (704, 1216),  # 0.86 MP
    (1024, 832),  # 0.85 MP
    (832, 1024),  # 0.85 MP
    (1088, 768),  # 0.84 MP
    (768, 1088),  # 0.84 MP
    (1280, 640),  # 0.82 MP
    (640, 1280),  # 0.82 MP
    (1152, 704),  # 0.81 MP
    (704, 1152),  # 0.81 MP
    (896, 896),  # 0.80 MP
    (960, 832),  # 0.80 MP
    (832, 960),  # 0.80 MP
    (1024, 768),  # 0.79 MP
    (768, 1024),  # 0.79 MP
    (1216, 640),  # 0.78 MP
    (640, 1216),  # 0.78 MP
    (1088, 704),  # 0.77 MP
    (704, 1088),  # 0.77 MP
    (896, 832),  # 0.75 MP
    (832, 896),  # 0.75 MP
    (1152, 640),  # 0.74 MP
    (960, 768),  # 0.74 MP
    (768, 960),  # 0.74 MP
    (640, 1152),  # 0.74 MP
    (1024, 704),  # 0.72 MP
    (704, 1024),  # 0.72 MP
    (1088, 640),  # 0.70 MP
    (640, 1088),  # 0.70 MP
    (832, 832),  # 0.69 MP
    (896, 768),  # 0.69 MP
    (768, 896),  # 0.69 MP
    (960, 704),  # 0.68 MP
    (704, 960),  # 0.68 MP
    (1024, 640),  # 0.66 MP
    (640, 1024),  # 0.66 MP
    (832, 768),  # 0.64 MP
    (768, 832),  # 0.64 MP
    (896, 704),  # 0.63 MP
    (704, 896),  # 0.63 MP
    (960, 640),  # 0.61 MP
    (640, 960),  # 0.61 MP
    (768, 768),  # 0.59 MP
    (832, 704),  # 0.59 MP
    (704, 832),  # 0.59 MP
    (896, 640),  # 0.57 MP
    (640, 896),  # 0.57 MP
    (768, 704),  # 0.54 MP
    (704, 768),  # 0.54 MP
    (832, 640),  # 0.53 MP
    (640, 832),  # 0.53 MP
    (704, 704),  # 0.50 MP
    (768, 640),  # 0.49 MP
    (640, 768),  # 0.49 MP
    (704, 640),  # 0.45 MP
    (640, 704),  # 0.45 MP
    (640, 640),  # 0.41 MP
]


# High-res preset list. All multiples of 64, total (W+H) in [2560, 3072],
# neither dimension above 1856 (so the smaller side can go down to 704 at the
# 2560 floor). The [2560, 3072] window matches the img2img auto-resize target
# window, so every entry is already a valid final-size target: the auto-resize
# never downscales it. An entry already at the top of the window (sum 3072)
# stays unchanged; one lower in the window is nudged up toward 3072 by the same
# max-total rule. Sorted by total pixel count, largest first. Rolled only by
# the high-res Randomize toggle. On Send to img2img / Send to inpaint
# these share the same max-total auto-resize as every other source.
HIGH_RES_RESOLUTIONS = [
    (1536, 1536),  # 2.36 MP
    (1600, 1472),  # 2.36 MP
    (1472, 1600),  # 2.36 MP
    (1664, 1408),  # 2.34 MP
    (1408, 1664),  # 2.34 MP
    (1728, 1344),  # 2.32 MP
    (1344, 1728),  # 2.32 MP
    (1792, 1280),  # 2.29 MP
    (1280, 1792),  # 2.29 MP
    (1536, 1472),  # 2.26 MP
    (1472, 1536),  # 2.26 MP
    (1856, 1216),  # 2.26 MP
    (1216, 1856),  # 2.26 MP
    (1600, 1408),  # 2.25 MP
    (1408, 1600),  # 2.25 MP
    (1664, 1344),  # 2.24 MP
    (1344, 1664),  # 2.24 MP
    (1728, 1280),  # 2.21 MP
    (1280, 1728),  # 2.21 MP
    (1792, 1216),  # 2.18 MP
    (1216, 1792),  # 2.18 MP
    (1472, 1472),  # 2.17 MP
    (1536, 1408),  # 2.16 MP
    (1408, 1536),  # 2.16 MP
    (1600, 1344),  # 2.15 MP
    (1344, 1600),  # 2.15 MP
    (1856, 1152),  # 2.14 MP
    (1152, 1856),  # 2.14 MP
    (1664, 1280),  # 2.13 MP
    (1280, 1664),  # 2.13 MP
    (1728, 1216),  # 2.10 MP
    (1216, 1728),  # 2.10 MP
    (1472, 1408),  # 2.07 MP
    (1408, 1472),  # 2.07 MP
    (1792, 1152),  # 2.06 MP
    (1152, 1792),  # 2.06 MP
    (1536, 1344),  # 2.06 MP
    (1344, 1536),  # 2.06 MP
    (1600, 1280),  # 2.05 MP
    (1280, 1600),  # 2.05 MP
    (1664, 1216),  # 2.02 MP
    (1216, 1664),  # 2.02 MP
    (1856, 1088),  # 2.02 MP
    (1088, 1856),  # 2.02 MP
    (1728, 1152),  # 1.99 MP
    (1152, 1728),  # 1.99 MP
    (1408, 1408),  # 1.98 MP
    (1472, 1344),  # 1.98 MP
    (1344, 1472),  # 1.98 MP
    (1536, 1280),  # 1.97 MP
    (1280, 1536),  # 1.97 MP
    (1792, 1088),  # 1.95 MP
    (1088, 1792),  # 1.95 MP
    (1600, 1216),  # 1.95 MP
    (1216, 1600),  # 1.95 MP
    (1664, 1152),  # 1.92 MP
    (1152, 1664),  # 1.92 MP
    (1856, 1024),  # 1.90 MP
    (1024, 1856),  # 1.90 MP
    (1408, 1344),  # 1.89 MP
    (1344, 1408),  # 1.89 MP
    (1472, 1280),  # 1.88 MP
    (1280, 1472),  # 1.88 MP
    (1728, 1088),  # 1.88 MP
    (1088, 1728),  # 1.88 MP
    (1536, 1216),  # 1.87 MP
    (1216, 1536),  # 1.87 MP
    (1600, 1152),  # 1.84 MP
    (1152, 1600),  # 1.84 MP
    (1792, 1024),  # 1.84 MP
    (1024, 1792),  # 1.84 MP
    (1664, 1088),  # 1.81 MP
    (1088, 1664),  # 1.81 MP
    (1344, 1344),  # 1.81 MP
    (1408, 1280),  # 1.80 MP
    (1280, 1408),  # 1.80 MP
    (1472, 1216),  # 1.79 MP
    (1216, 1472),  # 1.79 MP
    (1856, 960),  # 1.78 MP
    (960, 1856),  # 1.78 MP
    (1728, 1024),  # 1.77 MP
    (1024, 1728),  # 1.77 MP
    (1536, 1152),  # 1.77 MP
    (1152, 1536),  # 1.77 MP
    (1600, 1088),  # 1.74 MP
    (1088, 1600),  # 1.74 MP
    (1792, 960),  # 1.72 MP
    (960, 1792),  # 1.72 MP
    (1344, 1280),  # 1.72 MP
    (1280, 1344),  # 1.72 MP
    (1408, 1216),  # 1.71 MP
    (1216, 1408),  # 1.71 MP
    (1664, 1024),  # 1.70 MP
    (1024, 1664),  # 1.70 MP
    (1472, 1152),  # 1.70 MP
    (1152, 1472),  # 1.70 MP
    (1536, 1088),  # 1.67 MP
    (1088, 1536),  # 1.67 MP
    (1856, 896),  # 1.66 MP
    (896, 1856),  # 1.66 MP
    (1728, 960),  # 1.66 MP
    (960, 1728),  # 1.66 MP
    (1600, 1024),  # 1.64 MP
    (1024, 1600),  # 1.64 MP
    (1280, 1280),  # 1.64 MP
    (1344, 1216),  # 1.63 MP
    (1216, 1344),  # 1.63 MP
    (1408, 1152),  # 1.62 MP
    (1152, 1408),  # 1.62 MP
    (1792, 896),  # 1.61 MP
    (896, 1792),  # 1.61 MP
    (1472, 1088),  # 1.60 MP
    (1088, 1472),  # 1.60 MP
    (1664, 960),  # 1.60 MP
    (960, 1664),  # 1.60 MP
    (1536, 1024),  # 1.57 MP
    (1024, 1536),  # 1.57 MP
    (1728, 896),  # 1.55 MP
    (896, 1728),  # 1.55 MP
    (1856, 832),  # 1.54 MP
    (832, 1856),  # 1.54 MP
    (1600, 960),  # 1.54 MP
    (960, 1600),  # 1.54 MP
    (1792, 832),  # 1.49 MP
    (832, 1792),  # 1.49 MP
    (1664, 896),  # 1.49 MP
    (896, 1664),  # 1.49 MP
    (1728, 832),  # 1.44 MP
    (832, 1728),  # 1.44 MP
    (1856, 768),  # 1.43 MP
    (768, 1856),  # 1.43 MP
    (1792, 768),  # 1.38 MP
    (768, 1792),  # 1.38 MP
    (1856, 704),  # 1.31 MP
    (704, 1856),  # 1.31 MP
]

# ---------------------------------------------------------------------------
# img2img size fit
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"Size\s*:\s*(\d+)\s*[x\u00d7]\s*(\d+)")


def parse_size_from_infotext(infotext):
    """``(width, height)`` an infotext records (``Size: WxH``), or None."""

    if not isinstance(infotext, str) or not infotext.strip():
        return None
    # Prefer the WebUI's own parser so quoting / escaping is handled the same way.
    try:
        from modules.infotext_utils import parse_generation_parameters

        params = parse_generation_parameters(infotext)
        width, height = params.get("Size-1"), params.get("Size-2")
        if width and height:
            return int(width), int(height)
    except Exception:  # the parser needs the full WebUI; fall back to the regex
        pass
    match = _SIZE_RE.search(infotext)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def upscale_target(width, height):
    """``(target_w, target_h)`` for an img2img source of ``width`` x
    ``height``, or None when no pair fits.

    Both are positive multiples of BUCKET (64); W+H lies in
    [TARGET_TOTAL_LOW, TARGET_TOTAL_HIGH], neither side exceeds
    MAX_SIDE_LIMIT, and the source is never downscaled (target_w >= width,
    target_h >= height).

    Among the candidates whose aspect-ratio drift from the source is within
    MAX_UPSCALE_DRIFT, the highest total (W+H) wins, ties broken by least
    drift: the size push toward 3072 is the goal, the drift cap only stops it
    from distorting the ratio. When no candidate is within the cap (very
    extreme ratios), the least-drift candidate is used. A source already at
    the within-cap maximum stays as it is (its own size is the highest-total,
    zero-drift pair)."""

    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    src_ratio = w / h
    candidates = {}  # (target_w, target_h) -> (drift, total, target_w, target_h)
    for tw in range(BUCKET, TARGET_TOTAL_HIGH + 1, BUCKET):
        if tw < w:
            continue
        if tw > MAX_SIDE_LIMIT:
            break
        h_floor = (int(tw / src_ratio) // BUCKET) * BUCKET
        # Flanks around the ideal height, plus MAX_SIDE_LIMIT itself when the
        # ideal is above the cap: without it very tall sources (640x1536 with
        # an 1856 cap) find no target while their wide mirror does.
        flanks = [h_floor - BUCKET, h_floor, h_floor + BUCKET, h_floor + 2 * BUCKET]
        if h_floor > MAX_SIDE_LIMIT:
            flanks.append(MAX_SIDE_LIMIT)
        for th in flanks:
            if th <= 0 or th < h or th > MAX_SIDE_LIMIT:
                continue
            total = tw + th
            if not TARGET_TOTAL_LOW <= total <= TARGET_TOTAL_HIGH:
                continue
            candidates[(tw, th)] = (abs(tw / th - src_ratio) / src_ratio, total, tw, th)

    if not candidates:
        logger.info("No upscale target for %dx%d with W+H in [%d, %d]", w, h, TARGET_TOTAL_LOW, TARGET_TOTAL_HIGH)
        return None
    in_cap = [candidate for candidate in candidates.values() if candidate[0] <= MAX_UPSCALE_DRIFT]
    if in_cap:
        drift, total, tw, th = min(in_cap, key=lambda c: (-c[1], c[0]))  # max total, then min drift
    else:
        drift, total, tw, th = min(candidates.values(), key=lambda c: (c[0], -c[1]))  # min drift, then max total
    logger.info("Upscale target %dx%d -> %dx%d (sum %d, drift %.2f%%)", w, h, tw, th, total, drift * 100)
    return tw, th


def _source_record(ctx, upload_id):
    """The file record of an upload/output id (``ctx.files``; None outside
    the server, or for an unknown id)."""

    files = getattr(ctx, "files", None)
    return files.get(upload_id) if files is not None else None


def _record_infotext(record):
    if record.infotext:
        return record.infotext
    try:
        from PIL import Image

        from modules import images

        with Image.open(record.path) as image:
            infotext, _items = images.read_info_from_image(image)
        return infotext
    except Exception:  # unreadable metadata: fall back to the image size
        return None


def source_size(ctx):
    """``(width, height)`` of the img2img source: the size its infotext
    records (the size it was generated at, before Hires. fix), else the
    image's own size; ``ctx.extra["source"]`` (``{width, height,
    infotext?}``) stands in for the stored file. None without a source."""

    given = ctx.extra.get("source") if isinstance(ctx.extra, dict) else None
    if isinstance(given, dict):
        size = parse_size_from_infotext(given.get("infotext"))
        return size or (given.get("width"), given.get("height"))
    record = _source_record(ctx, (ctx.core or {}).get("init_img"))
    if record is None or not record.is_image:
        return None
    return parse_size_from_infotext(_record_infotext(record)) or (record.width, record.height)


# ---------------------------------------------------------------------------
# Prompt history
# ---------------------------------------------------------------------------

# Session-only, per user (None: no login), most recent first; cleared when
# the WebUI process exits.
_prompt_history: dict[str | None, list[str]] = {}
# The job each user's latest entry came from (see ``remember_job_prompt``)
_last_job: dict[str | None, object] = {}
_history_lock = threading.Lock()


def _preview(text, max_chars=PROMPT_PREVIEW_CHARS):
    """Single-line truncated preview for the history list."""

    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


def remember_prompt(prompt, user: str | None = None) -> None:
    """Put ``prompt`` first in ``user``'s history (an earlier copy moves up)."""

    if not isinstance(prompt, str) or not prompt.strip():
        return
    clean = prompt.strip()
    with _history_lock:
        history = _prompt_history.setdefault(user, [])
        if clean in history:
            history.remove(clean)
        history.insert(0, clean)
        del history[PROMPT_HISTORY_MAX:]


def _running_job():
    """A token of the running job (``shared.state.begin`` stamps
    ``time_start`` once per job; script cells do not), None outside one."""

    try:
        from modules import shared

        state = shared.state
    except Exception:  # the WebUI state is not set up (tests, tools)
        return None
    if state is None or not getattr(state, "job", ""):
        return None
    return getattr(state, "time_start", None)


def remember_job_prompt(p) -> None:
    """Log ``p``'s prompt for its user, once per job: script cells (X/Y/Z,
    Prompt S/R, prompts from file) run ``before_process`` per cell, and only
    the first (the job's own prompt for X/Y/Z and S/R) is logged."""

    user = getattr(p, "user", None)
    job = _running_job()
    with _history_lock:
        if job is not None and _last_job.get(user) == job:
            return
        _last_job[user] = job
    remember_prompt(getattr(p, "prompt", None), user)


def history_entries(user: str | None = None) -> list[str]:
    with _history_lock:
        return list(_prompt_history.get(user, ()))


def history_choices(user: str | None = None) -> list[tuple[str, str]]:
    """``(preview, prompt)`` pairs of ``user``'s history. The schema lists the
    anonymous history (there is no user there); a logged-in user's own list
    comes with Refresh."""

    return [(_preview(prompt), prompt) for prompt in history_entries(user)]


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

_TXT2IMG_HELP = (
    "Standard presets keep W+H at or below 2176; high-res presets keep W+H within 2560\u20133072 "
    "(at most 1856 per side). Randomize (standard) rolls the standard list, Randomize (high-res) the "
    "high-res list; both on roll the combined pool."
)
_IMG2IMG_HELP = (
    "Sets Width and Height (the Resize to tab) to a multiple-of-64 size whose W+H is as close to 3072 "
    "as the aspect ratio allows (2560\u20133072, at most 1856 per side). The source size is the size "
    "its parameters record, else the image size."
)


def _sizes(table) -> list[list[int]]:
    return [[w, h] for w, h in table]


class AnimaRandomizeResolutionScript(scripts.Script):
    """Resolution presets, Randomize and prompt history on txt2img; the
    source size fit on img2img. When either Randomize toggle is on, every
    txt2img job's width and height are rolled in ``before_process`` from
    RESOLUTIONS, HIGH_RES_RESOLUTIONS, or both (the combined pool)."""

    section = "dimensions"

    def title(self):
        return "Anima random resolution"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        if is_img2img:
            # No processing argument is read on img2img
            self.param_order = None
            return ui.Group(
                "Anima resolution helper",
                ui.Markdown(_IMG2IMG_HELP),
                ui.Custom(
                    "fit_source",
                    "source-size-fit",
                    True,
                    props={"source": "core.init_img", "action": "fit_to_source", "manual_action": "fit_to_source_now"},
                    label="",
                ),
            )
        # The Randomize toggles are the processing arguments, first as before;
        # the preset picker and the history pick are form helpers.
        self.param_order = ("randomize", "randomize_highres", "presets", "history")
        return ui.Group(
            "Anima resolution helper",
            ui.Markdown(_TXT2IMG_HELP),
            ui.Custom(
                "presets",
                "resolution-presets",
                None,
                props={"standard": _sizes(RESOLUTIONS), "highres": _sizes(HIGH_RES_RESOLUTIONS), "width": "core.width", "height": "core.height"},
                label="",
            ),
            ui.Row(
                # allow_custom: the empty pick and entries the server no longer
                # has (an earlier session, another user's list) must still
                # validate on Generate; use_history ignores anything typed.
                ui.Choice(
                    "history",
                    history_choices,
                    "",
                    allow_custom=True,
                    label="Session prompt history (most recent first, max 100)",
                    info="Pick an entry to put it back into the prompt; Refresh adds the latest prompts.",
                    on_change="use_history",
                    width="fill",
                ),
                ui.Button("Refresh", "refresh_history", "tool"),
            ),
            ui.Row(
                ui.Toggle("randomize", False, label="Randomize resolution each generation (standard preset list)", preset="resolution.random_standard"),
                ui.Toggle(
                    "randomize_highres",
                    False,
                    label="Randomize high-res resolution each generation (high-res preset list; both on = combined pool)",
                    preset="resolution.random_highres",
                ),
            ),
        )

    @ui.action("use_history")
    def use_history(self, ctx, values):
        prompt = values.get("history")
        if not isinstance(prompt, str) or not prompt:
            return {}
        # Only the user's own entries: typed text does not replace the prompt
        if prompt not in history_entries(ctx.user):
            return {"history": {"value": ""}}
        # The pick goes back to empty, so the same entry can be picked again
        return {"core.prompt": {"value": prompt}, "history": {"value": ""}}

    @ui.action("refresh_history")
    def refresh_history(self, ctx, values):
        return {"history": {"choices": history_choices(ctx.user), "value": ""}}

    @ui.action("fit_to_source")
    def fit_to_source(self, ctx, values):
        """Width and height for the current img2img source (see
        :func:`upscale_target`), run when the source changes: quiet when
        there is nothing to fit."""

        return self._fit(ctx, manual=False)

    @ui.action("fit_to_source_now")
    def fit_to_source_now(self, ctx, values):
        """:meth:`fit_to_source` for the button: says why nothing changed."""

        return self._fit(ctx, manual=True)

    @staticmethod
    def _fit(ctx, *, manual):
        size = source_size(ctx)
        if size is None or None in size:
            return ui.toast("Add a source image first.", "warning") if manual else {}
        target = upscale_target(*size)
        if target is None:
            message = ui.message(
                "No size with W+H in {{low}}\u2013{{high}} fits a {{source_width}} x {{source_height}} source; the size is unchanged.",
                low=TARGET_TOTAL_LOW,
                high=TARGET_TOTAL_HIGH,
                source_width=size[0],
                source_height=size[1],
            )
            return ui.toast(message, "info") if manual else {}
        width, height = target
        return {
            "core.width": {"value": width},
            "core.height": {"value": height},
            **ui.toast(ui.message("Size set to {{width}} x {{height}} (source {{source_width}} x {{source_height}}).", width=width, height=height, source_width=size[0], source_height=size[1])),
        }

    def before_process(self, p, randomize_enabled=False, randomize_highres_enabled=False, *_helpers, **kwargs):
        if getattr(self, "is_img2img", False) or hasattr(p, "init_images"):
            return
        remember_job_prompt(p)
        std_on = bool(randomize_enabled)
        hr_on = bool(randomize_highres_enabled)
        if not (std_on or hr_on):
            return
        if std_on and hr_on:
            pool, pool_name = RESOLUTIONS + HIGH_RES_RESOLUTIONS, "combined"
        elif hr_on:
            pool, pool_name = HIGH_RES_RESOLUTIONS, "high-res"
        else:
            pool, pool_name = RESOLUTIONS, "standard"
        new_w, new_h = random.choice(pool)
        logger.info("Randomized (%s pool) %sx%s -> %dx%d", pool_name, getattr(p, "width", None), getattr(p, "height", None), new_w, new_h)
        p.width = new_w
        p.height = new_h
