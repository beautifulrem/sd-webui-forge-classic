# Very-early diagnostic: if you don't see this line in your console at startup,
# the file isn't being imported (Forge isn't finding it).
print("[anima-resolution] script file is being imported")

"""
anima_resolutions.py
--------------------
Forge Neo extension. Four features layered together:

  1. Inline resolution picker (two preset dropdowns) injected directly
     below the txt2img seed row. The standard list caps txt2img W+H at
     2176; a secondary high-res list offers larger first-pass targets
     with W+H in [2560, 3072] and each dimension capped at 1856.

  2. img2img "Resize to" auto-adjuster: on Send to img2img / Send to
     inpaint, computes target W and H that both stay multiples of 64
     and land the post-upscale W+H in [2560, 3072] (neither side over
     1856), choosing the pair with the HIGHEST total (closest to 3072)
     among those whose aspect-ratio drift from the source stays within
     MAX_UPSCALE_DRIFT. (If no pair is within the cap - only for very
     extreme source ratios - it falls back to the least-drift pair.)
     Applies uniformly to both standard and high-res sources, because
     the resize step reads the source W/H from the displayed image's
     PNG metadata (the hidden generation_info textbox), so it stays
     correct regardless of which preset list produced the image, and
     correct when Randomize overrides the txt2img W/H sliders
     mid-generation.

  3. Randomize toggles - two independent checkboxes, applied per
     txt2img generation via a `scripts.Script` subclass running
     before_process. "Randomize (standard)" rolls from the standard
     preset list; "Randomize (high-res)" rolls from the high-res
     preset list. If both are on, the roll draws from the combined
     pool of both lists. img2img is unaffected either way.

  4. Session prompt history - logs every positive prompt at Generate
     time into a session-only ring (cap 100, oldest dropped first), shown
     in a dropdown directly below the preset dropdown. Selecting an entry
     refills the positive prompt textbox.

All preset base dimensions and all computed upscale targets are
multiples of 64, so latent dimensions divide cleanly with no distortion.

Designed for Anima (https://huggingface.co/circlestone-labs/Anima).
"""

import random
import traceback

import gradio as gr

from modules import script_callbacks, scripts


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
    ("1088 x 1088", 1088, 1088),  # 1.18 MP
    ("1152 x 1024", 1152, 1024),  # 1.18 MP
    ("1024 x 1152", 1024, 1152),  # 1.18 MP
    ("1216 x 960", 1216,  960),  # 1.17 MP
    (" 960 x 1216",  960, 1216),  # 1.17 MP
    ("1280 x 896", 1280,  896),  # 1.15 MP
    (" 896 x 1280",  896, 1280),  # 1.15 MP
    ("1344 x 832", 1344,  832),  # 1.12 MP
    (" 832 x 1344",  832, 1344),  # 1.12 MP
    ("1088 x 1024", 1088, 1024),  # 1.11 MP
    ("1024 x 1088", 1024, 1088),  # 1.11 MP
    ("1152 x 960", 1152,  960),  # 1.11 MP
    (" 960 x 1152",  960, 1152),  # 1.11 MP
    ("1216 x 896", 1216,  896),  # 1.09 MP
    (" 896 x 1216",  896, 1216),  # 1.09 MP
    ("1408 x 768", 1408,  768),  # 1.08 MP
    (" 768 x 1408",  768, 1408),  # 1.08 MP
    ("1280 x 832", 1280,  832),  # 1.06 MP
    (" 832 x 1280",  832, 1280),  # 1.06 MP
    ("1024 x 1024", 1024, 1024),  # 1.05 MP
    ("1088 x 960", 1088,  960),  # 1.04 MP
    (" 960 x 1088",  960, 1088),  # 1.04 MP
    ("1472 x 704", 1472,  704),  # 1.04 MP
    (" 704 x 1472",  704, 1472),  # 1.04 MP
    ("1344 x 768", 1344,  768),  # 1.03 MP
    ("1152 x 896", 1152,  896),  # 1.03 MP
    (" 896 x 1152",  896, 1152),  # 1.03 MP
    (" 768 x 1344",  768, 1344),  # 1.03 MP
    ("1216 x 832", 1216,  832),  # 1.01 MP
    (" 832 x 1216",  832, 1216),  # 1.01 MP
    ("1408 x 704", 1408,  704),  # 0.99 MP
    (" 704 x 1408",  704, 1408),  # 0.99 MP
    ("1536 x 640", 1536,  640),  # 0.98 MP
    ("1280 x 768", 1280,  768),  # 0.98 MP
    ("1024 x 960", 1024,  960),  # 0.98 MP
    (" 960 x 1024",  960, 1024),  # 0.98 MP
    (" 768 x 1280",  768, 1280),  # 0.98 MP
    (" 640 x 1536",  640, 1536),  # 0.98 MP
    ("1088 x 896", 1088,  896),  # 0.97 MP
    (" 896 x 1088",  896, 1088),  # 0.97 MP
    ("1152 x 832", 1152,  832),  # 0.96 MP
    (" 832 x 1152",  832, 1152),  # 0.96 MP
    ("1344 x 704", 1344,  704),  # 0.95 MP
    (" 704 x 1344",  704, 1344),  # 0.95 MP
    ("1472 x 640", 1472,  640),  # 0.94 MP
    (" 640 x 1472",  640, 1472),  # 0.94 MP
    ("1216 x 768", 1216,  768),  # 0.93 MP
    (" 768 x 1216",  768, 1216),  # 0.93 MP
    (" 960 x 960",  960,  960),  # 0.92 MP
    ("1024 x 896", 1024,  896),  # 0.92 MP
    (" 896 x 1024",  896, 1024),  # 0.92 MP
    ("1088 x 832", 1088,  832),  # 0.91 MP
    (" 832 x 1088",  832, 1088),  # 0.91 MP
    ("1408 x 640", 1408,  640),  # 0.90 MP
    ("1280 x 704", 1280,  704),  # 0.90 MP
    (" 704 x 1280",  704, 1280),  # 0.90 MP
    (" 640 x 1408",  640, 1408),  # 0.90 MP
    ("1152 x 768", 1152,  768),  # 0.88 MP
    (" 768 x 1152",  768, 1152),  # 0.88 MP
    ("1344 x 640", 1344,  640),  # 0.86 MP
    (" 960 x 896",  960,  896),  # 0.86 MP
    (" 896 x 960",  896,  960),  # 0.86 MP
    (" 640 x 1344",  640, 1344),  # 0.86 MP
    ("1216 x 704", 1216,  704),  # 0.86 MP
    (" 704 x 1216",  704, 1216),  # 0.86 MP
    ("1024 x 832", 1024,  832),  # 0.85 MP
    (" 832 x 1024",  832, 1024),  # 0.85 MP
    ("1088 x 768", 1088,  768),  # 0.84 MP
    (" 768 x 1088",  768, 1088),  # 0.84 MP
    ("1280 x 640", 1280,  640),  # 0.82 MP
    (" 640 x 1280",  640, 1280),  # 0.82 MP
    ("1152 x 704", 1152,  704),  # 0.81 MP
    (" 704 x 1152",  704, 1152),  # 0.81 MP
    (" 896 x 896",  896,  896),  # 0.80 MP
    (" 960 x 832",  960,  832),  # 0.80 MP
    (" 832 x 960",  832,  960),  # 0.80 MP
    ("1024 x 768", 1024,  768),  # 0.79 MP
    (" 768 x 1024",  768, 1024),  # 0.79 MP
    ("1216 x 640", 1216,  640),  # 0.78 MP
    (" 640 x 1216",  640, 1216),  # 0.78 MP
    ("1088 x 704", 1088,  704),  # 0.77 MP
    (" 704 x 1088",  704, 1088),  # 0.77 MP
    (" 896 x 832",  896,  832),  # 0.75 MP
    (" 832 x 896",  832,  896),  # 0.75 MP
    ("1152 x 640", 1152,  640),  # 0.74 MP
    (" 960 x 768",  960,  768),  # 0.74 MP
    (" 768 x 960",  768,  960),  # 0.74 MP
    (" 640 x 1152",  640, 1152),  # 0.74 MP
    ("1024 x 704", 1024,  704),  # 0.72 MP
    (" 704 x 1024",  704, 1024),  # 0.72 MP
    ("1088 x 640", 1088,  640),  # 0.70 MP
    (" 640 x 1088",  640, 1088),  # 0.70 MP
    (" 832 x 832",  832,  832),  # 0.69 MP
    (" 896 x 768",  896,  768),  # 0.69 MP
    (" 768 x 896",  768,  896),  # 0.69 MP
    (" 960 x 704",  960,  704),  # 0.68 MP
    (" 704 x 960",  704,  960),  # 0.68 MP
    ("1024 x 640", 1024,  640),  # 0.66 MP
    (" 640 x 1024",  640, 1024),  # 0.66 MP
    (" 832 x 768",  832,  768),  # 0.64 MP
    (" 768 x 832",  768,  832),  # 0.64 MP
    (" 896 x 704",  896,  704),  # 0.63 MP
    (" 704 x 896",  704,  896),  # 0.63 MP
    (" 960 x 640",  960,  640),  # 0.61 MP
    (" 640 x 960",  640,  960),  # 0.61 MP
    (" 768 x 768",  768,  768),  # 0.59 MP
    (" 832 x 704",  832,  704),  # 0.59 MP
    (" 704 x 832",  704,  832),  # 0.59 MP
    (" 896 x 640",  896,  640),  # 0.57 MP
    (" 640 x 896",  640,  896),  # 0.57 MP
    (" 768 x 704",  768,  704),  # 0.54 MP
    (" 704 x 768",  704,  768),  # 0.54 MP
    (" 832 x 640",  832,  640),  # 0.53 MP
    (" 640 x 832",  640,  832),  # 0.53 MP
    (" 704 x 704",  704,  704),  # 0.50 MP
    (" 768 x 640",  768,  640),  # 0.49 MP
    (" 640 x 768",  640,  768),  # 0.49 MP
    (" 704 x 640",  704,  640),  # 0.45 MP
    (" 640 x 704",  640,  704),  # 0.45 MP
    (" 640 x 640",  640,  640),  # 0.41 MP
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
    ("1536 x 1536", 1536, 1536),  # 2.36 MP
    ("1600 x 1472", 1600, 1472),  # 2.36 MP
    ("1472 x 1600", 1472, 1600),  # 2.36 MP
    ("1664 x 1408", 1664, 1408),  # 2.34 MP
    ("1408 x 1664", 1408, 1664),  # 2.34 MP
    ("1728 x 1344", 1728, 1344),  # 2.32 MP
    ("1344 x 1728", 1344, 1728),  # 2.32 MP
    ("1792 x 1280", 1792, 1280),  # 2.29 MP
    ("1280 x 1792", 1280, 1792),  # 2.29 MP
    ("1536 x 1472", 1536, 1472),  # 2.26 MP
    ("1472 x 1536", 1472, 1536),  # 2.26 MP
    ("1856 x 1216", 1856, 1216),  # 2.26 MP
    ("1216 x 1856", 1216, 1856),  # 2.26 MP
    ("1600 x 1408", 1600, 1408),  # 2.25 MP
    ("1408 x 1600", 1408, 1600),  # 2.25 MP
    ("1664 x 1344", 1664, 1344),  # 2.24 MP
    ("1344 x 1664", 1344, 1664),  # 2.24 MP
    ("1728 x 1280", 1728, 1280),  # 2.21 MP
    ("1280 x 1728", 1280, 1728),  # 2.21 MP
    ("1792 x 1216", 1792, 1216),  # 2.18 MP
    ("1216 x 1792", 1216, 1792),  # 2.18 MP
    ("1472 x 1472", 1472, 1472),  # 2.17 MP
    ("1536 x 1408", 1536, 1408),  # 2.16 MP
    ("1408 x 1536", 1408, 1536),  # 2.16 MP
    ("1600 x 1344", 1600, 1344),  # 2.15 MP
    ("1344 x 1600", 1344, 1600),  # 2.15 MP
    ("1856 x 1152", 1856, 1152),  # 2.14 MP
    ("1152 x 1856", 1152, 1856),  # 2.14 MP
    ("1664 x 1280", 1664, 1280),  # 2.13 MP
    ("1280 x 1664", 1280, 1664),  # 2.13 MP
    ("1728 x 1216", 1728, 1216),  # 2.10 MP
    ("1216 x 1728", 1216, 1728),  # 2.10 MP
    ("1472 x 1408", 1472, 1408),  # 2.07 MP
    ("1408 x 1472", 1408, 1472),  # 2.07 MP
    ("1792 x 1152", 1792, 1152),  # 2.06 MP
    ("1152 x 1792", 1152, 1792),  # 2.06 MP
    ("1536 x 1344", 1536, 1344),  # 2.06 MP
    ("1344 x 1536", 1344, 1536),  # 2.06 MP
    ("1600 x 1280", 1600, 1280),  # 2.05 MP
    ("1280 x 1600", 1280, 1600),  # 2.05 MP
    ("1664 x 1216", 1664, 1216),  # 2.02 MP
    ("1216 x 1664", 1216, 1664),  # 2.02 MP
    ("1856 x 1088", 1856, 1088),  # 2.02 MP
    ("1088 x 1856", 1088, 1856),  # 2.02 MP
    ("1728 x 1152", 1728, 1152),  # 1.99 MP
    ("1152 x 1728", 1152, 1728),  # 1.99 MP
    ("1408 x 1408", 1408, 1408),  # 1.98 MP
    ("1472 x 1344", 1472, 1344),  # 1.98 MP
    ("1344 x 1472", 1344, 1472),  # 1.98 MP
    ("1536 x 1280", 1536, 1280),  # 1.97 MP
    ("1280 x 1536", 1280, 1536),  # 1.97 MP
    ("1792 x 1088", 1792, 1088),  # 1.95 MP
    ("1088 x 1792", 1088, 1792),  # 1.95 MP
    ("1600 x 1216", 1600, 1216),  # 1.95 MP
    ("1216 x 1600", 1216, 1600),  # 1.95 MP
    ("1664 x 1152", 1664, 1152),  # 1.92 MP
    ("1152 x 1664", 1152, 1664),  # 1.92 MP
    ("1856 x 1024", 1856, 1024),  # 1.90 MP
    ("1024 x 1856", 1024, 1856),  # 1.90 MP
    ("1408 x 1344", 1408, 1344),  # 1.89 MP
    ("1344 x 1408", 1344, 1408),  # 1.89 MP
    ("1472 x 1280", 1472, 1280),  # 1.88 MP
    ("1280 x 1472", 1280, 1472),  # 1.88 MP
    ("1728 x 1088", 1728, 1088),  # 1.88 MP
    ("1088 x 1728", 1088, 1728),  # 1.88 MP
    ("1536 x 1216", 1536, 1216),  # 1.87 MP
    ("1216 x 1536", 1216, 1536),  # 1.87 MP
    ("1600 x 1152", 1600, 1152),  # 1.84 MP
    ("1152 x 1600", 1152, 1600),  # 1.84 MP
    ("1792 x 1024", 1792, 1024),  # 1.84 MP
    ("1024 x 1792", 1024, 1792),  # 1.84 MP
    ("1664 x 1088", 1664, 1088),  # 1.81 MP
    ("1088 x 1664", 1088, 1664),  # 1.81 MP
    ("1344 x 1344", 1344, 1344),  # 1.81 MP
    ("1408 x 1280", 1408, 1280),  # 1.80 MP
    ("1280 x 1408", 1280, 1408),  # 1.80 MP
    ("1472 x 1216", 1472, 1216),  # 1.79 MP
    ("1216 x 1472", 1216, 1472),  # 1.79 MP
    ("1856 x 960", 1856, 960),  # 1.78 MP
    ("960 x 1856", 960, 1856),  # 1.78 MP
    ("1728 x 1024", 1728, 1024),  # 1.77 MP
    ("1024 x 1728", 1024, 1728),  # 1.77 MP
    ("1536 x 1152", 1536, 1152),  # 1.77 MP
    ("1152 x 1536", 1152, 1536),  # 1.77 MP
    ("1600 x 1088", 1600, 1088),  # 1.74 MP
    ("1088 x 1600", 1088, 1600),  # 1.74 MP
    ("1792 x 960", 1792, 960),  # 1.72 MP
    ("960 x 1792", 960, 1792),  # 1.72 MP
    ("1344 x 1280", 1344, 1280),  # 1.72 MP
    ("1280 x 1344", 1280, 1344),  # 1.72 MP
    ("1408 x 1216", 1408, 1216),  # 1.71 MP
    ("1216 x 1408", 1216, 1408),  # 1.71 MP
    ("1664 x 1024", 1664, 1024),  # 1.70 MP
    ("1024 x 1664", 1024, 1664),  # 1.70 MP
    ("1472 x 1152", 1472, 1152),  # 1.70 MP
    ("1152 x 1472", 1152, 1472),  # 1.70 MP
    ("1536 x 1088", 1536, 1088),  # 1.67 MP
    ("1088 x 1536", 1088, 1536),  # 1.67 MP
    ("1856 x 896", 1856, 896),  # 1.66 MP
    ("896 x 1856", 896, 1856),  # 1.66 MP
    ("1728 x 960", 1728, 960),  # 1.66 MP
    ("960 x 1728", 960, 1728),  # 1.66 MP
    ("1600 x 1024", 1600, 1024),  # 1.64 MP
    ("1024 x 1600", 1024, 1600),  # 1.64 MP
    ("1280 x 1280", 1280, 1280),  # 1.64 MP
    ("1344 x 1216", 1344, 1216),  # 1.63 MP
    ("1216 x 1344", 1216, 1344),  # 1.63 MP
    ("1408 x 1152", 1408, 1152),  # 1.62 MP
    ("1152 x 1408", 1152, 1408),  # 1.62 MP
    ("1792 x 896", 1792, 896),  # 1.61 MP
    ("896 x 1792", 896, 1792),  # 1.61 MP
    ("1472 x 1088", 1472, 1088),  # 1.60 MP
    ("1088 x 1472", 1088, 1472),  # 1.60 MP
    ("1664 x 960", 1664, 960),  # 1.60 MP
    ("960 x 1664", 960, 1664),  # 1.60 MP
    ("1536 x 1024", 1536, 1024),  # 1.57 MP
    ("1024 x 1536", 1024, 1536),  # 1.57 MP
    ("1728 x 896", 1728, 896),  # 1.55 MP
    ("896 x 1728", 896, 1728),  # 1.55 MP
    ("1856 x 832", 1856, 832),  # 1.54 MP
    ("832 x 1856", 832, 1856),  # 1.54 MP
    ("1600 x 960", 1600, 960),  # 1.54 MP
    ("960 x 1600", 960, 1600),  # 1.54 MP
    ("1792 x 832", 1792, 832),  # 1.49 MP
    ("832 x 1792", 832, 1792),  # 1.49 MP
    ("1664 x 896", 1664, 896),  # 1.49 MP
    ("896 x 1664", 896, 1664),  # 1.49 MP
    ("1728 x 832", 1728, 832),  # 1.44 MP
    ("832 x 1728", 832, 1728),  # 1.44 MP
    ("1856 x 768", 1856, 768),  # 1.43 MP
    ("768 x 1856", 768, 1856),  # 1.43 MP
    ("1792 x 768", 1792, 768),  # 1.38 MP
    ("768 x 1792", 768, 1792),  # 1.38 MP
    ("1856 x 704", 1856, 704),  # 1.31 MP
    ("704 x 1856", 704, 1856),  # 1.31 MP
]
# Combined label lookup, populated from both lists.
_LABEL_TO_DIMS = {label: (w, h) for label, w, h in RESOLUTIONS}
_LABEL_TO_DIMS.update({label: (w, h) for label, w, h in HIGH_RES_RESOLUTIONS})


# ---------------------------------------------------------------------------
# Tracking state
# ---------------------------------------------------------------------------

_TRACKED_IDS = (
    "txt2img_width",
    "txt2img_height",
    "txt2img_prompt",
    "txt2img_generate",
    "txt2img_seed_row",
    "img2img_width",
    "img2img_height",
    "txt2img_send_to_img2img",
    "txt2img_send_to_inpaint",
    # The hidden textbox holding the displayed image's PNG metadata. The
    # elem_id naming differs across WebUI versions; we track both common
    # spellings and use whichever fires.
    "generation_info_txt2img",
    "txt2img_generation_info",
)

_refs = {k: None for k in _TRACKED_IDS}
_picker_anchor = None
_resize_wired_buttons = set()
_history_wired_buttons = set()

# Components we create inside _build_picker_ui that we need to wire to
# external components later (when those external refs become available).
_picker_components = {"history_dropdown": None}

# Module-level state shared with the AnimaRandomizeResolutionScript.
_state = {"randomize_enabled": False, "randomize_highres_enabled": False}

# Session-only prompt history; most recent first. Cleared on webui restart
# because module-level state doesn't persist.
_prompt_history = []


# ---------------------------------------------------------------------------
# Resize-by scale computation
# ---------------------------------------------------------------------------

# Module-level imports for the infotext parser. Tried at module load; the
# fallback regex is used if neither name resolves.
try:
    from modules import infotext_utils as _paste_module  # neo
except ImportError:
    try:
        from modules import generation_parameters_copypaste as _paste_module  # legacy
    except ImportError:
        _paste_module = None

import re as _re
_SIZE_RE = _re.compile(r"Size\s*:\s*(\d+)\s*[x\u00d7]\s*(\d+)")


def _parse_size_from_infotext(infotext):
    """Pull (width, height) out of an infotext blob. Returns (w, h) ints or None."""
    if not isinstance(infotext, str) or not infotext.strip():
        return None

    # Prefer the WebUI's own parser so we handle quoting / escaping the same way.
    if _paste_module is not None:
        try:
            params = _paste_module.parse_generation_parameters(infotext)
            w_raw = params.get("Size-1")
            h_raw = params.get("Size-2")
            if w_raw and h_raw:
                return int(w_raw), int(h_raw)
        except Exception:
            pass  # fall through to regex

    # Fallback: regex on "Size: WxH"
    m = _SIZE_RE.search(infotext)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except (TypeError, ValueError):
            pass
    return None


def _compute_resize_scale_from_infotext(infotext):
    """Read W, H from the image's PNG metadata (the generation_info textbox),
    then dispatch to _compute_upscale_target. Used so that Randomize-changed
    dimensions are honored, since the txt2img W/H sliders may not match the
    dimensions actually generated. Returns two gr.update objects: one for
    img2img_width, one for img2img_height (the 'Resize to' tab inputs)."""
    dims = _parse_size_from_infotext(infotext)
    if dims is None:
        print("[anima-resolution] could not parse Size from infotext; "
              "leaving Resize-to unchanged")
        return gr.update(), gr.update()
    w, h = dims
    print(f"[anima-resolution] parsed Size from metadata: {w} x {h}")
    return _compute_upscale_target(w, h)


def _compute_upscale_target(width, height):
    """Pick (target_W, target_H) for the img2img Resize-to inputs.

    Both dims are positive multiples of BUCKET (64). The target's W+H always
    falls in [TARGET_TOTAL_LOW, TARGET_TOTAL_HIGH], keeps each side <=
    MAX_SIDE_LIMIT (1856), and never downscales the source (target_W >=
    width, target_H >= height).

    Picking strategy (uniform for all sources): among the in-window
    multiple-of-64 candidates whose aspect-ratio drift from the source is
    within MAX_UPSCALE_DRIFT, pick the one with the HIGHEST total (W+H) - i.e.
    as close to TARGET_TOTAL_HIGH (3072) as the drift cap permits. Ties on
    total are broken by least drift. The size push toward 3072 is the goal;
    the drift cap only stops it from badly distorting the ratio.

    Fallback: if NO candidate is within the drift cap (only for very extreme
    source ratios, where even the best in-window pair distorts), the single
    least-drift candidate is used instead so a sane target is still emitted.

    Sources already in the window are pushed up toward 3072 too: a high-res
    source below the top of the window is upscaled to the largest within-cap
    pair, and a source already at the within-cap maximum stays put (its self-
    pair is the highest-total, zero-drift option). Standard-preset sources sit
    below the floor and always upscale. Nothing is ever downscaled.

    Returns two gr.update objects (for img2img_width, img2img_height) or two
    no-op updates if no valid pair exists at all (shouldn't happen for any
    standard or high-res preset; verified at list-generation time)."""
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return gr.update(), gr.update()
    if w <= 0 or h <= 0:
        return gr.update(), gr.update()

    src_ratio = w / h
    candidates = []  # list of (drift, total, target_w, target_h)

    # Sweep target_w across all multiples of BUCKET up to TARGET_TOTAL_HIGH.
    # For each, sweep a small window around the geometrically-ideal target_h
    # to catch valid alternatives the strict two-flank sweep would miss
    # (matters in the fallback when both flanks blow tolerance).
    for tw in range(BUCKET, TARGET_TOTAL_HIGH + 1, BUCKET):
        if tw < w:
            continue
        if tw > MAX_SIDE_LIMIT:
            break
        h_ideal = tw / src_ratio
        h_floor = (int(h_ideal) // BUCKET) * BUCKET
        # Flank sweep around the ideal height, plus MAX_SIDE_LIMIT itself when
        # the ideal sits above the cap. Without that extra candidate, very tall
        # sources (e.g. 640x1536 with a 1856 cap) get every flank rejected as
        # over-cap and find no target at all, while the mirrored wide source
        # still succeeds because the tw sweep stops naturally at the cap. The
        # clamped candidate restores the symmetry (it lands in the min-drift
        # fallback path, same as the wide mirror).
        th_flanks = [h_floor - BUCKET, h_floor, h_floor + BUCKET, h_floor + 2 * BUCKET]
        if h_floor > MAX_SIDE_LIMIT:
            th_flanks.append(MAX_SIDE_LIMIT)
        for th in th_flanks:
            if th <= 0 or th < h or th > MAX_SIDE_LIMIT:
                continue
            total = tw + th
            if not (TARGET_TOTAL_LOW <= total <= TARGET_TOTAL_HIGH):
                continue
            drift = abs(tw / th - src_ratio) / src_ratio
            candidates.append((drift, total, tw, th))

    # Dedupe (the four-flank sweep can produce duplicates when h_floor and
    # h_floor+BUCKET both map to the same valid pair via different tw values).
    candidates = list({(c[2], c[3]): c for c in candidates}.values())

    if not candidates:
        print(f"[anima-resolution] no valid upscale target for {w}x{h} "
              f"in W+H range [{TARGET_TOTAL_LOW}, {TARGET_TOTAL_HIGH}]; "
              f"leaving Resize-to unchanged")
        return gr.update(), gr.update()

    # Objective: get the total (W+H) as close to TARGET_TOTAL_HIGH (3072) as
    # possible, i.e. MAXIMIZE total - but only among candidates whose drift is
    # within MAX_UPSCALE_DRIFT, so the size push never throws the aspect ratio
    # badly off. Tiebreak on equal total: prefer the least-drift pair. The self-
    # pair (source unchanged) has drift 0 and is always eligible when the source
    # is in-window, so a source already at the within-cap maximum stays put,
    # while one lower in the window gets nudged up toward 3072.
    in_cap = [c for c in candidates if c[0] <= MAX_UPSCALE_DRIFT]
    if in_cap:
        in_cap.sort(key=lambda c: (-c[1], c[0]))  # max total, then min drift
        _, total, tw, th = in_cap[0]
        mode_note = "max-total within drift cap"
    else:
        # No in-window pair stays within the drift cap (only very extreme source
        # ratios). Fall back to the single least-drift pair so we still emit a
        # sane, in-window target rather than nothing.
        candidates.sort(key=lambda c: (c[0], -c[1]))  # min drift, then max total
        _, total, tw, th = candidates[0]
        mode_note = "min-drift fallback (no pair within drift cap)"

    if (tw, th) == (w, h):
        mode_note += ", source unchanged (no-op)"

    drift_pct = abs(tw / th - src_ratio) / src_ratio * 100
    print(f"[anima-resolution] auto-upscale target: {w}x{h} (ratio {src_ratio:.3f}) "
          f"-> {tw}x{th} (sum {total}, ratio {tw/th:.3f}, "
          f"drift {drift_pct:.2f}%, scale ~{tw/w:.3f}) [{mode_note}]")
    return gr.update(value=tw), gr.update(value=th)


# ---------------------------------------------------------------------------
# Picker UI handlers
# ---------------------------------------------------------------------------

def _apply_preset(label):
    dims = _LABEL_TO_DIMS.get(label)
    if dims is None:
        return gr.update(), gr.update()
    return gr.update(value=dims[0]), gr.update(value=dims[1])


# ---------------------------------------------------------------------------
# Prompt history helpers
# ---------------------------------------------------------------------------

def _preview(text, max_chars=PROMPT_PREVIEW_CHARS):
    """Single-line truncated preview for the dropdown label."""
    text = (text or "").strip()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rstrip() + "\u2026"  # ellipsis


def _format_history_choices():
    # Gradio Dropdown accepts (label, value) tuples. Label is shown in the
    # dropdown UI, value is what the change handler receives.
    return [(_preview(p), p) for p in _prompt_history]


def _log_prompt_on_generate(prompt):
    """Called as an additional click-handler on the Generate button. Reads the
    current value of the prompt textbox and prepends it to the session history
    (LRU dedup). Does NOT update the dropdown -- that only happens when the
    user clicks the Refresh button."""
    if isinstance(prompt, str) and prompt.strip():
        clean = prompt.strip()
        # LRU: remove existing occurrence anywhere in the list, then re-add at top.
        if clean in _prompt_history:
            _prompt_history.remove(clean)
        _prompt_history.insert(0, clean)
        # Cap the size, dropping oldest entries first.
        del _prompt_history[PROMPT_HISTORY_MAX:]
    # No return: outputs=[] in the click wiring.


def _refresh_history():
    """Refresh button handler. Just rebuilds the choices list."""
    return gr.update(choices=_format_history_choices(), value=None)


def _on_history_selected(selected):
    """Dropdown.change handler. If a real value was selected, fill the prompt
    textbox and reset the dropdown to None so the same entry can be picked
    again later. If the value just got cleared (our own reset), no-op so we
    don't loop."""
    if not isinstance(selected, str) or not selected:
        return gr.update(), gr.update()
    return gr.update(value=selected), gr.update(value=None)


def _set_randomize_enabled(value):
    _state["randomize_enabled"] = bool(value)
    print(f"[anima-resolution] randomize_enabled = {_state['randomize_enabled']}")


def _set_randomize_highres_enabled(value):
    _state["randomize_highres_enabled"] = bool(value)
    print(f"[anima-resolution] randomize_highres_enabled = "
          f"{_state['randomize_highres_enabled']}")


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def _build_picker_ui(width_comp, height_comp, prompt_comp):
    with gr.Group(elem_id="txt2img_anima_resolution_group"):
        gr.Markdown(
            "**Anima resolution helper** &nbsp; "
            "<small><i>standard preset total &le; 2176; high-res preset total "
            "2560&ndash;3072 (capped 1856 per side); on send to img2img/inpaint, "
            "Resize-to is auto-set to a multiple-of-64 target whose W+H "
            "lands as close to 3072 as possible (in 2560&ndash;3072, neither side over 1856) while keeping the aspect ratio close. Randomize (standard) rolls the standard list; Randomize (high-res) rolls the high-res list; both on = combined pool.</i></small>"
        )
        # Standard preset dropdown (W+H <= 2176; also the Randomize pool)
        dropdown = gr.Dropdown(
            choices=[label for label, _, _ in RESOLUTIONS],
            value=None,
            label="Preset",
            interactive=True,
            elem_id="txt2img_anima_preset",
        )

        # High-res preset dropdown (W+H 2560-3072, dims capped at 1856).
        # Shares _apply_preset with the standard dropdown via the merged
        # _LABEL_TO_DIMS lookup. Rolled only by the high-res Randomize toggle.
        highres_dropdown = gr.Dropdown(
            choices=[label for label, _, _ in HIGH_RES_RESOLUTIONS],
            value=None,
            label="High-res preset (txt2img total 2560\u20133072, max 1856 per side)",
            interactive=True,
            elem_id="txt2img_anima_preset_highres",
        )

        # Session prompt history dropdown + refresh button
        with gr.Row():
            history_dropdown = gr.Dropdown(
                choices=[],
                value=None,
                label="Session prompt history (most recent first, max 100, \u21bb Refresh to update)",
                interactive=True,
                elem_id="txt2img_anima_history",
                scale=4,
            )
            refresh_btn = gr.Button(
                "\u21bb Refresh",
                elem_id="txt2img_anima_history_refresh",
                scale=1,
            )
        _picker_components["history_dropdown"] = history_dropdown

        # Randomize toggles. Standard rolls RESOLUTIONS; high-res rolls
        # HIGH_RES_RESOLUTIONS; both on rolls the combined pool.
        with gr.Row():
            randomize_toggle = gr.Checkbox(
                label="Randomize resolution each generation (standard preset list)",
                value=_state["randomize_enabled"],
                elem_id="txt2img_anima_randomize",
            )
            randomize_highres_toggle = gr.Checkbox(
                label="Randomize high-res resolution each generation (high-res preset list; both on = combined pool)",
                value=_state["randomize_highres_enabled"],
                elem_id="txt2img_anima_randomize_highres",
            )

    # Wire preset dropdowns. Both call _apply_preset, which looks up labels
    # in the combined _LABEL_TO_DIMS table.
    dropdown.change(        fn=_apply_preset, inputs=[dropdown],         outputs=[width_comp, height_comp])
    highres_dropdown.change(fn=_apply_preset, inputs=[highres_dropdown], outputs=[width_comp, height_comp])

    # Wire history: select -> fill prompt textbox
    if prompt_comp is not None:
        history_dropdown.change(
            fn=_on_history_selected,
            inputs=[history_dropdown],
            outputs=[prompt_comp, history_dropdown],
        )
    refresh_btn.click(
        fn=_refresh_history,
        inputs=[],
        outputs=[history_dropdown],
    )

    # Wire randomize checkboxes -> module state (no UI output)
    randomize_toggle.change(
        fn=_set_randomize_enabled,
        inputs=[randomize_toggle],
        outputs=[],
    )
    randomize_highres_toggle.change(
        fn=_set_randomize_highres_enabled,
        inputs=[randomize_highres_toggle],
        outputs=[],
    )


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _try_inject_picker(seed_row_component):
    global _picker_anchor
    if _picker_anchor is seed_row_component:
        return
    w = _refs["txt2img_width"]
    h = _refs["txt2img_height"]
    if w is None or h is None:
        print(f"[anima-resolution] seed_row reached but W/H refs missing "
              f"(W={w is not None}, H={h is not None}); skipping inject")
        return
    prompt = _refs.get("txt2img_prompt")  # may be None; history just won't auto-fill prompt
    try:
        _build_picker_ui(w, h, prompt)
        _picker_anchor = seed_row_component
        print("[anima-resolution] picker injected after txt2img_seed_row")
    except Exception as e:
        print(f"[anima-resolution] inject failed: {e}")
        traceback.print_exc()


def _try_wire_resize_by():
    # Source W/H from the displayed image's actual metadata, not the txt2img
    # sliders (which may not match when Randomize is on). Targets the img2img
    # "Resize to" Width and Height inputs (multiple-of-64 targets, no scale
    # rounding error).
    #
    # We use .click(...).then(...) instead of a plain .click(...). The reason:
    # the WebUI's own "Send to img2img" handler also writes to img2img_width
    # and img2img_height (it sets them to the source image's exact dims).
    # When two handlers on the same click both target the same outputs,
    # Gradio's update reconciliation is order-dependent and occasionally
    # drops one of the writes -- which is the intermittent failure mode
    # users see. .then() chains our handler so it fires *after* the previous
    # chain step has fully resolved on the frontend, so our W/H write lands
    # on a settled state and never gets dropped.
    info = _refs.get("generation_info_txt2img") or _refs.get("txt2img_generation_info")
    iw = _refs.get("img2img_width")
    ih = _refs.get("img2img_height")
    if info is None or iw is None or ih is None:
        return
    for btn_id in ("txt2img_send_to_img2img", "txt2img_send_to_inpaint"):
        btn = _refs[btn_id]
        if btn is None or btn in _resize_wired_buttons:
            continue
        try:
            # Step 1: a no-op click handler we control. We don't need it to
            # do anything; its only job is to give us a return-value handle
            # we can chain .then() off of, and to ensure our chain starts
            # after the click event is dispatched.
            sentinel = btn.click(fn=_noop_sentinel, inputs=[], outputs=[])
            # Step 2: the actual compute, deferred. .then() waits for the
            # sentinel (and, in practice, the rest of the click's parallel
            # handlers) to finish before firing.
            sentinel.then(
                fn=_compute_resize_scale_from_infotext,
                inputs=[info],
                outputs=[iw, ih],
                # generation_info_txt2img is JSON containing one infotext per
                # gallery item. Resolve the currently selected item in the
                # browser before sending the string to Python; parsing the
                # whole JSON blob otherwise always found the first Size field.
                _js="""(info) => {
                    try {
                        const data = JSON.parse(info);
                        const texts = Array.isArray(data.infotexts) ? data.infotexts : [];
                        const index = (typeof selected_gallery_index === 'function')
                            ? selected_gallery_index() : 0;
                        return texts[index] || texts[0] || info;
                    } catch (_) {
                        return info;
                    }
                }""",
            )
            _resize_wired_buttons.add(btn)
            print(f"[anima-resolution] resize-to auto-adjuster wired on {btn_id} "
                  f"(reads image PNG metadata, writes img2img_width / img2img_height "
                  f"via .then() to avoid update races)")
        except Exception as e:
            print(f"[anima-resolution] failed to wire {btn_id}: {e}")
            traceback.print_exc()


def _noop_sentinel():
    """No-op click handler. Exists only so we have a return-value handle
    on which to chain .then() — see _try_wire_resize_by docstring."""
    return None


def _try_wire_history_logger():
    """Hook the Generate button so every click prepends the current positive
    prompt to the session history and refreshes the dropdown choices."""
    gen_btn = _refs.get("txt2img_generate")
    prompt = _refs.get("txt2img_prompt")
    history_dropdown = _picker_components.get("history_dropdown")
    if any(c is None for c in (gen_btn, prompt, history_dropdown)):
        return
    if gen_btn in _history_wired_buttons:
        return
    try:
        gen_btn.click(
            fn=_log_prompt_on_generate,
            inputs=[prompt],
            outputs=[],
        )
        _history_wired_buttons.add(gen_btn)
        print("[anima-resolution] prompt-history logger wired on txt2img_generate "
              "(log-only; use Refresh to update dropdown)")
    except Exception as e:
        print(f"[anima-resolution] failed to wire history logger: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Component watcher
# ---------------------------------------------------------------------------

def _on_after_component(component, **kwargs):
    elem_id = kwargs.get("elem_id")
    if elem_id not in _TRACKED_IDS:
        return

    # Detect a new UI build (Reload UI). If the new component is a different
    # object than the one we already had for this elem_id, every other stored
    # ref must also belong to the dead Blocks() and is invalid.
    if _refs[elem_id] is not None and component is not _refs[elem_id]:
        global _picker_anchor
        for k in _refs:
            _refs[k] = None
        _picker_anchor = None
        _resize_wired_buttons.clear()
        _history_wired_buttons.clear()
        _picker_components["history_dropdown"] = None

    _refs[elem_id] = component

    _try_wire_resize_by()
    _try_wire_history_logger()

    if elem_id == "txt2img_seed_row":
        _try_inject_picker(component)
        # After the picker is built, the history dropdown exists, so the
        # logger wiring may now be completable.
        _try_wire_history_logger()


script_callbacks.on_after_component(_on_after_component)


# ---------------------------------------------------------------------------
# Script subclass: randomize txt2img resolution per Generate click.
# ---------------------------------------------------------------------------

class AnimaRandomizeResolutionScript(scripts.Script):
    """When either randomize flag in _state is True, every txt2img
    generation has its width/height overwritten via before_process with a
    random choice from the corresponding preset list: RESOLUTIONS for the
    standard toggle, HIGH_RES_RESOLUTIONS for the high-res toggle, or the
    combined pool when both are on. img2img is unaffected."""

    def title(self):
        return "Anima random resolution"

    def show(self, is_img2img):
        # AlwaysVisible means it runs every generation (and shows an empty
        # accordion section -- minor cosmetic cost for the always-run hook).
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args, **kwargs):
        try:
            # img2img processing objects carry init_images; txt2img doesn't.
            if hasattr(p, "init_images"):
                return
            std_on = _state.get("randomize_enabled", False)
            hr_on = _state.get("randomize_highres_enabled", False)
            if not (std_on or hr_on):
                return
            if std_on and hr_on:
                pool, pool_name = RESOLUTIONS + HIGH_RES_RESOLUTIONS, "combined"
            elif hr_on:
                pool, pool_name = HIGH_RES_RESOLUTIONS, "high-res"
            else:
                pool, pool_name = RESOLUTIONS, "standard"
            _, new_w, new_h = random.choice(pool)
            old_w, old_h = getattr(p, "width", None), getattr(p, "height", None)
            p.width = new_w
            p.height = new_h
            print(f"[anima-resolution] randomized ({pool_name} pool) "
                  f"{old_w}x{old_h} -> {new_w}x{new_h}")
        except Exception as e:
            print(f"[anima-resolution] randomize failed in before_process: {e}")
            traceback.print_exc()


print("[anima-resolution] callback registered")
