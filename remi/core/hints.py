"""Tooltips of the core generation params: the static hint map of the
Gradio UI's ``javascript/hints.js`` (all 26 of its texts), now declared on the params
themselves (``hint=`` on inputs, ``hints=`` on choices; schema-format.md
§1.2). The texts are English catalog keys (``localizations/zh_CN.json``)."""

from __future__ import annotations

SAMPLER = "The algorithm used to refine each step of the image"
SCHEDULER = "The algorithm used to adjust the magnitude of refinement"
STEPS = "The number of times the image is iteratively refined"

BATCH_COUNT = "How many batches of images to generate (in sequence)"
BATCH_SIZE = "How many images to generate in a single batch (in parallel)"
FRAMES = "How many frames to generate for the video"

CFG_SCALE = "The strength used to calculate conditionings"
RESCALE_CFG = "Reduce the high-contrast burnt-color effects (mainly for v-pred checkpoints)"
MAHIRO = "An alternative algorithm used for CFG calculation"

SEED = 'Given the same prompts and parameters, you "should" generate the same image if the Seed is also the same'

RESIZE_MODES = {
    "Just Resize": "Resize the input image directly to the target resolution",
    "Crop and Resize": "Resize the image while maintaining the aspect ratio: Crop the excessive parts",
    "Resize and Fill": "Resize the image while maintaining the aspect ratio: Fill the empty parts with neighboring colors",
    "Latent Upscale": 'Same as "Just Resize" but resize the input latent instead of the input image',
    "Preserve Aspect Ratio": "Resize the image while maintaining the aspect ratio; Scale the short-side using the same ratio as the long-side",
}

MASK_BLUR = "How much feathering to apply to the mask (in pixels)"
MASKED_CONTENT = {
    "fill": "Fill the masked areas with neighboring colors",
    "original": "Keep whatever was within the masked areas",
    "latent noise": "Fill the masked areas with noise (requires high Denoising strength)",
    "latent nothing": "Fill the masked areas with zero values (requires high Denoising strength)",
}

DENOISING_STRENGTH = "How strong should the image be changed"

HIRES_FIX = "Automatically perform an additional pass of img2img"
HIRES_STEPS = "Sampling Steps for the img2img pass; use original if 0"
UPSCALE_BY = "Multiply the txt2img dimension by this ratio, to serve as the target dimension"
RESIZE_WIDTH_TO = 'Resize image to this width; use "Upscale by" if 0'
RESIZE_HEIGHT_TO = 'Resize image to this height; use "Upscale by" if 0'


def all_hints() -> list[str]:
    """Every text above (those of hints.js)."""

    texts: list[str] = []
    for name, value in globals().items():
        if name.isupper():
            texts += list(value.values()) if isinstance(value, dict) else [value]
    return texts
