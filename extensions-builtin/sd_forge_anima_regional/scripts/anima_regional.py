"""Forge Neo GUI integration for Anima regional prompt conditioning."""

from __future__ import annotations

import logging
from functools import partial
from types import SimpleNamespace

import gradio as gr

from backend.logging import setup_logger
from lib_anima_regional.regional import Region, RegionalState, apply_regional_patch
from modules import scripts
from modules.anima_feature_conflicts import FREEFUSE, REGIONAL, is_feature_active, register_exclusive_component, set_feature_active
from modules.anima_lora_support import model_block_set
from modules.anima_support import (
    conditioning_cache_key,
    encode_extra_prompts,
    is_anima_engine,
    process_ref,
    prompt_schedule,
    prompt_step,
    register_negpip_prompts,
    register_table_fields,
    schedule_steps,
    script_arguments,
)
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaRegional")
setup_logger(logger)

REGIONS = range(1, 4)
# One region's controls: (field, type written to infotext)
_REGION_FIELDS = (("active", bool), ("prompt", str), ("x", float), ("y", float), ("width", float), ("height", float), ("strength", float))

# One row per script argument, in the order ui() returns the components (the
# API's positional order): (name of the control, infotext key, preset name or
# None, type written to infotext). Region prompts are user content, which
# presets leave alone. process() and process_before_every_sampling() name the
# rows before the regions as parameters (Agent Scheduler stores a queued API
# task's script args by process()'s parameter names) and check the whole list
# with _arguments.
SCRIPT_ARGS = (
    ("enable", "Anima regional enabled", "regional.enabled", bool),
    ("blocks", "Anima regional blocks", "regional.blocks", str),
    ("start", "Anima regional start", "regional.start", float),
    ("end", "Anima regional end", "regional.end", float),
    ("feather", "Anima regional feather", "regional.feather", float),
    ("base_preserve", "Anima regional base preserve", "regional.base_preserve", float),
    *(
        (f"region{index}_{field}", f"Anima region {index} {field}", None if field == "prompt" else f"regional.region{index}_{field}", kind)
        for index in REGIONS
        for field, kind in _REGION_FIELDS
    ),
)
SCRIPT_ARG_NAMES = tuple(row[0] for row in SCRIPT_ARGS)
INFOTEXT_KEYS = tuple(row[1] for row in SCRIPT_ARGS)
_PROGRESS_INFO = (
    "A fraction of model time (sigma through the model's Shift), not of the sampling steps: "
    "the two agree only for evenly spaced schedules such as Simple."
)


def _arguments(values) -> SimpleNamespace:
    """The script args, ``enable`` first, one attribute per SCRIPT_ARGS name."""

    return script_arguments("Anima Regional Conditioning", SCRIPT_ARG_NAMES, values)


def generation_params(args: SimpleNamespace) -> dict:
    """Infotext for one regional pass from its script args ``args`` (see
    ``_arguments``); keys come from SCRIPT_ARGS. An inactive region records
    only its switch."""

    values = [True, args.blocks, args.start, args.end, args.feather, args.base_preserve]
    for active, *settings in _region_rows(args):
        values += [active, *(settings if active else [None] * len(settings))]
    return {key: kind(value) for (_, key, _, kind), value in zip(SCRIPT_ARGS, values, strict=True) if value is not None}


def _encode_cached(p, text: str):
    """:func:`_encode`, reused across the batches and passes of one generation."""

    cache = getattr(p, "_anima_regional_conds", None)
    if cache is None:
        cache = p._anima_regional_conds = {}
    key = (str(text), conditioning_cache_key(p))
    if key not in cache:
        cache[key] = _encode(p, text)
    return cache[key]


def _encode(p, text: str):
    """``(crossattn, negpip_mask or None)`` for one region prompt."""

    cond, mask = encode_extra_prompts(p, [text])
    if mask is not None:
        mask = mask.float().contiguous()
    return cond.float().contiguous(), mask


def _region(p, text: str, x, y, width, height, strength) -> Region:
    """A :class:`Region` for one region prompt. Forge's prompt editing in it
    (``[a:b:N]``, ``[a|b]``) resolves like the main prompt's: each distinct
    text of the pass's schedule is encoded once and a model call uses the
    entry of its step."""

    schedule = prompt_schedule([text], *schedule_steps(p))
    encoded = {texts[0]: _encode_cached(p, texts[0]) for _end, texts in schedule}
    entries = tuple(encoded[texts[0]] for _end, texts in schedule)
    (cond, negpip_mask), ends = entries[0], tuple(end for end, _texts in schedule)
    placement = (float(x), float(y), float(width), float(height), float(strength))
    if len(schedule) == 1:
        return Region(cond, *placement, negpip_mask)
    return Region(cond, *placement, negpip_mask, ends=ends, entries=entries)


def _region_rows(args: SimpleNamespace) -> list[tuple]:
    """Each region's values, in _REGION_FIELDS order."""

    return [tuple(getattr(args, f"region{index}_{field}") for field, _ in _REGION_FIELDS) for index in REGIONS]


def _region_is_used(active, text, width, height, strength) -> bool:
    return bool(active) and bool(str(text).strip()) and float(strength) > 0.0 and float(width) > 0.0 and float(height) > 0.0


class AnimaRegionalScript(scripts.Script):
    sorting_priority = 2027

    def title(self):
        return "Anima Regional Conditioning"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Routes independently encoded prompts into feathered rectangular regions at selected "
                "Anima cross-attention blocks. Regional, FreeFuse, and Artist Mixer are one-of-three; "
                "the active choice locks the others."
            )
            with gr.Row():
                blocks = gr.Textbox(
                    "0-27",
                    label="Anima blocks",
                    info="Written in the 2B (28-block) layout: on 40/52-block models each block also selects its inserted copies, so the default covers every block of any Anima model.",
                )
                start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Start progress", info=_PROGRESS_INFO)
                end = gr.Slider(0.0, 1.0, value=0.65, step=0.01, label="End progress", info=_PROGRESS_INFO)
                feather = gr.Slider(
                    0.0,
                    0.25,
                    value=0.03,
                    step=0.005,
                    label="Edge feather",
                    info="Fades a region's edges inside the image, centred on the edge, so regions that share an edge "
                    "meet without a gap; edges on the canvas border are not feathered.",
                )
                base_preserve = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Base preserve")
            region_controls = {}
            defaults = [
                (True, "", 0.0, 0.0, 0.5, 1.0, 1.0),
                (True, "", 0.5, 0.0, 0.5, 1.0, 1.0),
                (False, "", 0.25, 0.25, 0.5, 0.5, 1.0),
            ]
            for index, values in enumerate(defaults, start=1):
                with gr.Accordion(f"Region {index}", open=index <= 2):
                    with gr.Row():
                        active = gr.Checkbox(values[0], label="Active")
                        prompt = gr.Textbox(values[1], label="Regional prompt")
                        strength = gr.Slider(
                            0.0,
                            2.0,
                            value=values[6],
                            step=0.05,
                            label="Strength",
                            info="Blend opacity inside the rectangle (1 = fully regional, less Base preserve). "
                            "Above 1 is not stronger: it only brings the feathered edge to full opacity sooner "
                            "and outweighs weaker regions where rectangles overlap.",
                        )
                    with gr.Row():
                        x = gr.Slider(0.0, 1.0, value=values[2], step=0.01, label="X")
                        y = gr.Slider(0.0, 1.0, value=values[3], step=0.01, label="Y")
                        width = gr.Slider(0.0, 1.0, value=values[4], step=0.01, label="Width")
                        height = gr.Slider(0.0, 1.0, value=values[5], step=0.01, label="Height")
                    row = {"active": active, "prompt": prompt, "x": x, "y": y, "width": width, "height": height, "strength": strength}
                    region_controls.update({f"region{index}_{field}": component for field, component in row.items()})
        register_exclusive_component(
            tab=self.tabname,
            group="anima_spatial_conditioning",
            names=("freefuse", "regional", "artist"),
            name="regional",
            component=enable,
        )
        # Each control is the local variable or region control named in SCRIPT_ARGS.
        return register_table_fields(self, SCRIPT_ARGS, {**locals(), **region_controls})

    def process(self, p, enable, blocks, start, end, feather, base_preserve, *region_values, **kwargs):
        args = _arguments((enable, blocks, start, end, feather, base_preserve, *region_values))
        # Runs before NegPiP's process_batch: let a negative weight in a region
        # prompt that will actually be encoded enable NegPiP.
        prompts = []
        if args.enable and is_anima_engine(getattr(p, "sd_model", None)) and not is_feature_active(p, FREEFUSE):
            prompts = [
                text
                for active, text, x, y, width, height, strength in _region_rows(args)
                if _region_is_used(active, text, width, height, strength)
            ]
        register_negpip_prompts(p, REGIONAL, prompts)
        # Runs before every sampling hook: Artist Mixer sees this and yields,
        # and NegPiP then ignores the artist prompts it registered.
        set_feature_active(p, REGIONAL, bool(prompts))

    def postprocess(self, p, processed, *args, **kwargs):
        set_feature_active(p, REGIONAL, False)

    def process_before_every_sampling(self, p, enable, blocks, start, end, feather, base_preserve, *region_values, **kwargs):
        # Checked before any region is built or the patch installed.
        args = _arguments((enable, blocks, start, end, feather, base_preserve, *region_values))
        if not args.enable or not is_anima_engine(getattr(p, "sd_model", None)):
            return
        if is_feature_active(p, FREEFUSE):
            message = "disabled: Anima FreeFuse has priority in headless processing"
            p.extra_generation_params["Anima regional conditioning"] = message
            logger.warning("Anima Regional Conditioning %s", message)
            return
        unet = p.sd_model.forge_objects.unet
        total_blocks = len(unet.model.diffusion_model.blocks)
        selected_blocks = model_block_set(str(args.blocks), total_blocks, feature="Anima Regional Conditioning")
        regions = []
        for active, text, x, y, width, height, strength in _region_rows(args):
            if not _region_is_used(active, text, width, height, strength):
                continue
            regions.append(_region(p, str(text), x, y, width, height, strength))
        if not regions:
            logger.warning("Anima Regional Conditioning was enabled but no active non-empty region exists.")
            return
        predictor = unet.model.predictor
        state = RegionalState(
            regions=regions,
            blocks=selected_blocks,
            feather=float(args.feather),
            base_preserve=float(args.base_preserve),
            start_sigma=float(predictor.percent_to_sigma(float(args.start))),
            end_sigma=float(predictor.percent_to_sigma(float(args.end))),
            step_of=partial(prompt_step, process_ref(p)),
        )
        p.sd_model.forge_objects.unet = apply_regional_patch(unet, state)
        # Keys match ``self.infotext_fields`` so pasted parameters restore the panel.
        p.extra_generation_params.update(generation_params(args))
