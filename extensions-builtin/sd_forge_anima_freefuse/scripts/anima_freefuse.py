"""Forge Neo GUI and lifecycle integration for Anima FreeFuse."""

from __future__ import annotations

import logging
import uuid

import gradio as gr

from backend.args import dynamic_args
from backend.logging import setup_logger
from lib_anima_freefuse import (
    AdapterSpec,
    AnimaFreeFuseState,
    apply_freefuse_patch,
    apply_freefuse_rejection,
    concept_token_positions,
)
from lib_anima_freefuse.runtime import install_patch_metadata_hook
from modules import scripts
from modules.anima_support import is_anima_engine
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

logger = logging.getLogger("AnimaFreeFuse")
setup_logger(logger)


def _parse_blocks(text: str, total: int) -> set[int]:
    result = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            result.update(range(max(0, int(left)), min(total - 1, int(right)) + 1))
        else:
            index = int(item)
            if 0 <= index < total:
                result.add(index)
    if not result:
        raise ValueError("Anima FreeFuse attention-bias block selection is empty")
    return result


class AnimaFreeFuseScript(scripts.Script):
    # After Spectrum, Regional, Artist Mixer and Guidance so conflicts can be
    # diagnosed from the actual pass configuration.
    sorting_priority = 2040

    def title(self):
        return "Anima FreeFuse (multi-LoRA)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown(
                "Two-pass, same-noise FreeFuse for **2–3 Anima subject LoRAs**. "
                "Enabling it selects `Anima FreeFuse Euler`; each trigger phrase must "
                "appear verbatim in the positive prompt."
            )
            adapter_controls = []
            defaults = [
                (True, "", ""),
                (True, "", ""),
                (False, "", ""),
            ]
            for index, (active_default, selector_default, concept_default) in enumerate(
                defaults, start=1
            ):
                with gr.Row():
                    active = gr.Checkbox(active_default, label=f"Subject {index}")
                    selector = gr.Textbox(
                        selector_default,
                        label=f"LoRA {index} name",
                        placeholder="Filename/stem as used by <lora:name:weight>",
                    )
                    concept = gr.Textbox(
                        concept_default,
                        label=f"Trigger phrase {index}",
                        placeholder="Must occur verbatim in the main prompt",
                    )
                adapter_controls.extend([active, selector, concept])

            with gr.Accordion("Phase 1: automatic mask collection", open=True):
                with gr.Row():
                    collect_step = gr.Slider(
                        0, 12, value=3, step=1, label="Collect step"
                    )
                    collect_block = gr.Slider(
                        0, 27, value=18, step=1, label="Collect block"
                    )
                    top_k = gr.Slider(
                        0.01,
                        0.50,
                        value=0.10,
                        step=0.01,
                        label="Core-token top-k ratio",
                    )
                    temperature = gr.Slider(
                        10, 2000, value=300, step=10, label="Self-concept temperature"
                    )
                with gr.Row():
                    background = gr.Textbox(
                        "",
                        label="Background phrase (optional)",
                        placeholder="Empty uses Anima's T5 EOS token",
                    )
                    bg_scale = gr.Slider(
                        0.0, 2.0, value=0.95, step=0.05, label="Background scale"
                    )
                    balance_iterations = gr.Slider(
                        1, 50, value=15, step=1, label="Balance iterations"
                    )
                    feather = gr.Slider(
                        0, 8, value=1, step=1, label="Mask feather radius"
                    )

            with gr.Accordion(
                "Phase 2: LoRA routing and attention isolation", open=True
            ):
                with gr.Row():
                    routing_strength = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.05, label="Spatial routing strength"
                    )
                    routing_end = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.05, label="Routing end progress"
                    )
                    bias_scale = gr.Slider(
                        0.0,
                        20.0,
                        value=6.0,
                        step=0.5,
                        label="Wrong-concept suppression",
                    )
                    positive_bias = gr.Slider(
                        0.0, 10.0, value=1.0, step=0.25, label="Own-concept boost"
                    )
                bias_blocks = gr.Textbox(
                    "0-27",
                    label="Attention-bias blocks",
                    info="LoRA output routing applies to all spatial LoRA layers; this selects cross-attention bias blocks.",
                )

        controls = [
            enable,
            *adapter_controls,
            collect_step,
            collect_block,
            top_k,
            temperature,
            background,
            bg_scale,
            balance_iterations,
            feather,
            routing_strength,
            routing_end,
            bias_scale,
            positive_bias,
            bias_blocks,
        ]
        keys = [
            "Anima FreeFuse enabled",
            *[
                f"Anima FreeFuse subject {index} {field}"
                for index in range(1, 4)
                for field in ("active", "LoRA", "concept")
            ],
            "Anima FreeFuse collect step",
            "Anima FreeFuse collect block",
            "Anima FreeFuse top-k",
            "Anima FreeFuse temperature",
            "Anima FreeFuse background",
            "Anima FreeFuse background scale",
            "Anima FreeFuse balance iterations",
            "Anima FreeFuse feather",
            "Anima FreeFuse routing strength",
            "Anima FreeFuse routing end",
            "Anima FreeFuse bias scale",
            "Anima FreeFuse positive bias",
            "Anima FreeFuse bias blocks",
        ]
        self.infotext_fields = [
            PasteField(control, key) for control, key in zip(controls, keys)
        ]
        self.paste_field_names = keys
        return controls

    def before_process(self, p, enable, *values, **kwargs):
        if not enable:
            return
        current_model = getattr(p, "sd_model", None)
        if current_model is not None and not is_anima_engine(current_model):
            logger.warning(
                "Anima FreeFuse was enabled for a non-Anima checkpoint and has been skipped."
            )
            return
        install_patch_metadata_hook()
        borrowed_online = (
            getattr(p, "_anima_lora_stage_scheduler_state", None) is not None
        )
        p._anima_freefuse_original_online = bool(dynamic_args.online_lora)
        p._anima_freefuse_borrowed_online = borrowed_online
        p._anima_freefuse_enabled = True
        dynamic_args.online_lora = True

        old_sampler = str(getattr(p, "sampler_name", ""))
        p.sampler_name = "Anima FreeFuse Euler"
        if getattr(p, "enable_hr", False):
            p.hr_sampler_name = "Anima FreeFuse Euler"
        p.extra_generation_params["Anima FreeFuse sampler"] = "Anima FreeFuse Euler"
        if old_sampler and old_sampler != p.sampler_name:
            p.extra_generation_params["Anima FreeFuse replaced sampler"] = old_sampler

        # Force the currently selected prompt LoRAs to reload as online patches.
        sd_model = getattr(p, "sd_model", None)
        if sd_model is not None:
            sd_model.current_lora_hash = f"anima-freefuse:{uuid.uuid4().hex}"

    def process_before_every_sampling(self, p, enable, *values, **kwargs):
        if not enable:
            return
        if not getattr(p, "_anima_freefuse_enabled", False):
            return
        try:
            self._configure_sampling(p, values)
        except Exception as error:
            logger.exception("Anima FreeFuse configuration failed: %s", error)
            unet = p.sd_model.forge_objects.unet
            p.sd_model.forge_objects.unet = apply_freefuse_rejection(unet, error)
            p.extra_generation_params["Anima FreeFuse"] = (
                f"error: {type(error).__name__}"
            )

    @staticmethod
    def _configure_sampling(p, values):
        if "Anima regional conditioning" in p.extra_generation_params:
            raise ValueError(
                "Anima FreeFuse and Anima Regional Conditioning cannot run in the same pass"
            )
        if "Anima Artist Mixer" in p.extra_generation_params:
            raise ValueError(
                "Anima FreeFuse and Anima Artist Mixer cannot run in the same pass"
            )
        if getattr(p, "refiner_checkpoint", None) not in (None, "", "None", "none"):
            raise ValueError(
                "Anima FreeFuse does not support switching checkpoints with Refiner during its two-pass sampler"
            )

        adapter_values = values[:9]
        (
            collect_step,
            collect_block,
            top_k,
            temperature,
            background,
            bg_scale,
            balance_iterations,
            feather,
            routing_strength,
            routing_end,
            bias_scale,
            positive_bias,
            bias_blocks,
        ) = values[9:]
        adapters = []
        for offset in range(0, 9, 3):
            active, selector, concept = adapter_values[offset : offset + 3]
            if not active:
                continue
            selector = str(selector).strip()
            concept = str(concept).strip()
            if not selector or not concept:
                raise ValueError(
                    "Every active FreeFuse subject needs both a LoRA name and trigger phrase"
                )
            adapters.append(
                AdapterSpec(f"subject_{len(adapters) + 1}", selector, concept)
            )
        if not 2 <= len(adapters) <= 3:
            raise ValueError("Anima FreeFuse requires two or three active subjects")

        prompts = list(getattr(p, "prompts", None) or [getattr(p, "prompt", "")])
        engine = p.sd_model.text_processing_engine_anima
        concepts = {adapter.name: adapter.concept for adapter in adapters}
        token_positions = None
        background_positions = None
        for prompt in prompts:
            chunks = engine.tokenize_line(str(prompt))
            if len(chunks) != 1:
                raise ValueError(
                    "Anima FreeFuse requires a single Anima conditioning chunk"
                )
            current_positions, current_background = concept_token_positions(
                engine.t5_tokenizer,
                str(prompt),
                concepts,
                str(background),
                prompt_ids=chunks[0].t5_tokens,
            )
            if token_positions is None:
                token_positions = current_positions
                background_positions = current_background
            elif (
                current_positions != token_positions
                or current_background != background_positions
            ):
                raise ValueError(
                    "Anima FreeFuse currently requires identical concept token positions across a batch"
                )

        unet = p.sd_model.forge_objects.unet
        total_blocks = len(unet.model.diffusion_model.blocks)
        block = min(max(0, int(collect_block)), total_blocks - 1)
        state = AnimaFreeFuseState(
            adapters=adapters,
            token_positions=token_positions,
            background_positions=background_positions,
            collect_step=int(collect_step),
            collect_block=block,
            top_k_ratio=float(top_k),
            temperature=float(temperature),
            bg_scale=float(bg_scale),
            balance_iterations=int(balance_iterations),
            feather=int(feather),
            routing_strength=float(routing_strength),
            routing_end=float(routing_end),
            bias_scale=float(bias_scale),
            positive_bias=float(positive_bias),
            bias_blocks=_parse_blocks(str(bias_blocks), total_blocks),
        )
        p.sd_model.forge_objects.unet = apply_freefuse_patch(unet, state)
        p.extra_generation_params["Anima FreeFuse"] = (
            f"{len(adapters)} subjects, two-pass same-noise"
        )
        p.extra_generation_params["Anima FreeFuse collect"] = (
            f"step {int(collect_step)}, block {block}"
        )
        p.extra_generation_params["Anima FreeFuse routing"] = (
            f"strength {float(routing_strength):.2f}, end {float(routing_end):.2f}"
        )
        p.extra_generation_params["Anima FreeFuse bias"] = (
            f"negative {float(bias_scale):.2f}, positive {float(positive_bias):.2f}, blocks {bias_blocks}"
        )

    def postprocess(self, p, processed, *args, **kwargs):
        if not getattr(p, "_anima_freefuse_enabled", False):
            return
        # The stage scheduler owns the online-LoRA lease when it was active at
        # entry; otherwise FreeFuse restores the user's original global mode.
        if not getattr(p, "_anima_freefuse_borrowed_online", False):
            dynamic_args.online_lora = bool(
                getattr(p, "_anima_freefuse_original_online", False)
            )
        p._anima_freefuse_enabled = False
