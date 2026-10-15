"""Forge Neo GUI and lifecycle integration for Anima FreeFuse."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from backend.logging import setup_logger
from lib_anima_freefuse import (
    AdapterSpec,
    AnimaFreeFuseState,
    apply_freefuse_patch,
    apply_freefuse_rejection,
    concept_token_positions,
)
from lib_anima_freefuse.runtime import has_untagged_patches, install_patch_metadata_hook
from modules import scripts
from modules.anima_lora_support import REFERENCE_BLOCKS, anima_model_block, model_block_set
from modules.anima_online_lora import acquire_online_lora, release_online_lora
from modules.anima_feature_conflicts import (
    ARTIST_MIXER,
    FREEFUSE,
    SPATIAL_GROUP,
    is_feature_active,
    record_conflict_resolution,
    resolve_freefuse_refiner_conflicts,
    set_feature_active,
)
from modules.anima_support import effective_prompt_batch, is_anima_engine, prompt_schedule, schedule_steps, script_arguments, table_params
from modules.prompt_parser import get_multicond_prompt_list
from remi import ui

logger = logging.getLogger("AnimaFreeFuse")
setup_logger(logger)

# Name of FreeFuse's lease on Forge's online-LoRA switch
ONLINE_LORA_OWNER = "anima_freefuse"

SUBJECTS = range(1, 4)
# One subject's controls: (name suffix, infotext field, type written to infotext)
_SUBJECT_FIELDS = (("active", "active", bool), ("selector", "LoRA", str), ("concept", "concept", str))

# One row per script argument, in positional order (the API's): (param name,
# infotext key, preset field or None, type written to infotext). The subject LoRAs, trigger phrases and the
# background phrase are user content, which presets leave alone.
SCRIPT_ARGS = (
    ("enable", "Anima FreeFuse enabled", "freefuse.enabled", bool),
    *(
        (f"subject{index}_{suffix}", f"Anima FreeFuse subject {index} {field}", None, kind)
        for index in SUBJECTS
        for suffix, field, kind in _SUBJECT_FIELDS
    ),
    ("collect_step", "Anima FreeFuse collect step", "freefuse.collect_step", int),
    ("collect_block", "Anima FreeFuse collect block", "freefuse.collect_block", int),
    ("top_k", "Anima FreeFuse top-k", "freefuse.top_k", float),
    ("temperature", "Anima FreeFuse temperature", "freefuse.temperature", float),
    ("background", "Anima FreeFuse background", None, str),
    ("bg_scale", "Anima FreeFuse background scale", "freefuse.background_scale", float),
    ("balance_iterations", "Anima FreeFuse balance iterations", "freefuse.balance_iterations", int),
    ("feather", "Anima FreeFuse feather", "freefuse.feather", int),
    ("routing_strength", "Anima FreeFuse routing strength", "freefuse.routing_strength", float),
    ("routing_end", "Anima FreeFuse routing end", "freefuse.routing_end", float),
    ("bias_scale", "Anima FreeFuse bias scale", "freefuse.bias_scale", float),
    ("positive_bias", "Anima FreeFuse positive bias", "freefuse.positive_bias", float),
    ("bias_blocks", "Anima FreeFuse bias blocks", "freefuse.bias_blocks", str),
)
SCRIPT_ARG_NAMES = tuple(row[0] for row in SCRIPT_ARGS)
INFOTEXT_KEYS = tuple(row[1] for row in SCRIPT_ARGS)


def _arguments(values) -> SimpleNamespace:
    """The script args, ``enable`` first, one attribute per SCRIPT_ARGS name."""

    return script_arguments("Anima FreeFuse", SCRIPT_ARG_NAMES, values)


def _subject(args: SimpleNamespace, index: int) -> tuple:
    """``(active, selector, concept)`` of subject ``index``."""

    return tuple(getattr(args, f"subject{index}_{suffix}") for suffix, _, _ in _SUBJECT_FIELDS)


def generation_params(args: SimpleNamespace) -> dict:
    """Infotext restoring the panel from the script args ``args`` (see
    ``_arguments``); keys come from SCRIPT_ARGS. An inactive subject records
    only its switch, an empty background phrase nothing."""

    written = {name: kind(getattr(args, name)) for name, _, _, kind in SCRIPT_ARGS[1:]}
    written["enable"] = True
    for index in SUBJECTS:
        for name in (f"subject{index}_selector", f"subject{index}_concept"):
            if written[f"subject{index}_active"]:
                written[name] = written[name].strip()
            else:
                del written[name]
    if not written["background"].strip():
        del written["background"]
    return {key: written[name] for name, key, _, _ in SCRIPT_ARGS if name in written}


def _loaded_patches_untagged(sd_model) -> bool:
    """Whether the LoRA-patched unet a matching LoRA hash would reuse holds
    patches without FreeFuse's filename tag."""

    objects = getattr(sd_model, "forge_objects_after_applying_lora", None)
    if objects is None:
        objects = getattr(sd_model, "forge_objects", None)
    unet = getattr(objects, "unet", None)
    return unet is not None and has_untagged_patches(unet)


class AnimaFreeFuseScript(scripts.Script):
    # Panel position only (sorting_priority orders the UI, not the hooks):
    # below Spectrum, Regional and Guidance, whose options conflict with it.
    sorting_priority = 2040

    def title(self):
        return "Anima FreeFuse (multi-LoRA)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        # The Refiner lock is the core conflict rule of the "freefuse.enabled"
        # preset field (anima_feature_conflicts.resolve_conflict_locks): the
        # client runs it after either side changes and shows its message.
        subjects = [
            ui.Row(
                ui.Toggle(f"subject{index}_active", index <= 2, label=f"Subject {index}"),
                ui.Text(
                    f"subject{index}_selector",
                    "",
                    label=f"LoRA {index} name",
                    placeholder="Filename/stem as used by <lora:name:weight>",
                    enabled_if=ui.param(f"subject{index}_active"),
                ),
                ui.Text(
                    f"subject{index}_concept",
                    "",
                    label=f"Trigger phrase {index}",
                    placeholder="Must occur verbatim in the main prompt",
                    enabled_if=ui.param(f"subject{index}_active"),
                ),
            )
            for index in SUBJECTS
        ]
        collection = ui.Group(
            "Phase 1: automatic mask collection",
            ui.Row(
                ui.Slider("collect_step", 0, 12, 3, 1, label="Collect step"),
                ui.Slider("collect_block", 0, 27, 18, 1, label="Collect block", info="In the 28-block layout; mapped to the same block on 40/52-block Anima."),
                ui.Slider("top_k", 0.01, 0.50, 0.10, 0.01, label="Core-token top-k ratio"),
                ui.Slider("temperature", 10, 2000, 300, 10, label="Self-concept temperature"),
            ),
            ui.Row(
                ui.Text("background", "", label="Background phrase (optional)", placeholder="Empty uses Anima's T5 EOS token"),
                ui.Slider("bg_scale", 0.0, 2.0, 0.95, 0.05, label="Background scale"),
                ui.Slider("balance_iterations", 1, 50, 15, 1, label="Balance iterations"),
                ui.Slider("feather", 0, 8, 1, 1, label="Mask feather radius"),
            ),
            open=True,
        )
        routing = ui.Group(
            "Phase 2: LoRA routing and attention isolation",
            ui.Row(
                ui.Slider("routing_strength", 0.0, 1.0, 1.0, 0.05, label="Spatial routing strength"),
                ui.Slider("routing_end", 0.0, 1.0, 1.0, 0.05, label="Routing end progress"),
                ui.Slider("bias_scale", 0.0, 20.0, 6.0, 0.5, label="Wrong-concept suppression"),
                ui.Slider("positive_bias", 0.0, 10.0, 1.0, 0.25, label="Own-concept boost"),
            ),
            ui.Text(
                "bias_blocks",
                "0-27",
                label="Attention-bias blocks",
                info="Written in the 2B (28-block) layout: on 40/52-block models each block also selects its inserted copies, so the default covers every block of any Anima model. LoRA output routing applies to all spatial LoRA layers.",
            ),
            open=True,
        )
        root = ui.Section(
            self.title(),
            enable=ui.Toggle("enable", False, label=self.title(), exclusive=(SPATIAL_GROUP, "freefuse")),
            children=[
                ui.Markdown(
                    "Two-pass, same-noise FreeFuse for **2–3 Anima subject LoRAs**. "
                    "Enabling it selects `Anima FreeFuse Euler`; each trigger phrase must "
                    "appear verbatim in the positive prompt. FreeFuse, Regional Conditioning, "
                    "and Artist Mixer are one-of-three; the active choice locks the others."
                ),
                *subjects,
                collection,
                routing,
            ],
        )
        self.param_order = table_params(SCRIPT_ARGS, root)
        return root

    def process(self, p, enable, *values, **kwargs):
        # A wrong argument count is refused before the sampler or the online-LoRA lease changes.
        args = _arguments((enable, *values))
        if not args.enable:
            return
        current_model = getattr(p, "sd_model", None)
        if current_model is not None and not is_anima_engine(current_model):
            logger.warning(
                "Anima FreeFuse was enabled for a non-Anima checkpoint and has been skipped."
            )
            return
        refiner_state = resolve_freefuse_refiner_conflicts(
            freefuse=True,
            refiner=getattr(p, "refiner_checkpoint", None)
            not in (None, "", "None", "none"),
            checkpoint=getattr(p, "refiner_checkpoint", None),
        )
        if refiner_state.messages:
            resolution = "; ".join(refiner_state.messages)
            record_conflict_resolution(p, *refiner_state.messages)
            logger.warning("Anima compatibility resolver: %s", resolution)
        p.refiner_checkpoint = None
        p.refiner_checkpoint_info = None
        p.refiner_switch_at = None
        hook_installed = install_patch_metadata_hook()
        set_feature_active(p, FREEFUSE)
        # Every subject LoRA must load as online patches
        acquire_online_lora(p, ONLINE_LORA_OWNER)

        old_sampler = str(getattr(p, "sampler_name", ""))
        p.sampler_name = "Anima FreeFuse Euler"
        if getattr(p, "enable_hr", False):
            p.hr_sampler_name = "Anima FreeFuse Euler"
        p.extra_generation_params["Anima FreeFuse sampler"] = "Anima FreeFuse Euler"
        if old_sampler and old_sampler != p.sampler_name:
            p.extra_generation_params["Anima FreeFuse replaced sampler"] = old_sampler

        # The online lease is part of the LoRA hash, so offline patches reload
        # by themselves. Force a reload only when the loaded patches predate
        # the filename tags FreeFuse routes by.
        sd_model = getattr(p, "sd_model", None)
        if sd_model is not None and (hook_installed or _loaded_patches_untagged(sd_model)):
            sd_model.current_lora_hash = "anima-freefuse:reload"

    def process_before_every_sampling(self, p, enable, *values, **kwargs):
        # A wrong argument count raises here, not as the sampler's configuration error.
        args = _arguments((enable, *values))
        if not args.enable or not is_feature_active(p, FREEFUSE):
            return
        try:
            self._configure_sampling(p, args)
        except Exception as error:
            logger.exception("Anima FreeFuse configuration failed: %s", error)
            unet = p.sd_model.forge_objects.unet
            p.sd_model.forge_objects.unet = apply_freefuse_rejection(unet, error)
            p.extra_generation_params["Anima FreeFuse"] = (
                f"error: {type(error).__name__}"
            )

    @staticmethod
    def _configure_sampling(p, args: SimpleNamespace):
        # Regional runs after FreeFuse (script load order) and yields to it;
        # Artist Mixer runs first, yields too, and is checked here.
        if is_feature_active(p, ARTIST_MIXER):
            raise ValueError(
                "Anima FreeFuse and Anima Artist Mixer cannot run in the same pass"
            )

        adapters = []
        for index in SUBJECTS:
            active, selector, concept = _subject(args, index)
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

        prompts, _negative_prompts = effective_prompt_batch(p)
        # Forge splits AND prompts into sub-prompt conds: the concept positions
        # and the one-mask-per-image collection only fit a single prompt.
        subprompt_indexes, _flat, _index = get_multicond_prompt_list(
            [str(prompt) for prompt in prompts]
        )
        if any(len(indexes) > 1 for indexes in subprompt_indexes):
            raise ValueError("Anima FreeFuse does not support AND prompts")
        engine = p.sd_model.text_processing_engine_anima
        concepts = {adapter.name: adapter.concept for adapter in adapters}
        token_positions = None
        background_positions = None
        # The model sees each prompt's text of the current step: its prompt
        # editing ([a:b:N], [a|b]) resolved like Forge's main conditioning.
        # One set of positions serves every step, so every text must agree.
        steps = schedule_steps(p)
        for prompt in prompts:
            schedule = prompt_schedule([prompt], *steps)
            prompt_positions = None
            for text in dict.fromkeys(texts[0] for _end, texts in schedule):
                chunks = engine.tokenize_line(text)
                if len(chunks) != 1:
                    raise ValueError(
                        "Anima FreeFuse requires a single Anima conditioning chunk"
                    )
                positions = concept_token_positions(
                    engine.t5_tokenizer,
                    text,
                    concepts,
                    str(args.background),
                    prompt_ids=chunks[0].t5_tokens,
                )
                if prompt_positions is None:
                    prompt_positions = positions
                elif positions != prompt_positions:
                    raise ValueError(
                        "Anima FreeFuse requires prompt editing ([a:b:N], [a|b]) to keep the concept "
                        "and background token positions of every step"
                    )
            current_positions, current_background = prompt_positions
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
        # The slider uses the 28-block (2B) layout; on the depth-expanded
        # 2.9B/3.8B models the same block sits deeper.
        block = anima_model_block(int(args.collect_block), REFERENCE_BLOCKS, total_blocks)
        block = min(max(0, block), total_blocks - 1)
        state = AnimaFreeFuseState(
            adapters=adapters,
            token_positions=token_positions,
            background_positions=background_positions,
            collect_step=int(args.collect_step),
            collect_block=block,
            top_k_ratio=float(args.top_k),
            temperature=float(args.temperature),
            bg_scale=float(args.bg_scale),
            balance_iterations=int(args.balance_iterations),
            feather=int(args.feather),
            routing_strength=float(args.routing_strength),
            routing_end=float(args.routing_end),
            bias_scale=float(args.bias_scale),
            positive_bias=float(args.positive_bias),
            # Written in the 2B layout: on 2.9B/3.8B a band brings the inserted copies along
            bias_blocks=model_block_set(args.bias_blocks, total_blocks, feature="Anima FreeFuse attention-bias"),
        )
        p.sd_model.forge_objects.unet = apply_freefuse_patch(unet, state)
        p.extra_generation_params["Anima FreeFuse"] = (
            f"{len(adapters)} subjects, two-pass same-noise"
        )
        p.extra_generation_params["Anima FreeFuse collect"] = (
            f"step {int(args.collect_step)}, block {block}"
        )
        p.extra_generation_params["Anima FreeFuse routing"] = (
            f"strength {float(args.routing_strength):.2f}, end {float(args.routing_end):.2f}"
        )
        p.extra_generation_params["Anima FreeFuse bias"] = (
            f"negative {float(args.bias_scale):.2f}, positive {float(args.positive_bias):.2f}, blocks {args.bias_blocks}"
        )
        # Keys are the params' infotext keys, so pasted parameters restore the panel.
        p.extra_generation_params.update(generation_params(args))

    @staticmethod
    def _cleanup_runtime(p):
        # The user's online-LoRA setting comes back once no other script
        # (the stage scheduler) holds a lease on it either.
        release_online_lora(p, ONLINE_LORA_OWNER)
        if not is_feature_active(p, FREEFUSE):
            return
        set_feature_active(p, FREEFUSE, False)

    # postprocess restores the online-LoRA mode before later scripts' postprocess
    # (which may start nested generations); cleanup covers runs that raise.
    def postprocess(self, p, processed, *args, **kwargs):
        self._cleanup_runtime(p)

    def cleanup(self, p, *args, **kwargs):
        self._cleanup_runtime(p)
