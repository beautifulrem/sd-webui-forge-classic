import torch

from modules import scripts, sd_models, shared
from remi import ui

LORA_REPLACEMENTS_HELP = """
Use this setting to load different LoRAs between the normal pass and the refiner pass
- Each line can contain one rule to process the LoRA names
- **Available Rules:**
    - `AAA=BBB` : the `AAA` within a LoRA would be replaced with `BBB` during the refiner pass
    - `-CCC` : any LoRA that contains `CCC` would be removed during the refiner pass
    - `DDD+EEE` : the LoRA named `DDD` would be added with strength `EEE` during the refiner pass
- **Example:**
    - The default value would replace any occurrence of "`high_noise`" with "`low_noise`"
"""

LORA_REPLACEMENTS_PLACEHOLDER = "\n".join(["high_noise=low_noise", "high-noise=low-noise", "-lora_to_remove", "lora_to_add+1.0"])


def checkpoint_tile(info) -> str:
    """The Checkpoint choice for ``info``, in the same form ``checkpoint_choices()`` lists it."""
    return info.short_title if shared.opts.sd_checkpoint_dropdown_use_short else info.name


def lookup_checkpoint(title):
    """Paste: map the infotext's ``Refiner: <short title>`` onto a Checkpoint choice."""
    info = sd_models.get_closet_checkpoint_match(title)
    return None if info is None else checkpoint_tile(info)


def checkpoint_choices() -> list[str]:
    return ["None"] + sorted(shared.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short))


class ScriptRefiner(scripts.ScriptBuiltinUI):
    section = "accordions"

    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.opts.show_refiner else None

    def params(self, is_img2img):
        if shared.opts.refiner_use_steps:
            switch_info = "(in steps) percentage of total steps"
        else:
            switch_info = "(in sigmas) " + ("Wan 2.2 I2V: 0.9" if is_img2img else "Wan 2.2 T2V: 0.875")
        architecture = "(use model of same architecture and quantization)" if shared.opts.refiner_fast_sd else "(use model of same architecture)"
        return ui.Section(
            "Refiner",
            enable=ui.Toggle("enable", False, label="Refiner", paste=lambda d: "Refiner" in d, preset="native.refiner"),
            children=[
                ui.Choice("checkpoint", checkpoint_choices, "None", same=ui.without_model_hash, label="Checkpoint", info=architecture, paste=lambda d: lookup_checkpoint(d.get("Refiner"))),
                ui.Row(
                    ui.Slider("switch_at", 0.0, 1.0, 0.875, 0.025, label="Switch at", info=switch_info, infotext="Refiner switch at"),
                    ui.Slider("cfg", 0.0, 24.0, 0.0, 0.5, label="Refiner CFG Scale", info="use original if < 1.0", infotext="Refiner CFG scale"),
                ),
                ui.Group(
                    "LoRA Replacements",
                    ui.Markdown(LORA_REPLACEMENTS_HELP),
                    ui.Text("replacements", lambda: shared.opts.refiner_lora_replacement, lines=4, max_lines=12, placeholder=LORA_REPLACEMENTS_PLACEHOLDER, label=""),
                    open=True,
                ),
            ],
        )

    def setup(self, p, enable_refiner: bool, refiner_checkpoint: str, refiner_switch_at: float, refiner_cfg: float, replacements: str):
        if not enable_refiner or refiner_checkpoint in (None, "", "None"):
            p.refiner_checkpoint = None
            p.refiner_switch_at = None
            return

        # actual implementation is in sd_samplers_common.py / apply_refiner()
        p.refiner_checkpoint = refiner_checkpoint
        p.refiner_switch_at = refiner_switch_at
        p.refiner_cfg = None if (refiner_cfg < 1.0) else refiner_cfg
        shared.opts.set("refiner_lora_replacement", replacements.strip())
        shared.opts.save(shared.config_filename)

    @torch.inference_mode()
    def postprocess(self, *args, **kwargs):
        from modules import sd_samplers_common, shared

        if sd_samplers_common.ORIGINAL_CHECKPOINT is None:
            return

        sd_model = shared.sd_model

        import huggingface_guess

        from backend.loader import preprocess_state_dict
        from backend.memory_management import LoadedModel, current_loaded_models
        from backend.patcher.unet import UnetPatcher
        from backend.state_dict import (
            convert_quantization,
            load_state_dict,
            try_filter_state_dict,
        )
        from backend.utils import load_torch_file
        from modules_forge.main_entry import logger

        idx = None

        for i, loaded_models in enumerate(current_loaded_models):
            if isinstance(loaded_models.model, UnetPatcher):
                idx = i
                break

        if idx is not None:
            mdl: LoadedModel = current_loaded_models.pop(idx)
            mdl.model_unload()
            del mdl

        model = sd_model.forge_objects.unet.model.diffusion_model

        sd, metadata = load_torch_file(sd_samplers_common.ORIGINAL_CHECKPOINT, return_metadata=True)
        sd, metadata = convert_quantization(sd, metadata)
        sd = preprocess_state_dict(sd)

        guess = huggingface_guess.guess(sd)

        sd = try_filter_state_dict(sd, guess.unet_key_prefix)

        logger.info("Restoring state_dict...")
        load_state_dict(model, sd, ignore_start="llm")

        if sd_samplers_common.ORIGINAL_CHECKPOINT.lower().endswith(".gguf"):

            from backend.memory_management import bake_gguf_model

            sd_model.forge_objects.unet.model.gguf_baked = False
            sd_model.forge_objects.unet.model = bake_gguf_model(sd_model.forge_objects.unet.model)

        sd_samplers_common.ORIGINAL_CHECKPOINT = None


shared.options_templates.update(
    shared.options_section(
        (None, "Refiner"),
        {
            "refiner_lora_replacement": shared.OptionInfo(default="high_noise=low_noise\nhigh-noise=low-noise"),
        },
    )
)
