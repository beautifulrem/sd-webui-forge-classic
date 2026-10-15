"""Model selection: the checkpoint, VAE / text encoder modules and
"Diffusion in Low Bits" choice, and the loading parameters they give
``sd_models.forge_model_reload``."""

from __future__ import annotations

import logging
import os.path

import torch
from rich import print_json

from backend.logging import setup_logger
from backend.nn.anima_quant import ANIMA_QUANT_INT8_CONVROT, ANIMA_QUANT_SCALED_FP8
from modules import (
    anima_online_lora,
    paths,
    processing,
    sd_models,
    shared,
    shared_items,
)

logger = logging.getLogger("ui_models")
setup_logger(logger)

forge_unet_storage_dtype_options: dict[str, tuple[torch.dtype, bool]] = {
    "Automatic": (None, False),
    "Automatic (fp16 LoRA)": (None, True),
    "float8-e4m3fn": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (fp16 LoRA)": (torch.float8_e4m3fn, True),
    "float8-e5m2": (torch.float8_e5m2, False),
    "float8-e5m2 (fp16 LoRA)": (torch.float8_e5m2, True),
    "float8-e4m3fn (Anima scaled)": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (Anima scaled, fp16 LoRA)": (torch.float8_e4m3fn, True),
    "int8 (Anima ConvRot)": (None, False),
    "int8 (Anima ConvRot, fp16 LoRA)": (None, True),
}
# Options that quantize an Anima DiT while loading (backend/nn/anima_quant.py);
# other models load with the option's plain storage dtype
anima_quant_options: dict[str, str] = {
    "float8-e4m3fn (Anima scaled)": ANIMA_QUANT_SCALED_FP8,
    "float8-e4m3fn (Anima scaled, fp16 LoRA)": ANIMA_QUANT_SCALED_FP8,
    "int8 (Anima ConvRot)": ANIMA_QUANT_INT8_CONVROT,
    "int8 (Anima ConvRot, fp16 LoRA)": ANIMA_QUANT_INT8_CONVROT,
}


module_list: dict[str, os.PathLike] = {}
module_map: dict[str, str] = {}


def find_files_with_extensions(base_path: os.PathLike, extensions: list[str]) -> dict[str, os.PathLike]:
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models() -> tuple[list[os.PathLike], list[os.PathLike]]:
    shared_items.refresh_checkpoints()
    shared_items.refresh_vae_list()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    module_list.clear()
    module_map.clear()

    module_paths: set[os.PathLike] = {
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
        *shared.cmd_opts.vae_dirs,
        *shared.cmd_opts.text_encoder_dirs,
    }

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

        for file, path in vae_files.items():
            rel = os.path.normpath(os.path.relpath(path, vae_path))
            module_map[file] = rel

    if shared.opts.sd_modules_dropdown_use_short:
        return sorted(ckpt_list), sorted(module_list.keys())
    else:
        return sorted(ckpt_list), sorted(module_map.values())


def forge_loading_parameters(checkpoint_info, unet_storage_option: str) -> dict:
    """What sd_models.forge_model_reload() loads (and compares to decide on a
    reload) for the "Diffusion in Low Bits" choice ``unet_storage_option``."""
    unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(unet_storage_option, (None, False))
    parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)
    if anima_quant := anima_quant_options.get(unet_storage_option):
        parameters["anima_quant"] = anima_quant
    return parameters


def refresh_model_loading_parameters(*, refresh: bool = True):
    if not refresh:
        return

    from modules.sd_models import model_data, select_checkpoint

    checkpoint_info = select_checkpoint()
    if checkpoint_info is None:
        logger.critical('You do not have any model... Please download models to "models/Stable-diffusion"')
        return

    unet_storage_option = shared.opts.forge_unet_storage_dtype
    parameters = model_data.forge_loading_parameters = forge_loading_parameters(checkpoint_info, unet_storage_option)
    lora_fp16 = forge_unet_storage_dtype_options.get(unet_storage_option, (None, False))[1]

    ckpt: str = checkpoint_info.filename
    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]
    dtype = str(parameters["unet_storage_dtype"] or [torch.float16, torch.bfloat16])

    selected = dict(checkpoint=os.path.basename(ckpt), modules=modules, dtype=dtype)
    if anima_quant := parameters.get("anima_quant"):
        selected["anima_quant"] = anima_quant  # Anima only; other models load with dtype
    logger.info("Model Selected:")
    print_json(data=selected)

    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        logger.warning("GGUF requires fp16 LoRA ; overriding option")
        lora_fp16 = True

    anima_online_lora.apply_user_setting(lora_fp16)  # held online-LoRA leases keep it on mid-run
    logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
    if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
        logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(getattr(shared.opts, "sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """`module_values` accepts file paths or just the module names; returns `True` if modules changed"""
    modules = []
    for v in module_values:
        module_name = os.path.basename(v)  # If the input is a filepath, extract the filename
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules.sort()

    # skip further processing if value unchanged
    if modules == getattr(shared.opts, "forge_additional_modules", []):
        return False

    shared.opts.set("forge_additional_modules", modules)
    if preset is not None:
        shared.opts.set(f"forge_additional_modules_{preset}", modules)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def dtype_change(dtype: str, preset: str, save=True, refresh=True) -> bool:
    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True
