import json
import os
import re
from enum import Enum

import torch
import tqdm
from huggingface_guess.detection import unet_prefix_from_state_dict
from safetensors.torch import save_file

from backend.state_dict import state_dict_prefix_replace
from backend.utils import load_torch_file
from modules import sd_models, shared


class InterpolationMethod(Enum):
    no_interpolation = "No Interpolation"
    weighted_sum = "Weighted Sum"
    add_difference = "Add Difference"

    @staticmethod
    def titles() -> list[str]:
        return [m.value for m in InterpolationMethod]

    @staticmethod
    def desc(value: str) -> str:
        match value:
            case InterpolationMethod.no_interpolation.value:
                return "Require 1 Model ; Mainly for format conversion"
            case InterpolationMethod.weighted_sum.value:
                return "Require 2 Model ; Result is calculated as A * (1 - M) + B * M"
            case InterpolationMethod.add_difference.value:
                return "Require 3 Model ; Result is calculated as A + (B - C) * M"


def read_metadata(primary_model_name: str, secondary_model_name: str, tertiary_model_name: str) -> dict[str, dict]:
    """The safetensors metadata of the chosen checkpoints, by checkpoint
    title ("Preview Metadata from Models" of the Checkpoint Merger)."""

    metadata = {}
    for checkpoint_name in (primary_model_name, secondary_model_name, tertiary_model_name):
        if checkpoint_name and (checkpoint_info := sd_models.checkpoint_aliases.get(checkpoint_name)) is not None:
            metadata[checkpoint_name] = checkpoint_info.metadata
    return metadata


def merge_output_path(filename: str) -> str:
    """Where a merged (or saved) checkpoint named ``filename`` goes: inside
    the checkpoint folder (``Stable-diffusion``), with a ``.safetensors``
    suffix. Raises ``ValueError`` for a name that leaves the folder."""

    name = (filename or "").strip()
    if not name:
        raise ValueError("Failed: Missing Filename")
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    root = os.path.realpath(sd_models.model_path)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.isabs(name) or os.path.commonpath([root, path]) != root or path == root:
        raise ValueError("Failed: The filename must stay inside the checkpoint folder")
    return path


def run_modelmerger(id_task, primary_model_name: str, secondary_model_name: str, tertiary_model_name: str, interp_method: str, multiplier: float, custom_name: str, discard_weights: str, save_metadata: bool, config_source: list[str], add_merge_recipe: bool, *, progress=None):
    """Merge up to three checkpoints and save the result; returns the status
    message (also on a missing model). The merge logic of the A1111
    "Checkpoint Merger" (the WebUI's Checkpoint Merger page,
    ``remi/core/merger.py``).

    ``progress(text, fraction)`` (optional) is called at each stage and
    while merging; an exception it raises (e.g. a cancelled page task)
    stops the merge. Other failures (shape or key mismatch, a bad regex)
    raise; ``shared.state`` is ended either way."""

    shared.state.begin(job="model-merge")
    try:
        return _run_modelmerger(primary_model_name, secondary_model_name, tertiary_model_name, interp_method, multiplier, custom_name, discard_weights, save_metadata, config_source, add_merge_recipe, progress)
    finally:
        shared.state.end()


def _run_modelmerger(primary_model_name: str, secondary_model_name: str, tertiary_model_name: str, interp_method: str, multiplier: float, custom_name: str, discard_weights: str, save_metadata: bool, config_source: list[str], add_merge_recipe: bool, progress) -> str:
    def stage(text: str, fraction: float | None = None) -> None:
        shared.state.textinfo = text
        if progress is not None:
            progress(text, fraction)

    def fail(message: str) -> str:
        shared.state.textinfo = message
        return message

    if not primary_model_name:
        return fail("Failed: Missing Primary Model")

    if interp_method != InterpolationMethod.no_interpolation.value and not secondary_model_name:
        return fail("Failed: Missing Secondary Model")

    if interp_method == InterpolationMethod.add_difference.value and not tertiary_model_name:
        return fail("Failed: Missing Tertiary Model")

    if interp_method not in InterpolationMethod.titles():
        return fail(f"Failed: Unknown Interpolation Method {interp_method!r}")

    for name in (primary_model_name, secondary_model_name if interp_method != InterpolationMethod.no_interpolation.value else None, tertiary_model_name if interp_method == InterpolationMethod.add_difference.value else None):
        if name and name not in sd_models.checkpoint_aliases:
            return fail(f"Failed: Unknown Model {name}")

    discard = re.compile(discard_weights) if discard_weights else None  # a bad pattern fails before any loading

    def weighted_sum(theta0: torch.Tensor, theta1: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.lerp(theta0, theta1, alpha)

    def get_difference(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        return torch.subtract(theta1, theta2)

    def add_difference(theta0: torch.Tensor, theta1_2_diff: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.add(theta0, theta1_2_diff, alpha=alpha)

    def filename_weighted_sum() -> str:
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    def filename_add_difference() -> str:
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        m = round(multiplier, 2)

        return f"{a} + {m}({b} - {c})"

    def filename_nothing() -> str:
        return primary_model_info.model_name

    THETA_FUNCS = {
        InterpolationMethod.no_interpolation.value: (filename_nothing, None, None),
        InterpolationMethod.weighted_sum.value: (filename_weighted_sum, None, weighted_sum),
        InterpolationMethod.add_difference.value: (filename_add_difference, get_difference, add_difference),
    }

    filename_generator, theta_func1, theta_func2 = THETA_FUNCS[interp_method]
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    primary_model_info = sd_models.checkpoint_aliases[primary_model_name]
    secondary_model_info = sd_models.checkpoint_aliases[secondary_model_name] if theta_func2 else None
    tertiary_model_info = sd_models.checkpoint_aliases[tertiary_model_name] if theta_func1 else None
    output_modelname = merge_output_path(custom_name or filename_generator())  # a bad name fails before any loading

    def _load_state_dict(filename: str) -> dict[str, torch.Tensor]:
        sd = load_torch_file(filename)
        prefix = unet_prefix_from_state_dict(sd)
        return state_dict_prefix_replace(sd, {prefix: "model.diffusion_model."})

    # steps: load B, (load C, merge B - C), load A, (merge A, B), save
    steps = 2 + (2 if theta_func1 else 0) + (2 if theta_func2 else 0)
    done = 0

    def loop_progress(text: str, index: int, total: int) -> None:
        if progress is not None and (index % 64 == 0 or index == total):
            progress(text, (done + index / max(total, 1)) / steps)

    if theta_func2:
        stage("Loading B", done / steps)
        print(f"Loading {secondary_model_info.filename}...")
        theta_1 = _load_state_dict(secondary_model_info.filename)
        done += 1
    else:
        theta_1 = None

    if theta_func1:
        _keys = list(theta_1.keys())

        stage("Loading C", done / steps)
        print(f"Loading {tertiary_model_info.filename}...")
        theta_2 = _load_state_dict(tertiary_model_info.filename)
        done += 1

        stage("Merging B and C", done / steps)
        shared.state.sampling_steps = total = len(_keys)
        missing = 0

        for index, key in enumerate(tqdm.tqdm(_keys), 1):
            shared.state.sampling_step += 1
            loop_progress("Merging B and C", index, total)

            if key in theta_2:
                a = theta_1.pop(key)
                b = theta_2.pop(key)

                if a.shape != b.shape:
                    raise ValueError(f"Shape Mismatch ({tuple(a.shape)} != {tuple(b.shape)})")

                theta_1[key] = theta_func1(a.float(), b.float()).to(a)
            else:
                w = theta_1.pop(key)
                theta_1[key] = torch.zeros_like(w)
                missing += 1

        del theta_2
        shared.state.nextjob()
        done += 1

        if missing > total * 0.25:
            raise SystemError("Keys Mismatch between B & C...")

    stage("Loading A", done / steps)
    print(f"Loading {primary_model_info.filename}...")
    theta_0 = _load_state_dict(primary_model_info.filename)
    done += 1

    if theta_func2:
        _keys = list(theta_0.keys())

        stage("Merging A and B", done / steps)
        shared.state.sampling_steps = total = len(_keys)
        missing = 0

        for index, key in enumerate(tqdm.tqdm(_keys), 1):
            shared.state.sampling_step += 1
            loop_progress("Merging A and B", index, total)

            if key not in theta_1:
                missing += 1
                continue

            a = theta_0.pop(key)
            b = theta_1.pop(key)

            if a.shape != b.shape:
                raise ValueError(f"Shape Mismatch ({tuple(a.shape)} != {tuple(b.shape)})")

            theta_0[key] = theta_func2(a.float(), b.float(), multiplier).to(a)

        del theta_1
        shared.state.nextjob()
        done += 1

        if missing > total * 0.25:
            raise SystemError("Keys Mismatch between A & B...")

    if discard is not None:
        for key in list(theta_0.keys()):
            if re.search(discard, key):
                theta_0.pop(key)

    os.makedirs(os.path.dirname(output_modelname), exist_ok=True)

    stage("Saving", done / steps)
    print(f"Saving to {output_modelname}...")

    metadata = {}

    if save_metadata:
        if "A" in config_source and primary_model_info is not None:
            metadata.update(primary_model_info.metadata)
        if "B" in config_source and secondary_model_info is not None:
            metadata.update(secondary_model_info.metadata)
        if "C" in config_source and tertiary_model_info is not None:
            metadata.update(tertiary_model_info.metadata)

        if add_merge_recipe:
            merge_recipe = {
                "type": "Neo",
                "interp_method": interp_method,
                "multiplier": multiplier,
                "discard_weights": discard_weights,
                "config_source": config_source,
            }

            sd_merge_models = {}

            def add_model_metadata(key: str, checkpoint_info: sd_models.CheckpointInfo):
                checkpoint_info.calculate_shorthash()
                merge_recipe[key] = checkpoint_info.sha256

                sd_merge_models[checkpoint_info.sha256] = {
                    "name": checkpoint_info.name,
                    "legacy_hash": checkpoint_info.hash,
                }

                if (r := checkpoint_info.metadata.get("sd_merge_recipe", None)) is not None:
                    sd_merge_models["sd_merge_recipe"] = r
                if (m := checkpoint_info.metadata.get("sd_merge_models", None)) is not None:
                    sd_merge_models["sd_merge_models"] = m

            if primary_model_info:
                add_model_metadata("primary_model_hash", primary_model_info)
            if secondary_model_info:
                add_model_metadata("secondary_model_hash", secondary_model_info)
            if tertiary_model_info:
                add_model_metadata("tertiary_model_hash", tertiary_model_info)

            metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
            metadata["sd_merge_models"] = json.dumps(sd_merge_models)

    def sanitize_metadata(meta_dict: dict | None) -> dict | None:
        if not meta_dict:
            return None

        sanitized = {}
        for key, value in meta_dict.items():
            if value is None:
                continue
            elif isinstance(value, str):
                sanitized[key] = value
            elif isinstance(value, (dict, list)):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)

        return sanitized

    save_file(theta_0, output_modelname, metadata=sanitize_metadata(metadata))
    print(f"Checkpoint saved to {output_modelname}")

    shared.state.textinfo = "Checkpoint Saved"

    sd_models.list_models()
    return f"Checkpoint saved to {output_modelname}"
