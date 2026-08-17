"""Small dependency-free helpers shared by built-in Anima LoRA controls."""

from __future__ import annotations

import re


ANIMA_28_TO_40_BLOCK_MAP = (
    0,
    1,
    1,
    2,
    3,
    3,
    4,
    5,
    5,
    6,
    7,
    7,
    8,
    9,
    9,
    10,
    11,
    11,
    12,
    13,
    14,
    14,
    15,
    16,
    16,
    17,
    18,
    18,
    19,
    20,
    20,
    21,
    22,
    22,
    23,
    24,
    24,
    25,
    26,
    27,
)
_ANIMA_LORA_BLOCK = re.compile(r"^lora_unet_blocks_(\d+)(?=_)")


def remap_anima_lora_blocks(lora: dict, *, target_blocks: int) -> bool:
    """Expand a complete legacy 28-block Anima LoRA for the 40-block model.

    Partial LoRAs are intentionally left untouched because their source model
    cannot be inferred safely from block names alone.
    """

    if target_blocks != len(ANIMA_28_TO_40_BLOCK_MAP):
        return False

    source_entries = []
    source_indices = set()
    for key, value in lora.items():
        match = _ANIMA_LORA_BLOCK.match(key)
        if match is None:
            continue
        source_index = int(match.group(1))
        source_indices.add(source_index)
        source_entries.append((key, source_index, value))

    if source_indices != set(range(28)):
        return False

    remapped = {}
    for target_index, source_index in enumerate(ANIMA_28_TO_40_BLOCK_MAP):
        for key, entry_index, value in source_entries:
            if entry_index != source_index:
                continue
            target_key = _ANIMA_LORA_BLOCK.sub(
                f"lora_unet_blocks_{target_index}", key, count=1
            )
            clone = getattr(value, "clone", None)
            remapped[target_key] = clone() if callable(clone) else value

    for key, _source_index, _value in source_entries:
        del lora[key]
    lora.update(remapped)
    return True


def active_template_parts(auto_timing: bool, parts):
    """Manual timing owns the single GUI window and must ignore preset parts."""

    return list(parts) if auto_timing else []


def current_sampling_position(params) -> tuple[int, int]:
    """Resolve Forge's current model-call step instead of its callback lag."""

    denoiser = getattr(params, "denoiser", None)
    callback_step = int(getattr(params, "sampling_step", 0))
    callback_steps = int(getattr(params, "total_sampling_steps", 1))
    denoiser_step = int(getattr(denoiser, "step", callback_step))
    denoiser_steps = int(getattr(denoiser, "total_steps", callback_steps))
    return max(callback_step, denoiser_step), max(
        1, callback_steps, denoiser_steps
    )


def merge_consistent_rules(target: dict, incoming: dict, *, label: str) -> None:
    """Merge alias rules, rejecting one adapter with conflicting batch rules."""

    for key, rule in incoming.items():
        existing = target.get(key)
        if existing is not None and existing != rule:
            raise ValueError(
                f"{label} {key!r} uses different inline weights in one batch; "
                "generate those prompts in separate batches"
            )
        target[key] = rule
