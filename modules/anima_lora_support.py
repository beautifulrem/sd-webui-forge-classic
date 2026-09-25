"""Small dependency-free helpers shared by built-in Anima LoRA controls."""

from __future__ import annotations


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


# Block layouts of the depth-expanded Anima models (2B = 28 blocks, 2.9B = 40,
# 3.8B = 52): each entry is the source block a target block was copied from.
# Kept identical to sd_forge_lora.networks.process_anima, which re-maps the
# LoRA weights themselves.
ANIMA_BLOCK_MAPPINGS = {
    (28, 40): (0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27),
    (28, 52): (0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24, 24, 24, 25, 26, 27),
    (40, 52): (0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 23, 24, 25, 26, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37, 38, 39),
}


def anima_lora_block_count(trained_blocks) -> int | None:
    """The Anima depth a LoRA was trained for, from its block indices."""

    if not trained_blocks:
        return None
    needed = max(trained_blocks) + 1
    return next((size for size in (28, 40, 52) if needed <= size), None)


def anima_source_block(index: int, lora_blocks: int | None, model_blocks: int | None) -> int:
    """The LoRA's own block index behind model block ``index``.

    A 2B LoRA on a depth-expanded model is copied onto the inserted blocks,
    so per-block weights written for the LoRA's layout must follow it.
    """

    mapping = ANIMA_BLOCK_MAPPINGS.get((lora_blocks, model_blocks))
    if mapping is None or not 0 <= index < len(mapping):
        return index
    return mapping[index]
