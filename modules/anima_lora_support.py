"""Small dependency-free helpers shared by built-in Anima LoRA controls."""

from __future__ import annotations

import re


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
# Kept identical to sd_forge_lora.networks.process_anima (upstream code, which
# re-maps the LoRA weights themselves); tests/test_anima_lora_support.py checks it.
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


def anima_model_block(index: int, reference_blocks: int, model_blocks: int | None) -> int:
    """Model block holding ``index`` of a ``reference_blocks``-deep layout.

    The original block, not an inserted copy of it, is the first model
    block mapped to it.
    """

    mapping = ANIMA_BLOCK_MAPPINGS.get((reference_blocks, model_blocks))
    if mapping is None or index not in mapping:
        return index
    return mapping.index(index)


def stage_window(stage: str, shift: float) -> tuple[float, float, float]:
    """``(start, end, peak)`` sampling-progress window of a template stage
    ("composition", "character" or "style") at sampler shift ``shift``.

    Shift 3.0 is Forge's common default; higher shift keeps useful structure
    later in the denoise path, so the windows move slightly later."""

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    move = clamp((clamp(float(shift), 1.0, 24.0) - 3.0) / 21.0, -0.10, 1.0) * 0.12
    if stage == "composition":
        start = 0.0
        end = clamp(0.32 + move, 0.22, 0.46)
        return start, end, clamp(0.08 + move * 0.25, start, end)
    if stage == "style":
        start = clamp(0.62 + move, 0.48, 0.82)
        end = 1.0
        return start, end, clamp(0.86 + move * 0.35, start, end)
    start = clamp(0.20 + move * 0.65, 0.12, 0.36)
    end = clamp(0.74 + move * 0.45, 0.58, 0.88)
    return start, end, clamp((start + end) * 0.5, start, end)


def lora_file_names(filename) -> frozenset[str]:
    """Lower-case names a LoRA file can be referred to by in prompts and
    target lists: its path, file name and stem, plus its network names and
    aliases in sd_forge_lora's index (refreshed in place, so not cached)."""

    import os
    import sys
    from pathlib import Path

    available = getattr(sys.modules.get("networks"), "available_networks", None) or {}
    target = os.path.abspath(str(filename))
    path = Path(str(filename))
    names = {str(path).lower(), path.name.lower(), path.stem.lower()}
    for name, entry in available.items():
        try:
            if os.path.abspath(str(entry.filename)) == target:
                names.update((str(name).lower(), str(getattr(entry, "alias", "")).lower(), str(entry.get_alias()).lower()))
        except Exception:
            continue
    return frozenset(name for name in names if name)


# region LoRA key layout
# Shared by the layer-weight and stage-scheduler extensions: where a LoRA key
# sits in the Anima DiT (block, module) or in the Qwen3 text encoder / adapter.

ANIMA_BLOCK_RE = re.compile(r"(?:^|\.)(?:diffusion_model\.)?(?:net\.)?blocks\.(\d+)\.")
QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.model\.layers\.(\d+)\.")
GENERIC_QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.(?:model\.)?layers\.(\d+)\.")
QWEN_ADAPTER_BLOCK_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.llm_adapter\.blocks\.(\d+)\.")
LEGACY_QWEN_ADAPTER_BLOCK_RE = re.compile(r"(?:^|\.)(?:diffusion_model\.)?llm_adapter\.blocks\.(\d+)\.")


def compact_text(text) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def is_plain_number(text) -> bool:
    return re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", str(text or "")) is not None


def module_key(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "self": "self_attn",
        "selfattn": "self_attn",
        "self_attention": "self_attn",
        "cross": "cross_attn",
        "crossattn": "cross_attn",
        "cross_attention": "cross_attn",
        "ada": "adaln",
        "adaln_modulation": "adaln",
        "modulation": "adaln",
        "norms": "norm",
        "layernorm": "norm",
        "feedforward": "mlp",
        "ffn": "mlp",
    }
    return aliases.get(key, key)


def block_index_from_key(key: str) -> int | None:
    key_text = str(key or "")
    match = ANIMA_BLOCK_RE.search(key_text)
    if match:
        return int(match.group(1))

    normalized = key_text.lower().replace(".", "_")
    match = re.search(r"(?:^|_)blocks_(\d+)(?:_|$)", normalized)
    if not match:
        return None
    return int(match.group(1))


def module_name_from_block_tail(module: str) -> str | None:
    if module.startswith("adaln_modulation"):
        return "adaln"
    if module.startswith("norm"):
        return "norm"
    if module.startswith("self_attn"):
        return "self_attn"
    if module.startswith("cross_attn"):
        return "cross_attn"
    if module.startswith("mlp"):
        return "mlp"
    return None


def module_from_key(key: str) -> str | None:
    key_text = str(key or "")
    marker = re.search(r"blocks\.\d+\.([^.]+)", key_text)
    if marker:
        module = module_name_from_block_tail(marker.group(1))
        return module or module_key(marker.group(1))

    normalized = key_text.lower().replace(".", "_")
    marker = re.search(r"(?:^|_)blocks_\d+_(.+)", normalized)
    if not marker:
        return None

    return module_name_from_block_tail(marker.group(1))


def module_factor(key: str, module_weights: dict[str, float]) -> float:
    module = module_from_key(key)
    if not module:
        return 1.0
    return module_weights.get(module, 1.0)


def model_key(key) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and key:
        return str(key[0])
    return str(key or "")


def qwen_layer_index(key: str) -> int | None:
    key_text = str(key or "")
    match = QWEN_LAYER_RE.search(key_text) or GENERIC_QWEN_LAYER_RE.search(key_text) or QWEN_ADAPTER_BLOCK_RE.search(key_text) or LEGACY_QWEN_ADAPTER_BLOCK_RE.search(key_text)
    if match:
        return int(match.group(1))

    normalized = key_text.lower().replace(".", "_")
    match = (
        re.search(r"qwen3_06b_(?:model_)?layers_(\d+)", normalized)
        or re.search(r"qwen3_06b_llm_adapter_blocks_(\d+)", normalized)
        or re.search(r"(?:^|_)llm_adapter_blocks_(\d+)", normalized)
    )
    if not match:
        return None
    return int(match.group(1))
