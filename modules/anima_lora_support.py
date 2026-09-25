"""Small dependency-free helpers shared by built-in Anima LoRA controls."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Implementations behind the Anima LoRA scripts' ModelPatcher.add_patches and
# extra_networks.activate wrappers, by name. "Reload UI" re-executes the
# script modules but keeps the wrappers already installed (install_patch finds
# them in the chain), so a wrapper calls what its script module registered
# here last: the live module, with its live per-run state.
script_hooks: dict = {}

# <lora:name> or <lora:name:args>
LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([^>]*))?>")


def active_template_parts(auto_timing: bool, parts):
    """Manual timing owns the single GUI window and must ignore preset parts."""

    return list(parts) if auto_timing else []


def manual_window_bans(parts) -> bool:
    """Whether manual timing's single window blocks rather than applies: it
    keeps the action of a template that only bans stages."""

    return bool(parts) and all(part.get("mode") == "ban" for part in parts)


def rewrite_batch_prompts(p, rewrite, rewrite_hires) -> None:
    """Rewrite the prompts of ``p``'s current batch: ``p.prompts`` and the
    batch's slice of ``p.all_prompts`` with ``rewrite``, and its slice of
    ``p.all_hr_prompts`` with ``rewrite_hires`` (None leaves them).

    p.hr_prompts is not rewritten: parse_extra_network_prompts re-slices it
    from all_hr_prompts after the script hooks. Every rewrite runs before any
    is stored, so one that raises leaves all prompts as they were."""

    start = getattr(p, "iteration", 0) * getattr(p, "batch_size", 1)
    batch = slice(start, start + getattr(p, "batch_size", 1))
    replacements = []
    if getattr(p, "prompts", None):
        replacements.append(("prompts", None, rewrite(list(p.prompts))))
    if getattr(p, "all_prompts", None):
        replacements.append(("all_prompts", batch, rewrite(p.all_prompts[batch])))
    if rewrite_hires is not None and getattr(p, "all_hr_prompts", None):
        replacements.append(("all_hr_prompts", batch, rewrite_hires(p.all_hr_prompts[batch])))
    for attribute, selection, values in replacements:
        if selection is None:
            setattr(p, attribute, values)
        else:
            getattr(p, attribute)[selection] = values


def invalidate_cond_caches(p) -> None:
    """Drop the conditioning processing caches across generations (hires
    ones included). Their key holds the prompts and extra network data but
    not the text-encoder LoRA weights a panel sets, so a change there must
    re-encode the prompts. The lists are shared with the class: cleared in place."""

    for name in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
        cache = getattr(p, name, None)
        if isinstance(cache, list) and cache:
            cache[0] = None


def sigma_progress(sigma, sigmas) -> float | None:
    """Sampling progress 0..1 of the model call at ``sigma``: the position of
    the nearest sigma in the full schedule ``sigmas``. Hires. fix and img2img
    run only its tail, so they start part-way. None when it cannot be told."""

    try:
        current = float(sigma.flatten()[0] if hasattr(sigma, "flatten") else sigma[0] if isinstance(sigma, (list, tuple)) else sigma)
        values = [float(value) for value in (sigmas.flatten().tolist() if hasattr(sigmas, "flatten") else sigmas)]
    except Exception:
        return None
    if len(values) <= 1:
        return None
    index = min(range(len(values)), key=lambda i: abs(values[i] - current))
    return index / (len(values) - 1)


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


def window_curve_factor(progress: float, start: float, end: float, peak: float, shape: str, floor: float = 0.0) -> float:
    """Strength factor (0..1) at sampling ``progress`` of the window
    [start, end] (either order), 0 outside it; a zero-width window is full
    strength at its point. ``shape``: "hold" is 1 throughout, "front" falls
    from 1 to ``floor``, "back" rises from ``floor`` to 1, "triangle" rises
    linearly to ``peak`` and falls, anything else is the triangle eased.

    The stage scheduler's front / back curves fade out completely (floor 0);
    the artist mixer's keep a quarter of the artist (floor 0.25)."""

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    start, end = sorted((clamp(start, 0.0, 1.0), clamp(end, 0.0, 1.0)))
    if progress < start or progress > end:
        return 0.0
    if end - start <= 1e-6:
        return 1.0
    local = (progress - start) / (end - start)
    if shape == "hold":
        return 1.0
    if shape == "front":
        return 1.0 - (1.0 - floor) * local
    if shape == "back":
        return floor + (1.0 - floor) * local
    peak_local = (clamp(peak, start, end) - start) / (end - start)
    if local <= peak_local:
        raw = local / max(peak_local, 1e-6)
    else:
        raw = (1.0 - local) / max(1.0 - peak_local, 1e-6)
    raw = clamp(raw, 0.0, 1.0)
    return raw if shape == "triangle" else raw * raw * (3.0 - 2.0 * raw)


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


def name_keys(name) -> set[str]:
    """A LoRA name as written, lower-cased, plus its file name and stem."""

    from pathlib import Path

    clean = str(name or "").strip().lower()
    if not clean:
        return set()
    path = Path(clean)
    return {key for key in (clean, path.name.lower(), path.stem.lower()) if key}


def parse_targets(text) -> set[str]:
    """``name_keys`` of every name in a comma / semicolon / line separated list."""

    targets = set()
    for part in re.split(r"[,;\n]+", str(text or "").replace("，", ",").replace("；", ";")):
        targets |= name_keys(part)
    return targets


def resolve_network(name):
    """The sd_forge_lora index entry ``<lora:name:...>`` loads, looked up the
    way networks.load_networks does (forbidden aliases by network name, others
    by alias, then a rescan by file stem); None when there is none."""

    import sys

    networks = sys.modules.get("networks")
    try:
        if networks is None:
            import networks  # type: ignore
        key = str(name or "").strip()
        if not key:
            return None

        def lookup():
            if key.lower() in networks.forbidden_network_aliases:
                return networks.available_networks.get(key)
            return networks.available_network_aliases.get(key)

        entry = lookup()
        if entry is None:
            networks.update_available_networks_by_names([key])
            entry = lookup()
        return entry
    except Exception:
        return None


# region LoRA key layout
# Shared by the layer-weight and stage-scheduler extensions: where a LoRA key
# sits in the Anima DiT (block, module) or in the Qwen3 text encoder / adapter.

ANIMA_BLOCK_RE = re.compile(r"(?:^|\.)(?:diffusion_model\.)?(?:net\.)?blocks\.(\d+)\.")
GENERIC_QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:text_encoders\.)?qwen3_06b\.(?:model\.)?layers\.(\d+)\.")
# Also matches text_encoders.qwen3_06b.llm_adapter.blocks.N
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
    match = GENERIC_QWEN_LAYER_RE.search(key_text) or LEGACY_QWEN_ADAPTER_BLOCK_RE.search(key_text)
    if match:
        return int(match.group(1))

    normalized = key_text.lower().replace(".", "_")
    match = re.search(r"qwen3_06b_(?:model_)?layers_(\d+)", normalized) or re.search(r"(?:^|_)llm_adapter_blocks_(\d+)", normalized)
    if not match:
        return None
    return int(match.group(1))


# region Layer weight syntax
# "0-9=0.6,10-18=1.0,all=0.8", or 28 bare numbers in block order; the
# optional "block_lrs=" prefix of some trainers' configs is accepted.


@dataclass
class LayerWeights:
    """Per-index weights (DiT blocks or text-encoder layers) with a default."""

    default: float = 1.0
    values: dict[int, float] = field(default_factory=dict)

    def factor(self, index: int) -> float:
        return self.values.get(index, self.default)


def _normalized_weight_text(text) -> str:
    source = str(text or "").strip().replace("，", ",").replace("；", ";").replace("：", "=")
    return re.sub(r"\bblock_lrs?\s*=", "", source, flags=re.IGNORECASE)


def looks_like_layer_weights(text) -> bool:
    value = _normalized_weight_text(text)
    if not value or is_plain_number(value):
        return False
    numeric_count = 0
    for token in [t for t in re.split(r"[\s,;]+", value) if t]:
        if "=" in token:
            left = token.split("=", 1)[0].strip().lower()
            if left in {"*", "all", "default"} or left.isdigit() or re.fullmatch(r"\d+\s*-\s*\d+", left):
                return True
            continue
        if is_plain_number(token):
            numeric_count += 1
    return numeric_count >= 2


# Past the deepest Anima DiT (52 blocks) and Qwen3 text encoder (28 layers);
# bounds what a prompt's "0-2000000000=1" can allocate.
MAX_LAYER_INDEX = 255


def _finite_weight(text) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _layer_index(text: str) -> int:
    # Longer digit strings than any index could have are not converted at all
    return int(text) if len(text) <= 3 else MAX_LAYER_INDEX + 1


def parse_layer_weights(text) -> LayerWeights:
    values: dict[int, float] = {}
    default = 1.0
    bare_values = []
    for token in [t for t in re.split(r"[\s,;]+", _normalized_weight_text(text)) if t]:
        if "=" not in token:
            value = _finite_weight(token)
            if value is None:
                logger.warning("Ignored invalid Anima layer weight token: %s", token)
            else:
                bare_values.append(value)
            continue
        left, right = token.split("=", 1)
        left = left.strip().lower()
        value = _finite_weight(right)
        if value is None:
            logger.warning("Ignored invalid Anima layer weight: %s", token)
            continue
        if left in {"*", "all", "default"}:
            default = value
        elif range_match := re.match(r"^(\d+)\s*-\s*(\d+)$", left):
            start, end = sorted((_layer_index(range_match.group(1)), _layer_index(range_match.group(2))))
            if start > MAX_LAYER_INDEX:
                logger.warning("Ignored Anima layer range past index %d: %s", MAX_LAYER_INDEX, left)
                continue
            values.update(dict.fromkeys(range(start, min(end, MAX_LAYER_INDEX) + 1), value))
        elif left.isdecimal() and _layer_index(left) <= MAX_LAYER_INDEX:
            values[int(left)] = value
        else:
            logger.warning("Ignored invalid Anima layer selector: %s", left)
    if bare_values and not values:
        values.update(enumerate(bare_values))
    return LayerWeights(default=default, values=values)


def split_weight_sections(text) -> dict[str, str]:
    """``blocks`` / ``modules`` / ``qwen`` sections of a weight text such as
    "blocks=0-9=0.8,10-27=1;modules=mlp=0.5;qwen=0-3=0". Parts without a
    section name (and the "block_lrs=" form) are block weights."""

    sections: dict[str, str] = {}
    bare_parts = []
    for part in [p.strip() for p in re.split(r"[;；]+", str(text or "")) if p.strip()]:
        if "=" in part or "：" in part:
            # The first separator ends the section name: "modules：mlp=0.5"
            left, right = re.split(r"[=：]", part, maxsplit=1)
            key = left.strip().lower()
            if key in {"blocks", "block", "layers", "layer"}:
                sections["blocks"] = right.strip()
                continue
            if key in {"modules", "module", "mods", "mod"}:
                sections["modules"] = right.strip()
                continue
            if key in {"qwen", "text", "text_encoder", "text_layers", "te"}:
                sections["qwen"] = right.strip()
                continue
        bare_parts.append(part)
    if bare_parts:
        sections["blocks"] = ";".join([sections["blocks"], *bare_parts] if "blocks" in sections else bare_parts)
    return sections


def parse_module_weights(text) -> dict[str, float]:
    """Explicitly written ``module=weight`` entries (module names normalised)."""

    weights = {}
    source = str(text or "").replace("，", ",").replace("；", ";").replace("：", "=")
    for token in [t for t in re.split(r"[\n,;]+", source) if t.strip()]:
        if "=" not in token:
            continue
        left, right = token.split("=", 1)
        try:
            value = float(right)
        except ValueError:
            value = 1.0
        if not math.isfinite(value):
            logger.warning("Ignored non-finite Anima module weight: %s", token.strip())
            continue
        weights[module_key(left)] = value
    return weights
