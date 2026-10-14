"""Shared logic of the built-in Anima LoRA controls (Layer Weight, Stage
Scheduler) and the Anima features that select DiT blocks: LoRA key layout and
per-layer weight syntax, block mappings of the depth-expanded models, stage
timing windows, prompt / patch / sampling-progress helpers, and the seams to
Forge's LoRA index and LoRA Control. Their Gradio UI helpers are in
``modules.anima_lora_ui``."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from modules.anima_support import block_index_token, clamp, fold_weight_text, is_plain_number, parse_block_list, plain_number, sigma_value, to_float

logger = logging.getLogger(__name__)

# Implementations behind the Anima LoRA scripts' ModelPatcher.add_patches and
# extra_networks.activate wrappers, by name. "Reload UI" re-executes the
# script modules but keeps the wrappers already installed (the install
# helpers below find them in the chain), so a wrapper calls what its script
# module registered here last: the live module, with its live per-run state.
script_hooks: dict = {}


def _install_hook(owner, attribute: str, key: str, implementation, make_wrapper) -> bool:
    """Register ``implementation`` as ``script_hooks[key]`` and, unless a
    wrapper of ``key`` is already in its chain (installed before a "Reload
    UI", or under a later wrapper of another extension), wrap
    ``owner.<attribute>`` with ``make_wrapper(original)``. True when it wrapped."""
    from backend.patcher.function_chain import function_chain_contains

    script_hooks[key] = implementation
    marker = "_anima_hook_" + re.sub(r"\W", "_", key)
    original = getattr(owner, attribute)
    if function_chain_contains(original, marker):
        return False
    wrapper = make_wrapper(original)
    wrapper.__qualname__ = wrapper.__name__ = f"{attribute}_with_{marker[len('_anima_hook_'):]}"
    setattr(wrapper, marker, True)
    wrapper.__wrapped__ = original
    setattr(owner, attribute, wrapper)
    return True


def install_add_patches_hook(key: str, implementation) -> bool:
    """Route ``ModelPatcher.add_patches`` through ``script_hooks[key]``, set to
    ``implementation(original, patcher, patches, strength_patch,
    strength_model, filename, online_mode)``; wraps once per process (see
    ``_install_hook``). True when it wrapped."""
    from backend.patcher.base import ModelPatcher

    def make_wrapper(original):
        def wrapper(self, patches: list[dict], strength_patch: float = 1.0, strength_model: float = 1.0, *, filename: str = None, online_mode: bool = None):
            return script_hooks[key](original, self, patches, strength_patch, strength_model, filename, online_mode)

        return wrapper

    return _install_hook(ModelPatcher, "add_patches", key, implementation, make_wrapper)


def install_activate_hook(key: str, implementation) -> bool:
    """Route ``extra_networks.activate`` through ``script_hooks[key]``, set to
    ``implementation(original, p, extra_network_data)``; wraps once per
    process (see ``_install_hook``). True when it wrapped."""
    from modules import extra_networks

    def make_wrapper(original):
        def wrapper(p, extra_network_data):
            return script_hooks[key](original, p, extra_network_data)

        return wrapper

    return _install_hook(extra_networks, "activate", key, implementation, make_wrapper)


# Forge's LoRA Control (sd_forge_lora_schedule/scripts/control.py): its
# ``LoRAControl.mapping`` (``{lora filename: [(weight, time), ...]}``) holds
# the files it schedules in this generation, filled from
# ``<lora:name:[w@t,...]>``. The Anima LoRA controls leave those files to it.
_LORA_CONTROL_SCRIPT = ("sd_forge_lora_schedule", "scripts", "control.py")


def lora_control_files() -> set:
    """Files Forge's LoRA Control schedules in this generation.

    Read from the script classes the WebUI loaded, at call time: "Reload UI"
    re-executes control.py into a new class (and leaves it out when the
    extension is disabled), and scripts are not kept in ``sys.modules``."""

    import sys
    from pathlib import PurePath

    webui_scripts = sys.modules.get("modules.scripts")
    for data in getattr(webui_scripts, "scripts_data", None) or ():
        try:
            if PurePath(data.path).parts[-3:] == _LORA_CONTROL_SCRIPT and data.script_class.__name__ == "LoRAControl":
                return set(getattr(data.script_class, "mapping", None) or ())
        except Exception:
            continue
    return set()


def note_lora_control_yield(noted: set | None, extra_generation_params, filename, *, feature: str, infotext_key: str) -> None:
    """Note once per run that ``feature`` leaves ``filename`` to LoRA Control:
    a warning, and the file's name in the infotext's ``infotext_key`` list.
    ``noted``: the files of the run already noted (None: nothing to note)."""
    if noted is None or filename in noted:
        return
    noted.add(filename)
    name = os.path.splitext(os.path.basename(str(filename)))[0]
    logger.warning("%s skips %s: it is scheduled with LoRA Control syntax", feature, name)
    if isinstance(extra_generation_params, dict):
        skipped = [value for value in str(extra_generation_params.get(infotext_key, "")).split(", ") if value]
        extra_generation_params[infotext_key] = ", ".join(sorted({*skipped, name}))


# <lora:name> or <lora:name:args>. Neither part holds a "<": a tag ends
# before the next one starts, so an unterminated "<lora:" is given up at the
# next "<" instead of rescanning the rest of the prompt (quadratic on
# "<lora:" * 4000).
LORA_TAG_RE = re.compile(r"<lora:([^:<>]+)(?::([^<>]*))?>")


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
    is stored, so one that raises leaves all prompts as they were.

    The prompts as written are noted on ``p``: restore_batch_prompts and
    restore_run_prompts put them back, so the infotext records the tags the
    user wrote rather than the ones the batch loaded."""

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
            _note_rewrite(p, attribute, start, getattr(p, attribute)[selection], values)
            getattr(p, attribute)[selection] = values


# ``p``'s record of the prompts rewrite_batch_prompts rewrote:
# {"owner": id(p), "all_prompts" / "all_hr_prompts": {index: (as written, as rewritten)}}
_PROMPT_RECORD = "_anima_rewritten_prompts"


def _prompt_record(p, create=False) -> dict | None:
    record = getattr(p, _PROMPT_RECORD, None)
    # A shallow copy of p (a nested run) carries the outer run's record
    if isinstance(record, dict) and record.get("owner") == id(p):
        return record
    if not create:
        return None
    record = {"owner": id(p)}
    setattr(p, _PROMPT_RECORD, record)
    return record


def _note_rewrite(p, attribute: str, start: int, before, after) -> None:
    entries = _prompt_record(p, create=True).setdefault(attribute, {})
    for index, (old, new) in enumerate(zip(before, after), start):
        entry = entries.get(index)
        # Rewritten again (the other adapter kind, another Anima LoRA script):
        # it keeps the text as the user wrote it
        written = entry[0] if entry is not None and entry[1] == old else old
        entries[index] = (written, new)


def _restore_prompts(values, entries: dict, offset: int = 0, indices=None) -> None:
    """Put back ``values[index - offset]`` as written for each recorded index
    (in ``indices``), unless another script has changed it since."""
    if not isinstance(values, list):
        return
    for index, (written, rewritten) in entries.items():
        position = index - offset
        if (indices is None or index in indices) and 0 <= position < len(values) and values[position] == rewritten:
            values[position] = written


def restore_batch_prompts(p) -> None:
    """Put the current batch's rewritten prompts back as written in
    ``p.prompts`` and ``p.all_hr_prompts``, which its infotexts (and "Hires
    prompt") are built from. For postprocess_batch_list: processing re-slices
    p.prompts from all_prompts after postprocess_batch.

    ``p.all_prompts`` stays rewritten until restore_run_prompts: a nested run
    (ADetailer, in postprocess_image) takes its prompt from there, and must
    load the LoRAs as the batch did."""
    record = _prompt_record(p)
    if record is None:
        return
    start = getattr(p, "iteration", 0) * getattr(p, "batch_size", 1)
    _restore_prompts(getattr(p, "prompts", None), record.get("all_prompts", {}), offset=start)
    batch = range(start, start + getattr(p, "batch_size", 1))
    _restore_prompts(getattr(p, "all_hr_prompts", None), record.get("all_hr_prompts", {}), indices=batch)


def restore_run_prompts(p) -> None:
    """Put every rewritten prompt of the run back as written in
    ``p.all_prompts`` and ``p.all_hr_prompts``, then forget them. For
    postprocess: the run's Processed holds the same lists."""
    record = _prompt_record(p)
    if record is None:
        return
    _restore_prompts(getattr(p, "all_prompts", None), record.get("all_prompts", {}))
    _restore_prompts(getattr(p, "all_hr_prompts", None), record.get("all_hr_prompts", {}))
    setattr(p, _PROMPT_RECORD, None)


def invalidate_cond_caches(p) -> None:
    """Drop the conditioning processing caches across generations (hires
    ones included). Their key holds the prompts and extra network data but
    not the text-encoder LoRA weights a panel sets, so a change there must
    re-encode the prompts. The lists are shared with the class: cleared in place."""

    for name in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
        cache = getattr(p, name, None)
        if isinstance(cache, list) and cache:
            cache[0] = None


# (schedule tensor, its values): the schedule of the pass being sampled
_SCHEDULE_VALUES: tuple | None = None


def _schedule_values(sigmas) -> list[float]:
    """``sigmas`` as floats; a tensor schedule is copied to the host once per
    pass, not on every model call."""
    global _SCHEDULE_VALUES

    if not hasattr(sigmas, "flatten"):
        return [float(value) for value in sigmas]
    if _SCHEDULE_VALUES is None or _SCHEDULE_VALUES[0] is not sigmas:
        _SCHEDULE_VALUES = (sigmas, [float(value) for value in sigmas.flatten().tolist()])
    return _SCHEDULE_VALUES[1]


def sigma_progress(sigma, sigmas) -> float | None:
    """Sampling progress 0..1 of the model call at ``sigma``: the position of
    the nearest sigma in the full schedule ``sigmas``. Hires. fix and img2img
    run only its tail, so they start part-way. None when it cannot be told."""

    try:
        current = sigma_value(sigma)
        values = _schedule_values(sigmas)
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
# Kept identical to backend.nn.anima.BLOCK_MAPPINGS (upstream code: the LoRA and
# ControlLLLite loaders re-map weights with it), which this module does not import
# to stay free of the backend; tests/test_anima_lora_support.py checks it.
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


# The layout block selections are written in
REFERENCE_BLOCKS = 28


def model_block_set(text, total: int, *, feature: str = "Anima", parse=None) -> set[int]:
    """Model blocks of a block selection such as ``"0-3,7"`` written in the
    2B (28-block) layout, on a model of ``total`` blocks. On the depth-expanded
    2.9B/3.8B models a block brings its inserted copies along, so a band
    covers the same depth of any model; other depths take the indices as
    written.

    ``parse(text, total)`` reads the selection: by default
    ``anima_support.parse_block_list``, which rejects a bad one naming
    ``feature``."""

    if parse is None:
        def parse(text, total):
            return parse_block_list(text, total, feature=feature)

    if (REFERENCE_BLOCKS, total) not in ANIMA_BLOCK_MAPPINGS:
        return parse(text, total)
    selected = parse(text, REFERENCE_BLOCKS)
    return {index for index in range(total) if anima_source_block(index, REFERENCE_BLOCKS, total) in selected}


def stage_window(stage: str, shift: float) -> tuple[float, float, float]:
    """``(start, end, peak)`` sampling-progress window of a template stage
    ("composition", "character" or "style") at sampler shift ``shift``.

    Shift 3.0 is Forge's common default; higher shift keeps useful structure
    later in the denoise path, so the windows move slightly later."""

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


def stage_window_of(stage, shift, *, composition, style) -> tuple[float, float, float]:
    """``stage_window`` of a panel's stage value: ``composition`` and ``style``
    are the panel's values of those stages, any other value is the character
    stage. ``shift`` is read leniently (3.0 when unset)."""
    kind = {composition: "composition", style: "style"}.get(stage, "character")
    return stage_window(kind, to_float(shift, 3.0))


def window_curve_factor(progress: float, start: float, end: float, peak: float, shape: str, floor: float = 0.0) -> float:
    """Strength factor (0..1) at sampling ``progress`` of the window
    [start, end] (either order), 0 outside it; a zero-width window is full
    strength at its point. ``shape``: "hold" is 1 throughout, "front" falls
    from 1 to ``floor``, "back" rises from ``floor`` to 1, "triangle" rises
    linearly to ``peak`` and falls, anything else is the triangle eased.

    The stage scheduler's front / back curves fade out completely (floor 0);
    the artist mixer's keep a quarter of the artist (floor 0.25)."""

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


# Names resolve_network rescanned for without finding them, noted on ``p``
_NETWORK_MISSES = "_anima_network_misses"


def network_misses(p) -> set[str]:
    """The names ``p``'s run looked up and a rescan did not find. Passed as
    resolve_network's ``misses``, an unknown name costs one rescan per run
    however many batches, passes and tabs look it up; the next run (a new
    ``p``) rescans again, and finds a file added since. A nested run on a
    shallow copy of ``p`` shares the set."""

    misses = getattr(p, _NETWORK_MISSES, None)
    if not isinstance(misses, set):
        misses = set()
        setattr(p, _NETWORK_MISSES, misses)
    return misses


def resolve_network(name, misses: set[str] | None = None):
    """The sd_forge_lora index entry ``<lora:name:...>`` loads, looked up the
    way networks.load_networks does (forbidden aliases by network name, others
    by alias, then a rescan by file stem); None when there is none.

    The rescan walks every LoRA directory: a name in ``misses`` (see
    network_misses) is not rescanned for, and one the rescan misses is added."""

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
        if entry is None and (misses is None or key not in misses):
            networks.update_available_networks_by_names([key])
            entry = lookup()
            if entry is None and misses is not None:
                misses.add(key)
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


def lora_key_factor(
    key,
    fallback=1.0,
    *,
    qwen_weights: LayerWeights | None = None,
    block_weights: LayerWeights | None = None,
    module_weights: dict[str, float] | None = None,
    lora_blocks: int | None = None,
    model_blocks: int | None = None,
):
    """Strength of the LoRA patch at ``key`` under per-layer weights: a
    text-encoder layer's from ``qwen_weights``, a DiT block's from
    ``block_weights`` times its module's from ``module_weights``. A part left
    None (not set) keeps ``fallback``, as does any other key.

    Block weights are written in the LoRA's own layout: on a depth-expanded
    model they follow ``anima_source_block``."""

    key = model_key(key)
    qwen_layer = qwen_layer_index(key)
    if qwen_layer is not None:
        return qwen_weights.factor(qwen_layer) if qwen_weights is not None else fallback

    block_index = block_index_from_key(key)
    if block_index is None:
        return fallback
    factor = block_weights.factor(anima_source_block(block_index, lora_blocks, model_blocks)) if block_weights is not None else fallback
    if module_weights is not None:
        factor = factor * module_factor(key, module_weights)
    return factor


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
    return re.sub(r"\bblock_lrs?\s*=", "", fold_weight_text(text).strip(), flags=re.IGNORECASE)


_RANGE_SELECTOR_SPACING = re.compile(r"(?<![\w.])(\d+)\s*-\s*(\d+)(?==)")


def _weight_entries(text, *, spaced_names: bool = False) -> list[tuple[str | None, str]]:
    """``(selector, value)`` entries of a weight text, both stripped; a token
    without "=" is ``(None, token)``.

    Without any "=" the text is a bare list split on commas, semicolons and
    whitespace ("0.5 0.8 1", "0.5, 0.8"). With one, entries are split on
    commas, semicolons and newlines, and spaces around "=" (and around the
    dash of a range selector) are dropped: "0 - 9 = 0.5" is one entry. Within
    an entry, whitespace still separates tokens ("12=2 all=0.8"), unless
    ``spaced_names``: a module name may contain spaces ("self attention=0.9")."""

    source = _normalized_weight_text(text)
    if "=" not in source:
        return [(None, token) for token in re.split(r"[\s,;]+", source) if token]
    entries = []
    for part in re.split(r"[,;\n]+", source):
        # A split, not re.sub(r"\s*=\s*"): that retries a whitespace run from
        # each of its spaces, quadratic in a prompt's "0=1      x"
        part = "=".join(piece.strip() for piece in part.split("="))
        part = _RANGE_SELECTOR_SPACING.sub(r"\1-\2", part)
        for token in [part] if spaced_names else part.split():
            if "=" in token:
                selector, value = token.split("=", 1)
                entries.append((selector.strip(), value.strip()))
            elif token:
                entries.append((None, token))
    return entries


_DEFAULT_SELECTORS = {"*", "all", "default"}
_RANGE_SELECTOR = re.compile(r"(\d+)\s*-\s*(\d+)")


def looks_like_layer_weights(text) -> bool:
    value = _normalized_weight_text(text)
    if not value or is_plain_number(value):
        return False
    numeric_count = 0
    for selector, token in _weight_entries(text):
        if selector is not None:
            selector = selector.lower()
            if selector in _DEFAULT_SELECTORS or _layer_index(selector) is not None or _RANGE_SELECTOR.fullmatch(selector):
                return True
            continue
        if is_plain_number(token):
            numeric_count += 1
    return numeric_count >= 2


# Past the deepest Anima DiT (52 blocks) and Qwen3 text encoder (28 layers);
# bounds what a prompt's "0-2000000000=1" can allocate.
MAX_LAYER_INDEX = 255


def _layer_index(text: str) -> int | None:
    # Leading zeros are allowed ("05"); a digit string longer than any index
    # reads as past MAX_LAYER_INDEX (anima_support.block_index_token)
    try:
        return block_index_token(text)
    except ValueError:
        return None


def parse_layer_weights(text) -> LayerWeights:
    """Block (or layer) weights of a weight text. Bare numbers are weights in
    block order only when no block is selected ("0.5 0.8 1", "0.5 all=0.8");
    next to a selector they are ignored with a warning ("0.8, 3=0.2" is
    {3: 0.2}). Whitespace separates entries ("12=2 all=0.8"), so "0-9=0 5" is
    blocks 0-9 at 0 and a stray "5", warned about, not a value "0 5" (which
    plain_number would refuse)."""

    values: dict[int, float] = {}
    default = 1.0
    bare: list[tuple[str, float]] = []
    for selector, token in _weight_entries(text):
        if selector is None:
            value = plain_number(token)
            if value is None:
                logger.warning("Ignored invalid Anima layer weight token: %s", token)
            else:
                bare.append((token, value))
            continue
        left = selector.lower()
        value = plain_number(token)
        if value is None:
            logger.warning("Ignored invalid Anima layer weight: %s=%s", selector, token)
            continue
        if left in _DEFAULT_SELECTORS:
            default = value
        elif range_match := _RANGE_SELECTOR.fullmatch(left):
            start, end = sorted((block_index_token(range_match.group(1)), block_index_token(range_match.group(2))))
            if start > MAX_LAYER_INDEX:
                logger.warning("Ignored Anima layer range past index %d: %s", MAX_LAYER_INDEX, left)
                continue
            values.update(dict.fromkeys(range(start, min(end, MAX_LAYER_INDEX) + 1), value))
        elif (index := _layer_index(left)) is not None and index <= MAX_LAYER_INDEX:
            values[index] = value
        else:
            logger.warning("Ignored invalid Anima layer selector: %s", left)
    if bare and values:
        logger.warning("Ignored Anima layer weights without a selector: %s", " ".join(token for token, _ in bare))
    elif bare:
        values.update(enumerate(value for _, value in bare))
    return LayerWeights(default=default, values=values)


def split_weight_sections(text) -> dict[str, str]:
    """``blocks`` / ``modules`` / ``qwen`` sections of a weight text such as
    "blocks=0-9=0.8,10-27=1;modules=mlp=0.5;qwen=0-3=0". Parts without a
    section name (and the "block_lrs=" form) are block weights. The text is
    read, and returned, folded by fold_weight_text."""

    sections: dict[str, str] = {}
    bare_parts = []
    for part in [p.strip() for p in fold_weight_text(text).split(";") if p.strip()]:
        if "=" in part:
            # The first "=" ends the section name: "modules=mlp=0.5",
            # "modules：mlp=0.5", "ｍｏｄｕｌｅｓ＝ｍｌｐ＝０．５"
            left, right = part.split("=", 1)
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
    for name, token in _weight_entries(text, spaced_names=True):
        if name is None:
            continue
        value = plain_number(token)
        if value is None:
            logger.warning("Ignored invalid Anima module weight: %s=%s", name, token)
            continue
        weights[module_key(name)] = value
    return weights
