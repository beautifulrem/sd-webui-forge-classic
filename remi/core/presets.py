"""Preset patches as named values: the Forge "UI preset" (architecture
defaults, from ``modules_forge.main_entry.on_preset_change``) and Remi's
Anima presets (``modules.anima_presets``).

:func:`forge_preset_patch` and :func:`anima_preset_patch` are pure reads of
the options and the layouts. The UI preset also selects the model
(checkpoint, VAE/text encoders, low-bits dtype); :func:`apply_forge_preset`
does that on the server, as the old preset radio did.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError, ValidationError
from remi.ui.nodes import _NO_VALUE
from remi.ui.types import Patch


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _opts(opts: Any) -> Any:
    if opts is not None:
        return opts
    from modules import shared

    return shared.opts


def current_preset(opts: Any = None) -> str:
    return str(getattr(_opts(opts), "forge_preset", "sd") or "sd")


def check_preset(preset: Any) -> str:
    """``preset`` when it names a UI preset, else :class:`ValidationError`."""

    from modules_forge.presets import PresetArch

    if not isinstance(preset, str) or preset not in PresetArch.choices():
        raise ValidationError([ParamError(f"unknown UI preset {preset!r} (one of {', '.join(PresetArch.choices())})", owner="core", param="preset")])
    return preset


def forge_preset_patch(tab: str, preset: str | None = None, opts: Any = None) -> Patch:
    """What the UI preset ``preset`` (``sd``, ``xl``, ``anima``, ...; default:
    the current one) sets on ``tab``'s form, relative to the core owner.
    Values not above zero are left alone, as before. The tab schema applies
    it to the core defaults and the Sampler script's defaults follow it."""

    from modules_forge.presets import is_video, use_distill, use_shift

    opts = _opts(opts)
    preset = preset or current_preset(opts)
    get = lambda name, default: getattr(opts, name, default)  # noqa: E731
    prefix = "t2i" if tab == "txt2img" else "i2i"
    patch: Patch = {}

    def put(ref: str, **props: Any) -> None:
        patch.setdefault(ref, {}).update(props)

    if _positive(steps := get(f"{preset}_{prefix}_step", 20)):
        put("sampler.steps", value=steps)
    put("sampler.sampler_name", value=get(f"{preset}_{prefix}_sampler", "Euler"))
    put("sampler.scheduler", value=get(f"{preset}_{prefix}_scheduler", "Simple"))
    for name, option in (("width", "width"), ("height", "height"), ("cfg_scale", "cfg")):
        if _positive(value := get(f"{preset}_{prefix}_{option}", 1024 if option != "cfg" else 1.0)):
            put(name, value=value)

    if use_shift(preset):
        distilled = {"visible": bool(get(f"{preset}_show_shift", True)), "label": "Shift"}
    elif use_distill(preset):
        distilled = {"visible": True, "label": "Distilled CFG Scale"}
    else:
        distilled = {"visible": False}
    put("distilled_cfg_scale", value=get(f"{preset}_{prefix}_dcfg", 3.0), **distilled)

    if tab == "txt2img":
        if _positive(steps := get(f"{preset}_t2i_hr_step", 20)):
            put("hr_second_pass_steps", value=steps)
        if _positive(cfg := get(f"{preset}_t2i_hr_cfg", 1.0)):
            put("hr_cfg", value=cfg)
        put("hr_distilled_cfg", value=get(f"{preset}_t2i_hr_dcfg", 3.0), **distilled)

    if (fps := is_video(preset)) > 1:
        put("batch_size", minimum=1, maximum=fps * 15 + 1, step=fps, label="Frames", value=get(f"{preset}_{prefix}_batch_size", 1))
    else:
        put("batch_size", minimum=1, maximum=8, step=1, label="Batch Size", value=get(f"{preset}_{prefix}_batch_size", 1))
    return patch


def preset_default(tab: str, ref: str, fallback: Any) -> Any:
    """The current UI preset's value for ``ref`` (e.g. ``"sampler.steps"``),
    or ``fallback`` when the preset leaves it alone."""

    return forge_preset_patch(tab).get(ref, {}).get("value", fallback)


def apply_form_patch(layout: Any, patch: Mapping[str, Any]) -> None:
    """Make ``patch``'s core entries (bare param names) the declared state of
    ``layout``'s nodes: defaults, labels, slider bounds, visibility. Only for
    layouts built per request (the core forms)."""

    for key, entry in patch.items():
        if key.startswith("__") or "." in key or key not in layout:
            continue
        node = layout.by_name[key]
        if "value" in entry:
            node.default = entry["value"]
        if "label" in entry:
            node.label = str(entry["label"])
        for bound in ("minimum", "maximum", "step"):
            if bound in entry and hasattr(node, bound):
                setattr(node, bound, entry[bound])
        if entry.get("visible") is False:
            node.visible_if = False


def _main_entry() -> Any:
    """``modules_forge.main_entry`` when something already loaded it (it
    imports the processing pipeline; ``modules.processing`` loads it)."""

    return sys.modules.get("modules_forge.main_entry")


def forge_preset_settings(preset: str, opts: Any = None) -> dict[str, Any]:
    """The settings the UI preset ``preset`` selects: itself, its checkpoint,
    VAE/text encoder modules and low-bits dtype."""

    opts = _opts(opts)
    get = lambda name, default: getattr(opts, name, default)  # noqa: E731
    modules = [os.path.basename(module) for module in get(f"forge_additional_modules_{preset}", []) or []]
    if not get("sd_modules_dropdown_use_short", True) and (main_entry := _main_entry()) is not None:
        modules = [main_entry.module_map.get(module, module) for module in modules]
    return {
        "forge_preset": preset,
        "sd_model_checkpoint": get(f"forge_checkpoint_{preset}", None) or get("sd_model_checkpoint", ""),
        "forge_additional_modules": modules,
        "forge_unet_storage_dtype": get(f"forge_unet_storage_dtype_{preset}", "Automatic"),
    }


def apply_forge_preset(preset: str) -> dict[str, Any]:
    """Select the UI preset ``preset`` on the server: store it, switch the
    model loading parameters to its checkpoint, modules and dtype (as
    ``main_entry._load_presets`` did) and save the options. Returns the
    settings as stored now."""

    from modules import shared
    from modules_forge import main_entry

    preset = check_preset(preset)
    settings = forge_preset_settings(preset)
    shared.opts.set("forge_preset", preset)
    changed = main_entry.dtype_change(settings["forge_unet_storage_dtype"], preset, save=False, refresh=False)
    changed = main_entry.modules_change(settings["forge_additional_modules"], preset, save=False, refresh=False) or changed
    if settings["sd_model_checkpoint"]:
        changed = main_entry.checkpoint_change(settings["sd_model_checkpoint"], preset, save=False, refresh=False) or changed
    shared.opts.save(shared.config_filename)
    if changed:
        main_entry.refresh_model_loading_parameters()
    return {key: getattr(shared.opts, key, value) for key, value in settings.items()}


def preset_bindings(layouts: Mapping[str, Any]) -> list[tuple[str, str, str, Any]]:
    """``[(field, owner, param, own value or None)]`` over ``layouts``,
    skipping fields the Anima preset table does not know."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET

    result = []
    for owner, layout in layouts.items():
        for field, (name, own) in layout.presets.items():
            if field in SAFE_BASE_AESTHETIC_PRESET:
                result.append((field, owner, name, None if own is _NO_VALUE else own))
    return result


def anima_preset_patch(
    tab: str,
    label: str,
    *,
    values: Mapping[str, Mapping[str, Any]] | None = None,
    layouts: Mapping[str, Any] | None = None,
    ctx: Context | None = None,
) -> Patch:
    """Applying the Anima preset ``label`` on ``tab``: every bound param's
    value, relative to the core owner, with the conflict locks resolved
    again for the result (as the old panels' ``.change`` chain did).

    ``values`` are the form's current values (``{"core": {...}, script_id:
    {...}}``, as for :func:`remi.core.conflicts.conflict_patch`); the
    preset's values are laid over them. Conflict-locked preset controls are
    unlocked unless a rule locks them again for the new values; the guidance
    mode always gets its allowed choices back. Controls the preset leaves
    alone keep their value unless ``values`` has it (only their lock and
    choices are patched)."""

    from modules.anima_presets import PRESET_CONFLICT_CONTROLS, PRESETS, SAFE_BASE_AESTHETIC_LABEL, preset_named_values
    from remi.core.conflicts import _layouts, _ref, conflict_patch

    label = label if label in PRESETS else SAFE_BASE_AESTHETIC_LABEL
    layouts = layouts if layouts is not None else _layouts(tab)
    bound = preset_bindings(layouts)
    preset_values = preset_named_values(label, bound)
    unlock = {(owner, name) for field, owner, name, _own in bound if field in PRESET_CONFLICT_CONTROLS}

    patch: Patch = {}
    for owner, owner_values in preset_values.items():
        for name, value in owner_values.items():
            patch[_ref(owner, name)] = {"value": value, **({"enabled": True} if (owner, name) in unlock else {})}

    merged: dict[str, dict[str, Any]] = {owner: dict(owner_values) for owner, owner_values in (values or {}).items() if isinstance(owner_values, Mapping)}
    for owner, owner_values in preset_values.items():
        merged.setdefault(owner, {}).update(owner_values)
    known = {_ref(owner, name) for owner, owner_values in merged.items() for name in owner_values}
    for key, entry in conflict_patch(tab, merged, ctx=ctx, layouts=layouts).items():
        if key.startswith("__"):
            continue  # the panels' compatibility notes explain the locks
        resolved = {prop: value for prop, value in entry.items() if prop != "value" or key in known}
        if resolved:
            patch.setdefault(key, {}).update(resolved)
    patch["__toast__"] = {"level": "success", "message": f"Applied: {label} ({len(bound)} controls)"}
    return patch
