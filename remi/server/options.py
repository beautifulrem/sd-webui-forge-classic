"""``PATCH /api/v2/options``: validate every value against its setting's
widget, then apply them (running ``onchange`` side effects), save
``config.json`` and report which keys changed.

The model selection (checkpoint, VAE/text encoder modules, low-bits dtype)
goes through the model manager, as the model bar did: it is remembered for
the current UI preset and the next generation loads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remi.ui import Context, ParamError, ValidationError

MODEL_KEYS = ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")


def _error(key: str, message: str) -> ParamError:
    return ParamError(message, owner="settings", param=key)


def frozen_reason(opts: Any, key: str, cmd_opts: Any) -> str | None:
    """Why ``PATCH /options`` may not change ``key`` (None when it may); the
    settings schema marks such settings ``locked`` with this reason."""

    info = opts.data_labels.get(key)
    if info is None:
        return "unknown setting"
    if getattr(cmd_opts, "freeze_settings", False):
        return "changing settings is disabled (--freeze-settings)"
    if info.do_not_save:
        return "not a stored setting"
    # ``restrict_api`` guards the A1111 API (/sdapi/v1/options); /api/v2 is the
    # WebUI's own API, behind the same login as the Gradio settings page was.
    args = info.component_args() if callable(info.component_args) else info.component_args
    if isinstance(args, dict) and args.get("visible", True) is False:
        return "this setting is hidden"
    sections = getattr(cmd_opts, "freeze_settings_in_sections", None)
    if sections and info.section and info.section[0] in [item.strip() for item in sections.split(",")]:
        return "the settings in this section are frozen"
    keys = getattr(cmd_opts, "freeze_specific_settings", None)
    if keys and key in [item.strip() for item in keys.split(",")]:
        return "this setting is frozen"
    if getattr(cmd_opts, "hide_ui_dir_config", False) and key in opts.restricted_opts:
        return "this setting is restricted (--hide-ui-dir-config)"
    return None


def coerce_values(opts: Any, values: Any, cmd_opts: Any) -> dict[str, Any]:
    """The coerced values, or :class:`ValidationError` listing every
    rejected one (nothing is applied then)."""

    from remi.ui import nodes
    from remi.ui.settings import setting_node

    if not isinstance(values, Mapping):
        raise ValidationError([ParamError("expected an object of setting values", owner="settings")])
    ctx = Context()
    result: dict[str, Any] = {}
    problems: list[ParamError] = []
    for key, value in values.items():
        reason = frozen_reason(opts, key, cmd_opts)
        if reason is not None:
            problems.append(_error(key, reason))
            continue
        if key in MODEL_KEYS:
            try:
                result[key] = _model_value(key, value)
            except ParamError as error:
                problems.append(_error(key, error.message))
            continue
        current = getattr(opts, key)
        node = setting_node(key, opts.data_labels[key], current)
        if not isinstance(node, nodes.Input):
            problems.append(_error(key, "not a setting"))
            continue
        try:
            result[key] = node.coerce(value, ctx)
        except ParamError as error:
            if value == current:  # a stored value outside stale choices
                result[key] = value
            else:
                problems.append(_error(key, error.message))
    if problems:
        raise ValidationError(problems)
    return result


def _model_value(key: str, value: Any) -> Any:
    if key == "sd_model_checkpoint":
        from modules import sd_models

        if not isinstance(value, str) or sd_models.get_closet_checkpoint_match(value) is None:
            raise ParamError(f"unknown checkpoint {value!r}")
        return value
    if key == "forge_additional_modules":
        from remi.networks import module_files

        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ParamError("expected a list of module names")
        known = {name for kind in ("vae", "text_encoders") for name, _path, _root in module_files(kind)}
        names = [item.replace("\\", "/").rsplit("/", 1)[-1] for item in value]
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ParamError(f"unknown module(s): {', '.join(unknown)}")
        return names
    from remi.server.runtime import unet_dtypes

    if value not in unet_dtypes():
        raise ParamError(f"unknown dtype {value!r}")
    return value


def apply_values(opts: Any, values: Mapping[str, Any], config_filename: str | None) -> tuple[list[str], list[ParamError]]:
    """Apply coerced values: ``(changed keys, failures)``; saves the
    options when something changed."""

    from remi.server import runtime

    changed: list[str] = []
    failed: list[ParamError] = []
    for key, value in values.items():
        try:
            if key == "sd_model_checkpoint":
                done = runtime.select_checkpoint(value)
            elif key == "forge_additional_modules":
                done = runtime.select_modules(value)
            elif key == "forge_unet_storage_dtype":
                done = runtime.select_unet_dtype(value)
            else:
                before = opts.data.get(key)
                done = opts.set(key, value)
                if not done and value != before and opts.data.get(key) != value:
                    failed.append(_error(key, "the setting rejected the value"))
                    continue
        except Exception as error:  # an onchange or model manager failure
            failed.append(_error(key, f"{type(error).__name__}: {error}"))
            continue
        if done:
            changed.append(key)
    if changed and config_filename:
        opts.save(config_filename)
    return changed, failed
