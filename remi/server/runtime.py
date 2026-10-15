"""Server-side startup before each app is created (also on "Reload UI"):
extension settings, the script runners and their layouts, the built-in tab
names and the model selection.

Scripts and Extras operations without ``params()`` (A1111's Gradio ``ui()``)
and script files that fail to import are left out; bootstrap lists them in
``not_ported``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

BUILTIN_TABS = (("txt2img", "txt2img"), ("img2img", "img2img"), ("extras", "Extras"), ("png-info", "PNG Info"), ("gallery", "Gallery"), ("networks", "Networks"))
"""``(route id, title)`` of the built-in tabs; titles are what the
``ui_tab_order`` / ``hidden_tabs`` options store."""


def prepare(*, models: bool = True) -> None:
    """Settings, runners and model selection for a new app."""

    from modules import script_callbacks, shared, shared_items
    from modules.options import options_section
    from remi.server.settings import register_options

    script_callbacks.ui_settings_callback()
    shared.opts.data_labels.update(options_section(("callbacks", "Callbacks", "system"), {**shared_items.callbacks_order_settings()}))
    register_options()
    shared.tab_names[:] = [title for _id, title in BUILTIN_TABS]
    initialize_runners()
    shared.tab_names.extend(_page_titles())  # extension pages, as their tabs were
    if models:
        initialize_model_selection()


def _page_titles() -> list[str]:
    """Registered extension pages' titles (their scripts are loaded by now)."""

    try:
        from remi import pages
    except ImportError:
        return []
    builtin = {title for _id, title in BUILTIN_TABS}
    return [title for title in dict.fromkeys(pages.tab_titles()) if title not in builtin]


def initialize_runners() -> None:
    """Build the txt2img / img2img runners and assign every script's and
    Extras operation's argument slots."""

    from modules import scripts
    from remi.core.extras import assign_args

    for runner, is_img2img in ((scripts.scripts_txt2img, False), (scripts.scripts_img2img, True)):
        runner.initialize_scripts(is_img2img=is_img2img)
        runner.ensure_arg_slots()

    postproc = scripts.scripts_postproc
    if postproc is not None:
        assign_args(postproc)


def _extension_of(path: str | None) -> str | None:
    from modules.paths_internal import extensions_builtin_dir, extensions_dir

    if not path:
        return None
    real = os.path.realpath(path)
    for root in (extensions_dir, extensions_builtin_dir):
        root = os.path.realpath(root)
        if real.startswith(root + os.sep):
            return os.path.relpath(real, root).split(os.sep)[0]
    return None


def not_ported() -> list[dict[str, Any]]:
    """What could not be loaded: script files that failed to import, scripts
    and Extras operations left out by their runner (no ``params()``, or it
    failed), with the reason."""

    from modules import scripts

    found: list[dict[str, Any]] = []
    for scriptfile, reason in getattr(scripts, "scripts_not_loaded", []):
        found.append({"kind": "script", "id": scriptfile.filename, "title": scriptfile.filename, "extension": _extension_of(scriptfile.path), "status": "disabled", "reason": reason})
    for tab, runner in (("txt2img", scripts.scripts_txt2img), ("img2img", scripts.scripts_img2img)):
        for script, reason in getattr(runner, "disabled_scripts", None) or []:
            found.append({"kind": "script", "id": getattr(script, "script_id", "") or "", "title": _title(script), "tab": tab, "extension": _extension_of(script.filename), "status": "disabled", "reason": reason})
    for operation, reason in getattr(scripts.scripts_postproc, "disabled_scripts", None) or []:
        found.append({"kind": "extras", "id": "", "title": operation.name or type(operation).__name__, "tab": "extras", "extension": _extension_of(operation.filename), "status": "disabled", "reason": reason})
    return found


def _title(script: Any) -> str:
    try:
        return str(script.title())
    except Exception:
        return os.path.basename(script.filename or "script")


# region Model selection


def initialize_model_selection() -> None:
    """Pick a checkpoint when none is set, scan the module folders and set
    the loading parameters."""

    from modules import sd_models, shared

    if shared.opts.sd_model_checkpoint in (None, "", "None", "none"):
        if not sd_models.checkpoints_list:
            sd_models.list_models()
        if sd_models.checkpoints_list:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)
    main_entry = _main_entry()
    main_entry.refresh_models()
    main_entry.refresh_model_loading_parameters()


def _main_entry() -> Any:
    from modules_forge import main_entry

    return main_entry


def load_selected_model(*, force: bool = False) -> None:
    """Load the selected checkpoint, modules and dtype now (call on the
    worker). ``force``: load again even when they are already loaded."""

    from modules import sd_models

    _main_entry().refresh_model_loading_parameters()
    if force:
        sd_models.model_data.forge_hash = ""
    sd_models.forge_model_reload()


def unet_dtypes() -> list[str]:
    import sys

    main_entry = sys.modules.get("modules_forge.main_entry")
    if main_entry is not None:
        return list(main_entry.forge_unet_storage_dtype_options)
    from modules import shared

    return [shared.opts.forge_unet_storage_dtype] if getattr(shared.opts, "forge_unet_storage_dtype", None) else ["Automatic"]


def select_checkpoint(name: str) -> bool:
    from modules import shared

    return _main_entry().checkpoint_change(name, shared.opts.forge_preset)


def select_modules(names: list[str]) -> bool:
    from modules import shared

    main_entry = _main_entry()
    if not main_entry.module_list:
        main_entry.refresh_models()
    return main_entry.modules_change(names, shared.opts.forge_preset)


def select_unet_dtype(dtype: str) -> bool:
    from modules import shared

    return _main_entry().dtype_change(dtype, shared.opts.forge_preset)


def refresh_models() -> None:
    """Re-scan checkpoints, VAE and text encoders (once each)."""

    _main_entry().refresh_models()


# endregion
