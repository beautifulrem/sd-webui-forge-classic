"""Server-side startup that used to happen while the Gradio UI was built:
extension settings, the script runners and their layouts, the built-in tab
names and the model selection. :func:`prepare` runs before each app is
created (also on "Reload UI").

Scripts still on Gradio's ``ui()`` (ported in a later wave) never get a UI:
when Gradio is importable their ``ui()`` runs once in a throwaway
``gr.Blocks`` only to read their argument count and defaults; otherwise they
are disabled with a log line. Bootstrap lists them as not yet ported.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

NO_GRADIO = "not yet ported to params(), and Gradio is not installed"

BUILTIN_TABS = (("txt2img", "txt2img"), ("img2img", "img2img"), ("extras", "Extras"), ("png-info", "PNG Info"), ("gallery", "Gallery"), ("networks", "Networks"))
"""``(route id, title)`` of the built-in tabs; titles are what the
``ui_tab_order`` / ``hidden_tabs`` options store."""

_extras_disabled: list[tuple[Any, str]] = []


def gradio_available() -> bool:
    try:
        return importlib.util.find_spec("gradio") is not None
    except (ImportError, ValueError):
        return False


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
    from modules import scripts

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    scripts.scripts_current = None
    prepare_legacy_scripts()


def prepare_legacy_scripts(*, gradio: bool | None = None) -> None:
    """Read (or, without Gradio, disable) the scripts and Extras operations
    still on ``ui()``."""

    from modules import scripts
    from modules.scripts import has_params, overrides_ui

    gradio = gradio_available() if gradio is None else gradio
    for runner in (scripts.scripts_txt2img, scripts.scripts_img2img):
        for script in list(runner.scripts):
            if has_params(script) or not overrides_ui(script):
                continue
            if gradio:
                runner.layout_for(script)  # throwaway ui(); disables the script if it fails
            else:
                log.warning("Script %s (%s) is disabled: %s", script.title(), script.filename, NO_GRADIO)
                runner.disable_script(script, NO_GRADIO)
        runner.ensure_arg_slots()

    _extras_disabled.clear()
    postproc = scripts.scripts_postproc
    if postproc is None:
        return
    postproc.scripts_in_preferred_order()  # creates the operations
    if not gradio:
        for operation in list(postproc.scripts):
            if not callable(getattr(operation, "params", None)) and operation.controls is None:
                log.warning("Extras operation %s (%s) is disabled: %s", operation.name, operation.filename, NO_GRADIO)
                postproc.scripts.remove(operation)
                _extras_disabled.append((operation, NO_GRADIO))
    from remi.core.extras import assign_args

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
    """Everything still on the Gradio API: legacy scripts (kept with
    positional params, or disabled), legacy Extras operations and extension
    tabs (``on_ui_tabs``, ported to pages later)."""

    from modules import script_callbacks, scripts

    found: list[dict[str, Any]] = []
    for scriptfile in getattr(scripts, "scripts_needing_gradio", []):
        found.append({"kind": "script", "id": scriptfile.filename, "title": scriptfile.filename, "extension": _extension_of(scriptfile.path), "status": "disabled", "reason": NO_GRADIO})
    for tab, runner in (("txt2img", scripts.scripts_txt2img), ("img2img", scripts.scripts_img2img)):
        if runner is None:
            continue
        for script in runner.scripts:
            layout = script.layout if script.layout is not None else runner.layout_for(script)
            if layout.legacy:
                found.append({"kind": "script", "id": script.script_id, "title": script.title(), "tab": tab, "extension": _extension_of(script.filename), "status": "legacy"})
        for script, reason in runner.disabled_scripts:
            found.append({"kind": "script", "id": getattr(script, "script_id", "") or "", "title": _title(script), "tab": tab, "extension": _extension_of(script.filename), "status": "disabled", "reason": reason})
    postproc = scripts.scripts_postproc
    if postproc is not None and postproc.scripts:
        from remi.core.extras import assign_args

        for operation, layout in assign_args(postproc):
            if layout.legacy:
                found.append({"kind": "extras", "id": layout.owner, "title": operation.name or type(operation).__name__, "tab": "extras", "extension": _extension_of(operation.filename), "status": "legacy"})
    for operation, reason in _extras_disabled:
        found.append({"kind": "extras", "id": "", "title": operation.name or type(operation).__name__, "tab": "extras", "extension": _extension_of(operation.filename), "status": "disabled", "reason": reason})
    for callback in script_callbacks.callback_map["callbacks_ui_tabs"]:
        extension = _extension_of(callback.script)
        name = extension or os.path.basename(callback.script or "")
        found.append({"kind": "page", "id": callback.name, "title": name, "extension": extension, "status": "disabled", "reason": "extension tab (on_ui_tabs) not yet ported to a page"})
    return found


def _title(script: Any) -> str:
    try:
        return str(script.title())
    except Exception:
        return os.path.basename(script.filename or "script")


# region Model selection


def initialize_model_selection() -> None:
    """Pick a checkpoint when none is set, scan the module folders and set
    the loading parameters (what the Gradio model bar did on build)."""

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
    # The model selection state still lives next to the Gradio model bar
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
