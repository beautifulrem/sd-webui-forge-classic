"""Prompt tools for the Remi WebUI (the successor of Prompt All-in-One):
tag chips, prompt history, favorites and collections, quick tag groups and
translation. This extension contributes the tag groups (``group_tags/``)
and the ``prompt_tools_*`` settings; the tools are part of the SPA's prompt
editor (``webui/src/features/prompt-tools``), served by
``/api/v2/prompt-tools`` (``remi/prompt_tools``). Disabling the extension
turns them off.
"""

from pathlib import Path

from modules import script_callbacks, scripts
from modules.paths_internal import data_path
from remi import prompt_tools
from remi.prompt_tools import options

BASE = Path(scripts.basedir())
LEGACY_STORAGE = (
    Path(data_path) / "extension-data" / "sd-webui-prompt-all-in-one-neo" / "storage",
    BASE / "storage",
)

GROUP_TAGS = BASE / "group_tags"

prompt_tools.register_tool_data(GROUP_TAGS)


def on_app_started(_blocks, app):
    services = getattr(app.state, "services", None)
    if services is None:
        return
    tools = prompt_tools.attach(services)
    tools.migrate_translation_key()
    for path in LEGACY_STORAGE:
        if path.is_dir() and tools.store.import_prompt_all_in_one(path):
            break


def on_script_unloaded():
    # a Reload UI runs the enabled extensions again: a disabled one stays off
    prompt_tools.unregister_tool_data(GROUP_TAGS)


script_callbacks.on_ui_settings(options.register_tool_options)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_script_unloaded(on_script_unloaded)
