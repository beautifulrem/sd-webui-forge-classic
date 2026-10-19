"""Tag autocompletion for the Remi WebUI.

This extension contributes its tag lists, translations and chants (``tags/``)
and the ``tac_*`` settings; the completion itself runs in the SPA
(``webui/src/features/prompt-tools``) against ``/api/v2/prompt-tools``
(``remi/prompt_tools``). Disabling the extension turns completion off.
"""

from pathlib import Path

from modules import script_callbacks, scripts
from modules.paths_internal import data_path
from remi import prompt_tools
from remi.prompt_tools import options

TAGS = Path(scripts.basedir()) / "tags"
LEGACY_FREQUENCY_DB = (
    Path(data_path) / "extension-data" / "sd-webui-tagcomplete-neo" / "tag_frequency.db",
    TAGS / "tag_frequency.db",
)

prompt_tools.register_tag_data(TAGS)


def on_ui_settings():
    options.register_tag_options(prompt_tools.tag_dir, prompt_tools.refresh)


def on_app_started(_blocks, app):
    services = getattr(app.state, "services", None)
    if services is None:
        return
    tools = prompt_tools.attach(services)
    for path in LEGACY_FREQUENCY_DB:
        if tools.store.import_tag_frequency(path):
            break


def on_script_unloaded():
    # a Reload UI runs the enabled extensions again: a disabled one stays off
    prompt_tools.unregister_tag_data(TAGS)


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_script_unloaded(on_script_unloaded)
