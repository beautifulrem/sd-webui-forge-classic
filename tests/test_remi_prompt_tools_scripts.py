"""The scripts of the two prompt-tool extensions (tag autocomplete, Prompt
All-in-One's successor): they register their data folders with
``remi.prompt_tools``, contribute their settings, import the old extensions'
data once on app start, and unregister on a Reload UI that leaves them off.

The scripts are loaded as Forge loads them (``scripts.basedir()`` is the
extension's folder), with the WebUI's callback registration recorded instead
of installed, so nothing leaks into other tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from remi.testing import webui_state

REPO = Path(__file__).resolve().parents[1]
PROMPT_TOOLS = REPO / "extensions-builtin" / "sd-webui-prompt-all-in-one-neo"
TAG_COMPLETE = REPO / "extensions-builtin" / "sd-webui-tagcomplete-neo"


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def registry(monkeypatch):
    """remi.prompt_tools' registry, restored afterwards."""

    from remi import prompt_tools

    monkeypatch.setattr(prompt_tools.registry, "tag_dirs", [])
    monkeypatch.setattr(prompt_tools.registry, "group_dirs", [])
    monkeypatch.setattr(prompt_tools._groups, "directories", [])
    return prompt_tools.registry


@pytest.fixture
def callbacks(monkeypatch):
    """The callbacks a script registers at import, by hook name."""

    from modules import script_callbacks

    seen: dict[str, list] = {}
    for hook in ("on_ui_settings", "on_app_started", "on_script_unloaded"):
        monkeypatch.setattr(script_callbacks, hook, lambda callback, *args, _hook=hook, **kwargs: seen.setdefault(_hook, []).append(callback))
    return seen


def load(extension: Path, script: str, monkeypatch):
    from modules import scripts

    monkeypatch.setattr(scripts, "current_basedir", str(extension))
    spec = importlib.util.spec_from_file_location(f"remi_test_{script}", extension / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStore:
    def __init__(self, imported: dict[str, int]):
        self.imported = imported
        self.calls: list[str] = []

    def import_prompt_all_in_one(self, path: Path) -> int:
        self.calls.append(str(path))
        return self.imported.get(str(path), 0)

    def import_tag_frequency(self, path: Path) -> int:
        self.calls.append(str(path))
        return self.imported.get(str(path), 0)


class FakeTools:
    def __init__(self, imported: dict[str, int]):
        self.store = FakeStore(imported)
        self.migrated = 0

    def migrate_translation_key(self) -> bool:
        self.migrated += 1
        return False


def fake_attach(monkeypatch, imported: dict[str, int]) -> list:
    from remi import prompt_tools

    attached = []

    def attach(services):
        tools = FakeTools(imported)
        attached.append((services, tools))
        return tools

    monkeypatch.setattr(prompt_tools, "attach", attach)
    return attached


def app(services=None):
    return SimpleNamespace(state=SimpleNamespace(services=services) if services is not None else SimpleNamespace())


# region Prompt tools (Prompt All-in-One's successor)


def test_prompt_tools_registers_its_tag_groups_and_settings(registry, callbacks, monkeypatch):
    from remi.prompt_tools import options

    module = load(PROMPT_TOOLS, "prompt_tools", monkeypatch)
    assert module.GROUP_TAGS == PROMPT_TOOLS / "group_tags" and module.GROUP_TAGS.is_dir()
    assert registry.group_dirs == [str(PROMPT_TOOLS / "group_tags")] and registry.tools_enabled
    assert callbacks["on_ui_settings"] == [options.register_tool_options]
    assert callbacks["on_app_started"] == [module.on_app_started] and callbacks["on_script_unloaded"] == [module.on_script_unloaded]
    # the old extension's storage: its extension-data folder first, then its own folder
    assert module.LEGACY_STORAGE[0].parts[-3:] == ("extension-data", "sd-webui-prompt-all-in-one-neo", "storage")
    assert module.LEGACY_STORAGE[1] == PROMPT_TOOLS / "storage"

    module.on_script_unloaded()  # a Reload UI with the extension disabled
    assert registry.group_dirs == [] and not registry.tools_enabled


def test_prompt_tools_imports_the_first_legacy_storage_with_data(registry, callbacks, monkeypatch, tmp_path):
    module = load(PROMPT_TOOLS, "prompt_tools", monkeypatch)
    missing, empty, used, never = tmp_path / "missing", tmp_path / "empty", tmp_path / "used", tmp_path / "never"
    for folder in (empty, used, never):
        folder.mkdir()
    monkeypatch.setattr(module, "LEGACY_STORAGE", (missing, empty, used, never))
    attached = fake_attach(monkeypatch, {str(used): 3, str(never): 5})

    module.on_app_started(None, app())  # no Remi services (a bare FastAPI app): nothing to attach to
    assert attached == []

    services = SimpleNamespace(root=str(tmp_path))
    module.on_app_started(None, app(services))
    ((given, tools),) = attached
    assert given is services and tools.migrated == 1
    assert tools.store.calls == [str(empty), str(used)]  # a missing folder is skipped; stops after the first import


# endregion

# region Tag autocomplete


def test_tag_autocomplete_registers_its_tags_and_settings(registry, callbacks, monkeypatch):
    from remi import prompt_tools
    from remi.prompt_tools import options

    module = load(TAG_COMPLETE, "tag_autocomplete", monkeypatch)
    assert module.TAGS == TAG_COMPLETE / "tags" and module.TAGS.is_dir()
    assert registry.tag_dirs == [str(TAG_COMPLETE / "tags")] and prompt_tools.tag_dir() == str(TAG_COMPLETE / "tags")
    assert module.LEGACY_FREQUENCY_DB[0].parts[-3:] == ("extension-data", "sd-webui-tagcomplete-neo", "tag_frequency.db")
    assert module.LEGACY_FREQUENCY_DB[1] == TAG_COMPLETE / "tags" / "tag_frequency.db"

    registered = []
    monkeypatch.setattr(options, "register_tag_options", lambda tag_dir, refresh: registered.append((tag_dir, refresh)))
    (on_ui_settings,) = callbacks["on_ui_settings"]
    on_ui_settings()
    assert registered == [(prompt_tools.tag_dir, prompt_tools.refresh)]
    assert callbacks["on_app_started"] == [module.on_app_started] and callbacks["on_script_unloaded"] == [module.on_script_unloaded]

    refreshed = []
    monkeypatch.setattr(prompt_tools, "refresh", lambda: refreshed.append(True))
    module.on_script_unloaded()
    assert registry.tag_dirs == [] and prompt_tools.tag_dir() is None and refreshed == [True]


def test_tag_autocomplete_imports_the_first_frequency_database(registry, callbacks, monkeypatch, tmp_path):
    module = load(TAG_COMPLETE, "tag_autocomplete", monkeypatch)
    first, second = tmp_path / "a.db", tmp_path / "b.db"
    monkeypatch.setattr(module, "LEGACY_FREQUENCY_DB", (first, second))

    attached = fake_attach(monkeypatch, {str(first): 12, str(second): 4})
    module.on_app_started(None, app())
    assert attached == []
    module.on_app_started(None, app(SimpleNamespace(root=str(tmp_path))))
    assert attached[0][1].store.calls == [str(first)]  # imported from the first: the second is not read

    attached = fake_attach(monkeypatch, {str(second): 4})
    module.on_app_started(None, app(SimpleNamespace(root=str(tmp_path))))
    assert attached[0][1].store.calls == [str(first), str(second)]
    assert attached[0][1].migrated == 0  # the translation key belongs to the prompt tools


# endregion
