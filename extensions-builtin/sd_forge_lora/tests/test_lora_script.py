"""sd_forge_lora's hooks; the card browser and the metadata editor live in
the SPA (``remi.networks``)."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_no_extra_networks_ui_left():
    for path in [*ROOT.glob("*.py"), *ROOT.glob("scripts/*.py")]:
        tree = ast.parse(path.read_text(encoding="utf8"))
        assert not {"ui_extra_networks", "ui_extra_networks_user_metadata"} & {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}, path
    assert not (ROOT / "ui_edit_user_metadata.py").exists() and not (ROOT / "ui_extra_networks_lora.py").exists()


class OnDisk:
    def __init__(self, name, alias):
        self.name, self.alias, self.filename, self.metadata = name, alias, f"/loras/{name}.safetensors", {}

    def get_alias(self):
        return self.alias


@pytest.fixture
def script(monkeypatch):
    from modules import shared

    fake_networks = types.ModuleType("networks")
    fake_networks.available_networks = {"style": OnDisk("style", "style_v2")}
    fake_networks.available_network_hash_lookup = {"abcdef123456": OnDisk("style", "style_v2")}
    fake_networks.list_available_networks = lambda: None
    fake_networks.infotext_pasted = lambda infotext, params: None
    fake_networks.extra_network_lora = None
    fake_extra = types.ModuleType("extra_networks_lora")
    fake_extra.ExtraNetworkLora = type("ExtraNetworkLora", (), {"name": "lora"})
    for name, module in {"networks": fake_networks, "extra_networks_lora": fake_extra, "lora": types.ModuleType("lora"), "network": types.ModuleType("network")}.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["network"].NetworkOnDisk = OnDisk
    monkeypatch.setattr(shared, "options_templates", dict(shared.options_templates or {}))
    callbacks = []
    from modules import script_callbacks

    monkeypatch.setattr(script_callbacks, "on_before_ui", lambda fn, **_: callbacks.append(("before_ui", fn)))
    monkeypatch.setattr(script_callbacks, "on_infotext_pasted", lambda fn, **_: callbacks.append(("infotext", fn)))
    monkeypatch.setattr(script_callbacks, "on_app_started", lambda fn, **_: callbacks.append(("app_started", fn)))
    spec = importlib.util.spec_from_file_location("lora_script_under_test", ROOT / "scripts" / "lora_script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(module=module, callbacks=callbacks, networks=fake_networks, options=shared.options_templates)


def test_options_use_remi_widgets(script):
    from remi import ui

    sd_lora = script.options["sd_lora"]
    assert sd_lora.component is ui.Choice and script.options["lora_preferred_name"].component is ui.Choice
    assert sd_lora.component_args()["choices"] == ["None", "style"]
    preferred = script.options["lora_preferred_name"]
    assert preferred.component_args == {"choices": ["Alias from file", "Filename"], "style": "radio"}
    assert script.options["lora_preset_filter"].default is False


def test_before_ui_registers_the_extra_network(script):
    from modules import extra_networks

    before_ui = next(fn for kind, fn in script.callbacks if kind == "before_ui")
    registered = []
    original = extra_networks.register_extra_network
    extra_networks.register_extra_network = registered.append
    try:
        before_ui()
    finally:
        extra_networks.register_extra_network = original
    assert registered == [script.networks.extra_network_lora]


def test_pasted_hashes_resolve_to_the_local_alias(script):
    params = {"Prompt": "a <lora:old_name:0.8> b <lora:other:1>", "Lora hashes": "old_name: abcdef123456, broken"}
    script.module.infotext_pasted("", params)
    assert params["Prompt"] == "a <lora:style_v2:0.8> b <lora:other:1>"


def test_refresh_invalidates_the_browser(script, monkeypatch):
    from remi import networks

    seen = []
    monkeypatch.setattr(networks, "notify_changed", lambda *kinds: seen.append(kinds))
    script.module.refresh_loras()
    assert seen == [("lora",)]
