"""The ported panel: params() declares what the Gradio ui() did."""

import inspect
import json
from pathlib import Path

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forge_never_oom.py"


@pytest.fixture(scope="module")
def never_oom():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "never_oom_params_under_test")


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(never_oom, tab):
    layout = harness.script_layout(never_oom.NeverOOMForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab))


def test_arguments_are_the_processing_signature(never_oom):
    layout = harness.script_layout(never_oom.NeverOOMForForge())
    hook = inspect.signature(never_oom.NeverOOMForForge.setup)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p")]
    assert layout.values_to_args({"vae_enabled": True}) == [False, True]


def test_setup_switches_the_offload_flags(never_oom, monkeypatch):
    memory = never_oom.memory_management
    monkeypatch.setattr(memory, "UNET_ALWAYS_OFFLOAD", False, raising=False)
    monkeypatch.setattr(memory, "VAE_ALWAYS_TILED", False, raising=False)
    monkeypatch.setattr(memory, "current_loaded_models", [])
    script = never_oom.NeverOOMForForge()
    script.setup(None, *harness.script_layout(script).values_to_args({"unet_enabled": True, "vae_enabled": True}))
    assert memory.UNET_ALWAYS_OFFLOAD is True and memory.VAE_ALWAYS_TILED is True
    assert script.previous_unet_enabled is True


def test_schema_export(never_oom):
    schema = harness.script_layout(never_oom.NeverOOMForForge()).schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["title"] == "Never OOM Integrated" and "enable" not in schema
    assert [child["name"] for child in schema["children"]] == ["unet_enabled", "vae_enabled"]


def test_zh_cn_covers_the_panel(never_oom):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(never_oom.NeverOOMForForge(), tab).schema() for tab in harness.TABS])


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT])
