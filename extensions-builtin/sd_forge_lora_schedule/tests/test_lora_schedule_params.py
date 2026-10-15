"""LoRA Control has no panel: no parameters, no legacy ui(), no Gradio."""

import sys
from pathlib import Path
from types import ModuleType

import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "control.py"
LORA = Path(__file__).resolve().parents[2] / "sd_forge_lora"


def test_the_runner_gives_it_an_empty_ported_layout(monkeypatch):
    from modules import script_callbacks, scripts
    from remi.testing import webui_state

    # control.py reads sd_forge_lora's index (the initialised WebUI): a stand-in
    networks = ModuleType("networks")
    networks.available_network_aliases = {}
    monkeypatch.setitem(sys.modules, "networks", networks)
    monkeypatch.setattr(script_callbacks, "on_cfg_denoiser", lambda callback: None)
    with webui_state():
        control = harness.load_script(SCRIPT, "lora_schedule_params_under_test")
    assert not scripts.has_params(control.LoRAControl) and not scripts.overrides_ui(control.LoRAControl)
    with webui_state(script_classes=[control.LoRAControl]) as webui:
        script = next(script for script in webui.txt2img.scripts if isinstance(script, control.LoRAControl))
        assert script.layout is not None and not script.layout.legacy and len(script.layout) == 0
        assert script.script_id not in {item["id"] for item in webui.txt2img.scripts_schema() if item["legacy"]}


def test_imports_without_gradio():
    harness.assert_gradio_free([SCRIPT], extra_paths=[LORA])
