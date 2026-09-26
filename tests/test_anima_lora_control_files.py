"""``lora_control_files`` reads Forge's LoRA Control (the upstream
sd_forge_lora_schedule/scripts/control.py, loaded the way the WebUI loads
scripts) at call time."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from modules import script_callbacks, script_loading, scripts
from modules.anima_lora_support import lora_control_files

_CONTROL = Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_lora_schedule" / "scripts" / "control.py"


def _load_control(monkeypatch):
    # control.py's heavy imports (sd_forge_lora's index, the processing
    # module) need the initialised WebUI: stand-ins for the names it uses
    networks = ModuleType("networks")
    networks.available_network_aliases = {"x": SimpleNamespace(filename="/loras/x.safetensors")}
    processing = ModuleType("modules.processing")
    processing.StableDiffusionProcessing = object
    monkeypatch.setitem(sys.modules, "networks", networks)
    monkeypatch.setitem(sys.modules, "modules.processing", processing)
    monkeypatch.setattr(script_callbacks, "on_cfg_denoiser", lambda callback: None)
    monkeypatch.setattr(script_loading, "loaded_scripts", {})
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "online_lora", True, raising=False)
    return script_loading.load_module(str(_CONTROL))


def _register(monkeypatch, *modules):
    monkeypatch.setattr(scripts, "scripts_data", [scripts.ScriptClassData(module.LoRAControl, str(_CONTROL), str(_CONTROL.parents[1]), module) for module in modules])


def test_files_lora_control_schedules_in_this_generation(monkeypatch):
    control = _load_control(monkeypatch)
    _register(monkeypatch, control)
    script = control.LoRAControl()

    script.before_process(SimpleNamespace(prompt="1girl <lora:x:[0@0,1@1]>"))
    assert lora_control_files() == {"/loras/x.safetensors"}

    script.before_process(SimpleNamespace(prompt="1girl <lora:x:0.8>"))
    assert lora_control_files() == set()


def test_reload_ui_reads_the_reloaded_script(monkeypatch):
    stale = _load_control(monkeypatch)
    stale.LoRAControl().before_process(SimpleNamespace(prompt="<lora:x:[0@0,1@1]>"))
    # "Reload UI" re-executes control.py into a new, empty class
    reloaded = _load_control(monkeypatch)
    _register(monkeypatch, reloaded)
    assert lora_control_files() == set()

    reloaded.LoRAControl().before_process(SimpleNamespace(prompt="<lora:x:[1@0,0@1]>"))
    assert lora_control_files() == {"/loras/x.safetensors"}


def test_nothing_is_scheduled_without_lora_control(monkeypatch):
    control = _load_control(monkeypatch)
    control.LoRAControl().before_process(SimpleNamespace(prompt="<lora:x:[0@0,1@1]>"))
    # The extension disabled (left out of the loaded scripts), or other scripts only
    monkeypatch.setattr(scripts, "scripts_data", [scripts.ScriptClassData(scripts.Script, "/elsewhere/scripts/control.py", "/elsewhere", None)])
    assert lora_control_files() == set()
