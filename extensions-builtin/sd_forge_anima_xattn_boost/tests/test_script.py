import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "anima_xattn_boost.py"


def _load():
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        # Its image helpers need the full WebUI install; the script only needs PasteField
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = lambda component, target: (component, target)
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
    spec = importlib.util.spec_from_file_location("anima_xattn_boost_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boost = _load()


class _Patcher:
    def __init__(self, blocks, options=None):
        dit = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(blocks)]))
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        clone = _Patcher(0, self.model_options)
        clone.model = self.model
        return clone

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def _process():
    unet = _Patcher(28)
    sd_model = SimpleNamespace(is_anima_engine=True, forge_objects=SimpleNamespace(unet=unet))
    return SimpleNamespace(sd_model=sd_model, extra_generation_params={}), unet


def test_written_infotext_restores_the_panel_controls():
    # The keys and their order come from the panel's paste fields, not from
    # the module's own key table: each value written must land on the control
    # that ``process_before_every_sampling`` read it from.
    import gradio as gr

    script = boost.AnimaXattnBoostScript()
    with gr.Blocks():
        controls = script.ui(False)
    fields = {target: component for component, target in script.infotext_fields}
    p, _ = _process()
    script.process_before_every_sampling(p, True, 2.0, 0.85, 0.5)
    written = p.extra_generation_params
    assert written == {"Anima xattn boost enabled": True, "Anima xattn boost strength": 2.0, "Anima xattn boost band": 0.85, "Anima xattn boost renorm": 0.5}
    assert [fields[key] for key in written] == controls


def test_process_patches_the_pass_and_writes_infotext():
    p, unet = _process()
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, True, 2.0, 0.85, 0.5)
    patched = p.sd_model.forge_objects.unet
    assert patched is not unet and "model_function_wrapper" in patched.model_options
    assert "model_function_wrapper" not in unet.model_options
    assert p.extra_generation_params == boost.generation_params(2.0, 0.85, 0.5)


def test_disabled_unit_strength_and_non_anima_are_no_ops():
    for enable, strength, anima in ((False, 2.0, True), (True, 1.0, True), (True, 2.0, False)):
        p, unet = _process()
        p.sd_model.is_anima_engine = anima
        boost.AnimaXattnBoostScript().process_before_every_sampling(p, enable, strength, 0.85, 0.5)
        assert p.sd_model.forge_objects.unet is unet
        assert p.extra_generation_params == {}
