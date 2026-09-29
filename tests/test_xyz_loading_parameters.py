import ast
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from backend.nn import anima_quant

# main_entry only imports inside the initialised WebUI: run its own source
# for these functions against stubs.
_ROOT = Path(__file__).parents[1]
_SOURCE = _ROOT / "modules_forge" / "main_entry.py"
_tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_names = {
    "forge_unet_storage_dtype_options",
    "anima_quant_options",
    "forge_loading_parameters",
    "refresh_model_loading_parameters",
}
_nodes = [n for n in _tree.body if getattr(n, "name", None) in _names or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) in _names)]
_ns = {
    "torch": torch,
    "ANIMA_QUANT_SCALED_FP8": anima_quant.ANIMA_QUANT_SCALED_FP8,
    "ANIMA_QUANT_INT8_CONVROT": anima_quant.ANIMA_QUANT_INT8_CONVROT,
    "os": __import__("os"),
    "logger": logging.getLogger("test_low_bits"),
    "print_json": lambda **kwargs: None,
    "anima_online_lora": SimpleNamespace(apply_user_setting=lambda value: None),
    "processing": SimpleNamespace(),
}
exec(compile(ast.Module(_nodes, []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
anima_quant_options = _ns["anima_quant_options"]


def _xyz_refresh():
    source = (_ROOT / "scripts" / "xyz_grid.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == "refresh_loading_params_for_xyz_grid")
    namespace = {}
    exec(compile(ast.Module([node], []), "xyz_grid.py", "exec"), namespace)  # noqa: S102 - our own source file
    return namespace


@pytest.mark.parametrize("option", ["int8 (Anima ConvRot)", "float8-e4m3fn (Anima scaled)", "float8-e4m3fn", "Automatic"])
def test_xyz_grid_rebuilds_the_same_loading_parameters(monkeypatch, option):
    """The X/Y/Z grid rebuilds the parameters for each checkpoint and after the
    grid; dropping ``anima_quant`` there reloaded INT8 ConvRot as bf16."""
    checkpoint = SimpleNamespace(filename="anima.safetensors")
    model_data = SimpleNamespace(forge_loading_parameters={})
    opts = SimpleNamespace(data={}, forge_unet_storage_dtype=option, forge_additional_modules=["vae.safetensors"])
    shared = SimpleNamespace(opts=opts)
    monkeypatch.setitem(_ns, "shared", shared)
    monkeypatch.setitem(_ns, "processing", SimpleNamespace())
    sd_models = ModuleType("modules.sd_models")
    sd_models.model_data, sd_models.select_checkpoint = model_data, lambda: checkpoint
    monkeypatch.setitem(sys.modules, "modules.sd_models", sd_models)

    _ns["refresh_model_loading_parameters"]()
    from_main_entry = model_data.forge_loading_parameters

    main_entry = ModuleType("modules_forge.main_entry")
    main_entry.forge_loading_parameters = _ns["forge_loading_parameters"]
    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", main_entry)
    xyz = _xyz_refresh()
    xyz.update(select_checkpoint=lambda: checkpoint, model_data=model_data, shared=shared)
    xyz["refresh_loading_params_for_xyz_grid"]()

    assert model_data.forge_loading_parameters == from_main_entry
    # sd_models compares str(parameters) to decide on a reload
    assert str(model_data.forge_loading_parameters) == str(from_main_entry)
    assert from_main_entry.get("anima_quant") == anima_quant_options.get(option)
