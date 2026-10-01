"""Samplers registered as functions (the Anima ones) get their extra settings:
``KDiffusionSampler`` looks ``sampler_extra_params`` up by the function's
name, else UniPC2's solver, PC3's gamma, CNS strength, ... never reach the
sampler while the infotext records them.

sd_samplers_kdiffusion cannot be imported outside the WebUI (circular
imports), so the table and the lookup are read from its source."""

import ast
import importlib
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "sd_samplers_kdiffusion.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))


def _assigned(name):
    return next(node.value for node in _TREE.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name)


def _lookup():
    """``KDiffusionSampler.__init__``'s ``self.extra_params = ...`` as a function of ``funcname``."""
    sampler = next(node for node in _TREE.body if getattr(node, "name", None) == "KDiffusionSampler")
    init = next(node for node in sampler.body if getattr(node, "name", None) == "__init__")
    value = next(node.value for node in ast.walk(init) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "self.extra_params")
    function = ast.Expression(ast.Lambda(ast.arguments(posonlyargs=[], args=[ast.arg("funcname")], kwonlyargs=[], kw_defaults=[], defaults=[]), value))
    return eval(compile(ast.fix_missing_locations(function), str(_SOURCE), "eval"), {"sampler_extra_params": EXTRA_PARAMS})  # noqa: S307 - our own source file


EXTRA_PARAMS = ast.literal_eval(_assigned("sampler_extra_params"))


def _registered_functions():
    """The functions ``samplers_k_diffusion`` registers by reference (``module.function``)."""
    modules = {"sd_samplers_anima", "sd_samplers_anima_speed", "sd_samplers_anima_freefuse", "sd_samplers_anima_cns"}
    for entry in _assigned("samplers_k_diffusion").elts:
        function = entry.elts[1]
        if isinstance(function, ast.Attribute) and getattr(function.value, "id", None) in modules:
            yield getattr(importlib.import_module(f"modules.{function.value.id}"), function.attr)


def test_samplers_registered_as_functions_get_their_extra_settings():
    lookup = _lookup()
    functions = list(_registered_functions())
    assert len(functions) >= 6

    with_settings = [function for function in functions if function.__name__ in EXTRA_PARAMS]
    assert {function.__name__ for function in with_settings} >= {"sample_anima_flow_unipc2", "sample_anima_flow_pc3", "sample_anima_er_sde_cns", "sample_anima_freefuse_euler"}
    for function in with_settings:
        assert lookup(function) == EXTRA_PARAMS[function.__name__], function.__name__
    assert lookup("sample_euler") == EXTRA_PARAMS["sample_euler"]  # names still work
