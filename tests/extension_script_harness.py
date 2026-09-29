"""Shared harness for the built-in Anima block-patch panels' script tests.

Not a test module. The DAVE, Cross-Attention Boost and preset-registration
suites load it by file path: each runs from its own extension folder, where
the repo ``tests`` directory is not an importable package (and putting it on
sys.path would shadow the real ``backend`` package with ``tests/backend``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def load_script(path, name):
    """Import one extension script the way Forge does, without the full WebUI."""

    # The WebUI modules parse sys.argv on first import: keep pytest's away
    sys.argv = sys.argv[:1]
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        # Its image helpers need the full WebUI install; the script only needs PasteField
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = lambda component, target: (component, target)
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Patcher:
    """The slice of a Forge UNet patcher the block patches touch."""

    def __init__(self, blocks, options=None):
        dit = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(blocks)]), patch_temporal=1)
        self.model = SimpleNamespace(diffusion_model=dit)
        self.model_options = dict(options or {})

    def clone(self):
        clone = Patcher(0, self.model_options)
        clone.model = self.model
        return clone

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def make_process(blocks=28):
    """A processing object on an Anima model, and its UNet patcher."""

    unet = Patcher(blocks)
    sd_model = SimpleNamespace(is_anima_engine=True, forge_objects=SimpleNamespace(unet=unet))
    return SimpleNamespace(sd_model=sd_model, extra_generation_params={}), unet


def build_ui(script, tab="txt2img"):
    """Build ``script``'s panel on a fresh preset registry; returns its controls."""

    import gradio as gr

    from modules.anima_presets import reset_preset_controls

    reset_preset_controls(tab)
    script.tabname = tab
    with gr.Blocks():
        return script.ui(tab == "img2img")


def assert_infotext_restores_controls(script, args, expected):
    """Each infotext value ``args`` writes lands on the control it was read from.

    The keys and their order come from the panel's paste fields, not from the
    module's own key table."""

    controls = build_ui(script)
    fields = {target: component for component, target in script.infotext_fields}
    p, _ = make_process()
    script.process_before_every_sampling(p, *args)
    written = p.extra_generation_params
    assert written == expected
    assert [fields[key] for key in written] == controls


def assert_registers_preset(script, prefix, tab="txt2img", unowned_labels=(), preset_order=True):
    """Every ``prefix.*`` preset field is one of the panel's controls, in
    control order, and its preset value is the panel's default and in range.

    ``unowned_labels``: labels of the controls the preset leaves alone (user
    content such as prompts and local file paths); every other control must be
    a preset field. ``preset_order=False``: the panel's control order need not
    be the order of the preset table's ``prefix.*`` rows."""

    from modules.anima_presets import SAFE_BASE_AESTHETIC_PRESET, preset_controls

    controls = build_ui(script, tab)
    labels = [getattr(control, "label", None) for control in controls]
    assert all(label in labels for label in unowned_labels), unowned_labels
    owned = [control for control, label in zip(controls, labels) if label not in unowned_labels]
    names, components, values = preset_controls(tab)
    expected = [name for name in SAFE_BASE_AESTHETIC_PRESET if name.startswith(f"{prefix}.")]
    if preset_order:
        assert list(names) == expected
    else:
        assert sorted(names) == sorted(expected)
    assert list(components) == owned
    for name, component, value in zip(names, components, values):
        assert value == SAFE_BASE_AESTHETIC_PRESET[name], name
        assert type(value) is type(component.value), name
        assert value == component.value, name
        minimum, maximum = getattr(component, "minimum", None), getattr(component, "maximum", None)
        if minimum is not None:
            assert minimum <= value <= maximum, name
    assert values[0] is False  # the switch: Apply preset turns the panel off
