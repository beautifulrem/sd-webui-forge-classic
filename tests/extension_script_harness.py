"""Shared harness for the built-in Anima panels' script tests.

Not a test module. The extension suites load it by file path: each runs from
its own extension folder, where the repo ``tests`` directory is not an
importable package (and putting it on sys.path would shadow the real
``backend`` package with ``tests/backend``). Each suite's ``conftest.py``
calls :func:`prepare_webui_imports` before collection.

Besides building panels and fake passes, it replays a run's infotext the
three ways Forge does, with Forge's own code read from its source (the
modules need the full WebUI install to import): :func:`ui_paste` (the paste
button), :func:`api_replay` (an API request's ``infotext``) and
:func:`send_to_components` (the "Send to" buttons).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]


def prepare_webui_imports():
    """Make the WebUI modules importable outside the WebUI, the same way for
    every suite, whatever file runs first.

    Each extension suite's ``tests/conftest.py`` calls it before collection;
    it is idempotent. The WebUI modules parse ``sys.argv`` on first import, so
    pytest's own arguments are dropped. ``modules.infotext_utils`` needs the
    full WebUI install (its image helpers): where it cannot be imported, a
    module holding Forge's own ``PasteField`` (read from its source) stands in
    for it for the whole session, since the scripts import it lazily inside
    ``ui()``."""

    sys.argv = sys.argv[:1]
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = forge_infotext().PasteField
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub


def load_script(path, name):
    """Import one extension script the way Forge does, without the full WebUI."""

    prepare_webui_imports()
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


def forge_source(path, names, namespace=None) -> dict:
    """``namespace`` with the top-level functions, classes and assignments
    ``names`` of the WebUI module at ``path`` (relative to the repo) run in
    it: for code whose module needs the full WebUI install to import."""

    path = REPO / path
    namespace = {} if namespace is None else namespace
    tree = ast.parse(path.read_text(encoding="utf8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names)
    ]
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)  # noqa: S102 - our own source file
    assert set(names) <= set(namespace), set(names) - set(namespace)
    return namespace


@functools.cache
def forge_infotext() -> SimpleNamespace:
    """Forge's infotext code: ``re_param_code``, ``quote``, ``unquote``,
    ``parse_info`` (``_parse_info``, the UI paste of one field) and
    ``PasteField`` of modules/infotext_utils.py, and ``pasted_value`` (the
    API replay's conversion) of modules/api/api.py."""

    import gradio as gr

    namespace = {"json": json, "gr": gr, "Any": object, "type_of_gr_update": type(gr.skip()), "errors": SimpleNamespace(report=print)}
    infotext = forge_source("modules/infotext_utils.py", {"re_param_code", "quote", "unquote", "_parse_info", "PasteField"}, namespace)
    api = forge_source("modules/api/api.py", {"pasted_value"})
    names = ("re_param_code", "quote", "unquote", "PasteField")
    return SimpleNamespace(**{name: infotext[name] for name in names}, parse_info=infotext["_parse_info"], pasted_value=api["pasted_value"])


def paste_fields(script) -> list:
    """``script.infotext_fields`` as the PasteFields Forge's
    ``add_paste_fields`` makes of them (``.component``, ``.label``,
    ``.function``), whatever the script built them with."""

    return [forge_infotext().PasteField(*field) for field in script.infotext_fields]


def through_infotext(params: dict) -> tuple[str, dict]:
    """``(line, parsed)``: the infotext parameters ``params`` in a parameters
    line as processing.create_infotext writes it (values quoted by
    ``infotext_utils.quote``), and that line read back as
    parse_generation_parameters reads it (``re_param_code``, ``unquote``)."""

    forge = forge_infotext()
    line = "Steps: 20, Sampler: Euler, " + ", ".join(f"{key}: {forge.quote(value)}" for key, value in params.items() if value is not None)
    return line, {key: forge.unquote(value) for key, value in re.findall(forge.re_param_code, line)}


def ui_paste_updates(script, params: dict, callbacks=()) -> list:
    """The update the paste button (infotext_utils.connect_paste) makes of
    each of ``script``'s paste fields, in field order, for a run that
    recorded ``params``; None where it leaves the control alone.

    ``callbacks``: the ``on_infotext_pasted`` callbacks the panel registers,
    each called with ``(line, parsed)`` before the fields read ``parsed``."""

    line, parsed = through_infotext(params)
    for callback in callbacks:
        callback(line, parsed)
    updates = [forge_infotext().parse_info(field.component, field[1], parsed) for field in paste_fields(script)]
    return [update if set(update) - {"__type__"} else None for update in updates]


def ui_paste(script, params: dict, callbacks=()) -> list:
    """The value the paste button sets each of ``script``'s paste fields to,
    in field order (None: left alone). See :func:`ui_paste_updates`."""

    return [update.get("value") if update is not None else None for update in ui_paste_updates(script, params, callbacks)]


def pasted_args(script, controls, params: dict, callbacks=(), start=None) -> list:
    """The script args of a panel with ``controls`` (as ``ui`` returns them)
    after the paste button pasted ``params`` into it. ``start``: the script
    args the panel is set to before the paste (default: the controls'
    defaults); a control the paste leaves alone keeps its ``start`` value."""

    args = [control.value for control in controls] if start is None else list(start)
    assert len(args) == len(controls)
    indexes = {id(control): index for index, control in enumerate(controls)}
    for field, value in zip(paste_fields(script), ui_paste(script, params, callbacks), strict=True):
        if value is not None and id(field.component) in indexes:
            args[indexes[id(field.component)]] = value
    return args


def api_replay(script, controls, params: dict) -> list:
    """The script args an API request whose ``infotext`` recorded ``params``
    runs with (modules/api/api.py ``apply_infotext``): the controls'
    defaults (the API's default script args), each paste field of a script
    control replaced by ``get_field_value`` (its function's value, else its
    label's, converted by ``pasted_value`` to the control's value type). No
    ``on_infotext_pasted`` callback runs."""

    forge = forge_infotext()
    _, parsed = through_infotext(params)
    args = [control.value for control in controls]
    indexes = {id(control): index for index, control in enumerate(controls)}
    for field in paste_fields(script):
        if id(field.component) not in indexes:
            continue
        value = field.function(parsed) if field.function else parsed.get(field.label)
        target_type = type(field.component.value)
        if value is None or target_type is type(None):
            continue
        value = forge.pasted_value(value, target_type)
        if value is not None:
            args[indexes[id(field.component)]] = value
    return args


def send_to_components(script) -> list:
    """The controls of ``script``'s panel the "Send to" buttons carry to the
    other tab's panel, in order (infotext_utils.connect_paste_params_buttons):
    those whose paste target is named in ``paste_field_names``."""

    names = getattr(script, "paste_field_names", [])
    return [field.component for field in paste_fields(script) if field[1] in names]


def assert_infotext_round_trip(script, controls, args, record, expected=None) -> dict:
    """A run with the script args ``args`` records the infotext
    ``record(args)`` (a dict); pasting it into a fresh panel (the paste
    button) and replaying it through the API both give the script args
    ``expected`` (default ``args``: pass the defaults of the controls the run
    does not record), and a run with those records the same infotext.
    Returns the recorded infotext."""

    params = record(list(args))
    expected = list(args) if expected is None else list(expected)
    for path, replayed in (("UI paste", pasted_args(script, controls, params)), ("API replay", api_replay(script, controls, params))):
        assert replayed == expected, path
        assert record(replayed) == params, path
    return params
