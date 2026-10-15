"""Shared fakes for the built-in Anima panels' script tests.

Not a test module. The extension suites load it by file path: each runs from
its own extension folder, where the repo ``tests`` directory is not an
importable package (and putting it on sys.path would shadow the real
``backend`` package with ``tests/backend``).

:func:`load_script` imports an extension script as the WebUI does,
:func:`make_process` fakes a pass on an Anima model, and :func:`forge_source`
runs selected definitions of a WebUI module that needs the full install to
import.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]


def prepare_webui_imports():
    """Make the WebUI modules importable outside the WebUI (idempotent): the
    WebUI modules parse ``sys.argv`` on first import, so pytest's own
    arguments are dropped."""

    sys.argv = sys.argv[:1]


def load_script(path, name):
    """Import one extension script the way Forge does, without the full WebUI."""

    prepare_webui_imports()
    # Forge puts every extension's folder on sys.path (modules/scripts.py), for its lib_* packages
    basedir = str(Path(path).resolve().parents[1])
    if basedir not in sys.path:
        sys.path.insert(0, basedir)
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
