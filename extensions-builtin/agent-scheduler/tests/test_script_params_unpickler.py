import _codecs
import _compat_pickle
import ast
import collections
import dataclasses
import io
import os
import pickle
import sys
import types
import zlib
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

# task_helpers imports the WebUI at import time; load only the unpickler.
SOURCE = Path(__file__).parents[1] / "agent_scheduler" / "task_helpers.py"
_NAMES = {"_SAFE_PICKLE_GLOBALS", "_SAFE_REDUCE_GLOBALS"}
_FUNCS = {"_ScriptArgsUnpickler", "_load_script_args", "_is_plain_data_class", "_is_safe_reduce"}
_tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
_nodes = [
    node
    for node in _tree.body
    if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") in _NAMES for t in node.targets))
    or (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in _FUNCS)
]
_ns = {
    "pickle": pickle, "zlib": zlib, "io": io, "sys": sys, "Enum": Enum,
    "dataclasses": dataclasses, "Image": Image, "_codecs": _codecs, "_compat_pickle": _compat_pickle,
}
exec(compile(ast.Module(_nodes, []), str(SOURCE), "exec"), _ns)


def _load(obj, protocol=4):
    return _ns["_load_script_args"](zlib.compress(pickle.dumps(obj, protocol=protocol)))


# Stand-in for Forge's lib_controlnet.external_code.ControlNetUnit.
_pkg = types.ModuleType("fake_cnet")
_mod = types.ModuleType("fake_cnet.external_code")
sys.modules.setdefault("fake_cnet", _pkg)
sys.modules.setdefault("fake_cnet.external_code", _mod)
_pkg.external_code = _mod


@dataclasses.dataclass
class ControlNetUnit:
    enabled: bool = False
    image: object = None


ControlNetUnit.__module__ = "fake_cnet.external_code"
ControlNetUnit.__qualname__ = "ControlNetUnit"
_mod.ControlNetUnit = ControlNetUnit


class Mode(Enum):
    A = "a"


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_plain_script_args_round_trip(protocol):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "PNG")
    buffer.seek(0)
    png = Image.open(buffer)
    png.load()
    data = [ControlNetUnit(True, np.arange(3, dtype=np.float32)), png, Mode.A,
            collections.OrderedDict(a=1), {1, 2}, (1, "x", None), b"raw"]

    out = _load(data, protocol)

    assert out[0].enabled and out[0].image.tolist() == [0, 1, 2]
    assert out[1].size == (4, 4)
    assert out[2] is Mode.A and out[3] == collections.OrderedDict(a=1) and out[4] == {1, 2}
    assert out[6] == b"raw"


class _Shell:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


class _OpenHostFile:
    def __reduce__(self):
        return (PngImagePlugin.PngImageFile, (os.devnull,))


class _CallDataclass:
    def __reduce__(self):
        return (ControlNetUnit, (True, "x"))


class _Codec:
    def __reduce__(self):
        return (_codecs.encode, ("x", "zlib"))


@pytest.mark.parametrize("payload", [_Shell(), _OpenHostFile(), _CallDataclass(), _Codec()])
@pytest.mark.parametrize("protocol", [2, 4])
def test_code_execution_payloads_are_rejected(payload, protocol):
    with pytest.raises(pickle.UnpicklingError):
        _load([payload], protocol)
