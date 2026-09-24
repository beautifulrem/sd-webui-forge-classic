"""The restricted unpickler for script params without a valid signature.

Rows stored by versions before signing (or signed with a key that has since
been replaced) cannot be told apart from crafted imports those versions
accepted, so they are loaded with this filter: plain data only, never calls
that act on the host.
"""

import _codecs
import _compat_pickle
import copyreg
import dataclasses
import io
import pickle
import sys
import zlib
from enum import Enum

from PIL import Image, ImageFile


_SAFE_PICKLE_GLOBALS = {
    ("builtins", name)
    for name in (
        "list", "dict", "tuple", "set", "frozenset", "str", "bytes", "bytearray",
        "int", "float", "bool", "complex", "slice", "range", "object",
    )
} | {
    ("collections", "OrderedDict"),
    ("copyreg", "_reconstructor"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("PIL.Image", "Image"),
    ("_codecs", "encode"),  # protocol < 3 bytes; REDUCE restricts it to latin1
}


# Image classes script params may hold; their pickles rebuild through the
# stock __setstate__ (pixel data), never through a file-opening __init__.
_PIL_MODULES = {"PIL.Image", "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin", "PIL.MpoImagePlugin"}


def _is_plain_data_class(cls, module):
    if issubclass(cls, Enum):
        return True
    if module in _PIL_MODULES and issubclass(cls, Image.Image):
        return cls.__setstate__ in (Image.Image.__setstate__, ImageFile.ImageFile.__setstate__)
    return (
        dataclasses.is_dataclass(cls)
        and "__setstate__" not in vars(cls)
        and "__reduce__" not in vars(cls)
        and "__reduce_ex__" not in vars(cls)
    )


# Callables REDUCE may invoke. Classes resolved by find_class are only rebuilt
# via NEWOBJ (cls.__new__) + BUILD (state), never called with payload
# arguments: e.g. PngImageFile("/any/path") would open host files.
_SAFE_REDUCE_GLOBALS = {
    ("copyreg", "_reconstructor"),
    ("collections", "OrderedDict"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
} | {
    ("builtins", name)
    for name in ("set", "frozenset", "bytearray", "complex", "slice", "range", "list", "dict", "tuple", "str", "bytes", "int", "float", "bool")
}


def _is_safe_reduce(func, args):
    if isinstance(func, type) and issubclass(func, Enum):
        return True  # Enum members pickle as Enum(value): a lookup, no side effects.
    if func is copyreg._reconstructor:
        # _reconstructor(cls, base, state) runs base.__init__(obj, state):
        # only the plain object base is a no-op (protocol 0/1 pickles).
        return isinstance(args, tuple) and len(args) == 3 and args[1] is object
    return (getattr(func, "__module__", None), getattr(func, "__qualname__", None)) in _SAFE_REDUCE_GLOBALS


class _ScriptArgsUnpickler(pickle._Unpickler):
    """Only rebuild plain data: script params can come from /import, so an
    unrestricted pickle.loads would execute attacker-controlled code.

    The pure-Python unpickler is used so REDUCE can be restricted too."""

    dispatch = dict(pickle._Unpickler.dispatch)

    def find_class(self, module, name):
        if self.proto < 3 and self.fix_imports:
            # Map Python 2 names (e.g. __builtin__) before checking, as the
            # base class would.
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        if (module, name) in _SAFE_PICKLE_GLOBALS:
            return super().find_class(module, name)
        # Plain data classes of already-loaded modules are data too: enums,
        # dataclasses without custom (de)serialisation hooks (e.g. Forge's
        # ControlNetUnit) and PIL image subclasses. Never import a module on
        # behalf of a payload (``vars`` avoids lazy-module __getattr__ imports).
        loaded = sys.modules.get(module)
        candidate = vars(loaded).get(name) if loaded is not None and "." not in name else None
        if isinstance(candidate, type) and _is_plain_data_class(candidate, module):
            return candidate
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from task script params")

    def load_reduce(self):
        func, args = self.stack[-2], self.stack[-1]
        safe = _is_safe_reduce(func, args) or (
            func is _codecs.encode and len(args) == 2 and args[1] in ("latin1", "latin-1")
        )
        if not safe:
            raise pickle.UnpicklingError(f"Refusing to call {func!r} from task script params")
        super().load_reduce()

    dispatch[pickle.REDUCE[0]] = load_reduce


def legacy_load(data: bytes):
    """Load zlib-compressed pickled script params with the legacy filter."""
    return _ScriptArgsUnpickler(io.BytesIO(zlib.decompress(data))).load()

