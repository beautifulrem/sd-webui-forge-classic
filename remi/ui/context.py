"""Per-request resolution context: dynamic choices and defaults, uploads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from remi.ui.nodes import Input


UploadResolver = Callable[["Input", str], Any]
"""``(input, upload_id) -> object``: a PIL image for ``Image``/``ImageList``,
a filesystem path for ``File``. Raise ``ParamError`` for an unknown id."""


@dataclass
class Context:
    """What resolving a layout needs besides the layout itself.

    Callables given as ``choices`` or ``default`` are evaluated once per
    context (one request), so one schema export or one job submission sees a
    consistent checkpoint/sampler list. Without ``resolve_upload`` upload ids
    pass through unchanged (tests, schema export).

    ``store_upload`` (PIL image -> upload id) lets ``from_arg`` turn images
    nested in a processing argument (e.g. a JSON value's pictures) into
    uploads, as job importers need; without it they are dropped."""

    resolve_upload: UploadResolver | None = None
    store_upload: Callable[[Any], str] | None = None
    cache: dict[tuple[int, str], tuple[Any, Any]] = field(default_factory=dict)

    def resolve(self, node: Any, slot: str, value: Any) -> Any:
        """``value()`` when callable (cached per node and slot), else ``value``."""

        if not callable(value):
            return value
        return self.memo(node, slot, value)

    def memo(self, node: Any, slot: str, compute) -> Any:
        """``compute()``, once per node and slot. The entry keeps the node
        alive, so a recycled ``id()`` cannot alias another node's entry."""

        key = (id(node), slot)
        entry = self.cache.get(key)
        if entry is None or entry[0] is not node:
            entry = self.cache[key] = (node, compute())
        return entry[1]


def ensure_context(ctx: Context | None) -> Context:
    return ctx if ctx is not None else Context()
