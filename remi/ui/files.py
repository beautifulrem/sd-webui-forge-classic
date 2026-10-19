"""Files for actions: ``ctx.files`` (an :class:`ActionFiles`).

The server fills ``ActionContext.files`` for every action it runs, so a
script or page reads the images the user uploaded (or a generated output the
form points at) and stores the images it makes without knowing the server:

    @ui.action
    def preview(self, ctx, values):
        image = ctx.files.image(values.get("source"))      # PIL image or None
        if image is None:
            return ui.toast("Upload an image first", "warning")
        file_id = ctx.files.store_image(make_preview(image), "preview.png")
        return {"__outputs__": {"preview": file_id}}

Files are addressed by opaque id only: the upload ids image inputs carry
(``POST /api/v2/uploads``) and output ids. ``ctx.files`` is ``None`` outside
a server (e.g. a unit test that builds an ``ActionContext`` itself; tests pass
``files=ActionFiles(store)`` around a ``remi.files.FileStore``).
"""

from __future__ import annotations

import io
from typing import Any

READABLE_KINDS = ("upload", "output")
"""What actions may read: uploads and generated outputs (not network previews)."""


class ActionFiles:
    """The server's file store as seen by actions.

    ``store`` is a ``remi.files.FileStore`` (anything with ``get(id)``,
    ``add_upload_bytes(data, name)`` and ``resolve_upload(node, id)``).
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def get(self, file_id: Any) -> Any:
        """The record of an upload or output (``id``, ``name``, ``path``,
        ``mime``, ``size``, ``width``, ``height``, ``infotext``,
        ``is_image``), or ``None`` for an unknown id."""

        if not isinstance(file_id, str) or not file_id:
            return None
        record = self._store.get(file_id)
        if record is None or getattr(record, "kind", None) not in READABLE_KINDS:
            return None
        return record

    def path(self, file_id: Any) -> str | None:
        """The file's path on the server, or ``None``."""

        record = self.get(file_id)
        return record.path if record is not None else None

    def image_size(self, file_id: Any) -> tuple[int, int] | None:
        """``(width, height)`` of an image file as indexed (no decoding), or ``None``."""

        record = self.get(file_id)
        if record is None or not record.is_image or not record.width or not record.height:
            return None
        return record.width, record.height

    def image(self, file_id: Any) -> Any:
        """The image as a PIL image (``image.filename`` = the upload's name),
        or ``None`` when there is none, it is not an image or cannot be read."""

        from remi.ui.errors import ParamError

        if self.get(file_id) is None:
            return None
        try:
            return self._store.resolve_upload(None, file_id)
        except (ParamError, OSError, ValueError):
            return None

    def store_bytes(self, data: bytes, name: str) -> str:
        """Stores ``data`` as an upload (content-sniffed, deduplicated);
        returns its id, which image inputs and outputs accept."""

        return self._store.add_upload_bytes(data, name).id

    def store_image(self, image: Any, name: str = "image.png") -> str:
        """Stores a PIL image (or an ``HxWxC`` uint8-compatible array) as a
        PNG upload; returns its id."""

        from PIL import Image

        if not isinstance(image, Image.Image):
            import numpy as np

            image = Image.fromarray(np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8)))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.store_bytes(buffer.getvalue(), name)

    def resolve_upload(self, node: Any, file_id: str) -> Any:
        """The ``remi.ui`` upload resolver (``ui.Context(resolve_upload=...)``):
        a PIL image for image inputs, the path for ``File``."""

        return self._store.resolve_upload(node, file_id)
