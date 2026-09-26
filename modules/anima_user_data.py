"""Per-user JSON state of the bundled Anima extensions (templates, panel values).

It lives under ``<data dir>/extension-data/<extension>/``, like the other
bundled extensions' state, instead of inside the (possibly read-only, git
tracked) install tree.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from modules import paths_internal

logger = logging.getLogger(__name__)

# One lock per data file, so concurrent edits (two UI requests, the API) of the
# same file apply one after the other instead of the last write dropping the other
_EDIT_LOCKS: dict[str, threading.Lock] = {}
_EDIT_LOCKS_GUARD = threading.Lock()


def _edit_lock(path: Path) -> threading.Lock:
    key = os.path.abspath(path)
    with _EDIT_LOCKS_GUARD:
        return _EDIT_LOCKS.setdefault(key, threading.Lock())


class UnreadableDataFile(ValueError):
    """The file exists but is not a JSON object; it must not be overwritten."""


def data_file(extension: str, filename: str, *, seed=None) -> Path:
    """``<data dir>/extension-data/<extension>/<filename>``. If it does not exist
    yet, ``seed`` (a default shipped with the extension, or the file an older
    version kept in its own folder) is copied there first."""

    path = Path(paths_internal.data_path) / "extension-data" / extension / filename
    if not path.exists() and seed is not None and Path(seed).is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed, path)
    return path


def read_json_object(path) -> dict:
    """The JSON object in ``path``, ``{}`` when the file is missing.

    A file that cannot be parsed raises UnreadableDataFile instead of reading
    as ``{}``: a later read-modify-write would otherwise replace it with just
    the one new entry."""

    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as error:
        raise UnreadableDataFile(f"{path} cannot be read: {error}") from error
    if not isinstance(data, dict):
        raise UnreadableDataFile(f"{path} does not hold a JSON object")
    return data


def write_json_atomic(path, data) -> None:
    """Write ``data`` as JSON so that ``path`` is always either the old or the
    new complete file, keeping an existing file's permissions."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            mode = os.stat(path).st_mode & 0o777
        except OSError:
            mode = 0o644  # what a plain write of a new file gives (mkstemp makes 0600)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class TemplateStore:
    """An extension's saved templates: one JSON object of ``name: template``
    in ``data_file(extension, filename, seed=seed)``."""

    def __init__(self, extension: str, filename: str, *, seed=None):
        self.extension = extension
        self.filename = filename
        self.seed = seed

    def path(self) -> Path:
        return data_file(self.extension, self.filename, seed=self.seed)

    def load(self) -> dict:
        """The templates for display; an unreadable file reads as ``{}`` and is left alone."""

        try:
            return read_json_object(self.path())
        except UnreadableDataFile:
            logger.exception("Failed to read the %s templates", self.extension)
            return {}

    def edit(self, change):
        """Apply ``change(templates)`` to the templates on disk, write them back
        atomically if it changed them, and return its result. Edits of the same
        file are serialised. An unreadable file raises UnreadableDataFile and is
        never overwritten."""

        path = self.path()
        with _edit_lock(path):
            data = read_json_object(path)
            before = copy.deepcopy(data)
            result = change(data)
            if data != before:
                write_json_atomic(path, data)
        return result
