"""The gallery's view of the output index (``files.db``): every generated
image with what the gallery filters and sorts by.

:class:`~remi.files.FileStore` owns the ``files`` table; this module keeps a
side table ``gallery_meta`` (one row per output: the model read from its
infotext, its category — ``txt2img``, ``img2img``, ``extras``, ``grids``,
``saved`` or ``other``, from the job that made it or the folder it is in —
and the star) in the same database, on a connection of its own. Rows are
filled lazily for outputs that have none yet, so files indexed by the
folder scan or by older versions join in on the next query.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import time
import weakref
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from remi.files import FileRecord, FileStore, connect

Sort = Literal["newest", "oldest", "name", "largest"]
SORTS: tuple[Sort, ...] = ("newest", "oldest", "name", "largest")
CATEGORIES = ("txt2img", "img2img", "extras", "grids", "saved", "other")
MAX_QUERY_TERMS = 8
MAX_ZIP_FILES = 2000
EXPORT_MAX_AGE = 3600.0
MAX_SCAN_ROUNDS = 10
"""pages of rows :meth:`GalleryIndex.list` reads past missing files before it
returns a short page (with a cursor to go on)"""

# Output folder options per category (``modules.shared_options``), most specific first
CATEGORY_FOLDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grids", ("outdir_grids", "outdir_txt2img_grids", "outdir_img2img_grids")),
    ("saved", ("outdir_save",)),
    ("extras", ("outdir_extras_samples",)),
    ("img2img", ("outdir_img2img_samples",)),
    ("txt2img", ("outdir_txt2img_samples", "outdir_hires_samples")),
)

_MODEL_RE = re.compile(r'(?:^|,\s*)Model:\s*("(?:\\.|[^"\\])*"|[^,\n]*)')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gallery_meta (
    file_id TEXT PRIMARY KEY,
    model TEXT,
    category TEXT NOT NULL,
    starred INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS gallery_meta_model ON gallery_meta(model);
"""

_COLUMNS = "f.id, f.kind, f.path, f.name, f.mime, f.size, f.width, f.height, f.created, f.job_id, f.tab, f.infotext"


def model_of(infotext: str | None) -> str | None:
    """The ``Model`` entry of an infotext's parameter line."""

    if not infotext:
        return None
    last = infotext.strip().rsplit("\n", 1)[-1]
    match = _MODEL_RE.search(last)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value[:200] or None


def output_folders() -> list[tuple[str, str]]:
    """``(category, absolute folder)`` of the configured output folders."""

    try:
        from modules import shared

        opts = shared.opts
    except Exception:
        return []
    found = []
    for category, names in CATEGORY_FOLDERS:
        for name in names:
            value = getattr(opts, name, "") or ""
            if value:
                found.append((category, os.path.realpath(os.path.abspath(value))))
    return found


def category_of(record_path: str, tab: str | None, folders: Iterable[tuple[str, str]]) -> str:
    path = os.path.realpath(record_path)
    for category, folder in folders:
        if path == folder or path.startswith(folder.rstrip(os.sep) + os.sep):
            if category in ("grids", "saved") or tab is None:
                return category
            break
    return tab if tab in ("txt2img", "img2img", "extras") else "other"


def _like(query: str | None) -> list[str]:
    terms = [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in (query or "").split() if term]
    return [f"%{term}%" for term in terms[:MAX_QUERY_TERMS]]


@dataclass(frozen=True)
class GalleryItem:
    record: FileRecord
    model: str | None
    category: str
    starred: bool


@dataclass(frozen=True)
class GalleryQuery:
    query: str | None = None
    category: str | None = None
    model: str | None = None
    since: float | None = None
    until: float | None = None
    starred: bool | None = None
    sort: Sort = "newest"


class GalleryIndex:
    def __init__(self, files: FileStore, *, folders: Callable[[], list[tuple[str, str]]] = output_folders) -> None:
        self.files = files
        self.folders = folders
        self._lock = threading.RLock()
        self._db = connect(files.root / "files.db")
        self._db.executescript(_SCHEMA)
        self._synced: int | None = None
        """``PRAGMA data_version`` at the last complete :meth:`sync` (it
        changes when another connection — the file store's — commits)"""

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # region Index

    def sync(self, batch: int = 2000, *, force: bool = False) -> int:
        """Fill ``gallery_meta`` for outputs that have no row yet; a no-op
        while the file index is unchanged since the last sync."""

        with self._lock:
            version = self._db.execute("PRAGMA data_version").fetchone()[0]
            if not force and version == self._synced:
                return 0
        folders = self.folders()
        added = 0
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT f.id, f.path, f.tab, f.infotext FROM files f LEFT JOIN gallery_meta g ON g.file_id = f.id WHERE f.kind = 'output' AND g.file_id IS NULL LIMIT ?",
                    (batch,),
                ).fetchall()
                if rows:
                    self._db.executemany(
                        "INSERT OR IGNORE INTO gallery_meta (file_id, model, category) VALUES (?, ?, ?)",
                        [(file_id, model_of(infotext), category_of(path, tab, folders)) for file_id, path, tab, infotext in rows],
                    )
                if len(rows) < batch:
                    self._synced = version
                    return added + len(rows)
            added += len(rows)

    def _prune_meta(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM gallery_meta WHERE file_id NOT IN (SELECT id FROM files WHERE kind = 'output')")

    def _forget(self, ids: list[str]) -> None:
        """Drop the index rows (and thumbnails) of deleted files; unlike
        :meth:`FileStore.delete`, the paths are not marked removed, so the
        folder scan finds them again should they come back."""

        with self._lock:
            self._db.executemany("DELETE FROM files WHERE id = ? AND kind = 'output'", [(file_id,) for file_id in ids])
            self._db.executemany("DELETE FROM gallery_meta WHERE file_id = ?", [(file_id,) for file_id in ids])
        for file_id in ids:
            for thumb in self.files.thumbs_dir.glob(f"{file_id}-*.webp"):
                thumb.unlink(missing_ok=True)

    # endregion

    # region Queries

    def _filters(self, q: GalleryQuery) -> tuple[list[str], list[Any]]:
        where, args = ["f.kind = 'output'"], []
        if q.category:
            where.append("g.category = ?")
            args.append(q.category)
        if q.model:
            where.append("g.model = ?")
            args.append(q.model)
        if q.since is not None:
            where.append("f.created >= ?")
            args.append(float(q.since))
        if q.until is not None:
            where.append("f.created < ?")
            args.append(float(q.until))
        if q.starred is not None:
            where.append("g.starred = ?")
            args.append(int(q.starred))
        for pattern in _like(q.query):
            where.append("(f.infotext LIKE ? ESCAPE '\\' OR f.name LIKE ? ESCAPE '\\' OR g.model LIKE ? ESCAPE '\\')")
            args += [pattern] * 3
        return where, args

    def list(self, q: GalleryQuery = GalleryQuery(), *, cursor: str | None = None, limit: int = 100) -> tuple[list[GalleryItem], str | None, int | None]:
        """A page: ``(items, next_cursor, total)``; ``total`` only for the
        first page (no ``cursor``). Outputs whose file cannot be found are
        left out (see the module docstring)."""

        self.sync()
        sort = q.sort if q.sort in SORTS else "newest"
        key, direction = {"newest": ("f.created", "DESC"), "oldest": ("f.created", "ASC"), "name": ("f.name", "ASC"), "largest": ("f.width * f.height", "DESC")}[sort]
        where, args = self._filters(q)
        op = "<" if direction == "DESC" else ">"
        after = _position(cursor, text=sort == "name")
        limit = min(max(int(limit), 1), 200)
        total: int | None = None
        if cursor is None:
            with self._lock:
                total = self._db.execute(f"SELECT COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE {' AND '.join(where)}", args).fetchone()[0]

        items: list[GalleryItem] = []
        deleted: list[str] = []
        last: tuple[Any, int] | None = None
        more = False
        for _round in range(MAX_SCAN_ROUNDS):
            page_where, page_args = list(where), list(args)
            if after is not None:
                page_where.append(f"(COALESCE({key}, 0) {op} ? OR (COALESCE({key}, 0) = ? AND f.rowid {op} ?))")
                page_args += [after[0], after[0], after[1]]
            sql = (
                f"SELECT {_COLUMNS}, g.model, g.category, g.starred, COALESCE({key}, 0), f.rowid FROM files f JOIN gallery_meta g ON g.file_id = f.id "
                f"WHERE {' AND '.join(page_where)} ORDER BY COALESCE({key}, 0) {direction}, f.rowid {direction} LIMIT ?"
            )
            with self._lock:
                rows = self._db.execute(sql, [*page_args, limit + 1]).fetchall()
            more = False
            for index, row in enumerate(rows):
                if len(items) == limit:
                    more = True  # a row past the last one returned
                    break
                if index == limit:  # the probe row of a batch that came up short: read on from it
                    more = True
                    break
                last = (row[-2], row[-1])
                record = FileRecord(*row[:12])
                if os.path.isfile(record.path):
                    items.append(GalleryItem(record, row[12], row[13], bool(row[14])))
                elif os.path.isdir(os.path.dirname(record.path)):
                    deleted.append(record.id)
            if len(items) == limit or not more:
                break
            after = last
        else:
            more = True  # gave up reading past missing files: the client goes on with the cursor
        if deleted:
            self._forget(deleted)
            if total is not None:
                total = max(0, total - len(deleted))
        next_cursor = _cursor(*last) if more and last is not None else None
        return items, next_cursor, total

    def get(self, file_id: str) -> GalleryItem | None:
        self.sync()
        with self._lock:
            row = self._db.execute(
                f"SELECT {_COLUMNS}, g.model, g.category, g.starred FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE f.id = ? AND f.kind = 'output'",
                (file_id,),
            ).fetchone()
        if row is None or not os.path.isfile(row[2]):
            return None
        return GalleryItem(FileRecord(*row[:12]), row[12], row[13], bool(row[14]))

    def facets(self) -> dict[str, Any]:
        """Filter values with counts: models, categories, date bounds, stars."""

        self.sync()
        with self._lock:
            models = self._db.execute(
                "SELECT g.model, COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE f.kind = 'output' AND g.model IS NOT NULL GROUP BY g.model ORDER BY COUNT(*) DESC, g.model LIMIT 500"
            ).fetchall()
            categories = dict(self._db.execute("SELECT g.category, COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE f.kind = 'output' GROUP BY g.category").fetchall())
            oldest, newest, total, starred = self._db.execute(
                "SELECT MIN(f.created), MAX(f.created), COUNT(*), COALESCE(SUM(g.starred), 0) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE f.kind = 'output'"
            ).fetchone()
        return {
            "models": [{"name": name, "count": count} for name, count in models],
            "categories": [{"name": name, "count": categories.get(name, 0)} for name in CATEGORIES],
            "oldest": oldest,
            "newest": newest,
            "total": total,
            "starred": starred,
        }

    # endregion

    # region Changes

    def set_starred(self, ids: list[str], starred: bool) -> int:
        self.sync()
        ids = list(dict.fromkeys(ids))[:MAX_ZIP_FILES]
        if not ids:
            return 0
        with self._lock:
            return self._db.execute(f"UPDATE gallery_meta SET starred = ? WHERE file_id IN ({','.join('?' * len(ids))})", (int(starred), *ids)).rowcount

    def delete(self, ids: list[str], *, delete_files: bool = False) -> int:
        """Remove outputs from the gallery (and from disk with ``delete_files``)."""

        removed = 0
        for file_id in dict.fromkeys(ids):
            record = self.files.get(file_id)
            if record is None or record.kind != "output":
                continue
            if self.files.delete(file_id, delete_file=delete_files):
                removed += 1
        if removed:
            self._prune_meta()
        return removed

    def prune_exports(self, max_age: float = EXPORT_MAX_AGE, exports_dir: Path | None = None) -> int:
        """Delete zips (and leftovers of failed ones) older than ``max_age``
        seconds, and the file index rows of exports that are gone or old."""

        exports = exports_dir or self.files.root / "exports"
        cutoff = time.time() - max_age
        removed = 0
        if exports.is_dir():
            for old in [*exports.glob("*.zip"), *exports.glob("*.tmp")]:
                try:
                    if old.stat().st_mtime < cutoff:
                        old.unlink()
                        removed += 1
                except OSError:
                    pass
        with self._lock:
            rows = self._db.execute("SELECT id, path, created FROM files WHERE kind = 'export'").fetchall()
        for file_id, path, created in rows:
            if created < cutoff or not os.path.isfile(path):
                self.files.delete(file_id)  # an export: only the row
                Path(path).unlink(missing_ok=True)
        return removed

    def export_zip(self, ids: list[str], exports_dir: Path | None = None) -> tuple[Path, int]:
        """Pack the outputs ``ids`` (originals, as stored) into a new zip
        under ``exports/``; files gone from disk are skipped. Exports older
        than an hour are removed first (and by the hourly housekeeping)."""

        exports = exports_dir or self.files.root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        self.prune_exports(exports_dir=exports)
        records = [record for record in (self.files.get(file_id) for file_id in dict.fromkeys(ids[:MAX_ZIP_FILES])) if record is not None and record.kind == "output"]
        if not records:
            raise ValueError("nothing to pack")
        path = exports / f"gallery-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}.zip"
        used: set[str] = set()
        temp = path.with_suffix(".tmp")
        packed = 0
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for record in records:
                    try:
                        archive.write(record.path, _unique(record.name, used))
                    except (FileNotFoundError, PermissionError, IsADirectoryError):
                        continue  # deleted (or locked) since: raised before the entry is written
                    packed += 1
            if not packed:
                raise ValueError("nothing to pack")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path, packed

    # endregion


def _cursor(value: Any, rowid: int) -> str:
    """A page cursor: the sort value and rowid of the last row read (JSON:
    file names may contain any character)."""

    return base64.urlsafe_b64encode(json.dumps([value, rowid]).encode()).decode().rstrip("=")


def _position(cursor: str | None, *, text: bool) -> tuple[Any, int] | None:
    if not cursor:
        return None
    try:
        value, rowid = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (ValueError, TypeError, binascii.Error):
        return None
    if not isinstance(rowid, int) or isinstance(value, bool) or not isinstance(value, str if text else (int, float)):
        return None
    return value, rowid


def _unique(name: str, used: set[str]) -> str:
    base, ext = os.path.splitext(name or "image")
    candidate, index = base + ext, 1
    while candidate.lower() in used:
        candidate = f"{base}-{index}{ext}"
        index += 1
    used.add(candidate.lower())
    return candidate


_indexes: weakref.WeakKeyDictionary[FileStore, GalleryIndex] = weakref.WeakKeyDictionary()
_indexes_lock = threading.Lock()


def gallery_for(files: FileStore) -> GalleryIndex:
    """One index per file store (process-wide, like the services)."""

    with _indexes_lock:
        index = _indexes.get(files)
        if index is None:
            index = _indexes[files] = GalleryIndex(files)
        return index


def item_json(item: GalleryItem, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    record = item.record
    return {
        "id": record.id,
        "name": record.name,
        "width": record.width,
        "height": record.height,
        "size": record.size,
        "mime": record.mime,
        "created_at": iso(record.created),
        "job_id": record.job_id,
        "tab": record.tab,
        "category": item.category,
        "model": item.model,
        "starred": item.starred,
        "infotext": record.infotext,
    }


def reveal(path: str) -> None:
    """Show ``path`` in the file manager of the machine running the WebUI
    (selected where the platform can)."""

    import platform
    import subprocess

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    system = platform.system()
    if system == "Windows":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif system == "Darwin":
        subprocess.Popen(["open", "-R", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        windows = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
        subprocess.Popen(["explorer.exe", "/select,", windows])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


__all__ = ["CATEGORIES", "SORTS", "GalleryIndex", "GalleryItem", "GalleryQuery", "category_of", "gallery_for", "item_json", "model_of", "reveal"]
