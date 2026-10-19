"""Files the WebUI serves by opaque id only: uploads, generation outputs and
extra-network previews (``/api/v2/files/{id}``, ``/api/v2/thumbs/{id}``).

The browser never sees a filesystem path: every file is registered in a
small SQLite index (``files.db`` under the Remi data directory) and
addressed by an id. Uploads are sniffed by content (not by the declared type
or name), size-limited, deduplicated by SHA-256 and stored under
``uploads/``. Outputs keep their infotext for the gallery search.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
THUMB_WIDTHS = (64, 128, 192, 256, 320, 384, 512, 768, 1024)
DEFAULT_MAX_UPLOAD = 64 * 1024 * 1024
MAX_UPLOAD_PIXELS = 2 * 89_478_485
"""Pillow's own decompression-bomb limit, enforced here even when something
else lifted ``Image.MAX_IMAGE_PIXELS`` for the process (X/Y/Z plot does)"""
MAX_UPLOAD_FRAMES = 2_000
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif", ".jxl", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
TEXT_EXTENSIONS = (".txt", ".csv", ".json", ".yaml", ".yml", ".cube")  # .cube: 3D LUTs (Post Processing Suite)


class FileError(ValueError):
    """A rejected upload; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class FileRecord:
    id: str
    kind: str
    """``upload``, ``output`` or ``network`` (a preview image)"""
    path: str
    name: str
    mime: str
    size: int
    width: int | None
    height: int | None
    created: float
    job_id: str | None = None
    tab: str | None = None
    infotext: str | None = None

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")


@dataclass(frozen=True)
class Sniffed:
    kind: str  # "image" | "video" | "text" | "model"
    mime: str
    ext: str


def sniff(head: bytes, name: str = "") -> Sniffed | None:
    """The content type of a file from its first bytes (``name`` only
    disambiguates plain text). None: not a type the WebUI accepts."""

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniffed("image", "image/png", ".png")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniffed("image", "image/jpeg", ".jpg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Sniffed("image", "image/gif", ".gif")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Sniffed("image", "image/webp", ".webp")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return Sniffed("video", "video/x-msvideo", ".avi")
    if head.startswith(b"BM"):
        return Sniffed("image", "image/bmp", ".bmp")
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Sniffed("image", "image/tiff", ".tif")
    if head.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")):
        return Sniffed("image", "image/jxl", ".jxl")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return Sniffed("image", "image/avif", ".avif")
        if brand in (b"heic", b"heix", b"mif1", b"msf1"):
            return Sniffed("image", "image/heic", ".heic")
        if brand == b"qt  ":
            return Sniffed("video", "video/quicktime", ".mov")
        return Sniffed("video", "video/mp4", ".mp4")
    if head.startswith(b"\x1aE\xdf\xa3"):
        return Sniffed("video", "video/webm", ".webm")
    if len(head) >= 9 and head[8:9] == b"{" and int.from_bytes(head[:8], "little") < 100 * 1024 * 1024:
        return Sniffed("model", "application/octet-stream", ".safetensors")
    ext = os.path.splitext(name)[1].lower()
    if ext in TEXT_EXTENSIONS and b"\x00" not in head:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.start < len(head) - 4:  # not just a character cut at the end
                return None
        return Sniffed("text", "application/json" if ext == ".json" else "text/plain", ext)
    return None


def clean_name(name: str | None) -> str:
    """A display file name: the last path component, without control
    characters, at most 200 characters."""

    base = re.split(r"[\\/]", name or "")[-1]
    base = "".join(char for char in base if char.isprintable()).strip()
    return base[-200:] or "upload"


def encode_cursor(*parts: Any) -> str:
    return base64.urlsafe_b64encode(":".join(str(part) for part in parts).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None, count: int) -> list[str] | None:
    if not cursor:
        return None
    try:
        parts = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode().split(":")
    except (ValueError, UnicodeDecodeError):
        return None
    return parts if len(parts) == count else None


def _like_terms(query: str | None) -> list[str]:
    return [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in (query or "").split() if term][:8]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    created REAL NOT NULL,
    sha256 TEXT,
    job_id TEXT,
    tab TEXT,
    infotext TEXT
);
CREATE INDEX IF NOT EXISTS files_kind_created ON files(kind, created DESC);
CREATE UNIQUE INDEX IF NOT EXISTS files_kind_path ON files(kind, path);
CREATE INDEX IF NOT EXISTS files_sha ON files(sha256);
CREATE TABLE IF NOT EXISTS removed (path TEXT PRIMARY KEY);
"""
# ``removed``: outputs taken out of the gallery but kept on disk (the output
# folder scan skips them)

_COLUMNS = "id, kind, path, name, mime, size, width, height, created, job_id, tab, infotext"


def connect(path: Path) -> sqlite3.Connection:
    """A WAL-mode SQLite connection shared between threads (callers lock)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


class FileStore:
    """The id → file index plus the upload, output and thumbnail folders
    under ``root``. ``max_upload_bytes`` may be a callable (a live setting).
    ``outputs/`` holds images the save path did not write (saving off); they
    are temporary, see :meth:`prune_unsaved`."""

    def __init__(self, root: str | os.PathLike, *, max_upload_bytes: int | Callable[[], int] = DEFAULT_MAX_UPLOAD) -> None:
        self.root = Path(root)
        self.uploads_dir = self.root / "uploads"
        self.outputs_dir = self.root / "outputs"
        self.thumbs_dir = self.root / "thumbs"
        self._max_upload = max_upload_bytes
        self._lock = threading.RLock()
        self._db = connect(self.root / "files.db")
        self._db.executescript(_SCHEMA)

    @property
    def max_upload_bytes(self) -> int:
        limit = self._max_upload
        return int(limit() if callable(limit) else limit)

    @max_upload_bytes.setter
    def max_upload_bytes(self, value: int | Callable[[], int]) -> None:
        self._max_upload = value

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # region Records

    def _record(self, row: tuple | None) -> FileRecord | None:
        return FileRecord(*row) if row else None

    def get(self, file_id: str) -> FileRecord | None:
        """The record of ``file_id`` when its file still exists."""

        if not isinstance(file_id, str) or not ID_RE.match(file_id):
            return None
        with self._lock:
            record = self._record(self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE id = ?", (file_id,)).fetchone())
        if record is None or not os.path.isfile(record.path):
            return None
        return record

    def _insert(self, record: FileRecord, sha256: str | None = None) -> FileRecord:
        with self._lock:
            self._db.execute(
                f"INSERT INTO files ({_COLUMNS}, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.id, record.kind, record.path, record.name, record.mime, record.size, record.width, record.height, record.created, record.job_id, record.tab, record.infotext, sha256),
            )
        return record

    def _by_path(self, kind: str, path: str) -> FileRecord | None:
        with self._lock:
            return self._record(self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE kind = ? AND path = ?", (kind, path)).fetchone())

    def delete(self, file_id: str, *, delete_file: bool = False) -> bool:
        """Forget a file; uploads and unsaved outputs are deleted, other
        outputs only with ``delete_file`` (else the folder scan skips them
        from now on)."""

        with self._lock:
            row = self._db.execute("SELECT path, kind FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None:
                return False
            self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))
            path, kind = row
            remove = delete_file or kind == "upload" or Path(path).parent == self.outputs_dir
            if kind == "output" and not remove:
                self._db.execute("INSERT OR IGNORE INTO removed (path) VALUES (?)", (path,))
        if remove:
            Path(path).unlink(missing_ok=True)
        for thumb in self.thumbs_dir.glob(f"{file_id}-*.webp"):
            thumb.unlink(missing_ok=True)
        return True

    # endregion

    # region Uploads

    def add_upload(self, chunks: Iterable[bytes], name: str | None = None) -> FileRecord:
        """Store an upload (its bytes, in chunks). Raises :class:`FileError`
        (413 when too large, 415 for a type the WebUI does not accept)."""

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        head = b""
        temp = self.uploads_dir / f".incoming-{secrets.token_hex(8)}"
        try:
            with open(temp, "wb") as file:
                for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise FileError(f"the file is larger than {self.max_upload_bytes // (1024 * 1024)} MB", 413)
                    if len(head) < 64:
                        head += chunk[: 64 - len(head)]
                    digest.update(chunk)
                    file.write(chunk)
            if size == 0:
                raise FileError("the file is empty")
            name = clean_name(name)
            sniffed = sniff(head, name)
            if sniffed is None:
                raise FileError("unsupported file type", 415)
            width = height = None
            if sniffed.kind == "image":
                width, height = _image_size(temp, max_pixels=MAX_UPLOAD_PIXELS)
            sha = digest.hexdigest()
            target = self.uploads_dir / sha[:2] / f"{sha}{sniffed.ext}"
            with self._lock:  # one record per content, even for concurrent uploads
                existing = self._upload_by_sha(sha)
                if existing is not None:
                    return self._touch(existing)  # a re-upload: not old enough to prune
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp, target)
                self._db.execute("DELETE FROM files WHERE kind = 'upload' AND path = ?", (str(target),))  # its file was gone
                record = FileRecord(secrets.token_urlsafe(16), "upload", str(target), name, sniffed.mime, size, width, height, time.time())
                return self._insert(record, sha)
        finally:
            temp.unlink(missing_ok=True)

    def add_upload_bytes(self, data: bytes, name: str | None = None) -> FileRecord:
        return self.add_upload([data], name)

    def _touch(self, record: FileRecord) -> FileRecord:
        now = time.time()
        with self._lock:
            self._db.execute("UPDATE files SET created = ? WHERE id = ?", (now, record.id))
        return dataclasses.replace(record, created=now)

    def _upload_by_sha(self, sha: str) -> FileRecord | None:
        with self._lock:
            record = self._record(self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE kind = 'upload' AND sha256 = ?", (sha,)).fetchone())
        return record if record is not None and os.path.isfile(record.path) else None

    def resolve_upload(self, node: Any, file_id: str) -> Any:
        """``remi.ui`` upload resolver: a PIL image for image inputs (with
        the original name as ``image.filename``), the path for ``File``."""

        from remi.ui import ParamError
        from remi.ui.nodes import File

        record = self.get(file_id)
        if record is None or record.kind not in ("upload", "output"):
            raise ParamError(f"unknown upload {file_id!r}")
        if isinstance(node, File):
            return record.path
        if not record.is_image:
            raise ParamError("the upload is not an image")
        from modules import images

        image = images.read(record.path)
        image.load()
        image.filename = record.name
        return image

    def check_upload(self, node: Any, file_id: str) -> str:
        """Upload resolver for validation only: the id, if it names a file."""

        from remi.ui import ParamError
        from remi.ui.nodes import File

        record = self.get(file_id)
        if record is None or record.kind not in ("upload", "output"):
            raise ParamError(f"unknown upload {file_id!r}")
        if not isinstance(node, File) and not record.is_image:
            raise ParamError("the upload is not an image")
        return file_id

    def prune_uploads(self, max_age: float, keep: set[str] = frozenset()) -> int:
        """Delete uploads older than ``max_age`` seconds, except ``keep``."""

        cutoff = time.time() - max_age
        with self._lock:
            rows = self._db.execute("SELECT id FROM files WHERE kind = 'upload' AND created < ?", (cutoff,)).fetchall()
        removed = 0
        for (file_id,) in rows:
            if file_id not in keep and self.delete(file_id):
                removed += 1
        return removed

    def prune_unsaved(self, max_age: float) -> int:
        """Delete outputs older than ``max_age`` seconds that the save path
        did not write (``outputs/``: generated with saving turned off)."""

        cutoff = time.time() - max_age
        with self._lock:
            rows = self._db.execute("SELECT id, path FROM files WHERE kind = 'output' AND created < ?", (cutoff,)).fetchall()
        return sum(1 for file_id, path in rows if Path(path).parent == self.outputs_dir and self.delete(file_id))

    # endregion

    # region Outputs

    def register_output(self, path: str | os.PathLike, *, job_id: str | None = None, tab: str | None = None, infotext: str | None = None, width: int | None = None, height: int | None = None, created: float | None = None) -> FileRecord:
        """Index a generated file (idempotent per path)."""

        path = os.path.realpath(path)
        with self._lock:
            self._db.execute("DELETE FROM removed WHERE path = ?", (path,))
        existing = self._by_path("output", path)
        if existing is not None:
            return existing
        stat = os.stat(path)
        sniffed = sniff(_read_head(path), path)
        mime = sniffed.mime if sniffed else "application/octet-stream"
        if sniffed and sniffed.kind == "image" and (width is None or height is None):
            try:
                width, height = _image_size(path)
            except FileError:
                pass
        record = FileRecord(secrets.token_urlsafe(16), "output", path, os.path.basename(path), mime, stat.st_size, width, height, created or stat.st_mtime, job_id, tab, infotext)
        try:
            return self._insert(record)
        except sqlite3.IntegrityError:  # registered concurrently
            return self._by_path("output", path) or record

    def save_output_image(self, image: Image.Image, infotext: str | None, **fields: Any) -> FileRecord:
        """Store an image that the normal save path did not write (sample
        saving off) as PNG with its infotext, and index it."""

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        metadata = PngImagePlugin.PngInfo()
        texts = {key: value for key, value in (image.info or {}).items() if isinstance(key, str) and isinstance(value, str)}
        if infotext:
            texts.setdefault("parameters", infotext)
        for key, value in texts.items():
            metadata.add_text(key, value)
        path = self.outputs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}.png"
        image.save(path, pnginfo=metadata if texts else None)
        return self.register_output(path, infotext=infotext, width=image.width, height=image.height, **fields)

    def list_outputs(self, *, cursor: str | None = None, limit: int = 50, query: str | None = None, tab: str | None = None) -> tuple[list[FileRecord], str | None, int]:
        """Newest first: ``(records, next_cursor, total)``; ``query`` terms
        must all appear in the infotext or the file name."""

        where, args = ["kind = 'output'"], []
        if tab:
            where.append("tab = ?")
            args.append(tab)
        for term in _like_terms(query):
            where.append("(infotext LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')")
            args += [f"%{term}%", f"%{term}%"]
        filters = " AND ".join(where)
        page_where, page_args = filters, list(args)
        position = decode_cursor(cursor, 2)
        if position is not None:
            try:
                created, rowid = float(position[0]), int(position[1])
            except ValueError:
                position = None
            else:
                page_where += " AND (created < ? OR (created = ? AND rowid < ?))"
                page_args += [created, created, rowid]
        limit = min(max(int(limit), 1), 200)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM files WHERE {filters}", args).fetchone()[0]
            rows = self._db.execute(f"SELECT {_COLUMNS}, rowid FROM files WHERE {page_where} ORDER BY created DESC, rowid DESC LIMIT ?", [*page_args, limit + 1]).fetchall()
        records = [FileRecord(*row[:-1]) for row in rows[:limit]]
        next_cursor = encode_cursor(rows[limit - 1][8], rows[limit - 1][-1]) if len(rows) > limit else None
        return records, next_cursor, total

    def scan_outputs(self, directories: Iterable[str | os.PathLike], *, limit: int = 50_000) -> int:
        """Index images found in ``directories`` that are not indexed yet
        (outputs from before Remi, or saved by other tools); returns how many."""

        from modules import images

        added = 0
        for path in _walk_images(directories):
            if added >= limit:
                break
            real = os.path.realpath(path)
            if self._by_path("output", real) is not None or self._was_removed(real):
                continue
            infotext = width = height = None
            try:
                with Image.open(real) as image:
                    width, height = image.size
                    infotext, _ = images.read_info_from_image(image)
            except Exception:
                continue
            self.register_output(real, infotext=infotext, width=width, height=height)
            added += 1
        return added

    def _was_removed(self, path: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM removed WHERE path = ?", (path,)).fetchone() is not None

    def is_removed(self, path: str | os.PathLike) -> bool:
        """``path`` was taken out of the gallery (kept on disk)."""

        return self._was_removed(os.path.realpath(path))

    # endregion

    # region Network previews

    def register_path(self, path: str | os.PathLike, kind: str = "network") -> FileRecord | None:
        """Index an existing file under an id stable while the file is
        unchanged (path, size and mtime: a changed file gets a new id, so
        cached copies are not reused); None when it is gone."""

        path = os.path.realpath(path)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        existing = self._by_path(kind, path)
        if existing is not None:
            if existing.size == stat.st_size and existing.created == stat.st_mtime:
                return existing
            self.delete(existing.id)
        sniffed = sniff(_read_head(path), path)
        record = FileRecord(_stable_id(kind, path, stat.st_size, stat.st_mtime_ns), kind, path, os.path.basename(path), sniffed.mime if sniffed else "application/octet-stream", stat.st_size, None, None, stat.st_mtime)
        try:
            return self._insert(record)
        except sqlite3.IntegrityError:
            return self._by_path(kind, path)

    # endregion

    # region Thumbnails

    def thumbnail(self, record: FileRecord, width: int | None) -> Path:
        """A cached WebP thumbnail ``width`` wide (snapped to a fixed set)."""

        if not record.is_image:
            raise FileError("not an image", 415)
        width = snap_width(width)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        target = self.thumbs_dir / f"{record.id}-{width}.webp"
        source_mtime = os.path.getmtime(record.path)
        if target.is_file() and target.stat().st_mtime >= source_mtime:
            return target
        with Image.open(record.path) as image:
            if getattr(image, "is_animated", False):
                image.seek(0)
            image.draft("RGB", (width, width * 4))
            image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
            if image.width > width:
                image.thumbnail((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
            temp = target.with_suffix(f".{secrets.token_hex(4)}.tmp")
            image.save(temp, format="WEBP", quality=82, method=4)
        os.replace(temp, target)
        return target

    # endregion


def snap_width(width: int | None) -> int:
    if not width or width <= 0:
        return 256
    return next((size for size in THUMB_WIDTHS if size >= width), THUMB_WIDTHS[-1])


def _stable_id(kind: str, path: str, size: int, mtime_ns: int) -> str:
    return "n" + base64.urlsafe_b64encode(hashlib.sha256(f"{kind}\0{path}\0{size}\0{mtime_ns}".encode()).digest()[:18]).decode()


def _read_head(path: str, size: int = 64) -> bytes:
    try:
        with open(path, "rb") as file:
            return file.read(size)
    except OSError:
        return b""


def _image_size(path: str | os.PathLike, *, max_pixels: int | None = None) -> tuple[int, int]:
    """Width and height from the header (nothing is decoded). ``max_pixels``:
    refuse larger images (and animations of more frames than
    :data:`MAX_UPLOAD_FRAMES`) whatever ``Image.MAX_IMAGE_PIXELS`` is."""

    try:
        with Image.open(path) as image:  # raises DecompressionBombError past Pillow's pixel limit
            width, height = image.size
            if max_pixels is not None and (width * height > max_pixels or getattr(image, "n_frames", 1) > MAX_UPLOAD_FRAMES):
                raise Image.DecompressionBombError(f"{width}x{height}")
            return width, height
    except Image.DecompressionBombError as error:
        raise FileError("the image is too large", 413) from error
    except Exception as error:
        # Pillow's message names the file (a path in the data folder): logged,
        # never sent to the client
        log.debug("not a readable image: %s", error)
        raise FileError("not a readable image", 415) from error


def _walk_images(directories: Iterable[str | os.PathLike]) -> Iterator[str]:
    seen: set[str] = set()
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        real = os.path.realpath(directory)
        if real in seen:
            continue
        seen.add(real)
        for current, _dirs, files in os.walk(real):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    yield os.path.join(current, name)
