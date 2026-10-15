"""Everything PNG Info shows about an image: the generation parameters
(``images.read_info_from_image``: PNG text, EXIF UserComment, GIF comment,
stealth info, NovelAI), the parsed parameter line, the image's format and
size, its other text entries, every EXIF tag and the list of PNG chunks.

Values are strings for display; binary values become ``<N bytes>``. The
answer is bounded: at most :data:`MAX_INFO_ENTRIES` text entries of at most
:data:`MAX_VALUE_CHARS` each, and :data:`MAX_TOTAL_CHARS` in all (PIL alone
lets a PNG carry about 64 MB of text).
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from typing import Any

MAX_VALUE_CHARS = 20_000
MAX_CHUNKS = 400
MAX_EXIF_TAGS = 400
MAX_INFO_ENTRIES = 200
MAX_TOTAL_CHARS = 1_000_000
MAX_INFOTEXT_CHARS = 200_000


def _text(value: Any) -> tuple[str, str]:
    """``(display text, kind)``; kind ``text``, ``binary`` or ``number``."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{len(data)} bytes>", "binary"
        if decoded.isprintable() or all(char.isprintable() or char in "\n\r\t" for char in decoded):
            return decoded[:MAX_VALUE_CHARS], "text"
        return f"<{len(data)} bytes>", "binary"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), "number"
    if isinstance(value, tuple) and all(isinstance(item, (int, float)) for item in value):
        return " × ".join(f"{item:g}" if isinstance(item, float) else str(item) for item in value), "number"
    text = str(value)
    return text[:MAX_VALUE_CHARS], "text"


def png_chunks(path: str) -> list[dict[str, Any]]:
    """The chunks of a PNG file in order (consecutive ``IDAT`` chunks merged):
    ``{type, length, count, key?}``; text chunks carry their keyword."""

    chunks: list[dict[str, Any]] = []
    try:
        with open(path, "rb") as file:
            if file.read(8) != b"\x89PNG\r\n\x1a\n":
                return []
            for _ in range(100_000):
                header = file.read(8)
                if len(header) < 8:
                    break
                length, kind = struct.unpack(">I4s", header)
                name = kind.decode("latin-1")
                entry: dict[str, Any] = {"type": name, "length": length, "count": 1}
                if name in ("tEXt", "iTXt", "zTXt") and length:
                    head = file.read(min(length, 80))
                    entry["key"] = head.split(b"\x00", 1)[0].decode("latin-1")
                    file.seek(length - len(head) + 4, os.SEEK_CUR)
                else:
                    file.seek(length + 4, os.SEEK_CUR)
                last = chunks[-1] if chunks else None
                if last is not None and last["type"] == name and name in ("IDAT", "fdAT"):
                    last["length"] += length
                    last["count"] += 1
                else:
                    chunks.append(entry)
                if name == "IEND" or len(chunks) >= MAX_CHUNKS:
                    break
    except OSError:
        return chunks
    return chunks


def _exif_entries(image: Any) -> Iterator[dict[str, str]]:
    from PIL import ExifTags

    try:
        exif = image.getexif()
    except Exception:
        return
    seen = 0
    groups = [("", dict(exif))]
    for ifd, label in ((ExifTags.IFD.Exif, "Exif"), (ExifTags.IFD.GPSInfo, "GPS"), (ExifTags.IFD.Interop, "Interop")):
        try:
            nested = exif.get_ifd(ifd)
        except Exception:
            nested = {}
        if nested:
            groups.append((label, dict(nested)))
    for label, tags in groups:
        names = ExifTags.GPSTAGS if label == "GPS" else ExifTags.TAGS
        for tag, value in tags.items():
            if label == "" and tag in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo, ExifTags.IFD.Interop):
                continue
            name = names.get(tag, f"0x{tag:04x}")
            if name == "UserComment" and isinstance(value, (bytes, bytearray)):
                value = _user_comment(bytes(value))
            text, kind = _text(value)
            yield {"group": f"EXIF {label}".strip(), "key": name, "value": text, "kind": kind}
            seen += 1
            if seen >= MAX_EXIF_TAGS:
                return


def _user_comment(data: bytes) -> str:
    prefix, body = data[:8], data[8:]
    if prefix.startswith(b"UNICODE"):
        for encoding in ("utf-16-be", "utf-16-le"):
            try:
                text = body.decode(encoding)
            except UnicodeDecodeError:
                continue
            if (text and text.isprintable()) or "\n" in text:
                return text
    return body.decode("utf-8", errors="replace") if prefix.startswith((b"ASCII", b"\x00" * 8)) else data.decode("utf-8", errors="replace")


def read_metadata(path: str, name: str | None = None) -> dict[str, Any]:
    """PNG Info of the image file at ``path``."""

    from PIL import Image

    from modules import images

    size = os.path.getsize(path)
    with Image.open(path) as image:
        infotext, items = images.read_info_from_image(image)
        info = dict(image.info or {})
        entries: list[dict[str, str]] = []
        basics = [
            ("Format", image.format or "?"),
            ("Mode", image.mode),
            ("Size", f"{image.width} × {image.height}"),
            ("File size", f"{size:,} bytes"),
        ]
        frames = getattr(image, "n_frames", 1)
        if frames and frames > 1:
            basics.append(("Frames", str(frames)))
        for key, value in basics:
            entries.append({"group": "Image", "key": key, "value": value, "kind": "text"})
        budget = MAX_TOTAL_CHARS
        own = [(key, value) for key, value in info.items() if key not in ("exif", "parameters")]
        derived = [(key, value) for key, value in items.items() if key not in info]  # stealth info and other derived entries
        for key, value in [*own, *derived]:
            if len(entries) - len(basics) >= MAX_INFO_ENTRIES or budget <= 0:
                break
            if key == "icc_profile":
                entries.append({"group": "Metadata", "key": "ICC profile", "value": f"<{len(value or b'')} bytes>", "kind": "binary"})
                continue
            text, kind = _text(value)
            text = text[:budget]
            budget -= len(text)
            entries.append({"group": "Metadata", "key": str(key)[:200], "value": text, "kind": kind})
        if isinstance(infotext, str) and len(infotext) > MAX_INFOTEXT_CHARS:
            infotext = infotext[:MAX_INFOTEXT_CHARS]
        entries.extend(_exif_entries(image))
        result = {
            "name": name or os.path.basename(path),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "size": size,
            "infotext": infotext,
        }
    params: dict[str, str] = {}
    if infotext:
        from modules import infotext_utils

        try:
            parsed = infotext_utils.parse_generation_parameters(infotext, [])
        except Exception:
            parsed = {}
        params = {str(key): _text(value)[0] for key, value in parsed.items() if value is not None and not str(key).endswith(("-1", "-2"))}
    result["params"] = params
    result["entries"] = entries
    result["chunks"] = png_chunks(path) if (result["format"] or "").upper() == "PNG" else []
    return result
