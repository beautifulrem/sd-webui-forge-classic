"""Pinned, checksum-verified downloads for Anima features (modulation guidance
weights, the CNS gamma table)."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.parse
import urllib.request

from modules.anima_user_data import atomic_replace


# One lock per target file: a long download or first hash of one artifact
# must not block callers that need another, already verified one.
_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}
# (path, mtime, size) -> sha256 of files already hashed: guidance asks for its
# artifacts (~430 MB) on every sampling pass; re-hash only changed files.
_VERIFIED: dict[tuple, str] = {}


def file_version(path: str) -> tuple:
    """``(path, mtime_ns, size)``: changes whenever the file is replaced."""

    stat = os.stat(path)
    return path, stat.st_mtime_ns, stat.st_size


def _cached_sha256(path: str) -> str:
    version = file_version(path)
    digest = _VERIFIED.get(version)
    if digest is None:
        digest = sha256_file(path)
        _VERIFIED[version] = digest
    return digest


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to ``https://`` URLs: urllib's own handler also
    follows them to plain ``http://`` and ``ftp://``."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            raise urllib.error.HTTPError(newurl, code, f"refusing a redirect to a non-https URL: {newurl!r}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request, timeout: float):
    return urllib.request.build_opener(_HttpsOnlyRedirectHandler).open(request, timeout=timeout)


def ensure_pinned_download(*, path: str, url: str, sha256: str, maximum_bytes: int, timeout: float = 180, user_agent: str = "Forge-Neo-Anima-Mod-Guidance/1") -> str:
    """Return a verified artifact, downloading atomically when necessary.

    Only ``https://`` URLs, and only redirects to ``https://`` URLs: ``urlopen``
    would also read ``file://`` / ``ftp://`` and plain ``http://``, and follow
    redirects to ``http://`` / ``ftp://``. A refused redirect raises
    ``urllib.error.HTTPError`` before its target is requested."""
    if urllib.parse.urlsplit(url).scheme.lower() != "https":
        raise ValueError(f"refusing to download from a non-https URL: {url!r}")
    path = os.path.abspath(path)
    with _LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.setdefault(path, threading.Lock())
    with lock:
        if os.path.isfile(path) and _cached_sha256(path) == sha256:
            return path

        def download(output):
            digest = hashlib.sha256()
            total = 0
            request = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310 - https only, checked above
            with _open(request, timeout) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise RuntimeError(f"download exceeded the {maximum_bytes} byte safety limit")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != sha256:
                raise RuntimeError("downloaded artifact failed its SHA-256 check")

        # Also on Ctrl+C / shutdown: no leaked descriptor or temporary file.
        atomic_replace(path, download)
        _VERIFIED[file_version(path)] = sha256  # hashed while downloading
        return path
