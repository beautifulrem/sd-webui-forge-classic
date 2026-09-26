"""Pinned, checksum-verified downloads for Anima features (modulation guidance
weights, the CNS gamma table)."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.request


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


def ensure_pinned_download(*, path: str, url: str, sha256: str, maximum_bytes: int, timeout: float = 180, user_agent: str = "Forge-Neo-Anima-Mod-Guidance/1") -> str:
    """Return a verified artifact, downloading atomically when necessary."""
    path = os.path.abspath(path)
    with _LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.setdefault(path, threading.Lock())
    with lock:
        if os.path.isfile(path) and _cached_sha256(path) == sha256:
            return path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="anima-download-", suffix=".part", dir=os.path.dirname(path))
        try:
            digest = hashlib.sha256()
            total = 0
            request = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise RuntimeError(f"download exceeded the {maximum_bytes} byte safety limit")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != sha256:
                raise RuntimeError("downloaded artifact failed its SHA-256 check")
            os.replace(temporary, path)
            _VERIFIED[file_version(path)] = sha256  # hashed while downloading
        except BaseException:  # also Ctrl+C / shutdown: no leaked descriptor or .part file
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return path
