"""Pinned and checksum-verified artifacts used by Anima modulation guidance."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.request


_DOWNLOAD_LOCK = threading.Lock()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pinned_download(*, path: str, url: str, sha256: str, maximum_bytes: int) -> str:
    """Return a verified artifact, downloading atomically when necessary."""
    path = os.path.abspath(path)
    with _DOWNLOAD_LOCK:
        if os.path.isfile(path) and sha256_file(path) == sha256:
            return path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="anima-mod-", suffix=".part", dir=os.path.dirname(path))
        try:
            digest = hashlib.sha256()
            total = 0
            request = urllib.request.Request(url, headers={"User-Agent": "Forge-Neo-Anima-Mod-Guidance/1"})
            with urllib.request.urlopen(request, timeout=180) as response, os.fdopen(descriptor, "wb") as output:
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
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return path
