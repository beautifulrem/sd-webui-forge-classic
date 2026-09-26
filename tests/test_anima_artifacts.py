import hashlib
import threading

import pytest

from modules import anima_artifacts


def test_a_verified_file_is_not_blocked_by_another_download(tmp_path):
    ready = tmp_path / "gamma.npz"
    ready.write_bytes(b"cns")
    busy = tmp_path / "adapter.pt"

    lock = anima_artifacts._LOCKS_GUARD
    with lock:
        busy_lock = anima_artifacts._DOWNLOAD_LOCKS.setdefault(str(busy), threading.Lock())
    busy_lock.acquire()  # another thread is downloading adapter.pt
    try:
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                anima_artifacts.ensure_pinned_download(
                    path=str(ready), url="https://unused", sha256=hashlib.sha256(b"cns").hexdigest(), maximum_bytes=10
                )
            )
        )
        worker.start()
        worker.join(timeout=5)
        assert result == [str(ready)]
    finally:
        busy_lock.release()


@pytest.fixture
def _served(monkeypatch):
    """Payloads served for https URLs; no network."""
    import io

    payloads = {}

    def urlopen(request, timeout):
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(anima_artifacts.urllib.request, "urlopen", urlopen)
    return payloads


def _source(served, payload=b"weights"):
    url = "https://example.invalid/source.bin"
    served[url] = payload
    return url, hashlib.sha256(payload).hexdigest()


def _leftovers(folder):
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".part"))


def test_download_is_verified_and_written_atomically(tmp_path, monkeypatch, _served):
    url, digest = _source(_served)
    target = tmp_path / "cache" / "artifact.bin"

    assert anima_artifacts.ensure_pinned_download(path=str(target), url=url, sha256=digest, maximum_bytes=100) == str(target)
    assert target.read_bytes() == b"weights"
    assert _leftovers(target.parent) == []

    hashed = []
    monkeypatch.setattr(anima_artifacts, "sha256_file", lambda path: hashed.append(path) or digest)
    anima_artifacts.ensure_pinned_download(path=str(target), url="https://unused", sha256=digest, maximum_bytes=100)
    assert hashed == []  # the verified version is remembered, not re-hashed


def test_checksum_mismatch_keeps_the_existing_file_and_cleans_up(tmp_path, _served):
    url, _ = _source(_served)
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="SHA-256"):
        anima_artifacts.ensure_pinned_download(path=str(target), url=url, sha256="0" * 64, maximum_bytes=100)
    assert target.read_bytes() == b"old"
    assert _leftovers(tmp_path) == []


def test_oversized_download_is_aborted(tmp_path, _served):
    url, digest = _source(_served, b"x" * 64)
    target = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="safety limit"):
        anima_artifacts.ensure_pinned_download(path=str(target), url=url, sha256=digest, maximum_bytes=63)
    assert not target.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("url", ["http://example.com/a.bin", "file:///etc/passwd", "ftp://example.com/a.bin", "data:,weights", "/local/a.bin"])
def test_only_https_urls_are_downloaded(tmp_path, monkeypatch, url):
    opened = []
    monkeypatch.setattr(anima_artifacts.urllib.request, "urlopen", lambda *args, **kwargs: opened.append(args))
    target = tmp_path / "artifact.bin"

    with pytest.raises(ValueError, match="non-https"):
        anima_artifacts.ensure_pinned_download(path=str(target), url=url, sha256="0" * 64, maximum_bytes=100)
    assert opened == []
    assert not target.exists()
    assert _leftovers(tmp_path) == []


class _Interrupted(KeyboardInterrupt):
    pass


@pytest.mark.parametrize("stage", ["connect", "read"])
def test_an_interrupted_download_leaks_no_descriptor_or_part_file(tmp_path, monkeypatch, stage):
    """Ctrl+C / shutdown mid-download is a BaseException: a new .part per retry piled up."""
    import os
    import tempfile

    descriptors = []
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            raise _Interrupted

    def urlopen(request, timeout):
        if stage == "connect":
            raise _Interrupted
        return _Response()

    monkeypatch.setattr(anima_artifacts.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(anima_artifacts.urllib.request, "urlopen", urlopen)
    target = tmp_path / "artifact.bin"

    with pytest.raises(_Interrupted):
        anima_artifacts.ensure_pinned_download(path=str(target), url="https://unused", sha256="0" * 64, maximum_bytes=100)
    assert _leftovers(tmp_path) == []
    with pytest.raises(OSError):
        os.fstat(descriptors[0])  # closed
