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

    monkeypatch.setattr(anima_artifacts, "_open", urlopen)
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
    monkeypatch.setattr(anima_artifacts, "_open", lambda *args, **kwargs: opened.append(args))
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
    monkeypatch.setattr(anima_artifacts, "_open", urlopen)
    target = tmp_path / "artifact.bin"

    with pytest.raises(_Interrupted):
        anima_artifacts.ensure_pinned_download(path=str(target), url="https://unused", sha256="0" * 64, maximum_bytes=100)
    assert _leftovers(tmp_path) == []
    with pytest.raises(OSError):
        os.fstat(descriptors[0])  # closed


@pytest.fixture
def _redirecting(monkeypatch):
    """Serve https URLs through urllib's real opener chain (redirect handling
    included) from ``routes``: url -> payload bytes or ("redirect", location).
    Records every URL the chain opens, over http as well as https."""
    import email.message
    import io
    import urllib.response

    routes, opened = {}, []

    def serve(handler, request):
        opened.append(request.full_url)
        route = routes[request.full_url]
        headers = email.message.Message()
        if isinstance(route, tuple):
            headers["Location"] = route[1]
            response = urllib.response.addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
        else:
            response = urllib.response.addinfourl(io.BytesIO(route), headers, request.full_url, 200)
        response.msg = "Found" if isinstance(route, tuple) else "OK"
        return response

    monkeypatch.setattr(anima_artifacts.urllib.request.HTTPSHandler, "https_open", serve)
    monkeypatch.setattr(anima_artifacts.urllib.request.HTTPHandler, "http_open", serve)
    monkeypatch.setattr(anima_artifacts.urllib.request.FTPHandler, "ftp_open", serve)
    return routes, opened


def test_redirects_are_followed_only_to_https(tmp_path, _redirecting):
    routes, opened = _redirecting
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    routes["https://example.invalid/a.bin"] = ("redirect", "https://cdn.example.invalid/a.bin")
    routes["https://cdn.example.invalid/a.bin"] = payload
    target = tmp_path / "artifact.bin"

    anima_artifacts.ensure_pinned_download(path=str(target), url="https://example.invalid/a.bin", sha256=digest, maximum_bytes=100)
    assert target.read_bytes() == payload
    assert opened == ["https://example.invalid/a.bin", "https://cdn.example.invalid/a.bin"]


@pytest.mark.parametrize("location", ["http://example.invalid/a.bin", "HTTP://example.invalid/a.bin", "ftp://example.invalid/a.bin"])
def test_a_redirect_off_https_is_refused(tmp_path, _redirecting, location):
    import urllib.error

    routes, opened = _redirecting
    routes["https://example.invalid/a.bin"] = ("redirect", location)
    routes[location] = routes[location.lower()] = b"weights"
    target = tmp_path / "artifact.bin"

    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        anima_artifacts.ensure_pinned_download(path=str(target), url="https://example.invalid/a.bin", sha256=hashlib.sha256(b"weights").hexdigest(), maximum_bytes=100)
    assert opened == ["https://example.invalid/a.bin"]  # the redirect target was never requested
    assert not target.exists()
    assert _leftovers(tmp_path) == []
