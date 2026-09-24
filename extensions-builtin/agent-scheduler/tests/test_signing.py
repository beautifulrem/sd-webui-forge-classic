import importlib.util
import os
import pickle
import threading
import zlib
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "agent_scheduler_signing", Path(__file__).parents[1] / "agent_scheduler" / "signing.py"
)


def _load(tmp_path, monkeypatch, name="signing.key"):
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    monkeypatch.setattr(module, "key_file", lambda: str(tmp_path / name))
    monkeypatch.setattr(module, "_stale_key_files", lambda: [str(tmp_path / "old.key")])
    return module


@pytest.fixture()
def signing(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch)


def test_signed_blobs_round_trip_and_key_is_private(signing, tmp_path):
    blob = zlib.compress(pickle.dumps([1, "x"]))

    signed = signing.sign(blob)

    assert signing.verified_payload(signed) == blob
    assert signing.strip_signature(signed) == blob == signing.strip_signature(blob)
    key = tmp_path / "signing.key"
    assert len(key.read_bytes()) == 32
    if os.name != "nt":
        assert key.stat().st_mode & 0o077 == 0


def test_unsigned_tampered_and_foreign_blobs_are_refused(signing, tmp_path, monkeypatch):
    blob = zlib.compress(pickle.dumps([1]))
    signed = signing.sign(blob)
    tampered = signed[:-1] + bytes([signed[-1] ^ 1])
    foreign = _load(tmp_path, monkeypatch, "other.key").sign(blob)  # another install

    for data in (blob, tampered, foreign):
        assert signing.verified_payload(data) is None


def test_keys_of_earlier_versions_are_deleted_not_used(signing, tmp_path):
    old = tmp_path / "old.key"
    old.write_bytes(b"k" * 32)

    signing.sign(b"blob")

    assert not old.exists()
    assert (tmp_path / "signing.key").read_bytes() != b"k" * 32


def test_damaged_key_is_replaced(signing, tmp_path):
    (tmp_path / "signing.key").write_bytes(b"")

    assert signing.verified_payload(signing.sign(b"blob")) == b"blob"
    assert len((tmp_path / "signing.key").read_bytes()) == 32


def test_racing_creators_all_get_the_published_key(signing, tmp_path):
    path = str(tmp_path / "race.key")
    results, errors = [], []
    start = threading.Barrier(16)

    def worker():
        try:
            start.wait()
            results.append(signing._read_or_create_key(path))
        except Exception as e:  # reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert set(results) == {Path(path).read_bytes()}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["race.key"]
