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


def _load(tmp_path, monkeypatch, name="private.key"):
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    monkeypatch.setattr(module, "_private_key_file", lambda: str(tmp_path / "config" / name))
    monkeypatch.setattr(module, "legacy_key_file", lambda: str(tmp_path / "data" / "legacy.key"))
    monkeypatch.setattr(module, "_is_served", lambda path: False)
    return module


@pytest.fixture()
def signing(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch)


def test_signed_blobs_round_trip_and_key_is_private(signing, tmp_path):
    blob = zlib.compress(pickle.dumps([1, "x"]))

    signed = signing.sign(blob)

    assert signing.is_signed(signed)
    assert signing.verify(signed) == blob
    assert signing.key_file() == str(tmp_path / "config" / "private.key")
    if os.name != "nt":
        assert (tmp_path / "config" / "private.key").stat().st_mode & 0o077 == 0


def test_unsigned_tampered_and_foreign_blobs_are_refused(signing, tmp_path, monkeypatch):
    blob = zlib.compress(pickle.dumps([1]))
    signed = signing.sign(blob)

    with pytest.raises(ValueError):
        signing.verify(blob)  # legacy / crafted, unsigned
    with pytest.raises(ValueError):
        signing.verify(signed[:-1] + bytes([signed[-1] ^ 1]))  # tampered payload

    other = _load(tmp_path, monkeypatch, "other.key")  # another install
    with pytest.raises(ValueError):
        other.verify(signed)


def test_key_from_the_served_data_dir_is_moved_out(signing, tmp_path):
    legacy = tmp_path / "data" / "legacy.key"
    legacy.parent.mkdir()
    legacy.write_bytes(b"k" * 32)

    signed = signing.sign(b"blob")

    assert not legacy.exists()
    assert (tmp_path / "config" / "private.key").read_bytes() == b"k" * 32
    assert signing.verify(signed) == b"blob"


def test_concurrent_first_use_shares_one_complete_key(signing, tmp_path):
    path = str(tmp_path / "key")
    results, errors = [], []

    def worker():
        try:
            results.append(signing._read_or_create_key(path))
        except Exception as e:  # reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(set(results)) == 1 and len(results[0]) == 32
    assert [p.name for p in tmp_path.iterdir()] == ["key"]
