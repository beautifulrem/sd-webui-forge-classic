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


def _load(tmp_path, monkeypatch, name="private.key", served=False):
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    monkeypatch.setattr(module, "_private_key_file", lambda: str(tmp_path / "config" / name))
    monkeypatch.setattr(module, "legacy_key_file", lambda: str(tmp_path / "data" / "legacy.key"))
    monkeypatch.setattr(module, "_is_served", lambda path: served)
    return module


@pytest.fixture(autouse=True)
def quiet_log(monkeypatch):
    import sys
    import types

    helpers = types.ModuleType("agent_scheduler_signing_helpers")
    helpers.log = types.SimpleNamespace(warning=lambda *a: None)
    # signing imports ``.helpers`` lazily; give the standalone module a parent.
    package = types.ModuleType("agent_scheduler_pkg")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "agent_scheduler_pkg", package)
    monkeypatch.setitem(sys.modules, "agent_scheduler_pkg.helpers", helpers)


@pytest.fixture()
def signing(tmp_path, monkeypatch):
    module = _load(tmp_path, monkeypatch)
    module.__package__ = "agent_scheduler_pkg"
    return module


def test_signed_blobs_round_trip_and_key_is_private(signing, tmp_path):
    blob = zlib.compress(pickle.dumps([1, "x"]))

    signed = signing.sign(blob)

    assert signing.is_signed(signed)
    assert signing.verify(signed) == blob
    assert signing.strip_signature(signed) == blob
    assert signing.strip_signature(blob) == blob
    key = tmp_path / "config" / "private.key"
    assert len(key.read_bytes()) == 32
    if os.name != "nt":
        assert key.stat().st_mode & 0o077 == 0


def test_unsigned_tampered_and_foreign_blobs_are_refused(signing, tmp_path, monkeypatch):
    blob = zlib.compress(pickle.dumps([1]))
    signed = signing.sign(blob)

    with pytest.raises(ValueError):
        signing.verify(blob)  # crafted, unsigned
    with pytest.raises(ValueError):
        signing.verify(signed[:-1] + bytes([signed[-1] ^ 1]))  # tampered payload

    other = _load(tmp_path, monkeypatch, "other.key")  # another install
    other.__package__ = "agent_scheduler_pkg"
    with pytest.raises(ValueError):
        other.verify(signed)


def test_exposed_legacy_key_is_removed_and_never_used(signing, tmp_path):
    legacy = tmp_path / "data" / "legacy.key"
    legacy.parent.mkdir()
    legacy.write_bytes(b"k" * 32)

    signing.sign(b"blob")

    assert not legacy.exists()
    assert (tmp_path / "config" / "private.key").read_bytes() != b"k" * 32


def test_no_key_file_inside_served_folders(tmp_path, monkeypatch):
    signing = _load(tmp_path, monkeypatch, served=True)
    signing.__package__ = "agent_scheduler_pkg"

    assert signing.verify(signing.sign(b"blob")) == b"blob"  # session key
    assert not (tmp_path / "config" / "private.key").exists()


def test_damaged_key_is_replaced(signing, tmp_path):
    key = tmp_path / "config" / "private.key"
    key.parent.mkdir()
    key.write_bytes(b"")

    assert signing.verify(signing.sign(b"blob")) == b"blob"
    assert len(key.read_bytes()) == 32


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
    assert all(len(key) == 32 for key in results)
    assert [p.name for p in tmp_path.iterdir()] == ["key"]
