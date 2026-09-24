import importlib.util
import pickle
import zlib
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "agent_scheduler_signing", Path(__file__).parents[1] / "agent_scheduler" / "signing.py"
)


@pytest.fixture()
def signing(tmp_path, monkeypatch):
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    monkeypatch.setattr(module, "_key_file", lambda: str(tmp_path / "script_params.key"))
    return module


def test_signed_blobs_round_trip_and_key_is_private(signing, tmp_path):
    blob = zlib.compress(pickle.dumps([1, "x"]))

    signed = signing.sign(blob)

    assert signing.is_signed(signed)
    assert signing.verify(signed) == blob
    assert (tmp_path / "script_params.key").stat().st_mode & 0o077 == 0


def test_unsigned_tampered_and_foreign_blobs_are_refused(signing, tmp_path, monkeypatch):
    blob = zlib.compress(pickle.dumps([1]))
    signed = signing.sign(blob)

    with pytest.raises(ValueError):
        signing.verify(blob)  # legacy / crafted, unsigned
    with pytest.raises(ValueError):
        signing.verify(signed[:-1] + bytes([signed[-1] ^ 1]))  # tampered payload

    # Signed by another install (different key).
    monkeypatch.setattr(signing, "_KEY", None)
    monkeypatch.setattr(signing, "_key_file", lambda: str(tmp_path / "other.key"))
    with pytest.raises(ValueError):
        signing.verify(signed)
