"""HMAC signatures for pickled task script params.

Script params are pickled, and unpickling runs arbitrary code, so only blobs
this server produced may be loaded. A per-install secret signs every blob it
writes; anything unsigned or signed elsewhere (e.g. a crafted /import) is
refused before pickle ever sees it. Export/import on the same install still
round-trips because exported blobs carry the signature.
"""

import hashlib
import hmac
import os
import secrets

_MAGIC = b"ASSIG1"
_DIGEST_SIZE = hashlib.sha256().digest_size
_KEY = None


def _key_file() -> str:
    from .db.base import db_file

    return os.path.join(os.path.dirname(os.path.abspath(db_file)), "agent_scheduler_script_params.key")


def migration_marker() -> str:
    """File recording that pre-signing rows were signed (done once)."""
    return _key_file() + ".migrated"


def _load_key() -> bytes:
    global _KEY
    if _KEY is not None:
        return _KEY
    path = _key_file()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(32))
    with open(path, "rb") as handle:
        key = handle.read()
    if len(key) < 32:
        raise RuntimeError(f"Agent Scheduler signing key is invalid: {path}")
    _KEY = key
    return key


def _mac(blob: bytes, key: bytes = None) -> bytes:
    return hmac.new(key or _load_key(), blob, hashlib.sha256).digest()


def sign(blob: bytes) -> bytes:
    return _MAGIC + _mac(blob) + blob


def is_signed(data) -> bool:
    return isinstance(data, (bytes, bytearray, memoryview)) and bytes(data[: len(_MAGIC)]) == _MAGIC


def verify(data) -> bytes:
    """Return the signed payload, or raise ValueError if it is not ours."""

    data = bytes(data)
    if not is_signed(data):
        raise ValueError("task script params are not signed by this server")
    mac = data[len(_MAGIC) : len(_MAGIC) + _DIGEST_SIZE]
    blob = data[len(_MAGIC) + _DIGEST_SIZE :]
    if not hmac.compare_digest(mac, _mac(blob)):
        raise ValueError("task script params signature does not match this server")
    return blob
