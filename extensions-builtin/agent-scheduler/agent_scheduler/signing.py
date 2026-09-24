"""HMAC signatures for pickled task script params.

Script params are pickled, and unpickling runs arbitrary code. Params this
server stores are signed with a per-install secret; exports carry the
signature and /import refuses anything that does not verify, so a crafted
import never reaches pickle. Stored params without a valid signature (written
before signing existed, or under a replaced key) are only loaded through the
restricted legacy unpickler.

The key lives next to the database, so it travels with the data; the file
guard keeps both away from Gradio's /file= route, which serves the data dir.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time

_MAGIC = b"ASSIG1"
_DIGEST_SIZE = hashlib.sha256().digest_size
_KEY_SIZE = 32
_KEY = None
_KEY_LOCK = threading.Lock()


def _db_file() -> str:
    from .db.base import db_file

    return os.path.abspath(db_file)


def key_file() -> str:
    # A new name: keys of earlier versions may have been served by /file=.
    return os.path.join(os.path.dirname(_db_file()), "agent_scheduler_signing.key")


def _stale_key_files():
    """Keys of earlier versions, possibly exposed; never trusted again."""
    yield os.path.join(os.path.dirname(_db_file()), "agent_scheduler_script_params.key")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    install = hashlib.sha256(os.path.normcase(_db_file()).encode("utf-8")).hexdigest()[:16]
    yield os.path.join(base, "agent-scheduler", f"script-params-{install}.key")


def protected_files():
    """Files that must never be served: the key and the task database."""
    db_file = _db_file()
    return [key_file(), db_file, db_file + "-journal", db_file + "-wal", db_file + "-shm"]


def _load_key() -> bytes:
    global _KEY
    with _KEY_LOCK:
        if _KEY is None:
            for stale in _stale_key_files():
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass  # it is not trusted either way
            _KEY = _read_or_create_key(key_file())
        return _KEY


def _read_key(path: str, wait: bool):
    # On filesystems without hard links a key being created by another
    # process may briefly be empty (see below).
    for _ in range(50 if wait else 1):
        try:
            with open(path, "rb") as handle:
                key = handle.read()
        except FileNotFoundError:
            return None
        if len(key) >= _KEY_SIZE:
            return key
        if wait:
            time.sleep(0.02)
    return b""


def _read_or_create_key(path: str) -> bytes:
    key = _read_key(path, wait=False)
    if key:
        return key
    temp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(_KEY_SIZE))
            handle.flush()
            os.fsync(handle.fileno())
        if key is None:
            # Publish exclusively: when processes race, the first key wins
            # and everyone reads that one; nobody sees a partial key.
            try:
                os.link(temp, path)
            except FileExistsError:
                pass
            except OSError:
                # No hard links (e.g. exFAT): create it exclusively in place.
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(fd, "wb") as handle:
                        with open(temp, "rb") as source:
                            handle.write(source.read())
                        handle.flush()
                        os.fsync(handle.fileno())
            key = _read_key(path, wait=True)
        if not key:
            # Damaged (e.g. a crash while it was written): replace it. Params
            # signed with the lost key fall back to the legacy unpickler.
            os.replace(temp, path)
            key = _read_key(path, wait=False)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return key


def _mac(blob: bytes) -> bytes:
    return hmac.new(_load_key(), blob, hashlib.sha256).digest()


def sign(blob: bytes) -> bytes:
    return _MAGIC + _mac(blob) + blob


def is_signed(data) -> bool:
    return isinstance(data, (bytes, bytearray, memoryview)) and bytes(data[: len(_MAGIC)]) == _MAGIC


def verified_payload(data):
    """The payload if ``data`` carries this install's signature, else None."""
    data = bytes(data)
    if not is_signed(data):
        return None
    mac = data[len(_MAGIC) : len(_MAGIC) + _DIGEST_SIZE]
    blob = data[len(_MAGIC) + _DIGEST_SIZE :]
    return blob if hmac.compare_digest(mac, _mac(blob)) else None


def verify(data) -> bytes:
    """Return the signed payload, or raise ValueError if it is not ours."""
    if not is_signed(data):
        raise ValueError("task script params are not signed by this server")
    blob = verified_payload(data)
    if blob is None:
        raise ValueError("task script params signature does not match this server")
    return blob


def strip_signature(data) -> bytes:
    data = bytes(data)
    return data[len(_MAGIC) + _DIGEST_SIZE :] if is_signed(data) else data
