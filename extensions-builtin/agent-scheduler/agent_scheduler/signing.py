"""HMAC signatures for exported task script params.

Script params are pickled, and unpickling runs arbitrary code. The database is
written by this server only, but exported queues come back through /import,
so exports carry a signature made with a per-install secret and imports are
refused unless it matches: a crafted import never reaches pickle.
"""

import hashlib
import hmac
import os
import secrets
import threading

_MAGIC = b"ASSIG1"
_DIGEST_SIZE = hashlib.sha256().digest_size
_KEY_SIZE = 32
_KEY = None
_KEY_LOCK = threading.Lock()


def _db_file() -> str:
    from .db.base import db_file

    return os.path.abspath(db_file)


def legacy_key_file() -> str:
    """Where earlier versions kept the key: in the data dir, which Gradio's
    /file= route serves by default."""
    return os.path.join(os.path.dirname(_db_file()), "agent_scheduler_script_params.key")


def _private_key_file() -> str:
    """Per-install key in the user's config dir, outside every path the WebUI
    serves (Gradio's blocked_paths compare paths lexically, so blocking a file
    inside a served folder is not reliable on case-insensitive filesystems).
    AGENT_SCHEDULER_KEY_FILE overrides it (e.g. a persistent volume in Docker,
    so exports stay importable after the container is recreated)."""
    override = os.environ.get("AGENT_SCHEDULER_KEY_FILE")
    if override:
        return os.path.abspath(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    install = hashlib.sha256(os.path.normcase(_db_file()).encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, "agent-scheduler", f"script-params-{install}.key")


def _is_served(path: str) -> bool:
    try:
        from modules.paths_internal import data_path
        from modules.shared import cmd_opts

        roots = [data_path, *(getattr(cmd_opts, "gradio_allowed_path", None) or [])]
    except Exception:
        roots = [os.path.dirname(_db_file())]
    target = os.path.normcase(os.path.realpath(path))
    for root in roots:
        root = os.path.normcase(os.path.realpath(root))
        try:
            if os.path.commonpath([target, root]) == root:
                return True
        except ValueError:  # different drives
            continue
    return False


def _load_key() -> bytes:
    global _KEY
    with _KEY_LOCK:
        if _KEY is None:
            _KEY = _open_key()
        return _KEY


def _open_key():
    from .helpers import log

    legacy = legacy_key_file()
    if os.path.exists(legacy):
        # Earlier versions kept the key where /file= could serve it.
        try:
            os.remove(legacy)
        except OSError as error:
            log.warning(f"[AgentScheduler] Could not remove the exposed signing key {legacy}: {error}")
    path = _private_key_file()
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        if _is_served(path):
            raise OSError(f"{path} is inside a folder the WebUI serves")
        return _read_or_create_key(path)
    except OSError as error:
        # Only exports depend on the key: without a private place for it,
        # exports stay importable until the WebUI restarts.
        log.warning(f"[AgentScheduler] Queue exports will only import into this session ({error})")
        return secrets.token_bytes(_KEY_SIZE)


def _read_or_create_key(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            key = handle.read()
        if len(key) >= _KEY_SIZE:
            return key
    except FileNotFoundError:
        pass
    # Missing or damaged (e.g. a crash while it was written): publish a new
    # key atomically, so no reader ever sees a partial one.
    temp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(_KEY_SIZE))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    with open(path, "rb") as handle:
        return handle.read()


def _mac(blob: bytes, key: bytes = None) -> bytes:
    return hmac.new(key or _load_key(), blob, hashlib.sha256).digest()


def sign(blob: bytes) -> bytes:
    return _MAGIC + _mac(blob) + blob


def is_signed(data) -> bool:
    return isinstance(data, (bytes, bytearray, memoryview)) and bytes(data[: len(_MAGIC)]) == _MAGIC


def strip_signature(data) -> bytes:
    """The payload of a stored blob (rows stored by earlier versions may
    carry a signature)."""
    data = bytes(data)
    if is_signed(data):
        return data[len(_MAGIC) + _DIGEST_SIZE :]
    return data


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
