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
import threading
import time

_MAGIC = b"ASSIG1"
_DIGEST_SIZE = hashlib.sha256().digest_size
_KEY_SIZE = 32
_KEY = None
_KEY_PATH = None
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
    where the home directory does not survive the container)."""
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


def key_file() -> str:
    """The key file in use (created on first use)."""
    _load_key()
    return _KEY_PATH


def _load_key() -> bytes:
    global _KEY, _KEY_PATH
    with _KEY_LOCK:
        if _KEY is None:
            _KEY_PATH, _KEY = _open_key()
        return _KEY


def _open_key():
    path = _private_key_file()
    legacy = legacy_key_file()
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        if _is_served(path):
            from .helpers import log

            log.warning(f"[AgentScheduler] The script params signing key {path} is inside a folder the WebUI serves")
        seed = None
        if not os.path.exists(path) and os.path.exists(legacy):
            seed = _read_key(legacy)  # keep blobs signed by earlier versions valid
        key = _read_or_create_key(path, seed)
    except OSError as error:
        from .helpers import log

        log.warning(
            f"[AgentScheduler] Cannot keep the script params signing key private ({error}); "
            f"using {legacy}, which only Gradio's blocked_paths hides"
        )
        return legacy, _read_or_create_key(legacy)
    if os.path.exists(legacy):
        try:
            os.remove(legacy)
        except OSError:
            pass
    return path, key


def _read_key(path: str) -> bytes:
    # A key being created by another process may briefly be empty on
    # filesystems without hard links (see below).
    for _ in range(50):
        with open(path, "rb") as handle:
            key = handle.read()
        if len(key) >= _KEY_SIZE:
            return key
        time.sleep(0.02)
    raise RuntimeError(f"Agent Scheduler signing key is invalid (delete it to regenerate): {path}")


def _read_or_create_key(path: str, seed: bytes = None) -> bytes:
    if os.path.exists(path):
        return _read_key(path)
    key = seed or secrets.token_bytes(_KEY_SIZE)
    # Write the key to a private temp file first and publish it with an
    # exclusive link, so no reader (or a crash) ever sees a partial key.
    temp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
        except OSError:
            # No hard links here (e.g. exFAT): create the key exclusively in
            # place; concurrent readers wait for it in _read_key.
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(key)
                    handle.flush()
                    os.fsync(handle.fileno())
    finally:
        os.unlink(temp)
    return _read_key(path)


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
