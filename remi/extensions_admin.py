"""Extension administration without a UI: the installed list, enabling and
disabling, update checks and updates, installing from a git URL or the
extension index, uninstalling, and config-state backups ("Backup/Restore").

The HTTP routes (``remi/server/routes/extensions.py``) and the legacy Gradio
tab (``modules/ui_extensions.py``) both call this module. Functions raise
:class:`AdminError` subclasses carrying an HTTP status and a translatable
message template; nothing here imports Gradio or touches the request layer.

Extensions are arbitrary code: every function that changes anything first
calls :func:`check_access`, which refuses when the server is reachable from
other machines (``--listen`` & co.) without
``--enable-insecure-extension-access``.

Git runs non-interactively (:func:`run_git`): no terminal, no password or
host-key prompt (``GIT_TERMINAL_PROMPT=0``, ssh ``BatchMode=yes``) and a
timeout, so a private repository or an unknown ssh host fails the request
instead of blocking it (and the management lock) forever.

Backups restore the extensions only. The WebUI checkout itself is never
reset from here: a backup made before the Remi rewrite points at an
upstream commit, and ``git reset --hard`` would drop the working tree.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui-extensions/master/index.json"
INDEX_TIMEOUT = 20
INDEX_MAX_BYTES = 16 * 1024 * 1024
INDEX_CACHE_SECONDS = 600

GIT_TIMEOUT = 120
"""Seconds a network git command (fetch) may take."""
GIT_CLONE_TIMEOUT = 900
"""Seconds a clone (with submodules) may take."""
GIT_LOCAL_TIMEOUT = 60
"""Seconds a local git command (rev-parse, reset) may take."""

DIRNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
GIT_URL_RE = re.compile(r"^(https?|ssh|git)://[^\s]+$|^git@[^\s:]+:[^\s]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
STATE_ID_RE = re.compile(r"^[^/\\]{1,200}\.json$")
RESTORE_TARGETS = ("extensions",)

log = logging.getLogger(__name__)

lock = threading.RLock()
"""One management operation (install, update, uninstall, restore) at a time."""


# region Errors


def render(template: str, params: dict[str, Any]) -> str:
    """``"git {{command}} failed"`` with ``command="fetch"`` → ``"git fetch failed"``."""

    return re.sub(r"\{\{(\w+)\}\}", lambda match: str(params.get(match.group(1), match.group(0))), template)


class AdminError(Exception):
    """A refused or failed operation. ``template`` is English text with
    ``{{name}}`` placeholders (the key clients translate), ``params`` fills
    them, ``str(error)`` is the rendered English message."""

    status = 422

    def __init__(self, template: str, status: int | None = None, **params: Any) -> None:
        self.template = template
        self.params = {key: str(value) for key, value in params.items()}
        super().__init__(render(template, self.params))
        if status is not None:
            self.status = status

    def to_json(self) -> dict[str, Any]:
        return {"detail": str(self), "template": self.template, "params": self.params}


class AccessDenied(AdminError):
    status = 403


class NotFound(AdminError):
    status = 404


class Conflict(AdminError):
    status = 409


class GitError(AdminError):
    """Git failed or timed out (the remote is unreachable, needs credentials, ...)."""

    status = 502


# endregion

# region Access and helpers


def access_allowed() -> bool:
    from modules import shared

    return not shared.cmd_opts.disable_extension_access


def check_access() -> None:
    if not access_allowed():
        raise AccessDenied("extension access is disabled (the server is reachable from other machines; see --enable-insecure-extension-access)")


def default_index_url() -> str:
    return os.environ.get("WEBUI_EXTENSIONS_INDEX") or DEFAULT_INDEX_URL


def normalize_git_url(url: str | None) -> str:
    """``https://host/user/repo.git/`` → ``https://host/user/repo`` (for comparing remotes)."""

    if not url:
        return ""
    url = url.strip().rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def dirname_from_url(url: str) -> str:
    return normalize_git_url(url).rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _host(parsed: urllib.parse.SplitResult) -> str:
    host = parsed.hostname or ""
    if ":" in host:  # IPv6
        host = f"[{host}]"
    return host + (f":{parsed.port}" if parsed.port else "")


def web_url(remote: str | None) -> str | None:
    """A browsable https URL for a git remote (``git@github.com:a/b.git`` too),
    credentials removed; None when the remote is not a web host."""

    if not remote:
        return None
    remote = normalize_git_url(remote)
    match = re.match(r"^git@([^:/\s]+):(.+)$", remote)
    if match:
        return f"https://{match.group(1)}/{match.group(2)}"
    if remote.startswith(("https://", "http://")):
        try:
            parsed = urllib.parse.urlsplit(remote)
            host = _host(parsed)
        except ValueError:
            return None
        if not parsed.hostname:
            return None
        return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return None


def strip_credentials(remote: str | None) -> str | None:
    """A git remote as shown to clients: no password or token in any form.

    ``scheme://user:secret@host/p`` → ``scheme://host/p`` (http(s), git, ftp:
    any user part may be a token; ssh keeps a plain user name);
    ``user:secret@host:path`` → ``user@host:path``; local paths unchanged."""

    if not remote:
        return None
    remote = remote.strip()
    if "://" in remote:
        try:
            parsed = urllib.parse.urlsplit(remote)
            host = _host(parsed)
        except ValueError:
            return remote.split("://", 1)[0] + "://"
        user = parsed.username if parsed.scheme in ("ssh", "git+ssh", "ssh+git") else None
        netloc = f"{user}@{host}" if user else host
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    match = re.match(r"^([^@/\s]+)@([^:/\s]+):(.*)$", remote)
    if match:
        user = match.group(1).split(":", 1)[0]
        return f"{user}@{match.group(2)}:{match.group(3)}"
    return remote


def display_remote(remote: str | None) -> str | None:
    """A browsable URL when the remote has one, else the remote without credentials."""

    return web_url(remote) or strip_credentials(remote)


def commit_url(remote: str | None, commit: str | None) -> str | None:
    base = web_url(remote)
    if not base or not commit:
        return None
    host = urllib.parse.urlsplit(base).hostname or ""
    if host in ("github.com", "gitlab.com", "codeberg.org") or host.endswith(".github.com"):
        return f"{base}/commit/{commit}"
    return None


def iso(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


# endregion

# region Git


def git_executable() -> str:
    return os.environ.get("GIT", "git")


def git_environment() -> dict[str, str]:
    """The environment of every git command: never ask for anything.

    No credential prompt (``GIT_TERMINAL_PROMPT``, Git Credential Manager),
    no ssh askpass, ssh in batch mode (no password or host-key question).
    A ``GIT_SSH_COMMAND`` of the user's keeps its options."""

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["SSH_ASKPASS_REQUIRE"] = "never"
    if not env.get("GIT_SSH"):
        ssh = env.get("GIT_SSH_COMMAND") or "ssh"
        if "batchmode" not in ssh.lower():
            env["GIT_SSH_COMMAND"] = f"{ssh} -o BatchMode=yes"
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill git and what it started (ssh, remote helpers): they hold its pipes."""

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=10, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    with contextlib.suppress(OSError):
        process.kill()


def _reason(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return " ".join(lines[-3:])[:500] or "no output"


def run_git(args: list[str], *, cwd: str | os.PathLike | None = None, timeout: float = GIT_TIMEOUT) -> str:
    """Run ``git <args>`` without a terminal; its stdout, or :class:`GitError`
    (502) when it fails, cannot start or runs longer than ``timeout``."""

    command = next((arg for arg in args if not arg.startswith("-")), "git")
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True  # no controlling terminal: ssh cannot prompt
    try:
        process = subprocess.Popen(
            [git_executable(), *args],
            cwd=cwd,
            env=git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **options,
        )
    except OSError as error:
        raise GitError("git could not be started: {{reason}}", reason=error) from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            process.communicate(timeout=5)
        raise GitError("git {{command}} timed out after {{seconds}} s", command=command, seconds=int(timeout)) from None
    if process.returncode != 0:
        raise GitError("git {{command}} failed: {{reason}}", command=command, reason=_reason(stderr or stdout))
    return stdout


_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


@contextlib.contextmanager
def repo_lock(path: str | os.PathLike) -> Iterator[None]:
    """One git operation per repository at a time (a check never races an update)."""

    key = os.path.realpath(path)
    with _repo_locks_guard:
        held = _repo_locks.setdefault(key, threading.Lock())
    with held:
        yield


def _remote_name(path: str | os.PathLike) -> str:
    names = run_git(["remote"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).split()
    if not names:
        raise GitError("the repository has no remote")
    return "origin" if "origin" in names else names[0]


def _tracking_ref(remote: str, branch: str | None) -> str:
    return f"{remote}/{branch}" if branch else f"{remote}/HEAD"


def check_repo(path: str | os.PathLike, branch: str | None) -> tuple[bool, str]:
    """Fetch a repository's remote and compare: ``(can_update, status)``,
    status ``"latest"``, ``"new commits"`` (the remote is ahead) or
    ``"behind HEAD"`` (local commits the remote does not have; an update
    resets to the remote)."""

    remote = _remote_name(path)
    run_git(["fetch", "--quiet", remote], cwd=path)
    target = run_git(["rev-parse", "--verify", f"{_tracking_ref(remote, branch)}^{{commit}}"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    head = run_git(["rev-parse", "HEAD"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    if head == target:
        return False, "latest"
    behind = run_git(["rev-list", "--count", f"HEAD..{target}"], cwd=path, timeout=GIT_LOCAL_TIMEOUT).strip()
    if behind.isdigit() and int(behind) > 0:
        return True, "new commits"
    return True, "behind HEAD"


def reset_repo(path: str | os.PathLike, branch: str | None, commit: str | None = None) -> None:
    """Fetch every remote, then reset the checkout to ``commit`` (default: the
    remote's tip of ``branch``)."""

    remote = _remote_name(path)
    run_git(["fetch", "--all", "--quiet"], cwd=path)
    run_git(["reset", "--hard", commit or _tracking_ref(remote, branch)], cwd=path, timeout=GIT_LOCAL_TIMEOUT)


# endregion

# region Installed extensions


def all_extensions() -> list[Any]:
    from modules import extensions

    return list(extensions.extensions)


def find(name: str) -> Any:
    for ext in all_extensions():
        if ext.name == name:
            return ext
    raise NotFound("no extension {{name}}", name=name)


def info(ext: Any) -> dict[str, Any]:
    """One extension as JSON (reads its git metadata once)."""

    if not ext.have_info_from_repo:
        ext.read_info_from_repo()
    commit = getattr(ext, "commit_hash", "") or None
    return {
        "name": ext.name,
        "enabled": bool(ext.enabled),
        "builtin": bool(ext.is_builtin),
        "remote": display_remote(ext.remote),
        "remote_url": web_url(ext.remote),
        "branch": ext.branch,
        "version": ext.version or None,
        "commit_hash": commit,
        "commit_url": commit_url(ext.remote, commit),
        "commit_date": iso(ext.commit_date),
        "status": ext.status or None,
        "can_update": bool(ext.can_update),
    }


def disabled_reason() -> str | None:
    """Why extensions are not loaded at all (a command-line flag or the option)."""

    from modules import shared

    if shared.cmd_opts.disable_all_extensions:
        return "--disable-all-extensions"
    if shared.opts.disable_all_extensions == "all":
        return "all"
    if shared.cmd_opts.disable_extra_extensions:
        return "--disable-extra-extensions"
    if shared.opts.disable_all_extensions == "extra":
        return "extra"
    return None


def listing() -> dict[str, Any]:
    from modules import restart, shared

    return {
        "items": [info(ext) for ext in all_extensions()],
        "disable_all": shared.opts.disable_all_extensions,
        "access": access_allowed(),
        "restartable": restart.is_restartable(),
        "disabled_by": disabled_reason(),
        "index_url": default_index_url(),
    }


def configure(disabled: Iterable[str] | None = None, disable_all: str | None = None, *, config_filename: str | None = None) -> None:
    """Set which extensions are disabled (the others are enabled) and the
    "disable all" mode; saved to the config file; loaded on the next reload."""

    from modules import shared

    check_access()
    known = {ext.name for ext in all_extensions()}
    if disabled is not None:
        disabled = list(dict.fromkeys(disabled))
        unknown = sorted(set(disabled) - known)
        if unknown:
            raise AdminError("unknown extension(s): {{names}}", names=", ".join(unknown))
        shared.opts.disabled_extensions = disabled
    if disable_all is not None:
        if disable_all not in ("none", "extra", "all"):
            raise AdminError("disable_all must be none, extra or all, not {{value}}", value=disable_all)
        shared.opts.disable_all_extensions = disable_all
    shared.opts.save(config_filename or shared.config_filename)
    for ext in all_extensions():
        ext.enabled = ext.name not in shared.opts.disabled_extensions


def check_one(ext: Any) -> None:
    """Fetch an extension's remote and set its update status (a failure is
    the status ``"unknown (remote error)"``, never a prompt or a hang)."""

    from modules import errors

    if ext.remote is None:
        return
    with repo_lock(ext.path):
        try:
            ext.check_updates()
        except Exception as error:
            if isinstance(error, GitError):
                log.warning("Error checking updates for %s: %s", ext.name, error)
            else:
                errors.report(f"Error checking updates for {ext.name}", exc_info=True)
            ext.can_update = False
            ext.status = "unknown (remote error)"


def check_updates(names: Iterable[str] | None = None, *, on_progress: Callable[[Any], None] | None = None) -> list[Any]:
    """Check the given extensions (default: every enabled one with a remote),
    16 at a time. ``on_progress(ext)`` runs after each check."""

    check_access()
    if names is None:
        targets = [ext for ext in all_extensions() if ext.remote is not None and ext.enabled]
    else:
        targets = [find(name) for name in dict.fromkeys(names)]
    progress_lock = threading.Lock()

    def run(ext: Any) -> None:
        check_one(ext)
        if on_progress is not None:
            with progress_lock:
                on_progress(ext)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(run, targets))
    return targets


def update(names: Iterable[str]) -> None:
    """Update extensions to their remote's latest commit, after backing up
    the config state. Every one is tried; :class:`GitError` (502) names those
    that failed."""

    from modules import errors

    check_access()
    targets = [find(name) for name in dict.fromkeys(names)]
    builtin = [ext.name for ext in targets if ext.is_builtin or ext.remote is None]
    if builtin:
        raise AdminError("not updatable (built-in or not a git repository): {{names}}", names=", ".join(builtin))
    failed: list[tuple[str, str]] = []
    with lock:
        save_config_state("Backup (pre-update)")
        for ext in targets:
            try:
                with repo_lock(ext.path):
                    ext.fetch_and_reset_hard()
                ext.can_update = False
                ext.status = "latest"
                ext.read_info_from_repo()
            except Exception as error:
                if not isinstance(error, GitError):
                    errors.report(f"Error getting updates for {ext.name}", exc_info=True)
                failed.append((ext.name, str(error) or type(error).__name__))
    if failed:
        raise GitError("could not update {{names}}: {{reason}}", names=", ".join(name for name, _ in failed), reason=failed[0][1])


def install(url: str, branch: str | None = None, dirname: str | None = None) -> Any:
    """Clone ``url`` into ``extensions/<dirname>`` and run its installer.
    The extension loads after the next reload."""

    from modules import extensions, launch_utils, paths

    check_access()
    url = (url or "").strip()
    branch = (branch or "").strip() or None
    if not GIT_URL_RE.match(url):
        raise AdminError("expected a git URL (https://, ssh://, git:// or git@host:path)")
    dirname = (dirname or "").strip() or dirname_from_url(url)
    if not DIRNAME_RE.match(dirname):
        raise AdminError("not a valid folder name: {{name}}", name=dirname)
    if branch is not None and (not BRANCH_RE.match(branch) or branch.startswith("-")):
        raise AdminError("not a valid branch name: {{name}}", name=branch)
    with lock:
        target = os.path.join(extensions.extensions_dir, dirname)
        if os.path.exists(target):
            raise Conflict("the folder {{name}} already exists", name=dirname)
        if any(normalize_git_url(ext.remote) == normalize_git_url(url) for ext in all_extensions() if ext.remote):
            raise Conflict("an extension with this URL is installed")
        temp = os.path.join(paths.data_path, "tmp", dirname)
        try:
            shutil.rmtree(temp, True)
            os.makedirs(os.path.dirname(temp), exist_ok=True)
            args = ["clone", "--filter=blob:none", "--recurse-submodules"]
            if branch:
                args += ["--branch", branch]
            run_git([*args, "--", url, temp], timeout=GIT_CLONE_TIMEOUT)
            try:
                os.rename(temp, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.move(temp, target)  # tmp/ and extensions/ on different file systems
            launch_utils.run_extension_installer(target)
            extensions.list_extensions()
        finally:
            shutil.rmtree(temp, True)
    return find(dirname)


def _make_writable(function: Callable[..., Any], path: str, _excinfo: Any) -> None:
    # git marks pack files read-only; Windows refuses to delete them otherwise
    os.chmod(path, stat.S_IWRITE)
    function(path)


def uninstall(name: str) -> None:
    """Delete an installed (not built-in) extension's folder. Its scripts
    stay loaded until the next reload."""

    from modules import extensions, shared

    check_access()
    ext = find(name)
    if ext.is_builtin:
        raise AdminError("built-in extensions cannot be uninstalled; disable them instead")
    root = os.path.realpath(extensions.extensions_dir)
    path = os.path.realpath(ext.path)
    if os.path.dirname(path) != root:
        raise AdminError("the extension is not inside the extensions folder")
    with lock, repo_lock(path):
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_make_writable)
        else:
            shutil.rmtree(path, onerror=_make_writable)
        if name in shared.opts.disabled_extensions:
            shared.opts.disabled_extensions = [item for item in shared.opts.disabled_extensions if item != name]
            shared.opts.save(shared.config_filename)
        extensions.extensions[:] = [item for item in extensions.extensions if item.name != name]


# endregion

# region Extension index ("Available")


@dataclass
class _IndexCache:
    url: str
    fetched: float
    data: dict[str, Any]


_index_cache: _IndexCache | None = None


def fetch_index(url: str, *, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Download and parse an extension index (``{"tags", "extensions"}``)."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AdminError("the index URL must be http(s)://")
    request = urllib.request.Request(url, headers={"User-Agent": "Remi-WebUI", "Accept": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=INDEX_TIMEOUT) as response:
            text = response.read(INDEX_MAX_BYTES + 1)
    except Exception as error:
        raise AdminError("could not load the index: {{reason}}", status=502, reason=error) from error
    if len(text) > INDEX_MAX_BYTES:
        raise AdminError("the index is too large", status=502)
    try:
        data = json.loads(text)
    except ValueError as error:
        raise AdminError("the index is not JSON: {{reason}}", status=502, reason=error) from error
    if not isinstance(data, dict) or not isinstance(data.get("extensions"), list):
        raise AdminError("the index has no extension list", status=502)
    return data


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def index_entries(data: dict[str, Any]) -> dict[str, Any]:
    """The index normalised: entries with ``installed`` marked, and the tag
    descriptions (``installed`` added)."""

    names = {ext.name for ext in all_extensions()}
    remotes = {normalize_git_url(ext.remote) for ext in all_extensions() if ext.remote}
    tags = {str(key): _text(value) for key, value in (data.get("tags") or {}).items()} if isinstance(data.get("tags"), dict) else {}
    items = []
    for order, entry in enumerate(data["extensions"]):
        if not isinstance(entry, dict):
            continue
        url = _text(entry.get("url")).strip()
        if not url or not GIT_URL_RE.match(url):
            continue
        entry_tags = [str(tag) for tag in entry.get("tags") or [] if isinstance(tag, (str, int))]
        for tag in entry_tags:
            tags.setdefault(tag, "")
        try:
            stars = int(entry.get("stars") or 0)
        except (TypeError, ValueError):
            stars = 0
        items.append(
            {
                "name": _text(entry.get("name")) or dirname_from_url(url),
                "url": url,
                "description": _text(entry.get("description")),
                "tags": entry_tags,
                "added": _text(entry.get("added")) or None,
                "created_at": _text(entry.get("created_at")) or None,
                "commit_time": _text(entry.get("commit_time")) or None,
                "stars": stars,
                "order": order,
                "installed": dirname_from_url(url) in names or normalize_git_url(url) in remotes,
            }
        )
    tags.setdefault("installed", "extensions that are already installed")
    return {"items": items, "tags": tags}


def load_index(url: str | None = None, *, refresh: bool = False, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """The index at ``url`` (default: ``WEBUI_EXTENSIONS_INDEX`` or the
    A1111 index), cached for ten minutes."""

    global _index_cache

    check_access()  # the server fetches a URL the client names
    url = (url or "").strip() or default_index_url()
    cache = _index_cache
    if refresh or cache is None or cache.url != url or time.time() - cache.fetched > INDEX_CACHE_SECONDS:
        data = fetch_index(url, opener=opener)
        cache = _index_cache = _IndexCache(url, time.time(), data)
    return {"url": url, "fetched_at": iso(cache.fetched), **index_entries(cache.data)}


# endregion

# region Config states (backup / restore)


def _states_dir() -> str:
    from modules import config_states

    return config_states.config_states_dir


def save_config_state(name: str = "") -> dict[str, Any]:
    """Write the current WebUI and extension commits to ``config_states/``."""

    from modules import config_states

    state = config_states.get_config()
    state["name"] = os.path.basename((name or "").strip()) or "Config"
    folder = _states_dir()
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^\w .()-]+", "_", state["name"])[:80]
    filename = os.path.join(folder, f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_{safe}.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, ensure_ascii=False)
    config_states.list_config_states()
    state["filepath"] = filename
    return state_summary(state)


def _load_states() -> dict[str, dict[str, Any]]:
    """State file name → state (newest first; unreadable files skipped)."""

    from modules import config_states

    config_states.list_config_states()
    return {os.path.basename(state["filepath"]): state for state in config_states.all_config_states.values()}


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    webui = state.get("webui") if isinstance(state.get("webui"), dict) else {}
    extensions = state.get("extensions") if isinstance(state.get("extensions"), dict) else {}
    return {
        "id": os.path.basename(state.get("filepath", "")),
        "name": str(state.get("name") or "Config"),
        "created_at": iso(state.get("created_at")),
        "webui_commit": (webui or {}).get("commit_hash"),
        "extensions": len(extensions or {}),
    }


def list_config_states() -> list[dict[str, Any]]:
    return [state_summary(state) for state in _load_states().values()]


def _state(state_id: str) -> dict[str, Any]:
    if not STATE_ID_RE.match(state_id or ""):
        raise NotFound("no config state {{name}}", name=state_id)
    state = _load_states().get(state_id)
    if state is None:
        raise NotFound("no config state {{name}}", name=state_id)
    return state


def _repo_entry(entry: dict[str, Any]) -> dict[str, Any]:
    remote = entry.get("remote")
    commit = entry.get("commit_hash")
    return {
        "remote": display_remote(remote),
        "branch": entry.get("branch"),
        "commit_hash": commit,
        "commit_url": commit_url(remote, commit),
        "commit_date": iso(entry.get("commit_date")),
    }


def _current_entry(ext: Any) -> dict[str, Any]:
    if not ext.have_info_from_repo:
        ext.read_info_from_repo()
    return {**_repo_entry({"remote": ext.remote, "branch": ext.branch, "commit_hash": ext.commit_hash or None, "commit_date": ext.commit_date}), "enabled": bool(ext.enabled)}


def config_state_detail(state_id: str) -> dict[str, Any]:
    """A backup next to the current state: each extension's saved and
    current commit, branch, remote and enablement."""

    from modules import config_states

    state = _state(state_id)
    current_webui = config_states.get_webui_config()
    saved_webui = state.get("webui") if isinstance(state.get("webui"), dict) else {}
    current = {ext.name: ext for ext in all_extensions()}
    rows = []
    saved_exts = state.get("extensions") if isinstance(state.get("extensions"), dict) else {}
    for name, entry in sorted(saved_exts.items(), key=lambda item: item[0].lower()):
        if not isinstance(entry, dict):
            continue
        ext = current.get(name)
        rows.append({"name": name, "builtin": bool(entry.get("is_builtin")), "saved": {**_repo_entry(entry), "enabled": bool(entry.get("enabled"))}, "current": _current_entry(ext) if ext is not None else None})
    for name, ext in current.items():
        if name not in saved_exts and not ext.is_builtin:
            rows.append({"name": name, "builtin": False, "saved": None, "current": _current_entry(ext)})
    return {
        **state_summary(state),
        "webui": {"saved": _repo_entry(saved_webui or {}), "current": _repo_entry(current_webui)},
        "items": rows,
    }


def delete_config_state(state_id: str) -> None:
    check_access()
    state = _state(state_id)
    with lock:
        os.remove(state["filepath"])


def restore_config_state(state_id: str, what: str = "extensions") -> dict[str, Any]:
    """Restore the extensions of a backup: every extension is reset to its
    recorded commit (and enablement) while the next "Reload UI" runs
    (``restore_config_state_file``). Returns ``{"reload": True}``.

    The WebUI checkout is never restored (see the module docstring)."""

    from modules import shared

    check_access()
    if what not in RESTORE_TARGETS:
        raise AdminError("only the extensions of a backup can be restored; the WebUI checkout is never reset from here")
    state = _state(state_id)
    with lock:
        shared.opts.restore_config_state_file = state["filepath"]
        shared.opts.save(shared.config_filename)
    return {"reload": True}


# endregion
