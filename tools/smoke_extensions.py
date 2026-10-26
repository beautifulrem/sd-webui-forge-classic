"""Real-server smoke of extension management, and the SPA features that need no model.

    python tools/smoke_extensions.py                      # the API checks
    python tools/smoke_extensions.py --browser            # also the SPA's @fixture Playwright tests
    python tools/smoke_extensions.py --browser --report docs/research/smoke/extensions-<date>.json

Extension management needs a git host and an extension index, so a review cannot try it
on a real server without the network or without changing the install. This tool brings
both along, on loopback, and never loads a model:

1. A git fixture: two extension repositories (``remi-smoke-ext``, which ships a right-to-left
   ``localizations/ar.json``, and ``remi-smoke-index-ext``), bare clones served over git's
   smart HTTP protocol (``git http-backend`` behind ``http.server``, as a git host serves
   them) on ``127.0.0.1``, and an extension index
   (``index.json``) listing both. ``POST /__advance/<repo>`` on the fixture pushes a new
   commit (the browser tests use it to make an update appear).
2. Starts ``launch.py`` (``tools/smoke_real.py``'s ``Server``) with an isolated data folder,
   an empty models folder (nothing to load) and ``WEBUI_EXTENSIONS_INDEX`` pointing at the
   fixture's index. Three PNGs with infotext are put in its output folder for the gallery.
3. API checks against that server, as the SPA calls them: install from a URL; the index
   (entries, the ``installed`` mark) and install from it; a backup taken now compares equal
   to the current state on every row (built-ins included); the remote moves on, the
   per-extension check reports the update, the bulk check too; update (and its automatic
   "pre-update" backup); the backup now differs on that extension only; restore it, Reload
   UI, and the extension is back at the backed-up commit; the right-to-left locale the
   extension ships is offered after the reload; uninstall.
4. With ``--browser`` (build the SPA first: ``cd webui && pnpm build``): the SPA's
   ``@fixture`` Playwright tests (``webui/tests/e2e/extensions-real.spec.ts``) in Chromium
   against this server: the same flows through the UI (install from URL and from the index,
   Reload UI, Check for updates with its progress, Apply and reload, the backup's
   differences and Restore), the right-to-left layout of the shipped locale, the viewer's
   horizontal-wheel and gamepad navigation over the gallery, and the confirmation before
   leaving with unsaved settings.
5. Stops the server, removes the backups it made (``config_states/`` is in the repository
   folder; backups made before the run are left alone) and prints one row per check; exit
   1 when a check failed, 2 when the server could not be started.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_real
from smoke_real import ROOT, Report, Server

AR_CATALOG = {
    "txt2img": "نص إلى صورة",
    "Settings": "الإعدادات",
    "Extensions": "الإضافات",
    "Gallery": "المعرض",
    "Generate": "توليد",
}
REPOS = {
    "remi-smoke-ext": {"README.md": "# Remi smoke extension\n", "localizations/ar.json": json.dumps(AR_CATALOG, ensure_ascii=False, indent=1)},
    "remi-smoke-index-ext": {"README.md": "# Remi smoke extension from the index\n"},
}


def git(*args: str, cwd: Path) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "Remi smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid", "GIT_COMMITTER_NAME": "Remi smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(cwd)}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=60).stdout.strip()


class GitFixture:
    """Extension repositories served over HTTP from ``root``, and their index."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / "work"
        self.served = root / "served"
        self.lock = threading.Lock()
        self.served.mkdir(parents=True)
        for name, files in REPOS.items():
            work = self.work / name
            work.mkdir(parents=True)
            git("init", "-q", "-b", "main", cwd=work)
            for path, text in files.items():
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                (work / path).write_text(text, encoding="utf-8")
            git("add", "-A", cwd=work)
            git("commit", "-q", "-m", "first", cwd=work)
            git("clone", "-q", "--bare", str(work), str(self.served / f"{name}.git"), cwd=root)
            git("config", "uploadpack.allowFilter", "true", cwd=self.served / f"{name}.git")  # the installer's partial clone
        handler = partial(_Handler, fixture=self)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"
        index = {
            "tags": {"localization": "Localization", "script": "Script"},
            "extensions": [
                {"name": "Remi smoke extension", "url": self.url("remi-smoke-ext"), "description": "Ships a right-to-left catalog.", "tags": ["localization"], "added": "2026-10-01", "stars": 1},
                {"name": "Remi smoke index extension", "url": self.url("remi-smoke-index-ext"), "description": "Installed from the index.", "tags": ["script"], "added": "2026-10-02", "stars": 2},
            ],
        }
        (self.served / "index.json").write_text(json.dumps(index), encoding="utf-8")
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def url(self, name: str) -> str:
        return f"{self.base}/{name}.git"

    @property
    def index_url(self) -> str:
        return f"{self.base}/index.json"

    def head(self, name: str) -> str:
        return git("rev-parse", "HEAD", cwd=self.work / name)

    def advance(self, name: str) -> str:
        """A new commit on the remote's main branch; returns its hash."""

        with self.lock:
            work = self.work / name
            stamp = work / "CHANGES.txt"
            stamp.write_text((stamp.read_text() if stamp.exists() else "") + f"change {time.time_ns()}\n")
            git("add", "-A", cwd=work)
            git("commit", "-q", "-m", "a change", cwd=work)
            git("push", "-q", str(self.served / f"{name}.git"), "main", cwd=work)
            return self.head(name)

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


class _Handler(BaseHTTPRequestHandler):
    """``/index.json``; ``/<repo>.git/...`` through ``git http-backend`` (CGI);
    ``POST /__advance/<repo>`` pushes a new commit."""

    protocol_version = "HTTP/1.0"  # one request per connection: no keep-alive bookkeeping

    def __init__(self, *args, fixture: GitFixture, **kwargs) -> None:
        self.fixture = fixture
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        self._route()

    def do_POST(self) -> None:
        self._route()

    def _route(self) -> None:
        path, _, query = self.path.partition("?")
        if self.command == "POST" and path.startswith("/__advance/") and path.removeprefix("/__advance/") in REPOS:
            self._send(200, {"Content-Type": "application/json"}, json.dumps({"commit": self.fixture.advance(path.removeprefix("/__advance/"))}).encode())
        elif self.command == "GET" and path == "/index.json":
            self._send(200, {"Content-Type": "application/json"}, (self.fixture.served / "index.json").read_bytes())
        elif path.split("/", 2)[1].removesuffix(".git") in REPOS:
            self._git(path, query)
        else:
            self._send(404, {"Content-Type": "text/plain"}, b"not found")

    def _body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16):
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            self.rfile.readline()
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _git(self, path: str, query: str) -> None:
        body = self._body()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "GIT_PROJECT_ROOT": str(self.fixture.served),
            "GIT_HTTP_EXPORT_ALL": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "REQUEST_METHOD": self.command,
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": str(len(body)),
            "REMOTE_ADDR": "127.0.0.1",
            "HTTP_CONTENT_ENCODING": self.headers.get("Content-Encoding", ""),
            "GIT_PROTOCOL": self.headers.get("Git-Protocol", ""),
        }
        done = subprocess.run(["git", "http-backend"], input=body, env=env, capture_output=True, timeout=120, check=False)
        separator = b"\r\n\r\n" if b"\r\n\r\n" in done.stdout else b"\n\n"
        head, _, payload = done.stdout.partition(separator)
        status, headers = 200, {}
        for line in head.decode("latin-1").splitlines():
            key, _, value = line.partition(":")
            if key.lower() == "status":
                status = int(value.split()[0])
            elif key:
                headers[key.strip()] = value.strip()
        if done.returncode != 0 and not head:
            status, payload = 500, done.stderr
        self._send(status, headers, payload)

    def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() != "content-length":
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def seed_outputs(data_dir: Path, count: int = 3) -> None:
    """A few images with infotext in the txt2img output folder (for the gallery and its viewer)."""

    from PIL import Image, PngImagePlugin

    folder = data_dir / "output" / "txt2img-images" / "2026-10-21"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (256, 256), ((70 * index) % 256, 120, 200 - 50 * index))
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", f"smoke image {index + 1}\nSteps: 8, Sampler: Euler, CFG scale: 1, Seed: {index + 1}, Size: 256x256")
        image.save(folder / f"{index:05d}-{index + 1}.png", pnginfo=info)


def differs(saved: dict | None, current: dict | None) -> bool:
    """The SPA's comparison of a backup row (``backupDiffers`` in extensions-model.ts)."""

    if not saved or not current:
        return True
    return any((saved.get(key) or None) != (current.get(key) or None) for key in ("commit_hash", "branch")) or bool(saved.get("enabled")) != bool(current.get("enabled"))


def wait_for(predicate: Callable[[], bool], timeout: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except (OSError, RuntimeError):
            pass
        time.sleep(interval)
    return False


def extension(server: Server, name: str) -> dict | None:
    items = server.request("GET", "/api/v2/extensions")["items"]
    return next((item for item in items if item["name"] == name), None)


def reload_ui(server: Server, done: Callable[[], bool], timeout: float = 300) -> bool:
    """Reload UI and wait until ``done()`` holds (the reloaded app answers it)."""

    server.request("POST", "/api/v2/server/reload")
    time.sleep(2)
    return wait_for(done, timeout, 2)


def run_api(server: Server, fixture: GitFixture, report: Report) -> None:
    ext, index_ext = "remi-smoke-ext", "remi-smoke-index-ext"
    listing = server.request("GET", "/api/v2/extensions")
    if not report.check("extension management is allowed", listing.get("access") is True and not listing.get("disabled_by"), json.dumps({k: listing.get(k) for k in ("access", "disabled_by")})):
        return

    first = fixture.head(ext)
    installed = server.request("POST", "/api/v2/extensions/install", {"url": fixture.url(ext)})
    item = extension(server, ext)
    report.check("install from a URL", item is not None and item.get("commit_hash") == first, f"{installed.get('name')} at {(item or {}).get('version')}, the remote at {first[:8]}")

    index = server.request("GET", "/api/v2/extensions/index?refresh=true")
    entries = {entry["url"]: entry for entry in index["items"]}
    ok = index["url"] == fixture.index_url and entries.get(fixture.url(ext), {}).get("installed") is True and entries.get(fixture.url(index_ext), {}).get("installed") is False
    report.check("the index lists the extensions and marks the installed one", ok, f"{len(entries)} entries from {index['url']}")
    server.request("POST", "/api/v2/extensions/install", {"url": entries[fixture.url(index_ext)]["url"]})
    index = server.request("GET", "/api/v2/extensions/index")
    report.check("install from the index", extension(server, index_ext) is not None and all(entry["installed"] for entry in index["items"]), ", ".join(f"{entry['name']}: installed={entry['installed']}" for entry in index["items"]))

    backup = server.request("POST", "/api/v2/extensions/config-states", {"name": "smoke"})
    detail = server.request("GET", f"/api/v2/extensions/config-states/{backup['id']}")
    changed = [row["name"] for row in detail["items"] if differs(row["saved"], row["current"])]
    builtin = [row for row in detail["items"] if row["builtin"]]
    report.check("a backup taken now matches the current state", not changed and len(detail["items"]) == backup["extensions"], f"{len(detail['items'])} rows ({len(builtin)} built-in), differing: {changed or 'none'}")

    second = fixture.advance(ext)
    checked = server.request("POST", f"/api/v2/extensions/{ext}/check")
    report.check("the per-extension check finds the update", checked.get("can_update") is True, f"status {checked.get('status')!r}")
    bulk = {item["name"]: item for item in server.request("POST", "/api/v2/extensions/check-updates", {})["items"]}
    report.check("the bulk check finds it and nothing else", bulk[ext]["can_update"] is True and bulk[index_ext]["can_update"] is False and bulk[index_ext]["status"] == "latest", f"{ext}: {bulk[ext]['status']}, {index_ext}: {bulk[index_ext]['status']}")

    before = {state["id"] for state in server.request("GET", "/api/v2/extensions/config-states")}
    server.request("POST", "/api/v2/extensions/update", {"names": [ext]})
    item = extension(server, ext)
    made = [state for state in server.request("GET", "/api/v2/extensions/config-states") if state["id"] not in before]
    report.check("update moves to the remote's commit, after a backup", (item or {}).get("commit_hash") == second and any("pre-update" in state["name"] for state in made), f"{first[:8]} -> {(item or {}).get('version')}; new backups: {[state['name'] for state in made]}")

    detail = server.request("GET", f"/api/v2/extensions/config-states/{backup['id']}")
    changed = [row["name"] for row in detail["items"] if differs(row["saved"], row["current"])]
    report.check("the backup now differs on the updated extension only", changed == [ext], f"differing: {changed}")

    restored = server.request("POST", f"/api/v2/extensions/config-states/{backup['id']}/restore", {"what": "extensions"})
    reloaded = restored == {"reload": True} and reload_ui(server, lambda: (extension(server, ext) or {}).get("commit_hash") == first)
    item = extension(server, ext)
    report.check("restore + Reload UI puts the extension back at the backed-up commit", (item or {}).get("commit_hash") == first, f"reloaded={reloaded}, now at {(item or {}).get('version')}, backed up {first[:8]}")

    locales = {locale["id"] for locale in server.request("GET", "/api/v2/app").get("locales", [])}
    catalog = server.request("GET", "/api/v2/i18n/ar") if "ar" in locales else {}
    report.check("the locale an extension ships is offered after the reload", "ar" in locales and catalog.get("Settings") == AR_CATALOG["Settings"], f"locales {sorted(locales)}")

    for name in (ext, index_ext):
        server.request("DELETE", f"/api/v2/extensions/{name}")
    left = [name for name in (ext, index_ext) if extension(server, name) is not None]
    report.check("uninstall", not left, f"left: {left or 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter for launch.py (default: this one)")
    parser.add_argument("--port", type=int, default=7883)
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "remi-extensions-smoke", help="data folder, fixture, server log and browser report (emptied first)")
    parser.add_argument("--browser", action="store_true", help="also run the SPA's @fixture Playwright tests (build the SPA first)")
    parser.add_argument("--report", type=Path, help="write the run as JSON evidence (e.g. docs/research/smoke/extensions-<date>.json)")
    parser.add_argument("--startup-timeout", type=float, default=600)
    options = parser.parse_args()

    shutil.rmtree(options.out, ignore_errors=True)
    options.out.mkdir(parents=True)
    data_dir, models = options.out / "data", options.out / "models"
    models.mkdir()
    data_dir.mkdir()
    seed_outputs(data_dir)
    fixture = GitFixture(options.out / "git")
    states_dir = ROOT / "config_states"
    states_before = set(os.listdir(states_dir)) if states_dir.is_dir() else set()

    report = Report()
    print(f"git fixture at {fixture.base}; server on port {options.port} (log {options.out / 'server.log'})", flush=True)
    server = Server(options.python, options.port, data_dir, options.out / "server.log", model_ref=models, env={"WEBUI_EXTENSIONS_INDEX": fixture.index_url})
    try:
        if not server.wait_ready(options.startup_timeout):
            print(f"the server did not start; see {options.out / 'server.log'}", file=sys.stderr)
            return 2
        server.request("POST", "/api/v2/outputs/refresh")
        try:
            run_api(server, fixture, report)
        except (RuntimeError, OSError, KeyError) as error:
            report.check("API checks ran to the end", False, str(error)[:500])
        if options.browser:
            from smoke_browser import run_browser

            env = {"REMI_E2E_EXT_FIXTURE": json.dumps({"git": fixture.base, "repos": {name: fixture.url(name) for name in REPOS}})}
            run_browser(server.base, options.out, report.check, grep="@fixture", env=env)
    finally:
        server.stop()
        fixture.close()
        made = sorted(set(os.listdir(states_dir)) - states_before) if states_dir.is_dir() else []
        for name in made:
            (states_dir / name).unlink(missing_ok=True)
        if made:
            print(f"removed the {len(made)} backups this run made from {states_dir}", flush=True)

    if options.report:
        document = report.document(options.python, smoke_real.gradio_modules_in(options.python), features=False)
        document.update({"tool": "tools/smoke_extensions.py", "model": None, "browser": options.browser})
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(report.rows) - report.failed} of {len(report.rows)} checks passed", flush=True)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
