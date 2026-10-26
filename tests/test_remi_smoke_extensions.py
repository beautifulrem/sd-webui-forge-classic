"""tools/smoke_extensions.py: its git fixture serves extension repositories the way the
installer and the update check reach a git host (smart HTTP, partial clone), moves a remote
on, and serves the index; its backup comparison is the SPA's."""

import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from remi import extensions_admin as admin

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
_spec = importlib.util.spec_from_file_location("smoke_extensions", _TOOLS / "smoke_extensions.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_extensions", smoke)
_spec.loader.exec_module(smoke)

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def fixture(tmp_path):
    served = smoke.GitFixture(tmp_path / "git")
    yield served
    served.close()


def _get(url: str, method: str = "GET") -> tuple[int, bytes]:
    try:
        with _opener.open(urllib.request.Request(url, method=method), timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_installs_checks_and_updates_over_the_fixture(fixture, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    clone = tmp_path / "remi-smoke-ext"
    # the installer's clone (a partial clone, only the protocols it allows)
    admin.run_git(["clone", "--filter=blob:none", "--recurse-submodules", "--", fixture.url("remi-smoke-ext"), str(clone)], extra_env={"GIT_ALLOW_PROTOCOL": admin.INSTALL_PROTOCOLS})
    assert json.loads((clone / "localizations" / "ar.json").read_text(encoding="utf-8"))["Settings"] == smoke.AR_CATALOG["Settings"]
    first = fixture.head("remi-smoke-ext")
    assert admin.check_repo(clone, "main") == (False, "latest")

    status, body = _get(f"{fixture.base}/__advance/remi-smoke-ext", "POST")
    second = json.loads(body)["commit"]
    assert status == 200 and second != first and second == fixture.head("remi-smoke-ext")
    assert admin.check_repo(clone, "main") == (True, "new commits")
    admin.reset_repo(clone, "main")
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True, check=True).stdout.strip() == second
    admin.reset_repo(clone, "main", first)  # what a restore does
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True, check=True).stdout.strip() == first


def test_serves_the_index_and_nothing_else(fixture):
    status, body = _get(fixture.index_url)
    index = json.loads(body)
    assert status == 200 and [entry["url"] for entry in index["extensions"]] == [fixture.url(name) for name in smoke.REPOS]
    assert all(admin.valid_git_url(entry["url"]) for entry in index["extensions"])
    assert _get(f"{fixture.base}/other.git/info/refs")[0] == 404
    assert _get(f"{fixture.base}/__advance/other", "POST")[0] == 404


def test_backup_rows_compare_like_the_spa():
    # a built-in recorded with "" compares equal to its live None (BackupView's backupDiffers)
    assert not smoke.differs({"commit_hash": "", "branch": None, "enabled": True}, {"commit_hash": None, "branch": "", "enabled": True})
    assert smoke.differs({"commit_hash": "a", "enabled": True}, {"commit_hash": "b", "enabled": True})
    assert smoke.differs({"commit_hash": "a", "enabled": True}, {"commit_hash": "a", "enabled": False})
    assert smoke.differs(None, {"commit_hash": "a"})
