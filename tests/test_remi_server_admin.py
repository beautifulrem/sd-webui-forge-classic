"""Extensions management (remi.extensions_admin through /api/v2/extensions)
and the system routes of the Settings page (status, licenses, sysinfo,
model reload, settings validation and defaults)."""

import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from remi import extensions_admin as admin
from remi.server.testing import server_app
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def env(webui, tmp_path, monkeypatch):
    """A server with a temporary extensions folder, config file and backups folder."""

    from modules import config_states, extensions, shared

    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    states = tmp_path / "config_states"
    monkeypatch.setattr(extensions, "extensions_dir", str(ext_dir))
    monkeypatch.setattr(extensions, "extensions", [])
    monkeypatch.setattr(config_states, "config_states_dir", str(states))
    monkeypatch.setattr(shared, "config_filename", str(tmp_path / "config.json"))
    monkeypatch.setattr(shared.opts, "disabled_extensions", [], raising=False)
    monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", False)
    monkeypatch.setattr(admin, "_index_cache", None)
    with server_app(tmp_path / "remi") as server:
        yield SimpleNamespace(server=server, client=server.client, ext_dir=ext_dir, states=states, tmp=tmp_path)


def fake_ext(name, *, path, builtin=False, remote="https://github.com/u/repo.git", enabled=True, **extra):
    fields = {"name": name, "path": str(path), "enabled": enabled, "is_builtin": builtin, "remote": remote, "branch": "main", "version": "abcdef12", "commit_hash": "abcdef1234567890", "commit_date": 1_700_000_000, "status": "", "can_update": False, "have_info_from_repo": True}
    fields.update(extra)
    ext = SimpleNamespace(**fields)
    ext.read_info_from_repo = lambda: None
    return ext


# region Helpers


def test_git_url_helpers():
    assert admin.normalize_git_url("https://github.com/a/b.git/") == "https://github.com/a/b"
    assert admin.dirname_from_url("https://github.com/a/sd-ext.git") == "sd-ext"
    assert admin.dirname_from_url("git@github.com:a/sd-ext.git") == "sd-ext"
    assert admin.web_url("git@github.com:a/b.git") == "https://github.com/a/b"
    # credentials never leak into what the API shows
    assert admin.web_url("https://user:token@github.com/a/b.git") == "https://github.com/a/b"
    assert admin.web_url("ssh://git@host/x") is None
    # nor into remotes that are not web URLs
    assert admin.display_remote("ssh://user:token@host:2222/x.git") == "ssh://user@host:2222/x.git"
    assert admin.display_remote("ssh://git@host/x") == "ssh://git@host/x"
    assert admin.display_remote("git://token@host/x") == "git://host/x"
    assert admin.display_remote("https://token@example.org/x") == "https://example.org/x"
    assert admin.display_remote("deploy:secret@host:a/b.git") == "deploy@host:a/b.git"
    assert admin.display_remote("/srv/git/x") == "/srv/git/x"
    assert admin.commit_url("https://github.com/a/b.git", "abc") == "https://github.com/a/b/commit/abc"
    assert admin.commit_url("https://example.org/a/b", "abc") is None


# endregion

# region Installed list, toggles, checks, updates


def test_listing_marks_builtin_and_links_commits(env, monkeypatch):
    from modules import extensions

    monkeypatch.setattr(extensions, "extensions", [fake_ext("user-ext", path=env.ext_dir / "user-ext"), fake_ext("core-ext", path=env.tmp / "builtin" / "core-ext", builtin=True, remote=None)])
    data = env.client.get("/api/v2/extensions").json()
    user, core = data["items"]
    assert user["builtin"] is False and user["remote_url"] == "https://github.com/u/repo"
    assert user["commit_url"] == "https://github.com/u/repo/commit/abcdef1234567890" and user["commit_date"].startswith("2023-11-14")
    assert core["builtin"] is True and core["commit_url"] is None
    assert data["access"] is True and data["index_url"].startswith("https://") and data["disabled_by"] is None
    assert isinstance(data["restartable"], bool)


def test_disable_all_and_unknown_values(env, monkeypatch):
    from modules import extensions, shared

    monkeypatch.setattr(extensions, "extensions", [fake_ext("one", path=env.ext_dir / "one")])
    monkeypatch.setattr(shared.opts, "disable_all_extensions", "none", raising=False)
    data = env.client.patch("/api/v2/extensions", json={"disable_all": "extra"}).json()
    assert data["disable_all"] == "extra" and data["disabled_by"] == "extra"
    assert json.loads(Path(shared.config_filename).read_text())["disable_all_extensions"] == "extra"
    assert env.client.patch("/api/v2/extensions", json={"disable_all": "some"}).status_code == 422


def test_checking_one_extension_reports_its_status(env, monkeypatch):
    from modules import extensions

    ext = fake_ext("one", path=env.ext_dir / "one")

    def check_updates():
        ext.can_update, ext.status = True, "new commits"

    ext.check_updates = check_updates
    monkeypatch.setattr(extensions, "extensions", [ext])
    answer = env.client.post("/api/v2/extensions/one/check").json()
    assert answer["can_update"] is True and answer["status"] == "new commits"
    assert env.client.post("/api/v2/extensions/nope/check").status_code == 404


def test_a_failing_check_marks_the_remote_error(env, monkeypatch):
    from modules import extensions

    ext = fake_ext("one", path=env.ext_dir / "one")

    def broken():
        raise RuntimeError("offline")

    ext.check_updates = broken
    monkeypatch.setattr(extensions, "extensions", [ext])
    listing = env.client.post("/api/v2/extensions/check-updates", json={"names": ["one"]}).json()
    assert listing["items"][0]["status"] == "unknown (remote error)"


def test_updates_back_up_first_and_refuse_builtin(env, monkeypatch):
    from modules import config_states, extensions

    ext = fake_ext("one", path=env.ext_dir / "one", can_update=True)
    calls = []
    ext.fetch_and_reset_hard = lambda: calls.append("reset")
    monkeypatch.setattr(extensions, "extensions", [ext, fake_ext("core", path=env.tmp, builtin=True, remote=None)])
    monkeypatch.setattr(config_states, "get_webui_config", lambda: {"remote": None, "commit_hash": None, "commit_date": None, "branch": None})
    refused = env.client.post("/api/v2/extensions/update", json={"names": ["core"]})
    assert refused.status_code == 422
    assert refused.json() == {"detail": "not updatable (built-in or not a git repository): core", "template": "not updatable (built-in or not a git repository): {{names}}", "params": {"names": "core"}}
    listing = env.client.post("/api/v2/extensions/update", json={"names": ["one"]}).json()
    assert calls == ["reset"] and listing["items"][0]["can_update"] is False
    backups = env.client.get("/api/v2/extensions/config-states").json()
    assert [item["name"] for item in backups] == ["Backup (pre-update)"]


def test_a_failed_update_is_a_502_naming_the_extension(env, monkeypatch):
    from modules import config_states, extensions

    good, bad = fake_ext("good", path=env.ext_dir / "good", can_update=True), fake_ext("bad", path=env.ext_dir / "bad", can_update=True)
    good.fetch_and_reset_hard = lambda: None

    def offline():
        raise admin.GitError("git {{command}} failed: {{reason}}", command="fetch", reason="Could not resolve host")

    bad.fetch_and_reset_hard = offline
    monkeypatch.setattr(extensions, "extensions", [good, bad])
    monkeypatch.setattr(config_states, "get_webui_config", lambda: {"remote": None, "commit_hash": None, "commit_date": None, "branch": None})
    response = env.client.post("/api/v2/extensions/update", json={"names": ["good", "bad"]})
    assert response.status_code == 502
    body = response.json()
    assert body["template"] == "could not update {{names}}: {{reason}}" and body["params"]["names"] == "bad" and "Could not resolve host" in body["detail"]
    assert good.status == "latest" and bad.can_update is True


def test_writes_need_extension_access(env, monkeypatch):
    from modules import extensions, shared

    monkeypatch.setattr(extensions, "extensions", [fake_ext("one", path=env.ext_dir / "one")])
    saved = env.client.post("/api/v2/extensions/config-states", json={"name": "kept"}).json()
    monkeypatch.setattr(shared.cmd_opts, "disable_extension_access", True)
    assert env.client.get("/api/v2/extensions").json()["access"] is False
    for method, path, body in (
        ("PATCH", "/api/v2/extensions", {"disabled": ["one"]}),
        ("DELETE", "/api/v2/extensions/one", None),
        ("POST", "/api/v2/extensions/one/check", None),
        ("POST", "/api/v2/extensions/check-updates", None),
        ("POST", "/api/v2/extensions/update", {"names": ["one"]}),
        ("POST", "/api/v2/extensions/install", {"url": "https://example.com/x.git"}),
        ("GET", "/api/v2/extensions/index", None),
        ("POST", "/api/v2/extensions/config-states", {"name": "x"}),
        ("POST", f"/api/v2/extensions/config-states/{saved['id']}/restore", {"what": "extensions"}),
        ("DELETE", f"/api/v2/extensions/config-states/{saved['id']}", None),
    ):
        response = env.client.request(method, path, json=body)
        assert response.status_code == 403, path
        assert response.json()["template"].startswith("extension access is disabled"), path
    assert shared.opts.disabled_extensions == [] and [state["id"] for state in env.client.get("/api/v2/extensions/config-states").json()] == [saved["id"]]


# endregion

# region Install and uninstall


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def test_install_clones_runs_the_installer_and_refuses_duplicates(env, monkeypatch):
    from modules import extensions, launch_utils

    source = env.tmp / "source" / "sd-demo"
    source.mkdir(parents=True)
    (source / "README.md").write_text("demo")
    _git("init", "-q", "-b", "main", cwd=source)
    _git("add", ".", cwd=source)
    _git("commit", "-q", "-m", "init", cwd=source)
    url = source.as_uri()
    installed = []
    # a local repository stands in for the remote (file:// is not allowed for real installs)
    monkeypatch.setattr(admin, "valid_git_url", lambda value: value.startswith("file://"))
    monkeypatch.setattr(admin, "INSTALL_PROTOCOLS", admin.INSTALL_PROTOCOLS + ":file")
    monkeypatch.setattr(launch_utils, "run_extension_installer", lambda path: installed.append(path))
    monkeypatch.setattr(extensions, "list_extensions", lambda: extensions.extensions.append(fake_ext("sd-demo", path=env.ext_dir / "sd-demo", remote=url)))
    monkeypatch.setattr("modules.paths.data_path", str(env.tmp), raising=False)

    response = env.client.post("/api/v2/extensions/install", json={"url": url})
    assert response.status_code == 201, response.text
    assert response.json()["name"] == "sd-demo" and (env.ext_dir / "sd-demo" / "README.md").is_file()
    assert installed == [str(env.ext_dir / "sd-demo")]
    assert env.client.post("/api/v2/extensions/install", json={"url": url}).status_code == 409
    assert env.client.post("/api/v2/extensions/install", json={"url": url, "dirname": "other"}).status_code == 409
    bad_branch = env.client.post("/api/v2/extensions/install", json={"url": url, "dirname": "x", "branch": "a b"})
    assert bad_branch.status_code == 422


def test_install_rejects_bad_urls_and_folder_names(env):
    assert env.client.post("/api/v2/extensions/install", json={"url": "file:///etc"}).status_code == 422
    assert env.client.post("/api/v2/extensions/install", json={"url": "https://example.com/x.git", "dirname": "../up"}).status_code == 422
    (env.ext_dir / "taken").mkdir()
    assert env.client.post("/api/v2/extensions/install", json={"url": "https://example.com/taken.git"}).status_code == 409


def test_uninstall_deletes_user_extensions_only(env, monkeypatch):
    from modules import extensions, shared

    folder = env.ext_dir / "gone"
    (folder / ".git").mkdir(parents=True)
    pack = folder / ".git" / "pack"
    pack.write_text("x")
    pack.chmod(0o444)  # read-only like git's pack files
    builtin = fake_ext("core", path=env.tmp / "core", builtin=True)
    monkeypatch.setattr(extensions, "extensions", [fake_ext("gone", path=folder), builtin])
    monkeypatch.setattr(shared.opts, "disabled_extensions", ["gone"], raising=False)
    assert env.client.delete("/api/v2/extensions/core").status_code == 422
    assert env.client.delete("/api/v2/extensions/gone").status_code == 204
    assert not folder.exists() and shared.opts.disabled_extensions == []
    assert [item["name"] for item in env.client.get("/api/v2/extensions").json()["items"]] == ["core"]
    assert env.client.delete("/api/v2/extensions/gone").status_code == 404


def test_uninstall_refuses_folders_outside_the_extensions_folder(env, monkeypatch):
    from modules import extensions

    outside = env.tmp / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(extensions, "extensions", [fake_ext("odd", path=outside)])
    assert env.client.delete("/api/v2/extensions/odd").status_code == 422
    assert outside.exists()


# endregion

# region Index


INDEX = {
    "tags": {"script": "a script"},
    "extensions": [
        {"name": "Tagger", "url": "https://github.com/a/tagger.git", "description": "tags", "tags": ["script"], "added": "2023-01-01", "stars": "12"},
        {"name": "Mine", "url": "https://github.com/u/repo", "tags": ["tab"]},
        {"name": "No URL"},
        {"name": "Bad", "url": "javascript:alert(1)"},
        "junk",
    ],
}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_index_is_normalised_marks_installed_and_is_cached(env, monkeypatch):
    from modules import extensions

    monkeypatch.setattr(extensions, "extensions", [fake_ext("repo", path=env.ext_dir / "repo")])
    fetched = []

    def opener(request, timeout):
        fetched.append(request.full_url)
        return _Response(json.dumps(INDEX).encode())

    monkeypatch.setattr(admin, "host_is_public", lambda host: host == "example.org")
    monkeypatch.setattr(admin, "open_public_url", opener)
    data = env.client.get("/api/v2/extensions/index", params={"url": "https://example.org/index.json"}).json()
    assert [item["name"] for item in data["items"]] == ["Tagger", "Mine"]
    tagger, mine = data["items"]
    assert tagger["stars"] == 12 and tagger["order"] == 0 and tagger["installed"] is False
    assert mine["installed"] is True
    assert data["tags"] == {"script": "a script", "tab": "", "installed": "extensions that are already installed"}
    env.client.get("/api/v2/extensions/index", params={"url": "https://example.org/index.json"})
    assert len(fetched) == 1
    env.client.get("/api/v2/extensions/index", params={"url": "https://example.org/index.json", "refresh": True})
    assert len(fetched) == 2


def test_index_errors(env, monkeypatch):
    assert env.client.get("/api/v2/extensions/index", params={"url": "file:///etc/passwd"}).status_code == 422

    def offline(request, timeout):
        raise OSError("no route")

    monkeypatch.setattr(admin, "host_is_public", lambda host: host == "example.org")
    monkeypatch.setattr(admin, "open_public_url", offline)
    response = env.client.get("/api/v2/extensions/index", params={"url": "https://example.org/i.json"})
    assert response.status_code == 502 and "no route" in response.json()["detail"]
    monkeypatch.setattr(admin, "open_public_url", lambda request, timeout: _Response(b"[1, 2]"))
    assert env.client.get("/api/v2/extensions/index", params={"url": "https://example.org/i.json"}).status_code == 502


def test_client_named_index_urls_must_be_public(env, monkeypatch):
    # A URL the client names must not make the server probe its own network
    # (loopback, private, link-local addresses), neither directly nor through
    # a redirect; the configured index (WEBUI_EXTENSIONS_INDEX) may be local.
    fetched = []

    def opener(request, timeout):
        fetched.append(request.full_url)
        return _Response(json.dumps(INDEX).encode())

    monkeypatch.setattr(admin, "open_public_url", opener)
    for url in ("http://127.0.0.1:7860/api/v2/options", "http://localhost/", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://[::ffff:127.0.0.1]/"):
        response = env.client.get("/api/v2/extensions/index", params={"url": url, "refresh": True})
        assert response.status_code == 422, url
        assert "public internet address" in response.json()["detail"]
    assert fetched == []
    monkeypatch.setattr(admin.urllib.request, "urlopen", opener)
    monkeypatch.setenv("WEBUI_EXTENSIONS_INDEX", "http://127.0.0.1:9/index.json")
    assert env.client.get("/api/v2/extensions/index", params={"url": "http://127.0.0.1:9/index.json"}).status_code == 200
    assert env.client.get("/api/v2/extensions/index").status_code == 200
    assert fetched == ["http://127.0.0.1:9/index.json"]  # (cached the second time)


def test_index_redirects_are_checked_per_hop(monkeypatch):
    import urllib.request

    monkeypatch.setattr(admin, "host_is_public", lambda host: host == "example.org")
    handler = admin._PublicRedirects()
    request = urllib.request.Request("https://example.org/index.json")
    with pytest.raises(admin.AdminError, match="public internet address"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://127.0.0.1:7860/api/v2/options")
    with pytest.raises(admin.AdminError, match="http"):
        handler.redirect_request(request, None, 302, "Found", {}, "file:///etc/passwd")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://example.org/moved.json").full_url == "https://example.org/moved.json"


def test_public_hosts_are_global_addresses():
    assert not admin.host_is_public("127.0.0.1") and not admin.host_is_public("10.0.0.1") and not admin.host_is_public("::1")
    assert not admin.host_is_public("") and not admin.host_is_public(None)
    assert admin.host_is_public("8.8.8.8")


# endregion

# region Config states


def test_config_states_round_trip(env, monkeypatch):
    from modules import config_states, extensions, shared

    monkeypatch.setattr(config_states, "get_webui_config", lambda: {"remote": "https://github.com/me/webui.git", "commit_hash": "c0ffee", "commit_date": 1_700_000_000, "branch": "main"})
    one = fake_ext("one", path=env.ext_dir / "one")
    monkeypatch.setattr(extensions, "extensions", [one])
    saved = env.client.post("/api/v2/extensions/config-states", json={"name": "Before ../changes"}).json()
    assert saved["name"] == "changes" and saved["extensions"] == 1 and saved["id"].endswith(".json")
    # the extension moves on and another one appears
    one.commit_hash = "1234567890abcdef"
    monkeypatch.setattr(extensions, "extensions", [one, fake_ext("two", path=env.ext_dir / "two", remote="https://github.com/u/two")])
    states = env.client.get("/api/v2/extensions/config-states").json()
    assert [state["id"] for state in states] == [saved["id"]]
    detail = env.client.get(f"/api/v2/extensions/config-states/{saved['id']}").json()
    rows = {row["name"]: row for row in detail["items"]}
    assert rows["one"]["saved"]["commit_hash"] == "abcdef1234567890" and rows["one"]["current"]["commit_hash"] == "1234567890abcdef"
    assert rows["two"]["saved"] is None and rows["two"]["current"]["enabled"] is True
    assert detail["webui"]["saved"]["commit_url"] == "https://github.com/me/webui/commit/c0ffee"
    restored = env.client.post(f"/api/v2/extensions/config-states/{saved['id']}/restore", json={"what": "extensions"}).json()
    assert restored == {"reload": True}
    assert shared.opts.restore_config_state_file.endswith(saved["id"])
    shared.opts.restore_config_state_file = ""
    assert env.client.get("/api/v2/extensions/config-states/..%2Fconfig.json").status_code == 404
    assert env.client.get("/api/v2/extensions/config-states/nope.json").status_code == 404
    assert env.client.delete(f"/api/v2/extensions/config-states/{saved['id']}").status_code == 204
    assert env.client.get("/api/v2/extensions/config-states").json() == []


def test_the_webui_checkout_is_never_restored(env, monkeypatch):
    from modules import config_states, extensions, shared

    monkeypatch.setattr(config_states, "get_webui_config", lambda: {"remote": "https://github.com/me/webui.git", "commit_hash": "c0ffee", "commit_date": 1_700_000_000, "branch": "main"})
    monkeypatch.setattr(extensions, "extensions", [])
    resets = []
    monkeypatch.setattr(config_states, "restore_webui_config", lambda state: resets.append(state))
    saved = env.client.post("/api/v2/extensions/config-states", json={"name": "old"}).json()
    for what in ("webui", "both"):
        assert env.client.post(f"/api/v2/extensions/config-states/{saved['id']}/restore", json={"what": what}).status_code == 422
    with pytest.raises(admin.AdminError, match="never reset"):
        admin.restore_config_state(saved["id"], "webui")
    assert resets == [] and not shared.opts.restore_config_state_file


# endregion

# region Git (non-interactive, bounded)


def _repo(path, *files):
    path.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=path)
    for name in files or ("README.md",):
        (path / name).write_text(name)
    _git("add", ".", cwd=path)
    _git("commit", "-q", "-m", "init", cwd=path)
    return path


def test_git_never_prompts():
    env = admin.git_environment()
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell script")
def test_git_failures_and_timeouts_are_502(tmp_path, monkeypatch):
    with pytest.raises(admin.GitError) as failed:
        admin.run_git(["rev-parse", "HEAD"], cwd=tmp_path)
    assert failed.value.status == 502 and failed.value.params["command"] == "rev-parse"
    slow = tmp_path / "slow-git"
    slow.write_text("#!/bin/sh\nsleep 30\n")
    slow.chmod(0o755)
    monkeypatch.setenv("GIT", str(slow))
    import time

    started = time.monotonic()
    with pytest.raises(admin.GitError, match="timed out after 1 s"):
        admin.run_git(["fetch", "origin"], cwd=tmp_path, timeout=1)
    assert time.monotonic() - started < 10
    monkeypatch.setenv("GIT", str(tmp_path / "missing-git"))
    with pytest.raises(admin.GitError, match="could not be started"):
        admin.run_git(["status"])


def test_check_and_reset_follow_the_remote(tmp_path):
    upstream = _repo(tmp_path / "upstream")
    clone = tmp_path / "clone"
    _git("clone", "-q", str(upstream), str(clone), cwd=tmp_path)
    assert admin.check_repo(clone, "main") == (False, "latest")
    (upstream / "new.txt").write_text("x")
    _git("add", ".", cwd=upstream)
    _git("commit", "-q", "-m", "more", cwd=upstream)
    assert admin.check_repo(clone, "main") == (True, "new commits")
    admin.reset_repo(clone, "main")
    assert (clone / "new.txt").is_file() and admin.check_repo(clone, "main") == (False, "latest")


def test_an_unreachable_remote_is_a_check_failure_not_a_hang(env, monkeypatch, tmp_path):
    from modules import extensions

    clone = _repo(tmp_path / "local")
    _git("remote", "add", "origin", "https://user@127.0.0.1:9/nope.git", cwd=clone)
    ext = fake_ext("local", path=clone, remote="https://127.0.0.1:9/nope.git")
    ext.check_updates = lambda: extensions.Extension.check_updates(ext)
    monkeypatch.setattr(extensions, "extensions", [ext])
    answer = env.client.post("/api/v2/extensions/local/check").json()
    assert answer["status"] == "unknown (remote error)" and answer["can_update"] is False


def test_install_reports_git_errors_as_502(env, monkeypatch):
    monkeypatch.setattr("modules.paths.data_path", str(env.tmp), raising=False)
    response = env.client.post("/api/v2/extensions/install", json={"url": "https://127.0.0.1:9/nothing.git"})
    assert response.status_code == 502 and response.json()["template"] == "git {{command}} failed: {{reason}}"
    assert not (env.ext_dir / "nothing").exists() and not (env.tmp / "tmp" / "nothing").exists()


# endregion

# region System


def test_server_status(env):
    data = env.client.get("/api/v2/server/status").json()
    assert data["extension_access"] is True and data["settings_frozen"] is False and data["version"]
    assert data["model_loaded"] is None or isinstance(data["model_loaded"], str)


def test_licenses_are_structured_from_the_html_file(env):
    data = env.client.get("/api/v2/licenses").json()
    titles = [group["title"] for group in data["groups"]]
    assert titles == ["Special Thanks", "Diffusion Engine", "Misc. Components"]
    entries = [item for group in data["groups"] for item in group["items"]]
    assert len(entries) == 14 and all(item["license_url"].startswith("https://raw.githubusercontent.com/") for item in entries)
    comfy = entries[0]
    assert comfy == {"id": "comfy", "name": "ComfyUI", "url": "https://github.com/Comfy-Org/ComfyUI", "description": "Many inference code and structural optimizations were taken from this repository", "license_url": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/LICENSE"}


def test_license_parser_drops_markup_and_unsafe_links():
    from remi.server.routes.admin import parse_licenses

    data = parse_licenses("<style>h2{}</style><h2><a href='javascript:x'>A <b>B</b></a></h2><small>d &amp; e</small><pre data-license='http://insecure'></pre>")
    (group,) = data["groups"]
    assert group["items"] == [{"id": "0-0", "name": "A B", "url": None, "description": "d & e", "license_url": None}]


def test_license_texts_come_only_from_the_host_the_csp_allows():
    from remi.server.auth import LICENSE_TEXT_ORIGIN, html_headers
    from remi.server.routes.admin import parse_licenses

    data = parse_licenses("<h2>A</h2><pre data-license='https://example.com/LICENSE'></pre><h2>B</h2><pre data-license='https://raw.githubusercontent.com.evil.example/LICENSE'></pre><h2>C</h2><pre data-license='https://raw.githubusercontent.com/o/r/main/LICENSE'></pre>")
    assert [item["license_url"] for item in data["groups"][0]["items"]] == [None, None, "https://raw.githubusercontent.com/o/r/main/LICENSE"]
    assert f"connect-src 'self' {LICENSE_TEXT_ORIGIN};" in html_headers(b"")["Content-Security-Policy"]


def test_sysinfo_check(env):
    from modules import sysinfo

    text = json.dumps({"Checksum": sysinfo.checksum_token, "x": 1}, indent=4)
    import hashlib

    valid = text.replace(sysinfo.checksum_token, hashlib.sha256(text.encode()).hexdigest())
    assert env.client.post("/api/v2/sysinfo/check", json={"text": valid}).json() == {"valid": True}
    assert env.client.post("/api/v2/sysinfo/check", json={"text": valid.replace('"x": 1', '"x": 2')}).json() == {"valid": False}


def test_options_validate_applies_nothing(env, webui):
    before = webui.opts.data.get("jpeg_quality")
    assert env.client.post("/api/v2/options/validate", json={"jpeg_quality": 50}).json()["ok"] is True
    assert webui.opts.data.get("jpeg_quality") == before
    bad = env.client.post("/api/v2/options/validate", json={"jpeg_quality": "high", "nope": 1})
    assert bad.status_code == 422 and {error["param"] for error in bad.json()["errors"]} == {"jpeg_quality", "nope"}


def test_settings_schema_carries_the_builtin_defaults(env, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "jpeg_quality", 42)
    data = env.client.get("/api/v2/options").json()
    assert data["values"]["jpeg_quality"] == 42
    assert data["defaults"]["jpeg_quality"] == webui.opts.data_labels["jpeg_quality"].default
    assert set(data["defaults"]) == set(data["values"])


def test_model_reload_runs_behind_generation_and_reports(env, monkeypatch):
    from remi.server import runtime

    calls = []
    monkeypatch.setattr(runtime, "_main_entry", lambda: SimpleNamespace(refresh_model_loading_parameters=lambda: calls.append("params")))
    import modules.sd_models as sd_models

    monkeypatch.setattr(sd_models, "forge_model_reload", lambda: calls.append(("load", sd_models.model_data.forge_hash)))
    monkeypatch.setattr(sd_models.model_data, "forge_hash", "same", raising=False)
    events = []
    unsubscribe = env.server.services.bus.subscribe(SimpleNamespace(send_event=events.append, send_frame=lambda frame: None, wants_frames=lambda: False))
    try:
        assert env.client.post("/api/v2/models/reload").json()["ok"] is True
        # force: load again although the parameters did not change
        assert env.client.post("/api/v2/models/reload", json={"force": True}).json()["ok"] is True
    finally:
        unsubscribe()
    assert calls == ["params", ("load", "same"), "params", ("load", "")]
    assert [event["type"] for event in events if event["type"].startswith("model.")] == ["model.loading", "model.loaded"] * 2


def test_loaded_models_lists_the_memory_manager(env, monkeypatch):
    import sys

    entry = SimpleNamespace(model=SimpleNamespace(model=type("KModel", (), {})()), model_loaded_memory=lambda: 3 * 2**20, device="cuda:0")
    monkeypatch.setitem(sys.modules, "backend.memory_management", SimpleNamespace(current_loaded_models=[entry]))
    data = env.client.get("/api/v2/models/loaded").json()
    assert data["items"] == [{"name": "KModel", "memory_mb": 3, "device": "cuda:0"}]


def test_hashing_every_checkpoint_runs_in_the_background(env):
    import threading

    from remi.server.routes import admin as routes

    gate = threading.Event()
    hashed = []

    class Checkpoint:
        def __init__(self, name):
            self.name = name

        def calculate_shorthash(self):
            gate.wait(5)
            if self.name == "broken":
                raise OSError("unreadable")
            hashed.append(self.name)

    started = routes.hasher.start(2, [Checkpoint("a"), Checkpoint("b"), Checkpoint("broken")])
    assert started["running"] is True and started["total"] == 3
    assert env.client.post("/api/v2/models/hashes", json={"threads": 2}).status_code == 409
    gate.set()
    for _ in range(100):
        status = env.client.get("/api/v2/models/hashes").json()
        if not status["running"]:
            break
        threading.Event().wait(0.05)
    assert status["done"] == 3 and status["failed"] == ["broken"] and sorted(hashed) == ["a", "b"] and status["finished_at"]
    assert env.client.post("/api/v2/models/hashes", json={"threads": 99}).status_code == 422


# endregion
