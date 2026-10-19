"""The Prompt Workshop page (remi.pages): layout parity with the former Gradio
tab, schema export, actions (no network: requests are faked, and none is
made unless a grab / download action runs), credentials and zh_CN coverage."""

import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

EXT = Path(__file__).resolve().parents[1]
REPO = EXT.parents[1]
SCRIPT = EXT / "scripts" / "prompt_workshop.py"

_spec = importlib.util.spec_from_file_location("remi_page_harness", REPO / "tests" / "remi_page_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def pw(webui, tmp_path, monkeypatch):
    from remi import pages

    pages.clear()
    module = harness.load_page_script(SCRIPT, "prompt_workshop_under_test")
    monkeypatch.setattr(module, "_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(module, "_LEGACY_CONFIG_PATH", str(tmp_path / "legacy.json"))
    monkeypatch.setattr(module, "_STATE_DIR", str(tmp_path))

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected outbound request")

    monkeypatch.setattr(module.requests, "get", no_network)
    yield module
    pages.clear()


@pytest.fixture
def page(pw):
    from remi import pages

    return pages.get("prompt-workshop")


# region Layout

CORE = ["quality", "subject", "character", "series", "artist", "general"]
EXTRA = ["composition", "pose", "clothing", "setting", "lighting", "features", "style", "content"]
LEGACY_MAP = {
    0: "consolidated",
    **{1 + i: name for i, name in enumerate(CORE)},
    **{7 + i: name for i, name in enumerate(EXTRA)},
    15: "gel_api_key", 16: "gel_user_id", 17: "r34_api_key", 18: "r34_user_id",
    19: "source", 20: "rating", 21: "search_tag", 22: "blacklist_tags", 23: "blacklist_cats",
    24: "count_filter_enabled", 25: "count_filter_op", 26: "count_filter_threshold", 27: "count_filter_threshold_high",
    28: "hunt_mode", 29: "count_filter_targets", 30: "post_url", 31: "preview_image", 32: "grabbed_info",
    33: "grabbed", 34: "post_id",
    **{35 + i: f"grab_{name}" for i, name in enumerate(CORE)},
    41: "preview_url",
}


def test_layout_matches_the_legacy_tab(page):
    snapshot = harness.load_snapshot(SCRIPT)
    assert snapshot["title"] == page.title == "Prompt Workshop"
    # api_key fields are masked inputs now (write-only: a saved key is never sent back)
    secret = {"kind": "custom"}
    # "Source" alone is the SPA's own word for a source image ("源图")
    harness.assert_page_matches(page.layout(), snapshot, LEGACY_MAP, {15: secret, 17: secret, 19: {"label": "Booru source"}})
    names = page.layout().names
    order = [names.index(name) for name in LEGACY_MAP.values()]
    assert order == sorted(order)


def test_schema_export(page):
    data = page.schema()
    json.dumps(data, allow_nan=False)
    assert data["owner"] == "page:prompt-workshop" and data["on_load"] == "refresh" and "send_to" not in data
    widgets = {node["name"]: node for node in _walk(data["layout"]) if node.get("type") == "custom"}
    assert widgets["send_consolidated"]["props"] == {"label": "Send to txt2img Prompt", "to": "prompt", "source": "consolidated", "mode": "append"}
    assert widgets["replace_consolidated"]["props"]["mode"] == "replace"
    assert [widgets[f"send_{name}"]["props"]["navigate"] for name in CORE] == [False] * 6
    assert widgets["send_image"]["props"] == {"label": "📷 Send Image to JoyCaption", "to": "page", "page": "joycaption", "param": "image", "source": "preview_image", "action": "download", "variant": "outline"}
    assert widgets["open_post"]["props"]["to"] == "url" and widgets["gel_api_key"]["widget"] == "secret-text"
    # secrets and buttons never reach drafts or jobs
    assert widgets["gel_api_key"]["persist"] is False and widgets["gel_api_key"]["submit"] is False
    text = {node["name"]: node for node in _walk(data["layout"]) if node.get("type") == "text"}
    assert all(text[name]["prompt"] for name in ["consolidated", *CORE, *EXTRA, "grabbed"])  # tag completion


def _walk(node):
    if isinstance(node, dict):
        yield node
        for child in node.get("children", []):
            yield from _walk(child)


def test_no_tab_code_left():
    harness.assert_page_standalone(EXT, [SCRIPT])


# endregion

# region Actions


def test_build_and_clear(page, pw):
    ctx = harness.fake_ctx(page)
    values = {"quality": "masterpiece, best_quality", "subject": "1girl", "artist": "nnn yryr, @foo", "general": "smile, 1girl", "pose": "sitting"}
    built = harness.run_action(page, "build", ctx, values)
    expected = pw.build_prompt(*(values.get(name, "") for name in CORE + EXTRA))
    assert built == {"consolidated": {"value": expected}}
    assert expected == "masterpiece, best quality, 1girl, @nnn yryr, @foo, smile, sitting"
    cleared = harness.run_action(page, "clear_all", ctx, values)
    assert cleared["source"] == {"value": "Danbooru"} and cleared["count_filter_threshold"] == {"value": 50} and cleared["grab_general"] == {"value": ""}
    assert "gel_user_id" not in cleared and "preview_image" not in cleared  # credentials and the preview stay


def test_grab_maps_the_grabber_result(page, pw, monkeypatch):
    calls = []
    result = ("q, s, c", "q", "s", "c", "ser", "@a", "g", "Post: #1\nImage: https://img3.gelbooru.com/y.png", "https://x/posts/1")

    def fake_random(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(pw, "grab_random", fake_random)
    monkeypatch.setattr(pw, "grab_by_id", lambda *args: calls.append(args) or result)
    ctx = harness.fake_ctx(page)
    values = {"source": "Gelbooru", "rating": "Explicit", "search_tag": "long hair", "blacklist_cats": ["Artist"], "count_filter_enabled": True, "count_filter_op": "between", "count_filter_threshold": 10, "count_filter_threshold_high": 5, "hunt_mode": True}
    started = harness.run_action(page, "grab", ctx, values)
    assert started["__outputs__"]["task:grab"]["state"] in ("queued", "running", "completed")  # a task: progress and Cancel
    patch = harness.wait_task(page, ctx, "grab")
    assert calls[0] == ("Gelbooru", "Explicit", "long hair", "", ["Artist"], True, "between", 10, 5, [], True)
    assert patch["grabbed"] == {"value": "q, s, c"} and patch["grab_artist"] == {"value": "@a"} and patch["post_url"] == {"value": "https://x/posts/1"}
    assert pw._last_grab(None) == ("Gelbooru", "https://img3.gelbooru.com/y.png", "1")  # kept on the server
    harness.run_action(page, "grab_by_id", ctx, {"post_id": "123"})
    harness.wait_task(page, ctx, "grab_by_id")
    assert calls[1][:2] == ("Danbooru", "123") and len(calls[1]) == 9
    sent = harness.run_action(page, "send_grabbed", ctx, {f"grab_{name}": name.upper() for name in CORE})
    assert sent == {name: {"value": name.upper()} for name in CORE}
    harness.run_action(page, "clear_all", ctx)
    assert pw._last_grab(None) is None


def test_a_grab_can_be_cancelled_between_requests(page, pw, monkeypatch):
    import threading

    gate, requested = threading.Event(), []

    def slow_get(*args, **kwargs):
        requested.append(args)
        gate.wait(5)
        raise pw.requests.ConnectionError("down")

    def hunting(*args):
        for _ in range(50):  # like hunt mode: many requests, each failure swallowed
            try:
                pw._http_get("https://danbooru.donmai.us/tags.json")
            except Exception:
                pass
        return ("never",) * 9

    monkeypatch.setattr(pw.requests, "get", slow_get)
    monkeypatch.setattr(pw, "grab_random", hunting)
    ctx = harness.fake_ctx(page)
    harness.run_action(page, "grab", ctx, {"hunt_mode": True})
    harness.run_action(page, "__cancel__", ctx, task="grab")
    gate.set()
    done = harness.wait_task(page, ctx, "grab")
    assert done["__outputs__"]["task:grab"]["state"] == "cancelled" and "grabbed" not in done
    assert len(requested) <= 1


def test_credentials_are_write_only(page, pw):
    ctx = harness.fake_ctx(page)
    status = harness.run_action(page, "refresh", ctx)["__outputs__"]
    assert status["gel_status"] == {"text": "⚠ Not set — {{source}} grabs return 401.", "params": {"source": "Gelbooru"}}
    saved = harness.run_action(page, "save_gelbooru", ctx, {"gel_api_key": " KEY ", "gel_user_id": "42"})
    assert saved["gel_api_key"] == {"value": ""} and saved["__outputs__"]["gel_status"]["text"] == "✓ {{source}} credentials set."
    assert pw._get_credentials("gelbooru") == ("KEY", "42")
    # an empty key field keeps the saved key; the user id is shown (not secret)
    harness.run_action(page, "save_gelbooru", ctx, {"gel_api_key": "", "gel_user_id": "43"})
    assert pw._get_credentials("gelbooru") == ("KEY", "43")
    schema = page.schema()
    defaults = schema["defaults"]
    assert defaults["gel_api_key"] == "" and defaults["gel_user_id"] == "43"
    assert "KEY" not in json.dumps(schema)
    gel = next(node for node in _walk(schema["layout"]) if node.get("title") == "Gelbooru API credentials")
    assert gel["open"] is False  # set credentials fold away
    cleared = harness.run_action(page, "clear_rule34", ctx)
    assert pw._get_credentials("rule34") == ("", "") and cleared["r34_user_id"] == {"value": ""}
    harness.run_action(page, "clear_gelbooru", ctx)
    assert pw._get_credentials("gelbooru") == ("", "")


class FakeResponse:
    def __init__(self, data, headers=None, status=200):
        self.data, self.headers, self.status_code = data, headers or {}, status

    @property
    def is_redirect(self):
        return "Location" in self.headers and self.status_code in (301, 302, 303, 307, 308)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start : start + chunk_size]


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), "red").save(buffer, "PNG")
    return buffer.getvalue()


PUBLIC = {"cdn.donmai.us": "104.18.0.1", "img3.gelbooru.com": "104.18.0.2", "inside.donmai.us": "10.0.0.5", "mapped.donmai.us": "::ffff:127.0.0.1"}


@pytest.fixture
def dns(pw, monkeypatch):
    import ipaddress

    monkeypatch.setattr(pw, "_resolve_host", lambda host: {ipaddress.ip_address(PUBLIC[host])} if host in PUBLIC else (_ for _ in ()).throw(OSError("unknown host")))


def test_download_fills_the_preview_once(page, pw, tmp_path, monkeypatch, dns):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    requested = []

    def fake_get(url, headers, timeout, stream, allow_redirects):
        assert allow_redirects is False
        requested.append((url, headers.get("Referer")))
        return FakeResponse(_png_bytes())

    monkeypatch.setattr(pw.requests, "get", fake_get)
    missing = harness.run_action(page, "download", ctx, {"grabbed_info": "Image: https://cdn.donmai.us/data/abc.png"})
    # the page's text is never trusted: without a grab there is nothing to download
    assert missing["__outputs__"]["handoff_status"]["text"].startswith("❗ No image URL") and missing["__toast__"]["level"] == "warning"
    pw._remember_grab(None, "Danbooru", "Danbooru post #77\nImage: https://cdn.donmai.us/data/abc.png")
    patch = harness.run_action(page, "download", ctx, {"grabbed_info": "Image: https://169.254.169.254/latest"})
    image_id = patch["preview_image"]["value"]
    assert files.get(image_id).name == "abc.png" and patch["preview_url"] == {"value": "https://cdn.donmai.us/data/abc.png"}
    assert requested == [("https://cdn.donmai.us/data/abc.png", "https://cdn.donmai.us/")]
    again = harness.run_action(page, "download", ctx, {"preview_url": patch["preview_url"]["value"], "preview_image": image_id})
    assert "preview_image" not in again and len(requested) == 1  # reused, not fetched again
    # grabs are kept per user
    assert harness.run_action(page, "download", harness.fake_ctx(page, files, user="other"))["__toast__"]["level"] == "warning"


def test_download_rejects_non_images_and_clears_the_preview(page, pw, tmp_path, monkeypatch, dns):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    pw._remember_grab(None, "Danbooru", "Image: https://cdn.donmai.us/a.png")
    monkeypatch.setattr(pw.requests, "get", lambda *a, **k: FakeResponse(b"<html>secret internal page</html>"))
    patch = harness.run_action(page, "download", ctx, {"preview_image": "old-preview", "preview_url": "https://cdn.donmai.us/old.png"})
    error = patch["__outputs__"]["handoff_status"]["params"]["error"]
    assert patch["__toast__"]["level"] == "error" and "non-image content" in error and "secret" not in error  # the body is never echoed
    # a failed download leaves nothing stale to send to JoyCaption
    assert patch["preview_image"] == {"value": None} and patch["preview_url"] == {"value": ""}
    monkeypatch.setattr(pw.requests, "get", lambda *a, **k: FakeResponse(b"x" * 10, {"Content-Length": str(pw.MAX_IMAGE_DOWNLOAD_BYTES + 1)}))
    pw._remember_grab(None, "Danbooru", "Image: https://cdn.donmai.us/b.png")
    assert "larger than" in harness.run_action(page, "download", ctx)["__outputs__"]["handoff_status"]["params"]["error"]
    assert files.records == {}


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("file:///etc/passwd", "not an https URL"),
        ("http://cdn.donmai.us/a.png", "not an https URL"),
        ("https://127.0.0.1/a.png", "not an image host"),
        ("https://evil.example/a.png", "not an image host"),
        ("https://cdn.donmai.us.evil.example/a.png", "not an image host"),
        ("https://cdn.donmai.us:8080/a.png", "port"),
        ("https://user@cdn.donmai.us/a.png", "credentials"),
        ("https://inside.donmai.us/a.png", "non-public address"),
        ("https://mapped.donmai.us/a.png", "non-public address"),
        ("https://gone.donmai.us/a.png", "cannot resolve"),
    ],
)
def test_downloads_stay_on_the_boorus_public_image_hosts(pw, monkeypatch, dns, url, reason):
    monkeypatch.setattr(pw.requests, "get", lambda *a, **k: pytest.fail("fetched a refused URL"))
    data, _name, error = pw._download_image(url, "", pw._BOORU_SOURCES["Danbooru"]["image_hosts"])
    assert data is None and error.startswith("Download refused") and reason in error


def test_every_redirect_hop_is_checked(pw, monkeypatch, dns):
    hosts = pw._BOORU_SOURCES["Danbooru"]["image_hosts"]
    hops = {
        "https://cdn.donmai.us/a.png": FakeResponse(b"", {"Location": "/b.png"}, 302),
        "https://cdn.donmai.us/b.png": FakeResponse(_png_bytes()),
        "https://cdn.donmai.us/to-inside.png": FakeResponse(b"", {"Location": "https://inside.donmai.us/x"}, 301),
        "https://cdn.donmai.us/to-metadata.png": FakeResponse(b"", {"Location": "http://169.254.169.254/latest/meta-data"}, 307),
        "https://cdn.donmai.us/loop.png": FakeResponse(b"", {"Location": "/loop.png"}, 302),
    }
    fetched = []
    monkeypatch.setattr(pw.requests, "get", lambda url, **kwargs: fetched.append(url) or hops[url])
    data, _name, error = pw._download_image("https://cdn.donmai.us/a.png", "", hosts)
    assert error is None and data == _png_bytes() and fetched == ["https://cdn.donmai.us/a.png", "https://cdn.donmai.us/b.png"]
    assert "non-public address" in pw._download_image("https://cdn.donmai.us/to-inside.png", "", hosts)[2]
    assert "not an https URL" in pw._download_image("https://cdn.donmai.us/to-metadata.png", "", hosts)[2]
    assert "too many redirects" in pw._download_image("https://cdn.donmai.us/loop.png", "", hosts)[2]
    assert "inside.donmai.us/x" not in fetched and not any("169.254" in url for url in fetched)


def test_grab_results_are_shown_read_only(page):
    texts = {node["name"]: node for node in _walk(page.schema()["layout"]) if node.get("type") == "text"}
    assert texts["grabbed_info"]["enabled_if"] is False and texts["post_url"]["enabled_if"] is False


# endregion


def test_zh_cn_covers_the_page(page):
    port_spec = importlib.util.spec_from_file_location("remi_port_harness", REPO / "tests" / "remi_port_harness.py")
    port = importlib.util.module_from_spec(port_spec)
    port_spec.loader.exec_module(port)
    strings = harness.page_strings(page, SCRIPT)
    strings |= {"Quality/Meta/Safety", "Subject", "Character", "Series", "Artist", "General"}  # checkbox choices
    port.assert_localized(EXT, [], extra=strings)
