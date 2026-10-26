"""Routes of the Remi server: bootstrap, schemas, actions, options, infotext,
i18n, models, styles, the SPA and the OpenAPI export."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from remi.server.testing import server_app
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


# region Bootstrap and schemas


def test_bootstrap_describes_the_app(server):
    data = server.client.get("/api/v2/app").json()
    assert data["tabs"][:3] == ["txt2img", "img2img", "extras"]
    assert {"jobs.v1", "ws.v1"} <= set(data["features"]) and "auth" not in data["features"]
    assert data["models"]["preset"] == data["options"]["forge_preset"]
    assert data["auth"] == {"required": False, "user": None}
    assert data["queue"] == {"size": 0, "paused": False, "running": None}
    assert isinstance(data["not_ported"], list) and data["pages"] == []


def test_bootstrap_follows_tab_order_and_hidden_tabs(server, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "ui_tab_order", ["Extras", "PNG Info"])
    monkeypatch.setitem(webui.opts.data, "hidden_tabs", ["img2img"])
    tabs = server.client.get("/api/v2/app").json()["tabs"]
    assert tabs[:2] == ["extras", "png-info"] and "img2img" not in tabs and "txt2img" in tabs


@pytest.mark.parametrize("tab", ["txt2img", "img2img", "extras"])
def test_tab_schemas(server, tab):
    data = server.client.get(f"/api/v2/schema/{tab}").json()
    assert data["version"] == 1 and data["tab"] == tab and data["core"]["owner"] == "core"
    assert ("conflict_params" in data) == (tab != "extras")


def test_unknown_tab_and_unknown_api_paths(server):
    assert server.client.get("/api/v2/schema/nope").status_code == 422
    missing = server.client.get("/api/v2/does-not-exist")
    assert missing.status_code == 404 and missing.headers["content-type"] == "application/json"


# endregion

# region Actions, rules, presets


def test_core_actions(server):
    swapped = server.client.post("/api/v2/actions/core/swap_dimensions", json={"tab": "txt2img", "values": {"width": 832, "height": 1216}})
    assert swapped.json() == {"width": {"value": 1216}, "height": {"value": 832}}
    bad = server.client.post("/api/v2/actions/core/hr_resolution", json={"tab": "txt2img", "values": {"hr_scale": "big"}})
    assert bad.status_code == 422 and bad.json()["errors"]
    assert server.client.post("/api/v2/actions/core/nope", json={"values": {}}).status_code == 404
    assert server.client.post("/api/v2/actions/nobody/x", json={"tab": "txt2img", "values": {}}).status_code == 404


def test_conflicts_by_action_and_by_rules_route(server):
    body = {"tab": "txt2img", "core": {"enable_hr": True}, "scripts": {}, "selected": "core.enable_hr"}
    derived = server.client.post("/api/v2/rules/derive", json=body)
    assert derived.status_code == 200
    action = server.client.post("/api/v2/actions/core/__conflicts__", json={"tab": "txt2img", "values": {"enable_hr": True}, "scripts": {}, "selected": "core.enable_hr"})
    assert action.json() == derived.json()


def test_presets(server):
    listed = server.client.get("/api/v2/presets").json()
    assert listed["anima"] and "sd" in listed["forge"]
    label = listed["anima"][0]
    patch = server.client.get(f"/api/v2/presets/{label}", params={"tab": "txt2img"})
    assert patch.status_code == 200 and isinstance(patch.json(), dict)
    assert server.client.get("/api/v2/presets/no such preset").status_code == 404


def test_settings_refresh_action(server, webui):
    key = next(key for key, info in webui.opts.data_labels.items() if info.refresh is not None and info.section and info.section[0])
    response = server.client.post(f"/api/v2/actions/settings/refresh:{key}", json={"values": {}})
    assert response.status_code == 200 and key in response.json()
    assert server.client.post("/api/v2/actions/settings/refresh:nope", json={"values": {}}).status_code == 404


# endregion

# region Options


def test_options_schema(server):
    data = server.client.get("/api/v2/options").json()
    assert data["version"] == 1 and data["categories"] and "samples_format" in data["values"]


def test_options_patch_validates_applies_saves_and_broadcasts(server, webui, monkeypatch, tmp_path):
    from modules import shared

    config = tmp_path / "config.json"
    monkeypatch.setattr(shared, "config_filename", str(config))
    events = []
    server.services.bus.subscribe(type("R", (), {"send_event": lambda self, e: events.append(e), "send_frame": lambda self, f: None})())
    monkeypatch.setitem(webui.opts.data, "jpeg_quality", 80)

    response = server.client.patch("/api/v2/options", json={"jpeg_quality": 90, "samples_format": "png"})
    assert response.status_code == 200 and response.json()["values"]["jpeg_quality"] == 90
    assert json.loads(config.read_text())["jpeg_quality"] == 90
    assert {"type": "options.changed", "keys": ["jpeg_quality"]} in events


def test_options_patch_rejects_everything_when_one_value_is_bad(server, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "jpeg_quality", 80)
    response = server.client.patch("/api/v2/options", json={"jpeg_quality": 95, "samples_save": "maybe", "no_such_setting": 1})
    assert response.status_code == 422
    errors = {error["param"]: error for error in response.json()["errors"]}
    assert set(errors) == {"samples_save", "no_such_setting"} and errors["samples_save"]["owner"] == "settings"
    assert webui.opts.jpeg_quality == 80


def test_frozen_settings_are_refused_and_locked_in_the_schema(server, webui, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "freeze_specific_settings", "jpeg_quality")
    assert server.client.patch("/api/v2/options", json={"jpeg_quality": 70}).status_code == 422
    nodes = {node["name"]: node for category in server.client.get("/api/v2/options").json()["categories"] for section in category["sections"] for item in section["items"] for node in item.get("children", [item]) if "name" in node}
    assert nodes["jpeg_quality"]["enabled_if"] is False and nodes["jpeg_quality"]["locked"] == "this setting is frozen"
    assert "locked" not in nodes["samples_save"]
    # restrict_api guards /sdapi/v1 only: the WebUI's own API may change them
    restricted = next(key for key, info in webui.opts.data_labels.items() if info.restrict_api)
    assert server.client.patch("/api/v2/options", json={restricted: webui.opts.data.get(restricted, webui.opts.data_labels[restricted].default)}).status_code == 200
    assert "locked" not in nodes[restricted] and nodes[restricted]["restricted"] is True
    assert server.client.patch("/api/v2/options", json=[1, 2]).status_code == 422


# endregion

# region Infotext, i18n, models, styles


def test_infotext_parse(server):
    data = server.client.post("/api/v2/infotext/parse", json={"text": "a cat\nNegative prompt: dog\nSteps: 12, Size: 640x512, CFG scale: 5"}).json()
    assert data["tab"] == "txt2img" and data["core"]["prompt"] == "a cat" and data["core"]["width"] == 640
    assert data["scripts"]["sampler"]["steps"] == 12 and data["params"]["CFG scale"] == "5"


def test_image_info_of_an_upload(server):
    import io

    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a house\nSteps: 7, CFG scale: 5, Size: 64x64")
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="PNG", pnginfo=info)
    upload = server.client.post("/api/v2/uploads", files={"file": ("x.png", buffer.getvalue(), "image/png")}).json()
    data = server.client.post("/api/v2/infotext/image", json={"file_id": upload["id"]}).json()
    assert data["infotext"].startswith("a house") and data["parsed"]["scripts"]["sampler"]["steps"] == 7
    assert server.client.post("/api/v2/infotext/image", json={"file_id": "missingmissing"}).status_code == 404


@pytest.mark.parametrize("route", ["/api/v2/pnginfo", "/api/v2/infotext/image", "/api/v2/jobs"])
def test_a_binary_body_on_a_json_route_is_a_client_error(server, route):
    """A multipart PNG upload sent to a JSON route: the validation error echoes
    the raw body as its input, and encoding the binary bytes answered 500."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buffer, format="PNG")
    png = buffer.getvalue()
    binary = server.client.post(route, files={"file": ("x.png", png, "image/png")})
    assert binary.status_code == 422, binary.text
    assert "Internal server error" not in binary.text
    assert binary.json()["detail"][0]["input"] == f"<{len(binary.request.content)} bytes of binary data>"
    # binary bytes declared as JSON: FastAPI's own 400 (the body is not JSON)
    raw = server.client.post(route, content=png, headers={"content-type": "application/json"})
    assert raw.status_code == 400
    # a short text body is still echoed as is
    text = server.client.post(route, content=b"a=1", headers={"content-type": "text/plain"})
    assert text.status_code == 422 and text.json()["detail"][0]["input"] == "a=1"


def test_i18n_catalog_merges_and_caches(server, monkeypatch, tmp_path):
    from modules import localization

    core = tmp_path / "zh_CN.json"
    core.write_text(json.dumps({"Seed": "种子", "Steps": "步数", "n": 1}), encoding="utf8")
    extension = tmp_path / "ext_zh_CN.json"
    extension.write_text(json.dumps({"Steps": "采样步数", "Hello": "你好"}), encoding="utf8")
    monkeypatch.setattr(localization, "localizations", {"zh_CN": [str(core), str(extension)]})

    response = server.client.get("/api/v2/i18n/zh-CN")
    assert response.status_code == 200 and response.json() == {"Seed": "种子", "Steps": "采样步数", "Hello": "你好"}
    assert server.client.get("/api/v2/i18n/zh_CN", headers={"If-None-Match": response.headers["etag"]}).status_code == 304
    assert server.client.get("/api/v2/i18n/xx_YY").status_code == 404
    assert server.client.get("/api/v2/i18n/..%2Fsecrets").status_code == 404
    assert {"id": "zh_CN", "name": "简体中文"} in server.client.get("/api/v2/app").json()["locales"]


def test_the_real_zh_cn_catalog_is_served(server, monkeypatch):
    from modules import localization

    if not (ROOT / "localizations" / "zh_CN.json").is_file():
        pytest.skip("localizations/zh_CN.json is not in the tree")
    monkeypatch.setattr(localization, "localizations", {"zh_CN": [str(ROOT / "localizations" / "zh_CN.json")]})
    data = server.client.get("/api/v2/i18n/zh_CN").json()
    assert data and all(isinstance(value, str) for value in data.values())


def test_models(server, monkeypatch):
    from remi.server import runtime

    monkeypatch.setattr(runtime, "unet_dtypes", lambda: ["Automatic"])
    data = server.client.get("/api/v2/models").json()
    assert {"checkpoints", "modules", "samplers", "schedulers", "upscalers", "styles", "unet_dtypes", "presets"} <= set(data)
    assert data["samplers"] and data["schedulers"] and "sd" in data["presets"]


@pytest.mark.parametrize("route", ["/api/v2/models/refresh", "/api/v2/networks/refresh", "extras-refresh"])
def test_refresh_picks_up_an_upscaler_added_while_running(server, webui, monkeypatch, tmp_path, route):
    """A model dropped in the ESRGAN folder is a Hires. fix and Extras choice
    (schema and job validation) after a refresh, without a restart: the
    refresh endpoints and the Extras upscaler's own Refresh button."""

    from modules import modelloader, shared
    from remi.server import runtime

    folder = tmp_path / "ESRGAN"
    folder.mkdir()
    monkeypatch.setattr(shared.cmd_opts, "esrgan_models_path", str(folder))
    # only the upscalers matter here: the checkpoint and module scans are other tests'
    monkeypatch.setattr(runtime, "_main_entry", lambda: type("M", (), {"refresh_models": lambda self: None})())
    monkeypatch.setattr(shared, "refresh_checkpoints", lambda: None)
    monkeypatch.setattr(shared, "sd_upscalers", list(shared.sd_upscalers))  # restored after the test
    modelloader.load_upscalers()
    spec = importlib.util.spec_from_file_location("postprocessing_upscale_under_test", ROOT / "scripts" / "postprocessing_upscale.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operation = module.ScriptPostprocessingUpscale()
    operation.filename = "postprocessing_upscale.py"
    monkeypatch.setattr(webui.postproc, "scripts", [*webui.postproc.scripts, operation])

    def choices(tab: str, name: str) -> list[str]:
        nodes = [server.client.get(f"/api/v2/schema/{tab}").json()]
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                if node.get("name") == name and "choices" in node:
                    return [choice["value"] if isinstance(choice, dict) else choice for choice in node["choices"]]
                nodes.extend(node.values())
            elif isinstance(node, list):
                nodes.extend(node)
        raise AssertionError(f"{name} not in the {tab} schema")

    def job(upscaler: str) -> int:
        core = {"prompt": "a cat", "enable_hr": True, "hr_upscaler": upscaler}
        return server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": core, "scripts": {}}).status_code

    assert "RealESRGAN_x4plus_anime_6B" not in choices("txt2img", "hr_upscaler")
    assert job("RealESRGAN_x4plus_anime_6B") == 422

    (folder / "RealESRGAN_x4plus_anime_6B.pth").write_bytes(b"")
    events = []
    server.services.bus.subscribe(type("R", (), {"send_event": lambda self, e: events.append(e), "send_frame": lambda self, f: None})())
    if route == "extras-refresh":
        from remi.core.extras import assign_args

        owner = next(layout.owner for script, layout in assign_args(webui.postproc) if script is operation)
        patch = server.client.post(f"/api/v2/actions/{owner}/refresh_upscalers", json={"tab": "extras", "values": {}}).json()
        assert "RealESRGAN_x4plus_anime_6B" in [choice["value"] for choice in patch["upscaler_1_name"]["choices"]]
    else:
        assert server.client.post(route).json()["ok"] is True

    assert "RealESRGAN_x4plus_anime_6B" in choices("txt2img", "hr_upscaler")
    assert "RealESRGAN_x4plus_anime_6B" in choices("extras", "upscaler_1_name")
    assert "RealESRGAN_x4plus_anime_6B" in choices("extras", "upscaler_2_name")
    assert {"name": "RealESRGAN_x4plus_anime_6B"} in server.client.get("/api/v2/models").json()["upscalers"]
    assert job("RealESRGAN_x4plus_anime_6B") in (200, 201, 202)
    # clients refetch the tab schemas (and with them the choices) on options.changed
    assert any(event["type"] == "options.changed" and "upscalers" in event["keys"] for event in events)


def test_styles_crud(server, webui, tmp_path, monkeypatch):
    from modules import shared, styles

    monkeypatch.setattr(shared, "prompt_styles", styles.StyleDatabase([str(tmp_path / "styles.csv")]))
    saved = server.client.put("/api/v2/styles/Moody", json={"prompt": "{prompt}, dark", "negative_prompt": "bright"})
    assert saved.status_code == 200 and saved.json()["name"] == "Moody"
    assert server.client.get("/api/v2/styles").json() == [{"name": "Moody", "prompt": "{prompt}, dark", "negative_prompt": "bright"}]
    assert "Moody" in (tmp_path / "styles.csv").read_text(encoding="utf-8-sig")
    applied = server.client.post("/api/v2/styles/apply", json={"prompt": "cat", "negative_prompt": "", "styles": ["Moody"]}).json()
    assert applied == {"prompt": "cat, dark", "negative_prompt": "bright"}
    assert server.client.delete("/api/v2/styles/Moody").status_code == 204
    assert server.client.delete("/api/v2/styles/Moody").status_code == 404
    assert server.client.put("/api/v2/styles/%20", json={}).status_code == 422


def test_token_count(server, monkeypatch):
    from types import SimpleNamespace

    from modules import sd_models

    monkeypatch.setattr(sd_models.model_data, "sd_model", None)
    response = server.client.post("/api/v2/prompt/tokens", json={"text": "a cat", "steps": 20})
    assert response.status_code == 200 and response.json() == {"tokens": None, "max_length": None}

    seen = []
    model = SimpleNamespace(get_prompt_lengths_on_ui=lambda prompt: (seen.append(prompt) or len(prompt.split()), 75))
    monkeypatch.setattr(sd_models.model_data, "sd_model", model)
    response = server.client.post("/api/v2/prompt/tokens", json={"text": "[a:big cat:5] <lora:x:1>", "steps": 10})
    assert response.json() == {"tokens": 2, "max_length": 75}
    assert "<lora" not in " ".join(seen)


# endregion

# region Internal routes, SPA, OpenAPI


def test_internal_routes(server):
    instance = server.client.get("/internal/ping").json()["instance"]
    assert len(instance) == 32 and server.client.get("/internal/ping").json() == {"instance": instance}
    progress = server.client.post("/internal/progress", json={"id_task": "task(nope)"}).json()
    assert progress["active"] is False and progress["textinfo"] == "Waiting..."
    assert server.client.get("/internal/pending-tasks").json() == {"size": 0, "tasks": []}


def test_queued_jobs_show_in_the_progress_registry(server):
    server.services.queue.set_paused(True)
    job_id = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {}}).json()["job_id"]
    progress = server.client.post("/internal/progress", json={"id_task": f"task(remi-{job_id})"}).json()
    assert progress["queued"] is True and progress["textinfo"].startswith("In queue")
    server.client.delete(f"/api/v2/jobs/{job_id}")
    assert server.client.post("/internal/progress", json={"id_task": f"task(remi-{job_id})"}).json()["queued"] is False


def make_dist(root):
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<!doctype html><html><head><link rel="icon" href="/favicon.svg"><script type="module" src="/assets/app-1a2b.js"></script></head><body></body></html>')
    (dist / "assets" / "app-1a2b.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    (root / "secret.txt").write_text("secret")
    return dist


def test_the_spa_is_served_with_cache_headers_and_client_route_fallback(webui, tmp_path):
    dist = make_dist(tmp_path)
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        index = server.client.get("/")
        assert index.status_code == 200 and index.headers["cache-control"] == "no-cache" and "/assets/app-1a2b.js" in index.text
        assert "window.__REMI_ROOT__" in index.text
        asset = server.client.get("/assets/app-1a2b.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert server.client.get("/gallery/some/client/route").text == index.text
        assert server.client.get("/assets/missing.js").status_code == 404
        assert server.client.get("/../secret.txt").status_code in (200, 404) and "secret" not in server.client.get("/../secret.txt").text
        assert server.client.get("/%2e%2e/secret.txt").status_code == 404
        assert server.client.post("/").status_code == 404


def test_the_shell_carries_the_theme_of_launch_flag_before_its_own_script(tmp_path):
    """``--theme`` reaches the SPA as ``window.__REMI_THEME__``, set at the
    top of <head> so the shell's pre-paint script can read it."""

    from remi.server.auth import Auth
    from remi.server.spa import Spa

    dist = make_dist(tmp_path)
    (dist / "index.html").write_text("<!doctype html><html><head><script>applyTheme()</script></head><body></body></html>")
    auth = Auth({}, b"k" * 32)
    shell = Spa(auth, dist, theme="Light").index("").decode()
    assert shell.index('window.__REMI_THEME__="light"') < shell.index("applyTheme()")
    assert 'window.__REMI_THEME__=null' in Spa(auth, dist, theme="purple").index("").decode()
    assert 'window.__REMI_THEME__=null' in Spa(auth, dist).index("").decode()


def test_removed_extension_apis_answer_gone_instead_of_the_spa(webui, tmp_path):
    """Tag Autocomplete's /tacapi/v1, Prompt All-in-One's /physton_prompt,
    Agent Scheduler's and Prompt Queue's APIs are not kept (architecture.md):
    their clients get 410 naming the replacement, not the SPA shell."""

    dist = make_dist(tmp_path)
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        for method, path, replacement in (
            ("get", "/tacapi/v1/lora-info/x", "/api/v2/prompt-tools"),
            ("post", "/tacapi/v1/increase-use-count", "/api/v2/prompt-tools"),
            ("get", "/physton_prompt/get_version", "/api/v2/prompt-tools"),
            ("post", "/physton_prompt/translate", "/api/v2/prompt-tools"),
            ("get", "/agent-scheduler/v1/queue", "/api/v2/jobs"),
            ("get", "/prompt-queue/state", "/api/v2/jobs"),
        ):
            response = getattr(server.client, method)(path)
            assert response.status_code == 410, path
            assert replacement in response.json()["detail"] and "<html" not in response.text
        # look-alike client routes still get the shell
        assert server.client.get("/tacapix").status_code == 200


def test_assets_are_served_precompressed_when_the_client_accepts_it(webui, tmp_path):
    import gzip

    from remi.server.spa import accepted_codings

    dist = make_dist(tmp_path)
    source = "console.log(1)"
    (dist / "assets" / "app-1a2b.js.gz").write_bytes(gzip.compress(source.encode()))
    (dist / "assets" / "app-1a2b.js.br").write_bytes(b"not really brotli")
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        get = server.client.get
        br = get("/assets/app-1a2b.js", headers={"Accept-Encoding": "gzip, deflate, br, zstd"})
        assert br.headers["content-encoding"] == "br" and br.headers["content-type"].startswith("text/javascript") and br.content == b"not really brotli"
        assert "accept-encoding" in br.headers["vary"].lower() and "immutable" in br.headers["cache-control"]
        packed = get("/assets/app-1a2b.js", headers={"Accept-Encoding": "gzip, br;q=0"})
        assert packed.headers["content-encoding"] == "gzip" and packed.text == source  # the client inflates it
        plain = get("/assets/app-1a2b.js", headers={"Accept-Encoding": "identity"})
        assert "content-encoding" not in plain.headers and plain.text == source
        assert "content-encoding" not in get("/favicon.svg", headers={"Accept-Encoding": "gzip, br"}).headers
    assert accepted_codings("gzip;q=0.5, br;q=0, deflate") == {"gzip", "deflate"}
    assert accepted_codings("") == set() and accepted_codings("gzip;q=x") == set()


def test_the_spa_shell_gets_the_subpath(webui, tmp_path):
    from fastapi.testclient import TestClient

    dist = make_dist(tmp_path)
    with server_app(tmp_path / "data", spa_dir=dist) as server:
        client = TestClient(server.app, base_url="http://127.0.0.1:7860", root_path="/remi")
        text = client.get("/remi/").text
        assert '="/remi/assets/app-1a2b.js"' in text and 'href="/remi/favicon.svg"' in text and '"/remi"' in text
        assert client.get("/remi/api/v2/queue").status_code == 200


def test_a_missing_build_explains_itself(server):
    response = server.client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 503 and "pnpm build" in response.text


def test_openapi_covers_the_api_and_the_export_is_current(server):
    document = server.client.get("/openapi.json").json()
    for path in ("/api/v2/app", "/api/v2/jobs", "/api/v2/uploads", "/api/v2/options", "/api/v2/networks/{kind}", "/api/v2/outputs", "/api/v2/i18n/{locale}"):
        assert path in document["paths"]
    assert {"Bootstrap", "JobRequest", "Job", "Upload", "ValidationErrorBody"} <= set(document["components"]["schemas"])
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "export_openapi.py"), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


# endregion
