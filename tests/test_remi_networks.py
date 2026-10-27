"""Network browser: ``remi.networks`` (index, providers, extension pages),
``remi.network_metadata`` and the ``/api/v2/networks`` routes."""

import base64
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from remi import network_metadata as nm
from remi import networks as module
from remi.networks import ExtraPageProvider, KindInfo, LoraProvider, NetworkEntry, NetworkIndex, folder_tree, js_text
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path / "data") as server:
        yield server


def png_bytes(color=(10, 20, 30), size=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def safetensors(path, metadata, payload=b""):
    header = json.dumps({"__metadata__": metadata}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return path


TAGS = json.dumps({"10_char": {"1girl": 12, "smile": 5, "solo": 9}, "5_extra": {"smile": 2, "outdoors": 1}})


class FakeOnDisk:
    def __init__(self, name, filename, metadata, alias=None):
        self.name, self.filename, self.metadata = name, str(filename), metadata
        self.alias = alias or name
        self.shorthash = name[:4]

    def get_alias(self):
        return self.alias


class FakeNetworks:
    """sd_forge_lora's ``networks`` module, as ``LoraProvider`` reads it."""

    def __init__(self, root):
        self.root = root
        self.available_networks = {}
        self.listed = 0

    def list_available_networks(self):
        self.listed += 1
        self.available_networks.clear()
        for path in sorted(self.root.rglob("*.safetensors")):
            header = module.read_safetensors_metadata(str(path))
            self.available_networks[path.stem] = FakeOnDisk(path.stem, path, header, header.get("ss_output_name"))


@pytest.fixture
def lora_root(tmp_path, monkeypatch):
    root = tmp_path / "Lora"
    safetensors(root / "styles" / "watercolor.safetensors", {"ss_output_name": "wc_style", "ss_network_module": "networks.lora", "ss_base_model_version": "sdxl_base_v1-0", "ss_tag_frequency": TAGS, "ss_network_dim": "32", "ss_training_started_at": "1700000000"})
    safetensors(root / "styles" / "deep" / "ink.safetensors", {"ss_network_module": "lycoris.kohya", "ss_network_args": json.dumps({"algo": "lokr"}), "modelspec.architecture": "stable-diffusion-v1/lora"}, payload=b"x" * 4000)
    safetensors(root / "chars" / "alice.safetensors", {"modelspec.architecture": "anima/lora"})
    safetensors(root / "plain.safetensors", {})
    (root / "styles" / "watercolor.preview.png").write_bytes(png_bytes((200, 40, 40)))
    (root / "plain.txt").write_text("a plain description")
    old = time.time() - 1000
    os.utime(root / "plain.safetensors", (old, old))
    fake = FakeNetworks(root)
    fake.list_available_networks()
    monkeypatch.setitem(sys.modules, "networks", fake)
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "lora_dir", str(root), raising=False)
    monkeypatch.setattr(shared.cmd_opts, "lora_dirs", [], raising=False)
    return root


@pytest.fixture
def lora(server, lora_root):
    server.services.networks = NetworkIndex(server.services.files, lambda: {"lora": LoraProvider()}, current_arch=lambda: "xl")
    return server


def names(response):
    assert response.status_code == 200, response.text
    return [item["name"] for item in response.json()["items"]]


# region Metadata helpers


def test_archs_follow_the_presets():
    from modules_forge.presets import PresetArch

    assert tuple(PresetArch.choices()) == nm.ARCHS


def test_training_tags_are_summed_and_sorted():
    assert nm.build_tags({"ss_tag_frequency": TAGS}) == [("1girl", 12), ("solo", 9), ("smile", 7), ("outdoors", 1)]
    sentences = {"ss_tag_frequency": {"a": {"a photo of a red bicycle by the sea": 3, "a red car parked near the sea": 1}}}
    assert ("red", 4) in nm.build_tags(sentences) and all(len(tag) >= 3 for tag, _ in nm.build_tags(sentences))
    assert nm.build_tags({"ss_tag_frequency": "not json"}) == [] and nm.build_tags({}) == []


@pytest.mark.parametrize(
    ("metadata", "arch"),
    [
        ({"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"}, "xl"),
        ({"ss_base_model_version": "sdxl_base_v1-0"}, "xl"),
        ({"ss_base_model_version": "sd_v1"}, "sd"),
        ({"modelspec.architecture": "stable-diffusion-v1/lora"}, "sd"),
        ({"modelspec.architecture": "Flux.1-dev/lora"}, "flux"),
        ({"ss_base_model_version": "flux1"}, "flux"),
        ({"modelspec.architecture": "flux-2-klein/lora"}, "klein"),
        # FLUX.1 Krea [dev] is Flux.1, not Krea 2
        ({"modelspec.architecture": "flux.1-krea-dev/lora"}, "flux"),
        ({"ss_base_model_version": "flux1_krea"}, "flux"),
        ({"modelspec.architecture": "krea-2/lora"}, "krea"),
        ({"modelspec.architecture": "qwen-image/lora"}, "qwen"),
        ({"modelspec.architecture": "moss-image/lora"}, "moss"),
        ({"ss_base_model_version": "MOSS-Image-SFT"}, "moss"),
        ({"ss_network_module": "networks.lora_anima"}, "anima"),
        ({"ss_base_model_version": "sd_v2", "ss_v2": "True"}, None),
        ({"ss_network_module": "networks.lora"}, None),
        ({}, None),
    ],
)
def test_architecture_detection(metadata, arch):
    assert nm.detect_arch(metadata) == arch


def test_training_info_and_network_type():
    info = nm.training_info(
        {
            "ss_output_name": "wc",
            "ss_network_dim": "32",
            "ss_network_alpha": "16.0",
            "ss_clip_skip": "None",
            "ss_network_module": "networks.lora",
            "ss_training_started_at": "1700000000",
            "ss_bucket_info": json.dumps({"buckets": {"0": {"resolution": [832, 1216], "count": 10}, "1": {"resolution": [1024, 1024], "count": 30}}}),
            "ss_dataset_dirs": json.dumps({"a": {"img_count": 25}, "b": {"img_count": 15}}),
        }
    )
    assert info["output_name"] == "wc" and info["dim"] == 32 and info["alpha"] == 16 and "clip_skip" not in info
    assert info["network_type"] == "LoRA" and info["trained_at"].startswith("2023-11-14")
    assert info["resolutions"] == [{"resolution": "1024x1024", "count": 30}, {"resolution": "832x1216", "count": 10}]
    assert info["dataset_size"] == 40
    assert nm.network_type({"ss_network_module": "lycoris.kohya"}) == "LyCORIS"
    assert nm.network_type({"ss_network_module": "networks.lora", "ss_network_args": json.dumps({"dora_wd": "True"})}) == "DoRA"
    assert nm.network_type({}) is None


def test_user_metadata_edits_are_checked():
    assert nm.normalize_user_metadata({"preferred weight": "0.75", "tags": "a, b, a,", "sd version": "xl", "x": [1]}) == {"preferred weight": 0.75, "tags": ["a", "b"], "sd version": "xl", "x": [1]}
    assert nm.normalize_user_metadata({"notes": None}) == {"notes": None}
    for bad in ({"preferred weight": "heavy"}, {"preferred weight": float("nan")}, {"preferred weight": True}, {"sd version": "sd3"}, {"description": 5}, {"tags": 3}):
        with pytest.raises(ValueError):
            nm.normalize_user_metadata(bad)


def test_js_prompt_expressions_are_evaluated(webui):
    webui.opts.data["extra_networks_default_multiplier"] = 0.8
    try:
        assert js_text('"<lora:x:" + opts.extra_networks_default_multiplier + ">"') == "<lora:x:0.8>"
        assert js_text('"<hypernet:h:" + 1.5 + ">" + " trigger"') == "<hypernet:h:1.5> trigger"
        assert js_text('"tag \\"quoted\\""') == 'tag "quoted"'
        assert js_text("plain words") == "plain words"
        assert js_text('"a" + + "b"') == '"a" + + "b"'
        assert js_text(None) == "" and js_text("") == ""
    finally:
        webui.opts.data.pop("extra_networks_default_multiplier", None)


def test_folder_tree_counts_subfolders():
    tree = folder_tree(["a", "a/b", "a/b", "c", "", "a/B2"])
    assert [(node["path"], node["count"]) for node in tree] == [("a", 4), ("c", 1)]
    assert [(node["path"], node["count"]) for node in tree[0]["children"]] == [("a/b", 2), ("a/B2", 1)]


# endregion

# region LoRA through the API


def test_kinds_describe_how_cards_behave(lora):
    [kind] = lora.client.get("/api/v2/networks").json()
    assert kind["id"] == "lora" and kind["action"] == "prompt" and kind["negative"] is True and kind["builtin"] is True
    assert "activation text" in kind["fields"] and "preferred weight" in kind["fields"]


def test_lora_items_carry_folder_arch_and_prompt(lora, lora_root):
    items = {item["name"]: item for item in lora.client.get("/api/v2/networks/lora").json()["items"]}
    assert list(items) == ["alice", "ink", "plain", "watercolor"]
    wc = items["watercolor"]
    assert wc["folder"] == "styles" and wc["path"] == "styles/watercolor.safetensors" and wc["arch"] == "xl" and wc["subtype"] == "LoRA"
    assert wc["prompt"] == "<lora:wc_style:1>" and wc["alias"] == "wc_style" and wc["preview"] and wc["sd_version"] == "Unknown"
    assert items["ink"]["folder"] == "styles/deep" and items["ink"]["arch"] == "sd" and items["ink"]["subtype"] == "LyCORIS"
    assert items["plain"]["description"] == "a plain description" and items["plain"]["arch"] is None and items["plain"]["folder"] == ""
    assert str(lora_root) not in lora.client.get("/api/v2/networks/lora").text


def test_preferred_weight_and_activation_text_shape_the_prompt(lora, webui):
    client = lora.client
    edited = client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"preferred weight": 0.65, "activation text": "watercolor, soft", "negative text": "(blurry:1.2)", "sd version": "anima", "tags": ["paint", "style"]}}).json()
    assert edited["prompt"] == "<lora:wc_style:0.65> watercolor, soft" and edited["negative_prompt"] == "(blurry:1.2)"
    assert edited["preferred_weight"] == 0.65 and edited["activation_text"] == "watercolor, soft" and edited["negative_text"] == "(blurry:1.2)"
    # the user's sd version wins over the header
    assert edited["sd_version"] == "anima" and edited["arch"] == "anima" and edited["tags"] == ["paint", "style"]
    # the same JSON sidecar as sd_forge_lora's editor
    with open(lora.services.networks.entry("lora", "watercolor").base + ".json", encoding="utf8") as file:
        sidecar = json.load(file)
    assert sidecar["preferred weight"] == 0.65 and sidecar["activation text"] == "watercolor, soft"
    # weight 0 = the default multiplier, read per request
    client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"preferred weight": 0}})
    webui.opts.data["extra_networks_default_multiplier"] = 0.9
    try:
        item = client.get("/api/v2/networks/lora", params={"q": "watercolor"}).json()["items"][0]
        assert item["prompt"] == "<lora:wc_style:0.9> watercolor, soft" and item["preferred_weight"] is None
    finally:
        webui.opts.data.pop("extra_networks_default_multiplier", None)
    # searching finds activation text and tags
    assert names(client.get("/api/v2/networks/lora", params={"q": "SOFT paint"})) == ["watercolor"]


def test_bad_user_metadata_is_rejected(lora):
    response = lora.client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"preferred weight": "heavy"}})
    assert response.status_code == 422 and "preferred weight" in response.json()["detail"]
    assert lora.client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"sd version": "sd9"}}).status_code == 422


def test_filters_folder_arch_and_compatibility(lora):
    client = lora.client
    assert names(client.get("/api/v2/networks/lora", params={"folder": "styles"})) == ["ink", "watercolor"]
    assert names(client.get("/api/v2/networks/lora", params={"folder": "styles/deep"})) == ["ink"]
    assert names(client.get("/api/v2/networks/lora", params={"folder": "style"})) == []
    assert names(client.get("/api/v2/networks/lora", params={"arch": ["xl", "unknown"]})) == ["plain", "watercolor"]
    assert names(client.get("/api/v2/networks/lora", params={"arch": "sd,anima"})) == ["alice", "ink"]
    # current preset "xl": xl and unknown items
    assert names(client.get("/api/v2/networks/lora", params={"compatible": True})) == ["plain", "watercolor"]


def test_sorting_by_date_size_and_path(lora):
    client = lora.client
    assert names(client.get("/api/v2/networks/lora", params={"sort": "date"}))[0] == "plain"
    assert names(client.get("/api/v2/networks/lora", params={"sort": "mtime", "order": "desc"}))[-1] == "plain"
    assert names(client.get("/api/v2/networks/lora", params={"sort": "size", "order": "desc"}))[0] == "ink"
    assert names(client.get("/api/v2/networks/lora", params={"sort": "path"})) == ["alice", "plain", "ink", "watercolor"]
    assert names(client.get("/api/v2/networks/lora", params={"sort": "name", "order": "desc"})) == ["watercolor", "plain", "ink", "alice"]
    assert client.get("/api/v2/networks/lora", params={"sort": "random"}).status_code == 422


def test_cursor_pages_cover_every_item_once(lora):
    seen, cursor = [], None
    while True:
        page = lora.client.get("/api/v2/networks/lora", params={"limit": 3, **({"cursor": cursor} if cursor else {})}).json()
        seen += [item["name"] for item in page["items"]]
        assert page["total"] == 4
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert seen == ["alice", "ink", "plain", "watercolor"]


def test_facets_count_folders_and_archs(lora):
    facets = lora.client.get("/api/v2/networks/lora/facets").json()
    assert facets["total"] == 4 and facets["count"] == 4 and facets["current_arch"] == "xl"
    assert [(node["path"], node["count"]) for node in facets["folders"]] == [("chars", 1), ("styles", 2)]
    assert facets["folders"][1]["children"][0] == {"path": "styles/deep", "name": "deep", "count": 1, "children": []}
    assert {arch["id"]: arch["count"] for arch in facets["archs"]} == {"xl": 1, "sd": 1, "anima": 1, "unknown": 1}
    assert facets["archs"][-1]["id"] == "unknown"
    # each facet ignores its own filter
    narrowed = lora.client.get("/api/v2/networks/lora/facets", params={"folder": "styles", "arch": "xl"}).json()
    assert narrowed["total"] == 1
    assert {arch["id"]: arch["count"] for arch in narrowed["archs"]} == {"xl": 1, "sd": 1}
    assert [(node["path"], node["count"]) for node in narrowed["folders"]] == [("styles", 1)]
    assert lora.client.get("/api/v2/networks/nope/facets").status_code == 404


def test_meta_has_training_facts_and_dataset_tags(lora):
    meta = lora.client.get("/api/v2/networks/lora/watercolor/meta").json()
    assert meta["training"]["output_name"] == "wc_style" and meta["training"]["dim"] == 32 and meta["training"]["network_type"] == "LoRA"
    assert meta["dataset_tags"][:2] == [{"tag": "1girl", "count": 12}, {"tag": "solo", "count": 9}]
    # nested JSON strings come back decoded
    assert meta["metadata"]["ss_tag_frequency"]["10_char"]["smile"] == 5


def test_embedded_cover_images_become_previews(lora, lora_root):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), (0, 0, 255)).save(buffer, format="JPEG")
    cover = base64.b64encode(buffer.getvalue()).decode()
    safetensors(lora_root / "covered.safetensors", {"ssmd_cover_images": json.dumps([cover])})
    lora.client.post("/api/v2/networks/lora/refresh")
    item = lora.client.get("/api/v2/networks/lora", params={"q": "covered"}).json()["items"][0]
    assert item["preview"] and lora.client.get(f"/api/v2/thumbs/{item['preview']}").status_code == 200
    assert "ssmd_cover_images" not in lora.client.get("/api/v2/networks/lora/covered/meta").json()["metadata"]
    # extracted into the data folder, not next to the model
    assert not list(lora_root.glob("covered*.png")) and list((lora.services.files.root / "networks" / "covers").glob("*.png"))


def test_placeholders_are_computed_in_the_background(lora):
    first = lora.client.get("/api/v2/networks/lora", params={"q": "watercolor"}).json()["items"][0]
    lora.services.networks.placeholders.compute_now()
    again = lora.client.get("/api/v2/networks/lora", params={"q": "watercolor"}).json()["items"][0]
    assert first["placeholder"] in (None, again["placeholder"])
    assert again["placeholder"].startswith("data:image/webp;base64,") and len(again["placeholder"]) < 1200
    # kept on disk for the next start
    assert (lora.services.files.root / "networks" / "placeholders.json").is_file()


def test_set_preview_from_an_output_and_refresh(lora, lora_root):
    record = lora.services.files.save_output_image(Image.new("RGB", (1600, 800), (0, 128, 0)), "a forest\nSteps: 4")
    item = lora.client.put("/api/v2/networks/lora/plain/preview", json={"file_id": record.id}).json()
    assert item["preview"]
    with Image.open(lora_root / "plain.preview.png") as saved:
        assert saved.size == (1024, 512)
    before = sys.modules["networks"].listed
    assert lora.client.post("/api/v2/networks/refresh").json()["ok"] is True
    assert sys.modules["networks"].listed == before + 1


def test_open_folder(lora, monkeypatch):
    from modules import shared
    from remi.server import gallery

    opened = []
    monkeypatch.setattr(gallery, "open_folder", opened.append)
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert lora.client.post("/api/v2/networks/lora/ink/open-folder").status_code == 403
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    assert lora.client.post("/api/v2/networks/lora/ink/open-folder").json()["ok"] is True
    assert opened and opened[0].endswith(os.path.join("styles", "deep"))
    assert lora.client.post("/api/v2/networks/lora/nobody/open-folder").status_code == 404


# endregion

# region Other kinds


class StaticProvider:
    def __init__(self, entries, info=None):
        self.entries = entries
        if info:
            self.info = info

    def refresh(self):
        pass

    def list(self):
        return list(self.entries)


def test_names_with_slashes_are_addressable(server, tmp_path):
    root = tmp_path / "ckpt"
    path = safetensors(root / "anime" / "model.safetensors", {"modelspec.architecture": "stable-diffusion-xl-v1-base"})
    entry = NetworkEntry("checkpoints", "anime/model.safetensors", str(path), title="model", root=str(root), value="anime/model.safetensors [abcdef1234]")
    server.services.networks = NetworkIndex(server.services.files, lambda: {"checkpoints": StaticProvider([entry])})
    [item] = server.client.get("/api/v2/networks/checkpoints").json()["items"]
    assert item["value"] == "anime/model.safetensors [abcdef1234]" and item["arch"] == "xl" and item["title"] == "model"
    meta = server.client.get("/api/v2/networks/checkpoints/anime%2Fmodel.safetensors/meta")
    assert meta.status_code == 200 and meta.json()["name"] == "anime/model.safetensors"
    [kind] = server.client.get("/api/v2/networks").json()
    assert kind["action"] == "checkpoint" and kind["negative"] is False


class HypernetPage:
    """An extension's ExtraNetworksPage (no Gradio needed to adapt it)."""

    title = "Hypernetworks"
    name = "hypernetworks"
    extra_networks_tabname = "hypernetworks"
    allow_negative_prompt = False

    def __init__(self, root):
        self.root = root
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1

    def allowed_directories_for_previews(self):
        return [str(self.root)]

    def list_items(self):
        for path in sorted(self.root.rglob("*.pt")):
            yield {"name": path.stem, "filename": str(path), "prompt": f'"<hypernet:{path.stem}:" + opts.extra_networks_default_multiplier + ">"', "search_terms": ["secret-term"], "description": "from the page"}
        yield {"name": "", "filename": "ignored"}


def test_extension_pages_are_adapted(server, tmp_path, monkeypatch):
    root = tmp_path / "hyper"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "ink.pt").write_bytes(b"\0" * 10)
    page = HypernetPage(root)
    monkeypatch.setattr(module, "_extra_pages", [page])
    monkeypatch.setattr(module, "_providers", {"lora": StaticProvider([])})
    server.services.networks = NetworkIndex(server.services.files)
    kinds = server.client.get("/api/v2/networks").json()
    assert [kind["id"] for kind in kinds] == ["lora", "hypernetworks"]
    assert kinds[1] == {"id": "hypernetworks", "title": "Hypernetworks", "action": "prompt", "negative": False, "fields": ["description", "tags", "notes"], "builtin": False}
    [item] = server.client.get("/api/v2/networks/hypernetworks", params={"q": "secret-term"}).json()["items"]
    assert item["prompt"] == "<hypernet:ink:1>" and item["folder"] == "sub" and item["description"] == "from the page"
    server.client.post("/api/v2/networks/hypernetworks/refresh")
    assert page.refreshed == 1


def test_a_page_named_like_a_builtin_kind_gets_its_own_id():
    page = SimpleNamespace(title="VAE", extra_networks_tabname="vae", list_items=list)
    assert ExtraPageProvider(page).kind == "page_vae"


def test_built_in_extension_pages_are_not_listed_twice(monkeypatch):
    fake = SimpleNamespace(extra_pages=[SimpleNamespace(name="lora", title="Lora"), SimpleNamespace(name="checkpoints", title="Checkpoints"), SimpleNamespace(name="hypernetworks", title="Hypernetworks")])
    monkeypatch.setitem(sys.modules, "modules.extra_networks_pages", fake)
    monkeypatch.setattr(module, "_extra_pages", [])
    assert [page.name for page in module.extra_pages()] == ["hypernetworks"]


def test_registered_kinds_keep_their_info(server, tmp_path):
    info = KindInfo("wildcards", "Wildcards", "prompt", False, ("description",), 5)
    entry = NetworkEntry("wildcards", "colors", str(tmp_path / "colors.txt"), title="colors", prompt="__colors__")
    server.services.networks = NetworkIndex(server.services.files, lambda: {"lora": StaticProvider([]), "wildcards": StaticProvider([entry], info)})
    assert [kind["id"] for kind in server.client.get("/api/v2/networks").json()] == ["lora", "wildcards"]
    assert server.client.get("/api/v2/networks/wildcards").json()["items"][0]["prompt"] == "__colors__"


def test_embedding_architecture_from_vectors():
    assert module.embedding_arch(SimpleNamespace(vec={"clip_g": 0, "clip_l": 0}, shape=None)) == "xl"
    assert module.embedding_arch(SimpleNamespace(vec=None, shape=768)) == "sd"
    assert module.embedding_arch(SimpleNamespace(vec=None, shape=1024)) is None


# endregion


# region Hidden folders, sorting, windows, read-only items, caches


@pytest.fixture
def hidden_lora(lora, lora_root, webui):
    safetensors(lora_root / ".secret" / "deep" / "hush.safetensors", {})
    safetensors(lora_root / "styles" / ".wip" / "draft.safetensors", {})
    sys.modules["networks"].list_available_networks()
    lora.services.networks.invalidate("lora")
    yield lora
    for key in ("extra_networks_hidden_models", "extra_networks_show_hidden_directories"):
        webui.opts.data.pop(key, None)


def test_hidden_folders_follow_the_setting(hidden_lora, webui):
    client = hidden_lora.client
    webui.opts.data["extra_networks_hidden_models"] = "When searched"
    assert names(client.get("/api/v2/networks/lora")) == ["alice", "ink", "plain", "watercolor"]
    assert names(client.get("/api/v2/networks/lora", params={"q": "hus"})) == []
    [item] = client.get("/api/v2/networks/lora", params={"q": "hush"}).json()["items"]
    assert item["hidden"] is True and item["folder"] == ".secret/deep"
    webui.opts.data["extra_networks_hidden_models"] = "Never"
    assert names(client.get("/api/v2/networks/lora", params={"q": "draft"})) == []
    webui.opts.data["extra_networks_hidden_models"] = "Always"
    assert names(client.get("/api/v2/networks/lora")) == ["alice", "draft", "hush", "ink", "plain", "watercolor"]
    tree = client.get("/api/v2/networks/lora/facets").json()["folders"]
    assert [node["path"] for node in tree] == [".secret", "chars", "styles"]
    # hidden directories left out of the tree; their items count in the visible parent
    webui.opts.data["extra_networks_show_hidden_directories"] = False
    tree = client.get("/api/v2/networks/lora/facets").json()["folders"]
    assert [(node["path"], node["count"]) for node in tree] == [("chars", 1), ("styles", 3)]
    assert [node["path"] for node in tree[1]["children"]] == ["styles/deep"]


def test_folder_tree_can_hide_dot_folders():
    tree = folder_tree(["a/.b/c", ".x", "a"], hide_hidden=True)
    assert [(node["path"], node["count"], node["children"]) for node in tree] == [("a", 2, [])]


def test_all_row_counts_every_folder(lora):
    facets = lora.client.get("/api/v2/networks/lora/facets", params={"folder": "styles"}).json()
    assert facets["total"] == 2 and facets["folder_total"] == 4


def test_sorting_by_date_created(lora, lora_root, monkeypatch):
    created = {"alice": 30.0, "ink": 10.0, "plain": 40.0, "watercolor": 20.0}
    real = module._stat

    def fake(entry):
        size, mtime, _created, is_file = real(entry)
        return size, mtime, created[entry.name], is_file

    monkeypatch.setattr(module, "_stat", fake)
    lora.services.networks.invalidate("lora")
    assert names(lora.client.get("/api/v2/networks/lora", params={"sort": "created"})) == ["ink", "watercolor", "alice", "plain"]
    items = lora.client.get("/api/v2/networks/lora", params={"sort": "created", "order": "desc"}).json()["items"]
    assert items[0]["name"] == "plain" and items[0]["created"] == 40.0


def test_offset_jumps_straight_to_a_window(lora):
    page = lora.client.get("/api/v2/networks/lora", params={"offset": 2, "limit": 1}).json()
    assert [item["name"] for item in page["items"]] == ["plain"] and page["total"] == 4 and page["next_cursor"]
    assert names(lora.client.get("/api/v2/networks/lora", params={"offset": 10})) == []
    assert lora.client.get("/api/v2/networks/lora", params={"offset": -1}).status_code == 422


def test_video_previews_are_kept(lora, lora_root):
    # an MP4 header ("ftyp" box) is enough for the file index
    (lora_root / "chars" / "alice.preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    lora.services.networks.invalidate("lora")
    [item] = lora.client.get("/api/v2/networks/lora", params={"q": "alice"}).json()["items"]
    assert item["preview_type"] == "video" and item["preview"] and item["placeholder"] is None
    response = lora.client.get(f"/api/v2/files/{item['preview']}")
    assert response.status_code == 200 and response.headers["content-type"] == "video/mp4"
    [still] = lora.client.get("/api/v2/networks/lora", params={"q": "watercolor"}).json()["items"]
    assert still["preview_type"] == "image"


class PromptPage:
    """An extension page whose items have no model file."""

    title = "Wildcards"
    name = "wildcards"

    def list_items(self):
        yield {"name": "colors", "prompt": '"__colors__"'}


def test_items_without_a_file_are_read_only(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_extra_pages", [PromptPage()])
    monkeypatch.setattr(module, "_providers", {"lora": StaticProvider([])})
    server.services.networks = NetworkIndex(server.services.files)
    [item] = server.client.get("/api/v2/networks/wildcards").json()["items"]
    assert item["editable"] is False and item["path"] == "colors" and item["prompt"] == "__colors__"
    response = server.client.patch("/api/v2/networks/wildcards/colors/meta", json={"user_metadata": {"notes": "x"}})
    assert response.status_code == 409
    record = server.services.files.save_output_image(Image.new("RGB", (8, 8)), "x")
    assert server.client.put("/api/v2/networks/wildcards/colors/preview", json={"file_id": record.id}).status_code == 409
    assert server.client.post("/api/v2/networks/wildcards/colors/open-folder").status_code in (403, 409)
    # nothing written to (or read from) the working directory
    assert not list(tmp_path.glob("colors*"))


def test_set_preview_rejects_missing_and_unreadable_files(lora, lora_root):
    client = lora.client
    assert client.put("/api/v2/networks/lora/plain/preview", json={"file_id": "nosuchfile1234"}).status_code == 404
    text = lora.services.files.add_upload_bytes(b"just some words", "notes.txt")
    assert client.put("/api/v2/networks/lora/plain/preview", json={"file_id": text.id}).status_code == 404
    broken = lora.services.files.add_upload_bytes(png_bytes()[:60], "broken.png")
    response = client.put("/api/v2/networks/lora/plain/preview", json={"file_id": broken.id})
    assert response.status_code == 422, response.text
    assert not (lora_root / "plain.preview.png").exists()


def test_replace_drops_the_other_keys(lora):
    client = lora.client
    client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"notes": "keep?", "activation text": "wc"}})
    meta = client.patch("/api/v2/networks/lora/watercolor/meta", json={"user_metadata": {"description": "only this"}, "replace": True}).json()
    assert meta["user_metadata"] == {"description": "only this"} and meta["activation_text"] is None


def test_notify_changed_invalidates_and_tells_clients(server, lora_root):
    from remi.networks import notify_changed

    events = []
    unsubscribe = server.services.bus.subscribe(SimpleNamespace(send_event=events.append, send_frame=lambda frame: None))
    index = server.services.networks
    index._providers = lambda: {"lora": LoraProvider()}
    try:
        assert sorted(entry.name for entry in index.entries("lora")) == ["alice", "ink", "plain", "watercolor"]
        safetensors(lora_root / "zeta.safetensors", {})
        sys.modules["networks"].list_available_networks()
        notify_changed("lora")
        assert {"type": "networks.changed", "kind": "lora"} in events
        assert "zeta" in [entry.name for entry in index.entries("lora")]
    finally:
        unsubscribe()


class CountingProvider:
    def __init__(self, entries):
        self.entries = entries
        self.calls = 0
        self.gate = threading.Event()

    def refresh(self):
        pass

    def list(self):
        self.calls += 1
        self.gate.wait(2)
        return list(self.entries)


def test_one_listing_per_kind_at_a_time(server, tmp_path):
    provider = CountingProvider([NetworkEntry("lora", "a", str(tmp_path / "a.safetensors"))])
    index = NetworkIndex(server.services.files, lambda: {"lora": provider})
    results = []
    threads = [threading.Thread(target=lambda: results.append(index.records("lora"))) for _ in range(6)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    provider.gate.set()
    for thread in threads:
        thread.join()
    assert provider.calls == 1 and len(results) == 6 and all(result is results[0] for result in results)
    # invalidated while listing: that listing is not kept
    provider.gate.clear()
    index.invalidate("lora")
    lister = threading.Thread(target=index.records, args=("lora",))
    lister.start()
    time.sleep(0.05)
    index.invalidate("lora")
    provider.gate.set()
    lister.join()
    index.records("lora")
    assert provider.calls == 3


def test_placeholders_queued_while_the_worker_exits_are_not_lost(tmp_path):
    from remi.networks import PlaceholderCache

    image = tmp_path / "p.png"
    image.write_bytes(png_bytes())
    cache = PlaceholderCache(None)
    cache.IDLE_TIMEOUT = 0.001
    for number in range(150):
        cache.get(f"id{number}", str(image))
        time.sleep(0.002 if number % 3 else 0)
    deadline = time.time() + 10
    while len(cache) < 150 and time.time() < deadline:
        time.sleep(0.02)
    assert len(cache) == 150 and all(cache.get(f"id{number}", str(image)) for number in range(150))


def test_refresh_prunes_placeholders_and_covers(lora, lora_root):
    index = lora.services.networks
    cache = index.placeholders
    gone = lora_root / "gone.png"
    gone.write_bytes(png_bytes())
    cache.get("old-preview", str(gone))
    cache.compute_now()
    gone.unlink()
    covers = index.cache_dir / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    (covers / "stale0000000000000000000.png").write_bytes(png_bytes())
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), (0, 0, 255)).save(buffer, format="PNG")
    safetensors(lora_root / "covered.safetensors", {"ssmd_cover_images": json.dumps([base64.b64encode(buffer.getvalue()).decode()])})
    lora.client.post("/api/v2/networks/lora/refresh")
    lora.client.get("/api/v2/networks/lora", params={"q": "covered"})
    current = {path.name for path in covers.glob("*.png")} - {"stale0000000000000000000.png"}
    assert current
    index.prune_cache()
    assert {path.name for path in covers.glob("*.png")} == current
    assert cache.get("old-preview", str(gone)) is None and "old-preview" not in json.loads((index.cache_dir / "placeholders.json").read_text())


def test_bootstrap_says_whether_folders_open(server, monkeypatch):
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    assert "folders.open" in server.client.get("/api/v2/app").json()["features"]
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert "folders.open" not in server.client.get("/api/v2/app").json()["features"]


class FakeEmbeddingDb:
    def __init__(self, root):
        self.embedding_dirs = {str(root): None}
        self.word_embeddings = {
            "easyneg": SimpleNamespace(name="easyneg", filename=str(root / "neg" / "easyneg.safetensors"), shorthash="abc123", vec=None, shape=768),
            "xlstyle": SimpleNamespace(name="xlstyle", filename=str(root / "xlstyle.pt"), shorthash=None, vec={"clip_g": 0, "clip_l": 0}, shape=None),
            "inmemory": SimpleNamespace(name="inmemory", filename=None, shorthash=None, vec=None, shape=768),
        }

    def load_textual_inversion_embeddings(self, **_kwargs):
        pass


def test_embeddings_are_listed_from_the_database(server, tmp_path, monkeypatch):
    root = tmp_path / "embeddings"
    safetensors(root / "neg" / "easyneg.safetensors", {})
    (root / "xlstyle.pt").write_bytes(b"\0" * 8)
    monkeypatch.setattr(module, "_embedding_db", FakeEmbeddingDb(root))
    server.services.networks = NetworkIndex(server.services.files, lambda: {"embeddings": module.EmbeddingProvider()})
    items = {item["name"]: item for item in server.client.get("/api/v2/networks/embeddings").json()["items"]}
    assert list(items) == ["easyneg", "xlstyle"]
    assert items["easyneg"]["prompt"] == "easyneg" and items["easyneg"]["folder"] == "neg" and items["easyneg"]["arch"] == "sd" and items["easyneg"]["hash"] == "abc123"
    assert items["xlstyle"]["arch"] == "xl" and items["xlstyle"]["editable"] is True


# endregion
