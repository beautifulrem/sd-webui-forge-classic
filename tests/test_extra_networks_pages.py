"""The A1111 extra-networks page API without a UI: a page an extension
registers through ``modules.ui_extra_networks`` is a Networks kind."""

import json
from pathlib import Path

import pytest

from remi import networks
from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


def test_a_registered_page_becomes_a_network_kind(tmp_path, monkeypatch):
    from modules import extra_networks_pages, ui_extra_networks

    (tmp_path / "colors.txt").write_text("red\n")
    (tmp_path / "colors.preview.png").write_bytes(b"")

    class WildcardsPage(ui_extra_networks.ExtraNetworksPage):
        def __init__(self):
            super().__init__("Wildcards")

        def list_items(self):
            path = str(tmp_path / "colors")
            yield {"name": "colors", "filename": f"{path}.txt", "prompt": ui_extra_networks.quote_js("__colors__"), "preview": self.find_preview(path), "description": self.find_description(path), "search_terms": [self.search_terms_from_path(f"{path}.txt")]}

        def allowed_directories_for_previews(self):
            return [str(tmp_path)]

    monkeypatch.setattr(extra_networks_pages, "extra_pages", [])
    page = WildcardsPage()
    ui_extra_networks.register_page(page)
    ui_extra_networks.register_page(page)
    assert extra_networks_pages.extra_pages == [page]
    assert page.extra_networks_tabname == "wildcards" and page.find_preview(str(tmp_path / "colors")) == str(tmp_path / "colors.preview.png")
    assert page.find_description(str(tmp_path / "colors")) == "red\n"

    assert page in networks.extra_pages()
    provider = networks.ExtraPageProvider.for_page(page)
    (entry,) = provider.list()
    assert provider.kind == "wildcards" and entry.name == "colors" and entry.prompt == "__colors__"


def test_a_refresh_sees_files_added_since_the_last_listing(tmp_path):
    # the page's file cache (``lister``) is cleared on every refresh, as the
    # Gradio UI did before each listing: a preview added later is found
    from modules import ui_extra_networks

    (tmp_path / "colors.txt").write_text("red\n")

    class Page(ui_extra_networks.ExtraNetworksPage):
        def __init__(self):
            super().__init__("Refreshing")

        def list_items(self):
            path = str(tmp_path / "colors")
            yield {"name": "colors", "filename": f"{path}.txt", "preview": self.find_preview(path)}

    page = Page()
    provider = networks.ExtraPageProvider.for_page(page)
    assert page.find_preview(str(tmp_path / "colors")) is None
    (tmp_path / "colors.png").write_bytes(b"")
    assert page.find_preview(str(tmp_path / "colors")) is None  # cached listing
    provider.refresh()
    assert page.find_preview(str(tmp_path / "colors")) == str(tmp_path / "colors.png")


def bare_page(title="Things"):
    from modules import extra_networks_pages

    return extra_networks_pages.ExtraNetworksPage(title)


def test_page_registry_and_helpers(monkeypatch, tmp_path):
    from modules import extra_networks_pages

    pages = []
    monkeypatch.setattr(extra_networks_pages, "extra_pages", pages)
    page = bare_page("Text Inversions")
    extra_networks_pages.register_page(page)
    assert pages == [page] and page.name == "text inversions" and page.extra_networks_tabname == "text_inversions"
    extra_networks_pages.initialize()
    assert pages == []

    assert extra_networks_pages.quote_js('say "hi" 你好') == '"say \\"hi\\" 你好"'
    assert extra_networks_pages.path_is_parent(tmp_path, tmp_path / "a" / "b.txt")
    assert not extra_networks_pages.path_is_parent(tmp_path / "a", tmp_path / "b.txt")
    assert "png" in extra_networks_pages.allowed_preview_extensions()


def test_the_base_page_is_abstract():
    page = bare_page()
    assert page.refresh() is None and page.allowed_directories_for_previews() == []
    with pytest.raises(NotImplementedError):
        page.create_item("x")
    with pytest.raises(NotImplementedError):
        page.list_items()
    assert page.link_preview("/a/b.png") == "/a/b.png"


@pytest.mark.parametrize("use_cache", [True, False])
def test_read_user_metadata_copies_the_sidecar_json(tmp_path, use_cache):
    (tmp_path / "style.safetensors").write_bytes(b"")
    (tmp_path / "style.json").write_text(json.dumps({"description": "a style", "vae_te": "ae.safetensors", "sd_version_str": "Anima", "notes": "n"}), encoding="utf8")
    page = bare_page()
    item = {"filename": str(tmp_path / "style.safetensors")}
    page.read_user_metadata(item, use_cache=use_cache)
    assert item["description"] == "a style" and item["vae"] == ["ae.safetensors"] and item["sd_version_str"] == "Anima"
    assert item["user_metadata"]["notes"] == "n"


def test_read_user_metadata_keeps_a_vae_list_and_tolerates_no_sidecar(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"vae": ["x.safetensors", "y.safetensors"]}), encoding="utf8")
    page = bare_page()
    item = {"filename": str(tmp_path / "a.safetensors"), "description": "kept"}
    page.read_user_metadata(item)
    assert item["vae"] == ["x.safetensors", "y.safetensors"] and item["description"] == "kept" and "sd_version_str" not in item

    bare = {"filename": str(tmp_path / "missing.safetensors")}
    page.read_user_metadata(bare)
    assert bare == {"filename": str(tmp_path / "missing.safetensors"), "user_metadata": {}}
    nameless = {}
    page.read_user_metadata(nameless)
    assert nameless == {"user_metadata": {}}


def test_search_terms_are_relative_to_the_preview_folder_parent(tmp_path):
    lora = tmp_path / "models" / "Lora"
    file = lora / "sub" / "style.safetensors"
    page = bare_page()
    # relative to the parent of the matching folder, as A1111 did ("Lora/sub/style.safetensors")
    assert page.search_terms_from_path(str(file), [str(lora)]) == str(Path("Lora") / "sub" / "style.safetensors")
    assert page.search_terms_from_path(str(file), [str(tmp_path / "elsewhere" / "x")]) == ""
    assert page.search_terms_from_path(str(file)) == ""  # no allowed directories


def test_sort_keys_come_from_the_file_listing(tmp_path, monkeypatch):
    page = bare_page()
    monkeypatch.setattr(page.lister, "mctime", lambda path: (1700000000.9, 1600000000.2))
    keys = page.get_sort_keys(str(tmp_path / "Sub" / "Style.safetensors"))
    assert keys == {"date_created": 1700000000, "date_modified": 1600000000, "name": "style.safetensors", "path": str(tmp_path / "Sub" / "Style.safetensors").lower()}


def test_find_embedded_preview_needs_cover_images_and_the_file(tmp_path):
    (tmp_path / "style.safetensors").write_bytes(b"")
    page = bare_page()
    path = str(tmp_path / "style")
    assert page.find_embedded_preview(path, "style", {"ssmd_cover_images": json.dumps(["data:image/png;base64,AAAA"])}) == f"{path}.safetensors"
    assert page.find_embedded_preview(path, "style", {"ssmd_cover_images": "[]"}) is None
    assert page.find_embedded_preview(path, "style", {}) is None
    assert page.find_embedded_preview(str(tmp_path / "other"), "other", {"ssmd_cover_images": json.dumps(["x"])}) is None


def test_find_description_skips_an_unreadable_file(tmp_path):
    (tmp_path / "style.txt").mkdir()  # listed, but cannot be read as a file
    page = bare_page()
    assert page.find_description(str(tmp_path / "style")) is None
    (tmp_path / "style.description.txt").write_text("from the second name", encoding="utf8")
    page.lister.reset()
    assert page.find_description(str(tmp_path / "style")) == "from the second name"
