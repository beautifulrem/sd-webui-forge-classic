"""The A1111 extra-networks page API without a UI: a page an extension
registers through ``modules.ui_extra_networks`` is a Networks kind."""

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
