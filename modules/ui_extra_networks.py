"""Import path A1111 extensions use to register extra-networks pages
(``ui_extra_networks.register_page(MyPage())``). No UI: the data API lives in
:mod:`modules.extra_networks_pages`."""

from modules.extra_networks_pages import ExtraNetworksPage, allowed_preview_extensions, extra_pages, initialize, path_is_parent, quote_js, register_page  # noqa: F401
