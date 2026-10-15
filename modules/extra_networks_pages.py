"""The data side of A1111's extra-networks pages, without any UI.

Extensions subclass :class:`ExtraNetworksPage` (``list_items()`` yielding item
dicts: ``name``, ``filename``, ``prompt``, ``description``, ``search_terms``,
``metadata``, ...) and call :func:`register_page`. The Networks page reads
them through ``remi.networks`` (one kind per page); nothing here renders HTML.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modules import extra_networks, util

extra_pages: list[ExtraNetworksPage] = []
default_allowed_preview_extensions = {"png", "jpg", "jpeg", "webp", "jxl", "avif", "heif", "gif", "mp4", "webm"}


def allowed_preview_extensions() -> set[str]:
    return default_allowed_preview_extensions


def register_page(page: ExtraNetworksPage) -> None:
    """Register an extension's page (A1111 called this from ``on_before_ui``)."""

    if page not in extra_pages:
        extra_pages.append(page)


def initialize() -> None:
    extra_pages.clear()


def quote_js(s: str) -> str:
    """A JavaScript string literal: item prompts are JS expressions
    (``remi.networks.js_text`` evaluates them)."""

    return json.dumps(s, ensure_ascii=False)


def path_is_parent(parent_path, child_path) -> bool:
    return os.path.abspath(child_path).startswith(os.path.abspath(parent_path))


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.extra_networks_tabname = self.name.replace(" ", "_")
        self.allow_prompt = True
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}
        self.lister = util.MassFileLister()

    def refresh(self):
        pass

    def create_item(self, name, index=None):
        raise NotImplementedError()

    def list_items(self):
        raise NotImplementedError()

    def allowed_directories_for_previews(self):
        return []

    def read_user_metadata(self, item, use_cache=True):
        filename = item.get("filename", None)
        metadata = extra_networks.get_user_metadata(filename, lister=self.lister if use_cache else None)

        if (description := metadata.get("description", None)) is not None:
            item["description"] = description
        vae = metadata.get("vae_te", metadata.get("vae", None))
        if vae is not None:
            item["vae"] = [vae] if isinstance(vae, str) else vae
        if (version := metadata.get("sd_version_str", None)) is not None:
            item["sd_version_str"] = version

        item["user_metadata"] = metadata

    def link_preview(self, filename):
        """The preview of an item: its file (the Networks page serves
        previews by id, from the model's sidecar files)."""

        return filename

    def search_terms_from_path(self, filename, possible_directories=None):
        abspath = os.path.abspath(filename)
        for parentdir in possible_directories if possible_directories is not None else self.allowed_directories_for_previews():
            parentdir = os.path.dirname(os.path.abspath(parentdir))
            if abspath.startswith(parentdir):
                return os.path.relpath(abspath, parentdir)

        return ""

    def get_sort_keys(self, path):
        pth = Path(path)
        mtime, ctime = self.lister.mctime(path)
        return {
            "date_created": int(mtime),
            "date_modified": int(ctime),
            "name": pth.name.lower(),
            "path": str(pth).lower(),
        }

    def find_preview(self, path):
        """A preview file for ``path`` (without extension), or None."""

        for ext in allowed_preview_extensions():
            for file in (f"{path}.{ext}", f"{path}.preview.{ext}"):
                if self.lister.exists(file):
                    return self.link_preview(file)

        return None

    def find_embedded_preview(self, path, name, metadata):
        """Whether the safetensors header embeds cover images (the Networks
        page extracts them itself)."""

        file = f"{path}.safetensors"
        if self.lister.exists(file) and "ssmd_cover_images" in metadata and any(json.loads(metadata["ssmd_cover_images"])):
            return file

        return None

    def find_description(self, path):
        for file in (f"{path}.txt", f"{path}.description.txt"):
            if not self.lister.exists(file):
                continue

            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
        return None

