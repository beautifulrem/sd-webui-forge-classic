"""Quick tag groups: categories of commonly used tags with localized labels,
from YAML files (``<locale>.yaml``, else ``default.yaml``; a non-empty
``custom.yaml`` replaces them, ``prepend.yaml`` / ``append.yaml`` add
categories before / after).

    - name: 人物
      groups:
        - name: 对象
          color: rgba(255, 123, 2, .4)
          tags:
            1girl: 1女孩
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[_-][A-Za-z0-9]{2,8})*$")
COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\([0-9.,%\s/]+\)|[a-zA-Z]{3,20})$")


def _read(path: str) -> list[Any]:
    import yaml

    try:
        with open(path, encoding="utf8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        log.warning("Cannot read tag groups %s: %s", path, error)
        return []
    return data if isinstance(data, list) else []


def _category(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw.get("name"):
        return None
    groups = []
    for group in raw.get("groups") or []:
        if not isinstance(group, dict) or not group.get("name"):
            continue
        tags = group.get("tags") or {}
        if isinstance(tags, list):
            pairs = [(str(next(iter(item))), item[next(iter(item))]) if isinstance(item, dict) and item else (str(item), None) for item in tags]
        elif isinstance(tags, dict):
            pairs = list(tags.items())
        else:
            continue
        items = [{"tag": str(tag).strip(), "label": str(label).strip() if label not in (None, "") else None} for tag, label in pairs if str(tag).strip()]
        if not items:
            continue
        color = str(group.get("color") or "").strip()
        groups.append({"name": str(group["name"]), "color": color if COLOR_RE.match(color) else None, "tags": items})
    return {"name": str(raw["name"]), "groups": groups} if groups else None


def locale_file(locale: str | None) -> str:
    """``zh-CN`` → ``zh_CN``; anything unusual → ``default``."""

    if not locale or not LOCALE_RE.match(locale):
        return "default"
    return locale.replace("-", "_")


class TagGroups:
    """Loads and caches the groups of each locale (reloaded when a file changes)."""

    def __init__(self, directories: Sequence[str]) -> None:
        self.directories = list(directories)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple, list[dict[str, Any]]]] = {}
        self._tables: dict[str, tuple[list[dict[str, Any]], dict[str, str]]] = {}
        """``dictionary()`` per locale, with the groups it was built from"""

    def _path(self, name: str) -> str | None:
        for directory in self.directories:
            path = os.path.join(directory, f"{name}.yaml")
            if os.path.isfile(path):
                return path
        return None

    def _files(self, locale: str | None) -> list[str]:
        custom = self._path("custom")
        if custom and os.path.getsize(custom) > 0 and _read(custom):
            main = custom
        else:
            name = locale_file(locale)
            main = self._path(name) or self._path(name.split("_")[0]) or self._path("default")
        return [path for path in (self._path("prepend"), main, self._path("append")) if path]

    def groups(self, locale: str | None) -> list[dict[str, Any]]:
        files = self._files(locale)
        stamp = tuple((path, os.path.getmtime(path)) for path in files)
        key = locale_file(locale)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
        categories = [category for path in files for raw in _read(path) if (category := _category(raw))]
        with self._lock:
            self._cache[key] = (stamp, categories)
        return categories

    def dictionary(self, locale: str | None) -> dict[str, str]:
        """``tag → label`` of every group tag (a small offline translation
        table; built once per loaded set of groups, do not modify)."""

        categories = self.groups(locale)
        key = locale_file(locale)
        with self._lock:
            cached = self._tables.get(key)
            if cached and cached[0] is categories:
                return cached[1]
        table: dict[str, str] = {}
        for category in categories:
            for group in category["groups"]:
                for item in group["tags"]:
                    if item["label"]:
                        table.setdefault(re.sub(r"\s+", "_", item["tag"].lower()), item["label"])
        with self._lock:
            self._tables[key] = (categories, table)
        return table
