"""Locale catalogs for ``/api/v2/i18n/{locale}``: ``localizations/<id>.json``
merged with every extension's ``localizations/<id>.json`` (English text →
translation)."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any

log = logging.getLogger(__name__)

LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(?:_[A-Za-z0-9]{2,8})*$")
LOCALE_NAMES = {
    "de_DE": "Deutsch",
    "en": "English",
    "es_ES": "Español",
    "fr_FR": "Français",
    "it_IT": "Italiano",
    "ja_JP": "日本語",
    "ko_KR": "한국어",
    "pl_PL": "Polski",
    "pt_BR": "Português (Brasil)",
    "ru_RU": "Русский",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
}

_cache: dict[str, tuple[tuple, dict[str, str]]] = {}
_lock = threading.Lock()


def available() -> dict[str, list[str]]:
    """Locale id → catalog files (core first, then extensions)."""

    from modules import localization

    if not localization.localizations:
        from modules.shared_cmd_options import cmd_opts

        try:
            localization.list_localizations(cmd_opts.localizations_dir)
        except Exception:
            log.exception("listing localizations")
    return {name: list(files) for name, files in localization.localizations.items()}


def normalize(locale: str) -> str | None:
    """``zh-CN`` / ``zh_cn`` → ``zh_CN`` (an available id), None when unknown."""

    candidate = locale.replace("-", "_")
    if not LOCALE_RE.match(candidate):
        return None
    known = available()
    if candidate in known:
        return candidate
    lowered = candidate.lower()
    return next((name for name in known if name.lower() == lowered), None)


def locales() -> list[dict[str, str]]:
    return [{"id": name, "name": LOCALE_NAMES.get(name, name)} for name in sorted(available())]


def default_locale() -> str | None:
    from modules import shared

    value = getattr(shared.opts, "localization", "None")
    return value if value and value != "None" else None


def catalog(locale: str) -> tuple[dict[str, str], str]:
    """``(entries, etag)``; string → string entries only."""

    files = available().get(locale, [])
    key = tuple((path, _mtime(path)) for path in files)
    with _lock:
        cached = _cache.get(locale)
        if cached is not None and cached[0] == key:
            return cached[1], _etag(key)
    data: dict[str, str] = {}
    for path in files:
        try:
            with open(path, encoding="utf8") as file:
                loaded: Any = json.load(file)
        except (OSError, ValueError):
            log.exception("loading localization %s", path)
            continue
        if isinstance(loaded, dict):
            data.update({str(k): v for k, v in loaded.items() if isinstance(v, str)})
    with _lock:
        _cache[locale] = (key, data)
    return data, _etag(key)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _etag(key: tuple) -> str:
    import hashlib

    return '"' + hashlib.sha1(repr(key).encode()).hexdigest()[:20] + '"'
