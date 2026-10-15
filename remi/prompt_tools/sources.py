"""Completion sources besides booru tags: wildcard files (``__name__``, with
nested folders and Dynamic Prompts YAML collections), wildcard values,
chants (``<c:…>`` prompt presets from JSON files) and extra networks
(``<lora:…>``, embeddings) from the server's network index."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

WILDCARD_SCAN_TTL = 30.0
MAX_WILDCARD_VALUES = 2000
SKIPPED_WILDCARD_FILES = ("put wildcards here.txt",)


def _match_score(name: str, query: str) -> int | None:
    """0 exact, 1 prefix, 2 a path segment or word starts with it, 3 contains; None: no match."""

    if not query:
        return 1
    lowered = name.lower()
    if lowered == query:
        return 0
    if lowered.startswith(query):
        return 1
    for separator in ("/", "_", " ", "-"):
        if f"{separator}{query}" in lowered:
            return 2
    if query in lowered:
        return 3
    return None


def search_names(names: Iterable[tuple[str, Any]], query: str, limit: int) -> list[Any]:
    """Values whose name matches ``query``: exact, prefix, word start, then
    substring; alphabetical within each group."""

    q = query.strip().lower()
    scored = []
    for name, value in names:
        score = _match_score(name, q)
        if score is not None:
            scored.append((score, name.lower(), value))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [value for _, _, value in scored[:limit]]


# region Wildcards


@dataclass(frozen=True)
class Wildcard:
    name: str
    """``hair/colors`` (no extension; ``/`` separated)"""
    path: str | None = None
    """the text file (None for YAML collections)"""
    values: tuple[str, ...] | None = None
    """the values of a YAML collection"""
    source: str = ""
    """the folder it was found in (display only)"""


def _yaml_wildcards(path: str, source: str) -> list[Wildcard]:
    import yaml

    try:
        with open(path, encoding="utf8") as file:
            data = yaml.safe_load(file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        log.warning("Skipping wildcard file %s: %s", path, error)
        return []
    found: list[Wildcard] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}/{key}" if prefix else str(key))
        elif isinstance(node, list) and prefix and all(isinstance(item, (str, int, float)) for item in node):
            found.append(Wildcard(prefix, values=tuple(str(item) for item in node), source=source))

    walk(data, "")
    return found


YamlCache = dict[str, tuple[tuple[int, int], list[Wildcard]]]
"""``path → ((mtime_ns, size), wildcards)``: YAML files parsed by earlier scans"""


def _yaml_cached(path: str, source: str, cache: YamlCache | None) -> list[Wildcard]:
    if cache is None:
        return _yaml_wildcards(path, source)
    try:
        stat = os.stat(path)
    except OSError:
        return []
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    found = _yaml_wildcards(path, source)
    cache[path] = (stamp, found)
    return found


def scan_wildcards(roots: Sequence[str], excluded: Iterable[str] = (), *, yaml_cache: YamlCache | None = None) -> list[Wildcard]:
    """Text files (``*.txt``) and YAML collections under ``roots``; folders
    named in ``excluded`` are skipped. The first root wins for a name.
    Symlinked folders are followed once (a link back up the tree is not a
    loop); ``yaml_cache`` reuses the parse of unchanged YAML files."""

    skip = {name.strip() for name in excluded if name.strip()}
    found: dict[str, Wildcard] = {}
    seen_paths: set[str] = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        visited: set[str] = {os.path.realpath(root)}
        for directory, folders, files in os.walk(root, followlinks=True):
            kept = []
            for folder in sorted(folders):
                if folder in skip or folder.startswith("."):
                    continue
                real = os.path.realpath(os.path.join(directory, folder))
                if real in visited:
                    continue
                visited.add(real)
                kept.append(folder)
            folders[:] = kept
            for filename in sorted(files):
                full = os.path.join(directory, filename)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                lowered = filename.lower()
                if lowered.endswith(".txt") and lowered not in SKIPPED_WILDCARD_FILES:
                    name = rel[: -len(".txt")]
                    found.setdefault(name.lower(), Wildcard(name, path=full, source=root))
                elif lowered.endswith((".yaml", ".yml")):
                    seen_paths.add(full)
                    for wildcard in _yaml_cached(full, root, yaml_cache):
                        found.setdefault(wildcard.name.lower(), wildcard)
    if yaml_cache is not None:
        for path in [path for path in yaml_cache if path not in seen_paths]:
            del yaml_cache[path]
    return sorted(found.values(), key=lambda wildcard: wildcard.name.lower())


def wildcard_values(wildcard: Wildcard) -> list[str]:
    """A wildcard's values: non-empty, non-comment lines (YAML: its list)."""

    if wildcard.values is not None:
        return list(wildcard.values)
    if not wildcard.path:
        return []
    try:
        with open(wildcard.path, encoding="utf8", errors="replace") as file:
            lines = [line.strip() for line in file]
    except OSError:
        return []
    return [line for line in lines if line and not line.startswith("#")][:MAX_WILDCARD_VALUES]


class WildcardCache:
    """The wildcards of the roots. The first request scans; afterwards a
    request older than :data:`WILDCARD_SCAN_TTL` gets the current list at
    once and starts a rescan in the background (walking the folders; YAML
    files are parsed again only when they changed). A change of roots or
    exclusions, or :meth:`refresh`, scans before answering."""

    def __init__(self, roots: Callable[[], Sequence[str]], excluded: Callable[[], Iterable[str]] = lambda: (), *, ttl: float = WILDCARD_SCAN_TTL, background: bool = True) -> None:
        self._roots = roots
        self._excluded = excluded
        self._ttl = ttl
        self._background = background
        self._lock = threading.Lock()
        """guards the fields below"""
        self._scan_lock = threading.Lock()
        """one scan at a time (and ``_yaml``)"""
        self._items: list[Wildcard] | None = None
        self._key: tuple | None = None
        self._time = 0.0
        self._rescanning = False
        self._yaml: YamlCache = {}

    def _scan(self, key: tuple) -> list[Wildcard]:
        requested = time.monotonic()
        with self._scan_lock:
            with self._lock:
                # a scan that started after this request (concurrent first requests)
                if self._items is not None and self._key == key and self._time >= requested:
                    return self._items
            items = scan_wildcards(key[0], key[1], yaml_cache=self._yaml)
        with self._lock:
            # a refresh or another key while scanning: still the newest scan of this key
            self._items, self._key, self._time = items, key, time.monotonic()
            self._rescanning = False
        return items

    def _rescan(self, key: tuple) -> None:
        try:
            self._scan(key)
        except Exception:
            log.exception("scanning the wildcards")
            with self._lock:
                self._rescanning = False

    def items(self) -> list[Wildcard]:
        key = (tuple(self._roots()), tuple(self._excluded()))
        with self._lock:
            items = self._items
            current = items is not None and key == self._key
            stale = current and time.monotonic() - self._time > self._ttl
            start = stale and not self._rescanning
            if start:
                self._rescanning = True
        if current and items is not None:
            if start:
                if self._background:
                    threading.Thread(target=self._rescan, args=(key,), name="remi-wildcard-scan", daemon=True).start()
                else:
                    return self._scan(key)
            return items
        return self._scan(key)

    def refresh(self) -> None:
        with self._lock:
            self._items = None

    def find(self, name: str) -> Wildcard | None:
        key = name.strip().strip("_").lower()
        return next((wildcard for wildcard in self.items() if wildcard.name.lower() == key), None)

    def search(self, query: str, limit: int) -> list[Wildcard]:
        return search_names(((wildcard.name, wildcard) for wildcard in self.items()), query, limit)


# endregion

# region Chants


@dataclass(frozen=True)
class Chant:
    name: str
    content: str
    terms: str = ""
    color: int | None = None


def read_chants(path: str) -> list[Chant]:
    try:
        with open(path, encoding="utf8") as file:
            data = json.load(file)
    except (OSError, ValueError) as error:
        log.warning("Cannot read chants %s: %s", path, error)
        return []
    chants = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or not item.get("name") or not item.get("content"):
            continue
        color = item.get("color")
        chants.append(Chant(str(item["name"]), str(item["content"]).strip(), str(item.get("terms") or ""), color if isinstance(color, int) else None))
    return chants


@dataclass
class ChantIndex:
    path: str | None = None
    _mtime: float | None = None
    _chants: list[Chant] = field(default_factory=list)
    _haystacks: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _load(self, path: str | None) -> None:
        mtime = os.path.getmtime(path) if path and os.path.isfile(path) else None
        if path == self.path and mtime == self._mtime:
            return
        self.path, self._mtime = path, mtime
        self._chants = read_chants(path) if mtime is not None and path else []
        self._haystacks = [f"{chant.name} {chant.terms}".lower() for chant in self._chants]

    def search(self, path: str | None, query: str, limit: int) -> list[Chant]:
        with self._lock:
            self._load(path)
            chants, haystacks = self._chants, self._haystacks
        q = query.strip().lower()
        if not q:
            return chants[:limit]
        prefix, rest = [], []
        for chant, haystack in zip(chants, haystacks, strict=True):
            if chant.name.lower().startswith(q):
                prefix.append(chant)
            elif all(term in haystack for term in q.split()):
                rest.append(chant)
            if len(prefix) >= limit:
                break
        return (prefix + rest)[:limit]


# endregion

# region Extra networks


@dataclass(frozen=True)
class NetworkHit:
    record: Any
    """the index's :class:`remi.networks.Record` (entry, user metadata, description, base model)"""
    card: dict[str, Any]
    """the index's item (preview id, resolved prompt, …); empty when it cannot be built"""


def _shows_hidden(networks: Any, query: str) -> bool:
    """``extra_networks_hidden_models``, as the network browser applies it."""

    try:
        mode = networks.hidden_mode()
    except Exception:
        return True
    if mode == "Never":
        return False
    if mode == "When searched":
        from remi.networks import HIDDEN_SEARCH_MIN

        return len(query) >= HIDDEN_SEARCH_MIN
    return True


def network_items(networks: Any, kind: str, query: str, limit: int) -> list[NetworkHit]:
    """The records of the network index (``services.networks``) whose name,
    alias or path matches (cached by the index until the kind is
    refreshed), best first, with their cards."""

    try:
        records = networks.records(kind)
    except Exception:
        log.debug("network kind %s unavailable", kind, exc_info=True)
        return []
    q = query.strip().lower()
    hidden = _shows_hidden(networks, q)
    matched = []
    for record in records:
        if getattr(record, "hidden", False) and not hidden:
            continue
        entry = record.entry
        scores = [score for name in (entry.alias, entry.name, entry.relpath) if name and (score := _match_score(name, q)) is not None]
        if scores:
            matched.append((min(scores), (entry.alias or entry.name).lower(), record))
    matched.sort(key=lambda item: (item[0], item[1]))
    out = []
    for _, _, record in matched[:limit]:
        try:
            card = networks.item(record)
        except Exception:
            log.debug("no card for %s", record.entry.name, exc_info=True)
            card = {}
        out.append(NetworkHit(record, card if isinstance(card, dict) else {}))
    return out


def prompt_text(source: Any, user: Any) -> str | None:
    """A network's insert text: static, or computed from its user metadata
    (``remi.networks.PromptSource``); None when empty or failing."""

    try:
        value = source(user) if callable(source) else source
    except Exception:
        log.debug("network prompt failed", exc_info=True)
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def preferred_weight(user: Any, default_weight: float) -> float:
    """A LoRA's preferred weight (user metadata; unset or 0: the default)."""

    weight = user.get("preferred weight") if isinstance(user, dict) else None
    try:
        value = float(weight) if weight not in (None, "") else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return value if value and value == value else float(default_weight)


# endregion
