"""Persistent data of the prompt tools (``<remi data>/prompt-tools.db``):
completion usage counts, prompt history, favorites and their collections,
and cached translations. One SQLite connection behind a lock (WAL)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from remi.files import decode_cursor, encode_cursor

SCHEMA_VERSION = 1
FAVORITE_KINDS = ("prompt", "negative", "tags")
MAX_TEXT = 64 * 1024
MAX_NAME = 200
MAX_IMPORT = 50_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS usage (
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    uses_pos INTEGER NOT NULL DEFAULT 0,
    uses_neg INTEGER NOT NULL DEFAULT 0,
    last_used REAL NOT NULL,
    PRIMARY KEY (name, type)
);
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    tab TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL,
    negative TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    last_used REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 1,
    UNIQUE (prompt, negative)
);
CREATE INDEX IF NOT EXISTS history_last_used ON history (last_used DESC);
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS favorites (
    id TEXT PRIMARY KEY,
    collection_id TEXT REFERENCES collections (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    position REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS favorites_collection ON favorites (collection_id, position);
CREATE TABLE IF NOT EXISTS translations (
    provider TEXT NOT NULL,
    target TEXT NOT NULL,
    text TEXT NOT NULL,
    translation TEXT NOT NULL,
    created REAL NOT NULL,
    PRIMARY KEY (provider, target, text)
);
"""


class StoreError(ValueError):
    """A rejected change; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _like_terms(query: str | None) -> list[str]:
    terms = [term for term in (query or "").lower().split() if term]
    return ["%" + re.sub(r"([%_\\])", r"\\\1", term) + "%" for term in terms][:8]


def _clean_text(text: str, what: str = "text") -> str:
    if not isinstance(text, str):
        raise StoreError(f"{what} must be a string")
    if len(text) > MAX_TEXT:
        raise StoreError(f"{what} is too long")
    return text


def _clean_name(name: str | None) -> str:
    name = (name or "").strip()
    if len(name) > MAX_NAME:
        raise StoreError("name is too long")
    return name


class PromptToolsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
            self._db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)", (str(SCHEMA_VERSION),))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, params).fetchone()

    def meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def secret(self, name: str) -> str | None:
        """A secret (an API key): write-only for clients, never listed."""

        return self.meta(f"secret:{name}")

    def set_secret(self, name: str, value: str | None) -> None:
        """Stores a secret; None or "" removes it."""

        if value and len(value) > 4096:
            raise StoreError("the secret is too long")
        with self._tx() as db:
            if value:
                db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"secret:{name}", value))
            else:
                db.execute("DELETE FROM meta WHERE key = ?", (f"secret:{name}",))

    # region Usage (frequency sorting)

    def record_use(self, name: str, type: str, negative: bool = False, *, when: float | None = None) -> None:
        name = name.strip()
        if not name or len(name) > 512:
            raise StoreError("invalid name")
        column = "uses_neg" if negative else "uses_pos"
        with self._tx() as db:
            db.execute(
                f"INSERT INTO usage (name, type, {column}, last_used) VALUES (?, ?, 1, ?) "
                f"ON CONFLICT (name, type) DO UPDATE SET {column} = {column} + 1, last_used = excluded.last_used",
                (name, type, when if when is not None else time.time()),
            )

    def uses(self, names: Iterable[str], type: str, negative: bool = False, *, max_age_days: float = 0) -> dict[str, int]:
        """``name → uses`` (in the positive or negative prompt) of the
        ``names`` used within ``max_age_days`` (0: any time)."""

        names = list(dict.fromkeys(name for name in names if name))
        if not names:
            return {}
        column = "uses_neg" if negative else "uses_pos"
        since = time.time() - max_age_days * 86400 if max_age_days > 0 else 0
        found: dict[str, int] = {}
        for start in range(0, len(names), 500):
            chunk = names[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self._all(f"SELECT name, {column} AS n FROM usage WHERE type = ? AND last_used >= ? AND name IN ({marks})", (type, since, *chunk)):
                if row["n"]:
                    found[row["name"]] = int(row["n"])
        return found

    def reset_use(self, name: str, type: str) -> bool:
        with self._tx() as db:
            return db.execute("DELETE FROM usage WHERE name = ? AND type = ?", (name, type)).rowcount > 0

    def most_used(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._all("SELECT name, type, uses_pos, uses_neg, last_used FROM usage ORDER BY uses_pos + uses_neg DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    # endregion

    # region History

    @staticmethod
    def _upsert_history(db: sqlite3.Connection, prompt: str, negative: str, tab: str, when: float, name: str = "") -> None:
        db.execute(
            "INSERT INTO history (id, tab, prompt, negative, name, created, last_used, uses) VALUES (?, ?, ?, ?, ?, ?, ?, 1) "
            "ON CONFLICT (prompt, negative) DO UPDATE SET last_used = MAX(last_used, excluded.last_used), uses = uses + 1, tab = excluded.tab, "
            "name = CASE WHEN excluded.name != '' THEN excluded.name ELSE name END",
            (_new_id(), tab, prompt, negative, name, when, when),
        )

    @staticmethod
    def _trim_history(db: sqlite3.Connection, limit: int) -> None:
        if limit > 0:
            db.execute("DELETE FROM history WHERE name = '' AND id NOT IN (SELECT id FROM history ORDER BY last_used DESC LIMIT ?)", (limit,))

    def record_history(self, prompt: str, negative: str = "", tab: str = "", *, when: float | None = None, limit: int = 0, name: str = "") -> dict[str, Any] | None:
        """Adds a prompt pair, or moves an identical one to the top (``uses``
        counts the generations). ``limit`` > 0 keeps that many unnamed entries."""

        prompt, negative = _clean_text(prompt or "", "prompt"), _clean_text(negative or "", "negative prompt")
        if not prompt.strip() and not negative.strip():
            return None
        now = when if when is not None else time.time()
        with self._tx() as db:
            self._upsert_history(db, prompt, negative, tab, now, _clean_name(name))
            self._trim_history(db, limit)
            row = db.execute("SELECT * FROM history WHERE prompt = ? AND negative = ?", (prompt, negative)).fetchone()
        return self._history_item(row) if row else None

    @staticmethod
    def _history_item(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "tab": row["tab"],
            "prompt": row["prompt"],
            "negative_prompt": row["negative"],
            "name": row["name"],
            "created": row["created"],
            "last_used": row["last_used"],
            "uses": row["uses"],
        }

    def history(self, *, query: str | None = None, field: str | None = None, tab: str | None = None, cursor: str | None = None, limit: int = 50) -> tuple[list[dict[str, Any]], str | None, int]:
        """Newest first. ``field`` ``prompt`` / ``negative``: one item per
        distinct text of that field (the latest entry, uses summed); search
        terms must all appear in the prompts or the name."""

        where, params = [], []
        if field == "prompt":
            where.append("prompt != ''")
        elif field == "negative":
            where.append("negative != ''")
        if tab:
            where.append("tab = ?")
            params.append(tab)
        haystack = {"prompt": "prompt || ' ' || name", "negative": "negative || ' ' || name"}.get(field or "", "prompt || ' ' || negative || ' ' || name")
        for term in _like_terms(query):
            where.append(f"lower({haystack}) LIKE ? ESCAPE '\\'")
            params.append(term)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        position = decode_cursor(cursor, 1)
        offset = int(position[0]) if position and position[0].isdigit() else 0
        limit = min(max(int(limit), 1), 200)
        if field in ("prompt", "negative"):
            column = "prompt" if field == "prompt" else "negative"
            base = f"SELECT id, tab, prompt, negative, name, created, MAX(last_used) AS last_used, SUM(uses) AS uses FROM history {clause} GROUP BY {column}"
            total = self._one(f"SELECT COUNT(*) AS n FROM ({base})", params)["n"]
            rows = self._all(f"{base} ORDER BY last_used DESC LIMIT ? OFFSET ?", (*params, limit, offset))
        else:
            total = self._one(f"SELECT COUNT(*) AS n FROM history {clause}", params)["n"]
            rows = self._all(f"SELECT * FROM history {clause} ORDER BY last_used DESC LIMIT ? OFFSET ?", (*params, limit, offset))
        next_cursor = encode_cursor(offset + limit) if offset + limit < total else None
        return [self._history_item(row) for row in rows], next_cursor, int(total)

    def history_entry(self, entry_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM history WHERE id = ?", (entry_id,))
        return self._history_item(row) if row else None

    @staticmethod
    def _entry_scope(db: sqlite3.Connection, entry_id: str, field: str | None) -> tuple[str, tuple[Any, ...]] | None:
        """The rows an action on ``entry_id`` covers: the entry itself, or with
        ``field`` (a grouped list: one item per distinct prompt / negative
        prompt) every entry with the same text in that field."""

        if field not in ("prompt", "negative"):
            row = db.execute("SELECT id FROM history WHERE id = ?", (entry_id,)).fetchone()
            return ("id = ?", (entry_id,)) if row else None
        row = db.execute(f"SELECT {field} AS text FROM history WHERE id = ?", (entry_id,)).fetchone()
        return (f"{field} = ?", (row["text"],)) if row else None

    def rename_history(self, entry_id: str, name: str, *, field: str | None = None) -> dict[str, Any]:
        """Names an entry (``field``: every entry of its group, see :meth:`history`)."""

        with self._tx() as db:
            scope = self._entry_scope(db, entry_id, field)
            if scope is None:
                raise StoreError("no such history entry", 404)
            db.execute(f"UPDATE history SET name = ? WHERE {scope[0]}", (_clean_name(name), *scope[1]))
        return self.history_entry(entry_id)  # type: ignore[return-value]

    def delete_history(self, entry_id: str, *, field: str | None = None) -> int:
        """Removes an entry (``field``: its whole group); the number removed."""

        with self._tx() as db:
            scope = self._entry_scope(db, entry_id, field)
            if scope is None:
                return 0
            return db.execute(f"DELETE FROM history WHERE {scope[0]}", scope[1]).rowcount

    def clear_history(self, *, keep_named: bool = True) -> int:
        with self._tx() as db:
            return db.execute("DELETE FROM history" + (" WHERE name = ''" if keep_named else "")).rowcount

    # endregion

    # region Favorites and collections

    def collections(self) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT c.id, c.name, c.created, COUNT(f.id) AS count FROM collections c LEFT JOIN favorites f ON f.collection_id = c.id GROUP BY c.id ORDER BY lower(c.name)"
        )
        unsorted = self._one("SELECT COUNT(*) AS n FROM favorites WHERE collection_id IS NULL")["n"]
        return [{"id": None, "name": "", "created": 0.0, "count": int(unsorted)}, *({"id": row["id"], "name": row["name"], "created": row["created"], "count": row["count"]} for row in rows)]

    def create_collection(self, name: str) -> dict[str, Any]:
        name = _clean_name(name)
        if not name:
            raise StoreError("a collection needs a name")
        collection_id = _new_id()
        try:
            with self._tx() as db:
                db.execute("INSERT INTO collections (id, name, created) VALUES (?, ?, ?)", (collection_id, name, time.time()))
        except sqlite3.IntegrityError as error:
            raise StoreError("a collection with this name exists", 409) from error
        return {"id": collection_id, "name": name, "created": time.time(), "count": 0}

    def rename_collection(self, collection_id: str, name: str) -> None:
        name = _clean_name(name)
        if not name:
            raise StoreError("a collection needs a name")
        try:
            with self._tx() as db:
                if db.execute("UPDATE collections SET name = ? WHERE id = ?", (name, collection_id)).rowcount == 0:
                    raise StoreError("no such collection", 404)
        except sqlite3.IntegrityError as error:
            raise StoreError("a collection with this name exists", 409) from error

    def delete_collection(self, collection_id: str) -> bool:
        """Deletes the collection and its favorites."""

        with self._tx() as db:
            return db.execute("DELETE FROM collections WHERE id = ?", (collection_id,)).rowcount > 0

    def _check_collection(self, db: sqlite3.Connection, collection_id: str | None) -> None:
        if collection_id is not None and db.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone() is None:
            raise StoreError("no such collection", 404)

    @staticmethod
    def _favorite_item(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "collection_id": row["collection_id"], "kind": row["kind"], "text": row["text"], "name": row["name"], "created": row["created"], "position": row["position"]}

    def _favorite_query(self, *, collection_id: str | None, all_collections: bool, kinds: Iterable[str] | str | None, query: str | None) -> tuple[str, list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if not all_collections:
            if collection_id is None:
                where.append("collection_id IS NULL")
            else:
                where.append("collection_id = ?")
                params.append(collection_id)
        wanted = [kinds] if isinstance(kinds, str) else list(dict.fromkeys(kinds or ()))
        wanted = [kind for kind in wanted if kind]
        if wanted:
            where.append(f"kind IN ({','.join('?' * len(wanted))})")
            params.extend(wanted)
        for term in _like_terms(query):
            where.append("lower(text || ' ' || name) LIKE ? ESCAPE '\\'")
            params.append(term)
        return (f"WHERE {' AND '.join(where)}" if where else ""), params

    def favorites_page(
        self,
        *,
        collection_id: str | None = None,
        all_collections: bool = False,
        kinds: Iterable[str] | str | None = None,
        query: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        """A page of favorites in their order (``kinds``: any of them);
        the next cursor and the number of matches."""

        clause, params = self._favorite_query(collection_id=collection_id, all_collections=all_collections, kinds=kinds, query=query)
        position = decode_cursor(cursor, 1)
        offset = int(position[0]) if position and position[0].isdigit() else 0
        limit = min(max(int(limit), 1), 500)
        total = int(self._one(f"SELECT COUNT(*) AS n FROM favorites {clause}", params)["n"])
        rows = self._all(f"SELECT * FROM favorites {clause} ORDER BY position, created LIMIT ? OFFSET ?", (*params, limit, offset))
        next_cursor = encode_cursor(offset + limit) if offset + limit < total else None
        return [self._favorite_item(row) for row in rows], next_cursor, total

    def favorites(self, *, collection_id: str | None = None, all_collections: bool = False, kind: str | None = None, query: str | None = None, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Every matching favorite, in order (no paging)."""

        clause, params = self._favorite_query(collection_id=collection_id, all_collections=all_collections, kinds=kinds if kinds is not None else kind, query=query)
        rows = self._all(f"SELECT * FROM favorites {clause} ORDER BY position, created", params)
        return [self._favorite_item(row) for row in rows]

    def favorite(self, favorite_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM favorites WHERE id = ?", (favorite_id,))
        return self._favorite_item(row) if row else None

    def _insert_favorite(self, db: sqlite3.Connection, text: str, kind: str, name: str, collection_id: str | None, *, at_end: bool = False) -> str:
        self._check_collection(db, collection_id)
        if at_end:
            last = db.execute("SELECT MAX(position) AS p FROM favorites WHERE collection_id IS ?", (collection_id,)).fetchone()["p"]
            position = (last if last is not None else 0.0) + 1.0
        else:
            top = db.execute("SELECT MIN(position) AS p FROM favorites WHERE collection_id IS ?", (collection_id,)).fetchone()["p"]
            position = (top if top is not None else 0.0) - 1.0
        favorite_id = _new_id()
        db.execute(
            "INSERT INTO favorites (id, collection_id, kind, text, name, created, position) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (favorite_id, collection_id, kind, text, name, time.time(), position),
        )
        return favorite_id

    @staticmethod
    def _has_favorite(db: sqlite3.Connection, collection_id: str | None, kind: str, text: str) -> bool:
        return db.execute("SELECT 1 FROM favorites WHERE collection_id IS ? AND kind = ? AND text = ?", (collection_id, kind, text)).fetchone() is not None

    @staticmethod
    def _collection_id(db: sqlite3.Connection, name: str) -> str:
        """The id of the collection called ``name`` (created when missing)."""

        row = db.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        collection_id = _new_id()
        db.execute("INSERT INTO collections (id, name, created) VALUES (?, ?, ?)", (collection_id, name, time.time()))
        return collection_id

    def add_favorite(self, text: str, kind: str = "prompt", *, name: str | None = None, collection_id: str | None = None) -> dict[str, Any]:
        text = _clean_text(text).strip()
        if not text:
            raise StoreError("nothing to save")
        if kind not in FAVORITE_KINDS:
            raise StoreError(f"kind must be one of {', '.join(FAVORITE_KINDS)}")
        with self._tx() as db:
            favorite_id = self._insert_favorite(db, text, kind, _clean_name(name), collection_id)
        return self.favorite(favorite_id)  # type: ignore[return-value]

    def update_favorite(self, favorite_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        """``name``, ``text``, ``collection_id`` (None: unsorted) and
        ``before`` (an id in the same collection to move in front of; None:
        to the end)."""

        with self._tx() as db:
            row = db.execute("SELECT * FROM favorites WHERE id = ?", (favorite_id,)).fetchone()
            if row is None:
                raise StoreError("no such favorite", 404)
            sets, params = [], []
            collection_id = row["collection_id"]
            if "name" in changes:
                sets.append("name = ?")
                params.append(_clean_name(changes["name"]))
            if "text" in changes:
                text = _clean_text(changes["text"]).strip()
                if not text:
                    raise StoreError("nothing to save")
                sets.append("text = ?")
                params.append(text)
            if "collection_id" in changes:
                collection_id = changes["collection_id"]
                self._check_collection(db, collection_id)
                sets.append("collection_id = ?")
                params.append(collection_id)
            if "before" in changes or "collection_id" in changes:
                before = changes.get("before")
                if before:
                    target = db.execute("SELECT position FROM favorites WHERE id = ? AND collection_id IS ?", (before, collection_id)).fetchone()
                    if target is None:
                        raise StoreError("no such favorite to move before", 404)
                    previous = db.execute(
                        "SELECT MAX(position) AS p FROM favorites WHERE collection_id IS ? AND position < ? AND id != ?",
                        (collection_id, target["position"], favorite_id),
                    ).fetchone()["p"]
                    position = (target["position"] + previous) / 2 if previous is not None else target["position"] - 1.0
                else:
                    last = db.execute("SELECT MAX(position) AS p FROM favorites WHERE collection_id IS ? AND id != ?", (collection_id, favorite_id)).fetchone()["p"]
                    position = (last if last is not None else 0.0) + 1.0
                sets.append("position = ?")
                params.append(position)
            if sets:
                db.execute(f"UPDATE favorites SET {', '.join(sets)} WHERE id = ?", (*params, favorite_id))
        return self.favorite(favorite_id)  # type: ignore[return-value]

    def delete_favorite(self, favorite_id: str) -> bool:
        with self._tx() as db:
            return db.execute("DELETE FROM favorites WHERE id = ?", (favorite_id,)).rowcount > 0

    def export_favorites(self) -> dict[str, Any]:
        """Every favorite (with its collection's name), in order."""

        collections = {item["id"]: item["name"] for item in self.collections() if item["id"]}
        return {
            "version": 1,
            "favorites": [
                {"collection": collections.get(item["collection_id"], ""), "kind": item["kind"], "text": item["text"], "name": item["name"]}
                for item in self.favorites(all_collections=True)
            ],
        }

    def import_favorites(self, data: Mapping[str, Any]) -> int:
        """Adds the favorites of an export, in its order (collections by
        name, duplicates in the same collection and malformed items skipped;
        one transaction); returns how many were added."""

        items = data.get("favorites") if isinstance(data, Mapping) else None
        if not isinstance(items, list):
            raise StoreError("not a favorites export")
        if len(items) > MAX_IMPORT:
            raise StoreError(f"at most {MAX_IMPORT} favorites can be imported at once", 413)
        added = 0
        with self._tx() as db:
            for item in items:
                if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                    continue
                text = item["text"].strip()
                if not text or len(text) > MAX_TEXT:
                    continue
                kind = item.get("kind") if item.get("kind") in FAVORITE_KINDS else "prompt"
                collection = str(item.get("collection") or "").strip()[:MAX_NAME]
                collection_id = self._collection_id(db, collection) if collection else None
                if self._has_favorite(db, collection_id, kind, text):
                    continue
                self._insert_favorite(db, text, kind, str(item.get("name") or "").strip()[:MAX_NAME], collection_id, at_end=True)
                added += 1
        return added

    # endregion

    # region Translation cache

    def cached_translations(self, provider: str, target: str, texts: Iterable[str]) -> dict[str, str]:
        texts = list(dict.fromkeys(texts))
        found: dict[str, str] = {}
        for start in range(0, len(texts), 500):
            chunk = texts[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self._all(f"SELECT text, translation FROM translations WHERE provider = ? AND target = ? AND text IN ({marks})", (provider, target, *chunk)):
                found[row["text"]] = row["translation"]
        return found

    def cache_translations(self, provider: str, target: str, pairs: Mapping[str, str]) -> None:
        now = time.time()
        with self._tx() as db:
            db.executemany(
                "INSERT OR REPLACE INTO translations (provider, target, text, translation, created) VALUES (?, ?, ?, ?, ?)",
                [(provider, target, text, translation, now) for text, translation in pairs.items() if text and translation],
            )

    # endregion

    # region Legacy data (tag autocomplete and Prompt All-in-One)

    LEGACY_TYPES = {1: "tag", 2: "tag", 3: "embedding", 4: "wildcard-value", 5: "wildcard", 6: "wildcard", 7: "wildcard", 8: "lora", 9: "lora", 10: "chant"}
    """tag autocomplete's ``ResultType`` → usage type (extra tags count as tags,
    as :meth:`PromptTools.record_use` records them; ``styleName`` is dropped)"""

    def import_tag_frequency(self, path: str | Path) -> int:
        """Usage counts of the tag autocomplete extension's
        ``tag_frequency.db`` (once; one transaction; bad rows skipped)."""

        if self.meta("imported:tag_frequency") or not os.path.isfile(path):
            return 0
        try:
            source = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
            try:
                rows = source.execute("SELECT name, type, count_pos, count_neg, last_used FROM tag_frequency").fetchall()
            finally:
                source.close()
        except sqlite3.Error:
            return 0
        added = 0
        with self._tx() as db:
            for name, legacy_type, pos, neg, last_used in rows:
                try:
                    kind = self.LEGACY_TYPES.get(int(legacy_type or 0))
                    name = str(name or "").strip()
                    if not kind or not name or len(name) > 512:
                        continue
                    positive, negative = max(int(pos or 0), 0), max(int(neg or 0), 0)
                except (TypeError, ValueError):
                    continue
                try:
                    when = time.mktime(time.strptime(str(last_used)[:19], "%Y-%m-%d %H:%M:%S")) if last_used else time.time()
                except (ValueError, OverflowError):
                    when = time.time()
                db.execute(
                    "INSERT INTO usage (name, type, uses_pos, uses_neg, last_used) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (name, type) DO UPDATE SET uses_pos = uses_pos + excluded.uses_pos, uses_neg = uses_neg + excluded.uses_neg, last_used = MAX(last_used, excluded.last_used)",
                    (name, kind, positive, negative, when),
                )
                added += 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('imported:tag_frequency', ?)", (str(time.time()),))
        return added

    def import_prompt_all_in_one(self, storage_dir: str | Path) -> int:
        """History and favorites of Prompt All-in-One's JSON storage (once):
        ``history.<tab>[_neg].json`` and ``favorite.<tab>[_neg].json``, with
        their names, in their order. One transaction: malformed items are
        skipped, and the import is marked done together with its data."""

        if self.meta("imported:prompt_all_in_one"):
            return 0
        directory = Path(storage_dir)
        if not directory.is_dir():
            return 0

        def read(name: str) -> list[Any]:
            try:
                data = json.loads((directory / f"{name}.json").read_text(encoding="utf8"))
            except (OSError, ValueError):
                return []
            return data if isinstance(data, list) else []

        def text_of(item: Any) -> str | None:
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
                return None
            text = item["prompt"].strip()
            return text if text and len(text) <= MAX_TEXT else None

        def when_of(item: dict[str, Any]) -> float:
            try:
                value = float(item.get("time") or 0)
            except (TypeError, ValueError):
                return time.time()
            # seconds; a millisecond timestamp is scaled down
            if value > 1e11:
                value /= 1000
            return value if 0 < value < 1e11 else time.time()

        def name_of(item: dict[str, Any]) -> str:
            return str(item.get("name") or "").strip()[:MAX_NAME]

        added = 0
        with self._tx() as db:
            for tab in ("txt2img", "img2img"):
                for negative in (False, True):
                    suffix = "_neg" if negative else ""
                    for item in read(f"history.{tab}{suffix}"):
                        text = text_of(item)
                        if text is None:
                            continue
                        self._upsert_history(db, "" if negative else text, text if negative else "", tab, when_of(item), name_of(item))
                        added += 1
                    collection_id: str | None = None
                    kind = "negative" if negative else "prompt"
                    for item in read(f"favorite.{tab}{suffix}"):
                        text = text_of(item)
                        if text is None:
                            continue
                        if collection_id is None:
                            collection_id = self._collection_id(db, f"Prompt All-in-One · {tab}{' negative' if negative else ''}")
                        if self._has_favorite(db, collection_id, kind, text):
                            continue
                        self._insert_favorite(db, text, kind, name_of(item), collection_id, at_end=True)
                        added += 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('imported:prompt_all_in_one', ?)", (str(time.time()),))
        return added

    # endregion
