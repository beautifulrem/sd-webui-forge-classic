"""In-memory booru tag index for prompt autocompletion.

A tag file is the CSV format of the a1111 tag autocomplete lists
(``name,category,count,"alias1,alias2"``); an *extra* file adds a few custom
tags (``name,category,description,aliases,translation``) and a translation
file maps tags to another language (``tag,translation``, or the old
``tag,category,translation``).

Entries are kept in popularity order (post count, then file order), so an
entry's index is its rank. Lookups:

* name prefix, word prefix (``uni`` → ``school_uniform``) and alias prefix:
  sorted keys and ``bisect``, then the best ranks of the range;
* translation and substring (``*query``) matches: one newline-joined blob in
  rank order, scanned with ``str.find`` until enough hits;
* fuzzy subsequences (``detco`` → ``detached_collar``): a regex over the
  most popular tags that start with the query's first letter, only when the
  other tiers found too little.

Measured (Apple M-series, ``danbooru_e621_merged.csv``, 222k tags): building
takes about 4 s and 130–180 MB (twice that for a moment while a changed list
is rebuilt next to the old one); a prefix lookup takes under 1.5 ms, a
query that falls through to the substring and fuzzy tiers 2–7 ms.

:func:`rank` scores the matches (match tier, then post count, then the
user's own usage) and is pure, so the ranking is tested without files.
"""

from __future__ import annotations

import bisect
import csv
import heapq
import math
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

FUZZY_SCOPE = 60_000
"""Fuzzy matching only looks at this many of the most popular tags."""
MAX_QUERY = 120


class Tier(IntEnum):
    """How a tag matched; the value is the base score of the tier."""

    FUZZY = 0
    SUBSTRING = 20
    TRANSLATION = 34
    ALIAS_PREFIX = 38
    WORD = 46
    ALIAS_EXACT = 48
    PREFIX = 50
    EXACT = 100


@dataclass(slots=True)
class TagEntry:
    name: str
    """as in the file (underscores)"""
    category: int = 0
    count: int = 0
    aliases: tuple[str, ...] = ()
    extra: bool = False
    """from the extra file"""
    meta: str | None = None
    """the extra file's description (e.g. "Quality tag")"""


@dataclass(slots=True)
class TagMatch:
    entry: TagEntry
    rank: int
    """popularity rank in its list (extra entries: file order)"""
    tier: Tier
    alias: str | None = None
    """the alias that matched (alias tiers)"""
    score: float = 0.0
    uses: int = 0


@dataclass(frozen=True)
class FrequencyConfig:
    """``tac_frequency*`` settings (usage counts are already filtered by age)."""

    enabled: bool = True
    function: str = "Logarithmic (weak)"
    min_count: int = 3
    recommend_cap: int = 10


ARTIST_PREFIX_RE = re.compile(r"^@(?=[^\W_])")
"""Anima's ``@artist`` prefix (``@_@`` is a tag, not a prefix)."""


def strip_artist_prefix(text: str) -> str:
    """``@wlop`` → ``wlop``: the prefix is written by the insertion, the tag
    lists have none."""

    return ARTIST_PREFIX_RE.sub("", text.lstrip())


def normalize_query(text: str) -> str:
    """What a typed word searches for: lower case, ``\\(`` unescaped, spaces
    as underscores (tags are stored with underscores), no ``@`` artist
    prefix."""

    text = strip_artist_prefix(text).strip().lower().replace("\\", "")
    text = re.sub(r"\s+", "_", text)
    return text[:MAX_QUERY]


def abbreviation_pattern(query: str) -> re.Pattern[str]:
    """Fuzzy matching as abbreviations: each next character continues the
    current word or starts a later word (``detco`` → ``det``ached_``co``llar,
    ``lh`` → ``l``ong_``h``air); the first one starts the tag. Matches stay
    on one line of the newline-joined blob (one tag): neither the gap nor the
    word separator may be a line break."""

    first, rest = query[0], query[1:]
    parts = [re.escape(first)]
    for char in rest:
        escaped = re.escape(char)
        parts.append(f"(?:{escaped}|[^\\n]*?[_ \\t(:/-]{escaped})")
    return re.compile("^" + "".join(parts), re.MULTILINE)


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def _is_int(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"\s*-?\d+\s*", value) is not None


def _aliases(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(dict.fromkeys(alias.strip().lower() for alias in value.split(",") if alias.strip()))


def _rows(path: str) -> Iterable[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if row and row[0].strip() and not row[0].startswith("#"):
                yield row


def read_tag_file(path: str) -> list[TagEntry]:
    """A tag list, in file order."""

    entries = []
    for row in _rows(path):
        entries.append(TagEntry(row[0].strip(), _int(row[1] if len(row) > 1 else None), _int(row[2] if len(row) > 2 else None), _aliases(row[3] if len(row) > 3 else None)))
    return entries


def read_extra_file(path: str) -> tuple[list[TagEntry], dict[str, str]]:
    """Extra tags (``name,category,description,aliases,translation``; a plain
    tag-file row is accepted too) and the translations they carry."""

    entries, translations = [], {}
    for row in _rows(path):
        name = row[0].strip()
        category = _int(row[1] if len(row) > 1 else None)
        third = row[2].strip() if len(row) > 2 else ""
        if _is_int(third):
            entries.append(TagEntry(name, category, _int(third), _aliases(row[3] if len(row) > 3 else None), extra=True))
            continue
        entries.append(TagEntry(name, category, 0, _aliases(row[3] if len(row) > 3 else None), extra=True, meta=third or None))
        if len(row) > 4 and row[4].strip():
            translations[tag_key(name)] = row[4].strip()
    return entries, translations


def read_translation_file(path: str) -> dict[str, str]:
    """``tag → translation``; the old three-column format (``tag,category,
    translation``) is recognised by its numeric second column."""

    translations = {}
    for row in _rows(path):
        if len(row) >= 3 and _is_int(row[1]) and row[2].strip():
            value = row[2].strip()
        elif len(row) >= 2 and row[1].strip():
            value = row[1].strip()
        else:
            continue
        translations[tag_key(row[0])] = value
    return translations


def tag_key(name: str) -> str:
    """The lookup key of a tag: lower case with underscores."""

    return re.sub(r"\s+", "_", name.strip().lower())


class _Blob:
    """Newline-joined lines in rank order and the rank of each line."""

    def __init__(self, lines: Sequence[str], ranks: Sequence[int]) -> None:
        self.text = "\n".join(line.replace("\n", " ") for line in lines)
        self.ranks = list(ranks)
        self.starts: list[int] = []
        position = 0
        for line in lines:
            self.starts.append(position)
            position += len(line) + 1

    def line_at(self, position: int) -> int:
        return bisect.bisect_right(self.starts, position) - 1

    def find(self, needle: str, want: int) -> list[int]:
        """Ranks of the first ``want`` lines containing ``needle`` (in rank order)."""

        found: list[int] = []
        text, at = self.text, 0
        while len(found) < want:
            position = text.find(needle, at)
            if position < 0:
                break
            line = self.line_at(position)
            found.append(self.ranks[line])
            # continue at the next line: one hit per line
            at = self.starts[line + 1] if line + 1 < len(self.starts) else len(text)
        return found

    def search(self, pattern: re.Pattern[str], want: int, limit_chars: int | None = None) -> list[int]:
        found: list[int] = []
        end = len(self.text) if limit_chars is None else min(limit_chars, len(self.text))
        for match in pattern.finditer(self.text, 0, end):
            found.append(self.ranks[self.line_at(match.start())])
            if len(found) >= want:
                break
        return found


def _sorted_keys(pairs: Iterable[tuple[str, int]]) -> tuple[list[str], list[int]]:
    ordered = sorted(pairs)
    return [key for key, _ in ordered], [rank for _, rank in ordered]


def _range(keys: list[str], ranks: list[int], prefix: str) -> list[int]:
    lo = bisect.bisect_left(keys, prefix)
    hi = bisect.bisect_left(keys, prefix + "\U0010ffff", lo)
    return ranks[lo:hi]


class TagIndex:
    """The searchable tag list (plus extra tags and translations)."""

    def __init__(self, entries: Sequence[TagEntry], extras: Sequence[TagEntry] = (), translations: Mapping[str, str] | None = None) -> None:
        # popularity order; stable for equal counts (file order)
        self.entries: list[TagEntry] = sorted(entries, key=lambda entry: -entry.count)
        self.extras: list[TagEntry] = list(extras)
        self.translations: dict[str, str] = dict(translations or {})
        lowered = [entry.name.lower() for entry in self.entries]
        self._names = lowered
        self._by_name: dict[str, int] = {}
        for rank, name in enumerate(lowered):
            self._by_name.setdefault(tag_key(name), rank)

        self._prefix_keys, self._prefix_ranks = _sorted_keys((name, rank) for rank, name in enumerate(lowered))
        words: list[tuple[str, int]] = []
        for rank, name in enumerate(lowered):
            for match in re.finditer(r"[_\s(:/-]+", name):
                suffix = name[match.end() :]
                if suffix:
                    words.append((suffix.lstrip("(_"), rank))
        self._word_keys, self._word_ranks = _sorted_keys(words)
        self._alias_pairs = sorted((alias, rank) for rank, entry in enumerate(self.entries) for alias in entry.aliases)
        self._alias_keys = [alias for alias, _ in self._alias_pairs]
        self._names_blob = _Blob(lowered, range(len(lowered)))
        # fuzzy matches start with the query's first character: one small
        # blob per first character of the most popular tags (rank order)
        by_first: dict[str, list[int]] = {}
        for rank, name in enumerate(lowered[:FUZZY_SCOPE]):
            if name:
                by_first.setdefault(name[0], []).append(rank)
        self._fuzzy_blobs = {first: _Blob([lowered[rank] for rank in ranks], ranks) for first, ranks in by_first.items()}
        translated = [(rank, self.translations[key].lower()) for rank, name in enumerate(lowered) if (key := tag_key(name)) in self.translations]
        self._translation_blob = _Blob([text for _, text in translated], [rank for rank, _ in translated])

    def __len__(self) -> int:
        return len(self.entries)

    def translation(self, name: str) -> str | None:
        return self.translations.get(tag_key(name))

    def lookup(self, name: str) -> TagEntry | None:
        """The entry of an exact tag name (spaces or underscores), or None."""

        key = tag_key(name)
        rank = self._by_name.get(key)
        if rank is not None:
            return self.entries[rank]
        for entry in self.extras:
            if tag_key(entry.name) == key:
                return entry
        return None

    # region matching

    def _best(self, ranks: Iterable[int], want: int) -> list[int]:
        return heapq.nsmallest(want, set(ranks))

    def matches(self, query: str, limit: int = 10, *, aliases: bool = True, translations: bool = True, fuzzy: bool = True) -> list[TagMatch]:
        """Candidate matches of ``query`` (a typed word, normalised here), best
        tier per tag: up to ``3 * limit`` per tier (room for the usage
        ranking); substring and fuzzy matches only while fewer than ``limit``
        were found."""

        want = max(3 * limit, 12)

        query = strip_artist_prefix(query)
        substring_only = query.startswith("*")
        raw = query[1:] if substring_only else query
        q = normalize_query(raw)
        if not q:
            return []
        found: dict[int, TagMatch] = {}

        def add(rank: int, tier: Tier, alias: str | None = None) -> None:
            current = found.get(rank)
            if current is None or tier > current.tier:
                found[rank] = TagMatch(self.entries[rank], rank, tier, alias)

        if substring_only:
            for rank in self._names_blob.find(q, want):
                add(rank, Tier.SUBSTRING)
        else:
            exact = self._by_name.get(q)
            if exact is not None:
                add(exact, Tier.EXACT)
            for rank in self._best(_range(self._prefix_keys, self._prefix_ranks, q), want):
                add(rank, Tier.PREFIX)
            for rank in self._best(_range(self._word_keys, self._word_ranks, q), want):
                add(rank, Tier.WORD)
            if aliases:
                lo = bisect.bisect_left(self._alias_keys, q)
                hi = bisect.bisect_left(self._alias_keys, q + "\U0010ffff", lo)
                best: dict[int, str] = {}
                for alias, rank in self._alias_pairs[lo:hi]:
                    if rank not in best or alias == q:
                        best[rank] = alias
                for rank in heapq.nsmallest(want, best):
                    alias = best[rank]
                    add(rank, Tier.ALIAS_EXACT if alias == q else Tier.ALIAS_PREFIX, alias)
            if translations and self.translations:
                for rank in self._translation_blob.find(raw.strip().lower(), want):
                    add(rank, Tier.TRANSLATION)
            if len(found) < limit and len(q) >= 3:
                for rank in self._names_blob.find(q, want):
                    add(rank, Tier.SUBSTRING)
            if fuzzy and len(found) < limit and len(q) >= 3 and q.isascii() and (blob := self._fuzzy_blobs.get(q[0])) is not None:
                for rank in blob.search(abbreviation_pattern(q), want):
                    add(rank, Tier.FUZZY)
        return list(found.values())

    def extra_matches(self, query: str, *, aliases: bool = True) -> list[TagMatch]:
        """Matches among the extra tags (a short list, scanned)."""

        query = strip_artist_prefix(query)
        q = normalize_query(query.lstrip("*"))
        if not q:
            return []
        found = []
        for rank, entry in enumerate(self.extras):
            name = entry.name.lower()
            tier: Tier | None = None
            alias = None
            if name == q:
                tier = Tier.EXACT
            elif name.startswith(q):
                tier = Tier.PREFIX
            elif any(part.startswith(q) for part in re.split(r"[_\s]+", name)[1:]):
                tier = Tier.WORD
            elif aliases and (hit := next((a for a in entry.aliases if a.startswith(q)), None)):
                tier, alias = (Tier.ALIAS_EXACT if hit == q else Tier.ALIAS_PREFIX), hit
            elif len(q) >= 2 and q in name:
                tier = Tier.SUBSTRING
            elif (translation := self.translations.get(tag_key(name))) and query.strip().lower() in translation.lower():
                tier = Tier.TRANSLATION
            if tier is not None:
                found.append(TagMatch(entry, rank, tier, alias))
        return found

    # endregion


def usage_bias(uses: int, config: FrequencyConfig) -> float:
    if not config.enabled or uses < max(config.min_count, 1):
        return 0.0
    if config.function == "Usage first":
        return 1000.0 + uses
    if config.function == "Logarithmic (strong)":
        return 2 * math.log1p(uses)
    return math.log1p(uses)


def rank(matches: Iterable[TagMatch], uses: Mapping[str, int] | None = None, config: FrequencyConfig = FrequencyConfig(), *, extras_first: bool = True, limit: int = 20) -> list[TagMatch]:
    """Scores and orders matches: tier base + ``log(1 + posts)`` + usage
    bias (only the ``recommend_cap`` most used candidates get one). Extra tags
    go before (``extras_first``) or after every other match."""

    matches = list(matches)
    uses = uses or {}
    for match in matches:
        match.uses = int(uses.get(tag_key(match.entry.name), 0))
    boosted: set[int] = set()
    if config.enabled:
        used = [match for match in matches if match.uses >= max(config.min_count, 1)]
        used.sort(key=lambda match: -match.uses)
        cap = config.recommend_cap if config.recommend_cap > 0 else len(used)
        boosted = {id(match) for match in used[:cap]}
    for match in matches:
        bias = usage_bias(match.uses, config) if id(match) in boosted else 0.0
        score = float(match.tier) + math.log1p(max(match.entry.count, 0)) + bias
        if match.entry.extra:
            score += 10_000 if extras_first else -10_000
        match.score = score
    matches.sort(key=lambda match: (-match.score, match.entry.extra != extras_first, match.rank))
    return matches[: max(limit, 0)]


# region Loaded index (files, settings)


@dataclass
class TagFiles:
    tag_file: str | None = None
    extra_file: str | None = None
    translation_file: str | None = None
    extra_translations: Mapping[str, str] = field(default_factory=dict)
    """translations from elsewhere (tag groups), overridden by the files"""

    def signature(self) -> tuple:
        def stamp(path: str | None) -> tuple:
            if not path:
                return (None,)
            try:
                stat = os.stat(path)
                return (path, stat.st_mtime_ns, stat.st_size)
            except OSError:
                return (path, None)

        return (stamp(self.tag_file), stamp(self.extra_file), stamp(self.translation_file), len(self.extra_translations))


def load_index(files: TagFiles) -> TagIndex:
    entries = read_tag_file(files.tag_file) if files.tag_file and os.path.isfile(files.tag_file) else []
    extras: list[TagEntry] = []
    translations: dict[str, str] = dict(files.extra_translations)
    if files.translation_file and os.path.isfile(files.translation_file):
        translations.update(read_translation_file(files.translation_file))
    if files.extra_file and os.path.isfile(files.extra_file):
        extras, extra_translations = read_extra_file(files.extra_file)
        for key, value in extra_translations.items():
            translations.setdefault(key, value)
    return TagIndex(entries, extras, translations)


class IndexCache:
    """One loaded :class:`TagIndex`, rebuilt when its files (or their
    contents) change; concurrent first requests wait for one build."""

    def __init__(self, loader: Callable[[TagFiles], TagIndex] = load_index) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._index: TagIndex | None = None

    def get(self, files: TagFiles) -> TagIndex:
        signature = files.signature()
        with self._lock:
            if self._index is None or signature != self._signature:
                self._index = self._loader(files)
                self._signature = signature
            return self._index

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self._signature = None


# endregion
