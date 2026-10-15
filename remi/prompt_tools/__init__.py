"""Prompt tools: tag autocompletion (booru tags, aliases, translations,
LoRA / embeddings, wildcards, chants, usage ranking) and the prompt helpers
of Prompt All-in-One (prompt history, favorites and collections, quick tag
groups, translation). ``/api/v2/prompt-tools`` (remi/server/routes/
prompt_tools.py) serves them to the SPA (webui/src/features/prompt-tools).

Data and switches come from the built-in extensions: the tag autocomplete
extension registers its ``tags`` folder (:func:`register_tag_data`), Prompt
All-in-One its tag groups (:func:`register_tool_data`); a feature whose
extension is disabled is off. Per-user data lives in the server's data
folder (``prompt-tools.db``, :class:`~remi.prompt_tools.store.PromptToolsStore`).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remi.prompt_tools import options as opt
from remi.prompt_tools.groups import TagGroups
from remi.prompt_tools.sources import ChantIndex, WildcardCache, network_items, preferred_weight, prompt_text, wildcard_values
from remi.prompt_tools.store import PromptToolsStore, StoreError
from remi.prompt_tools.tags import FrequencyConfig, IndexCache, TagFiles, TagIndex, TagMatch, rank, tag_key, usage_bias
from remi.prompt_tools.translate import ProviderConfig, TranslationError, translate

__all__ = ["PromptTools", "StoreError", "TranslationError", "attach", "register_tag_data", "register_tool_data", "tools_for", "unregister_tag_data", "unregister_tool_data"]

log = logging.getLogger(__name__)

MODES = ("tag", "lora", "embedding", "chant", "network", "wildcard", "wildcard-value")


# region Registry (the extensions)


@dataclass
class Registry:
    tag_dirs: list[str] = field(default_factory=list)
    group_dirs: list[str] = field(default_factory=list)
    wildcard_dirs: list[str] = field(default_factory=list)
    """extra wildcard folders besides the default ones"""

    @property
    def tags_enabled(self) -> bool:
        return bool(self.tag_dirs)

    @property
    def tools_enabled(self) -> bool:
        return bool(self.group_dirs)


registry = Registry()


def register_tag_data(directory: str | os.PathLike[str]) -> None:
    """Enables tag autocompletion with the tag lists, translations and
    chants of ``directory``."""

    path = os.fspath(directory)
    if path not in registry.tag_dirs:
        registry.tag_dirs.append(path)


def register_tool_data(group_tags: str | os.PathLike[str]) -> None:
    """Enables the prompt tools (history, favorites, quick tag groups,
    translation) with the tag group files of ``group_tags``."""

    path = os.fspath(group_tags)
    if path not in registry.group_dirs:
        registry.group_dirs.append(path)
    _groups.directories = list(registry.group_dirs)


def unregister_tag_data(directory: str | os.PathLike[str]) -> None:
    """Undoes :func:`register_tag_data` (the extension's ``script_unloaded``:
    a Reload UI with the extension disabled leaves completion off)."""

    path = os.fspath(directory)
    if path in registry.tag_dirs:
        registry.tag_dirs.remove(path)
        refresh()


def unregister_tool_data(group_tags: str | os.PathLike[str]) -> None:
    """Undoes :func:`register_tool_data`."""

    path = os.fspath(group_tags)
    if path in registry.group_dirs:
        registry.group_dirs.remove(path)
    _groups.directories = list(registry.group_dirs)


def tag_dir() -> str | None:
    return registry.tag_dirs[0] if registry.tag_dirs else None


def _tag_path(name: Any) -> str | None:
    root = tag_dir()
    if not root or not isinstance(name, str) or name in ("", "None") or os.path.basename(name) != name:
        return None
    path = os.path.join(root, name)
    return path if os.path.isfile(path) else None


def default_wildcard_dirs() -> list[str]:
    """``scripts/wildcards`` of the WebUI, ``wildcards`` of every extension,
    Dynamic Prompts' ``wildcard_dir`` setting, then registered folders."""

    roots: list[str] = []
    try:
        from modules.paths_internal import extensions_builtin_dir, extensions_dir, script_path

        roots.append(os.path.join(script_path, "scripts", "wildcards"))
        for base in (extensions_dir, extensions_builtin_dir):
            if os.path.isdir(base):
                roots.extend(os.path.join(base, name, "wildcards") for name in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, name, "wildcards")))
    except Exception:
        log.debug("no WebUI paths", exc_info=True)
    custom = opt.option("wildcard_dir", None)
    if isinstance(custom, str) and custom.strip():
        roots.append(os.path.abspath(custom.strip()))
    roots.extend(registry.wildcard_dirs)
    return [root for root in dict.fromkeys(roots) if os.path.isdir(root)]


def _excluded_wildcard_folders() -> list[str]:
    return [name.strip() for name in str(opt.option("tac_wildcardExclusionList")).split(",") if name.strip()]


_index_cache = IndexCache()
_chants = ChantIndex()
_wildcards = WildcardCache(default_wildcard_dirs, _excluded_wildcard_folders)
_groups = TagGroups([])


def refresh() -> None:
    """Forget the loaded tag list, wildcards and chants (a settings refresh)."""

    _index_cache.clear()
    _wildcards.refresh()


# endregion

# region Settings as the client sees them


def _int_option(name: str) -> int:
    try:
        return int(float(opt.option(name)))
    except (TypeError, ValueError):
        return int(opt.DEFAULTS[name])


TRANSLATION_KEY = "translation_key"
"""the store secret holding the translation API key (never in the settings)"""


def translation_config(key: str = "") -> ProviderConfig:
    """The provider settings; ``key`` is the API key from the store."""

    provider = str(opt.option("prompt_tools_translation_provider") or "none")
    return ProviderConfig(provider, str(opt.option("prompt_tools_translation_url") or ""), key, str(opt.option("prompt_tools_translation_model") or ""))


def display_language() -> str:
    return str(opt.option("prompt_tools_translation_language") or "zh_CN")


def frequency_config() -> FrequencyConfig:
    return FrequencyConfig(bool(opt.option("tac_frequencySort")), str(opt.option("tac_frequencyFunction")), _int_option("tac_frequencyMinCount"), _int_option("tac_frequencyRecommendCap"))


def is_anima_model() -> bool:
    """Whether an Anima checkpoint is loaded or the Anima UI preset chosen
    (never loads a model)."""

    if str(opt.option("forge_preset", "") or "").lower() == "anima":
        return True
    sd_models = sys.modules.get("modules.sd_models")
    model = getattr(getattr(sd_models, "model_data", None), "sd_model", None) if sd_models else None
    return bool(getattr(model, "is_anima_engine", False))


def tag_source_name() -> str:
    """The family of the tag list for category colours: ``danbooru``,
    ``e621``, ``derpibooru`` or ``merged``."""

    name = str(opt.option("tac_tagFile") or "").lower()
    if "merged" in name:
        return "merged"
    for family in ("e621", "derpibooru"):
        if family in name:
            return family
    return "danbooru"


def client_config(translation: ProviderConfig | None = None) -> dict[str, Any]:
    """What the SPA needs to complete and insert (schema: ``PromptToolsConfig``);
    ``translation``: the provider with its key (:meth:`PromptTools.translation_config`)."""

    o = opt.option
    tags_on = registry.tags_enabled and bool(o("tac_active"))
    prefix = str(o("tac_animaArtistPrefix"))
    translation = translation or translation_config()
    toolbar = str(o("prompt_tools_toolbar"))
    try:
        comments = bool(o("enable_prompt_comments", True))
    except Exception:
        comments = True
    return {
        "tags": {
            "enabled": tags_on,
            "negative": bool(o("tac_activeIn_negativePrompts")),
            "source": tag_source_name(),
            "max_results": max(3, min(_int_option("tac_maxResults"), 50)),
            "loras": bool(o("tac_useLoras")),
            "embeddings": bool(o("tac_useEmbeddings")),
            "chants": _tag_path(o("tac_chantFile")) is not None,
            "wildcards": bool(o("tac_useWildcards")),
            "wildcard_mode": str(o("tac_wildcardCompletionMode")),
            "replace_underscores": bool(o("tac_replaceUnderscores")),
            "underscore_exclusions": [item.strip() for item in str(o("tac_undersocreReplacementExclusionList")).split(",") if item.strip()],
            "escape_parentheses": bool(o("tac_escapeParentheses")),
            "append_comma": bool(o("tac_appendComma")),
            "append_space": bool(o("tac_appendSpace")),
            "always_space_at_end": bool(o("tac_alwaysSpaceAtEnd")),
            "keywords": str(o("tac_modelKeywordCompletion")) != "Never",
            "keyword_location": str(o("tac_modelKeywordLocation")),
            "artist_prefix": prefix == "Always" or (prefix == "Auto" and is_anima_model()),
            "previews": bool(o("tac_showExtraNetworkPreviews")),
            "wiki_links": bool(o("tac_showWikiLinks")),
            "network_separator": str(o("extra_networks_add_text_separator", " ") or " "),
            "default_weight": float(o("extra_networks_default_multiplier", 1.0) or 1.0),
        },
        "tools": {
            "enabled": registry.tools_enabled,
            "history": registry.tools_enabled and bool(o("prompt_tools_history")),
            "groups": registry.tools_enabled,
            "translation": translation.provider if registry.tools_enabled and translation.enabled else None,
            "translation_provider": translation.provider if registry.tools_enabled and translation.provider in opt.TRANSLATION_PROVIDER_IDS else "none",
            "translation_key": bool(translation.key.strip()),
            "language": display_language(),
            "comments": comments,
            "toolbar": toolbar if toolbar in opt.TOOLBAR_SCOPES else opt.DEFAULTS["prompt_tools_toolbar"],
        },
    }


# endregion


def _tag_files() -> TagFiles:
    translation_choice = opt.option("tac_translation_translationFile")
    extra_translations: dict[str, str] = {}
    if translation_choice == opt.TAG_GROUPS_TRANSLATION:
        extra_translations = _groups.dictionary(display_language())
    return TagFiles(_tag_path(opt.option("tac_tagFile")), _tag_path(opt.option("tac_extra_extraFile")), _tag_path(translation_choice), extra_translations)


def tag_index() -> TagIndex:
    return _index_cache.get(_tag_files())


def _match_item(match: TagMatch, index: TagIndex) -> dict[str, Any]:
    entry = match.entry
    return {
        "type": "extra" if entry.extra else "tag",
        "text": entry.name,
        "category": entry.category,
        "count": entry.count or None,
        "alias": match.alias,
        "translation": index.translation(entry.name),
        "meta": entry.meta,
        "uses": match.uses or None,
    }


class PromptTools:
    """The prompt tools of one server (data under its ``root``)."""

    def __init__(self, root: str | Path, networks: Any = None) -> None:
        self.store = PromptToolsStore(Path(root) / "prompt-tools.db")
        self.networks = networks
        self._recorder: HistoryRecorder | None = None

    def close(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
        self.store.close()

    # region translation settings

    def translation_config(self) -> ProviderConfig:
        return translation_config(self.store.secret(TRANSLATION_KEY) or "")

    def client_config(self) -> dict[str, Any]:
        return client_config(self.translation_config())

    def set_translation_key(self, key: str) -> bool:
        """Stores (empty: forgets) the API key; True when one is set."""

        key = key.strip()
        self.store.set_secret(TRANSLATION_KEY, key or None)
        return bool(key)

    def migrate_translation_key(self) -> bool:
        """Moves a key a previous version kept in the settings file into the
        store (the settings file is saved without it); True when one moved."""

        legacy = opt.take_legacy_translation_key()
        if legacy is None:
            return False
        if not self.store.secret(TRANSLATION_KEY):
            self.store.set_secret(TRANSLATION_KEY, legacy)
        return True

    # endregion

    # region completion

    def _usage(self, names: Iterable[str], type: str, negative: bool) -> dict[str, int]:
        config = frequency_config()
        if not config.enabled:
            return {}
        uses = self.store.uses(names, type, negative, max_age_days=_int_option("tac_frequencyMaxAge"))
        return {tag_key(name): count for name, count in uses.items()}

    def _by_usage(self, items: list[dict[str, Any]], type: str, negative: bool, key: str = "text") -> list[dict[str, Any]]:
        """Network, wildcard and chant items: the match order, frequent ones first."""

        uses = self._usage([item[key] for item in items], type, negative)
        config = frequency_config()
        for item in items:
            count = uses.get(tag_key(item[key]), 0)
            item["uses"] = count or None
        if not uses:
            return items
        promoted = sorted((item for item in items if (item["uses"] or 0) >= max(config.min_count, 1)), key=lambda item: -(item["uses"] or 0))
        cap = config.recommend_cap if config.recommend_cap > 0 else len(promoted)
        keep = {id(item) for item in promoted[:cap]}
        return sorted(items, key=lambda item: -usage_bias(item["uses"] or 0, config) if id(item) in keep else 0.0)

    def complete(self, mode: str, query: str = "", *, negative: bool = False, limit: int | None = None, wildcard: str | None = None) -> list[dict[str, Any]]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if not registry.tags_enabled or not opt.option("tac_active"):
            return []
        limit = max(1, min(int(limit or _int_option("tac_maxResults")), 100))
        query = query[:200]
        if mode == "tag":
            return self._complete_tags(query, negative, limit)
        if mode == "wildcard":
            if not opt.option("tac_useWildcards"):
                return []
            items = [{"type": "wildcard", "text": item.name, "meta": "YAML" if item.path is None else None} for item in _wildcards.search(query, max(limit, 50))]
            return self._by_usage(items, "wildcard", negative)[: max(limit, 50)]
        if mode == "wildcard-value":
            found = _wildcards.find(wildcard or "")
            if found is None or not opt.option("tac_useWildcards"):
                return []
            values = wildcard_values(found)
            if opt.option("tac_sortWildcardResults"):
                values = sorted(values, key=str.lower)
            q = query.strip().lower()
            return [{"type": "wildcard-value", "text": value, "meta": found.name} for value in values if q in value.lower()][: max(limit, 100)]
        items: list[dict[str, Any]] = []
        if mode in ("lora", "network") and opt.option("tac_useLoras"):
            items.extend(self._complete_networks("lora", query, limit))
        if mode in ("embedding", "network") and opt.option("tac_useEmbeddings"):
            items.extend(self._complete_networks("embeddings", query, limit))
        if mode in ("chant", "network"):
            path = _tag_path(opt.option("tac_chantFile"))
            if path:
                items.extend({"type": "chant", "text": chant.content, "label": chant.name, "category": chant.color, "meta": chant.terms or None} for chant in _chants.search(path, query, limit))
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_type.setdefault(item["type"], []).append(item)
        ordered = [item for type, group in by_type.items() for item in self._by_usage(group, type, negative, "label" if type in ("lora", "chant") else "text")]
        return ordered[: limit * 3 if mode == "network" else limit]

    def _complete_tags(self, query: str, negative: bool, limit: int) -> list[dict[str, Any]]:
        index = tag_index()
        aliases = bool(opt.option("tac_alias_searchByAlias"))
        translations = bool(opt.option("tac_translation_searchByTranslation"))
        matches = index.matches(query, limit, aliases=aliases, translations=translations) + index.extra_matches(query, aliases=aliases)
        # an extra tag replaces the same tag of the main list
        extra_names = {tag_key(match.entry.name) for match in matches if match.entry.extra}
        matches = [match for match in matches if match.entry.extra or tag_key(match.entry.name) not in extra_names]
        uses = self._usage({match.entry.name for match in matches}, "tag", negative)
        ranked = rank(matches, uses, frequency_config(), extras_first=opt.option("tac_extra_addMode") != "Insert after", limit=limit)
        items = [_match_item(match, index) for match in ranked]
        if opt.option("tac_includeEmbeddingsInNormalResults") and opt.option("tac_useEmbeddings") and len(query.strip()) >= 2:
            items = self._complete_networks("embeddings", query.strip(), min(limit, 5)) + items
        return items

    def _complete_networks(self, kind: str, query: str, limit: int) -> list[dict[str, Any]]:
        if self.networks is None:
            return []
        weight_default = float(opt.option("extra_networks_default_multiplier", 1.0) or 1.0)
        previews = bool(opt.option("tac_showExtraNetworkPreviews"))
        items = []
        for hit in network_items(self.networks, kind, query, limit):
            record, card = hit.record, hit.card
            entry = record.entry
            user = record.user if isinstance(record.user, dict) else {}
            preview = card.get("preview") if previews and isinstance(card.get("preview"), str) else None
            description = card.get("description") or record.description
            if kind == "lora":
                sd_version = card.get("sd_version") or record.sd_version
                keywords = str(user.get("activation text") or "").strip() or None
                items.append(
                    {
                        "type": "lora",
                        "text": entry.alias or entry.name,
                        "label": entry.name,
                        "weight": preferred_weight(user, weight_default),
                        "keywords": keywords,
                        "meta": sd_version if isinstance(sd_version, str) and sd_version not in ("", "Unknown") else None,
                        "preview": preview,
                        "description": description if isinstance(description, str) else None,
                    }
                )
            else:
                text = card.get("prompt") if isinstance(card.get("prompt"), str) and card.get("prompt") else prompt_text(entry.prompt, user)
                items.append({"type": "embedding", "text": text or entry.name, "label": entry.name, "preview": preview, "description": description if isinstance(description, str) else None})
        return items

    def record_use(self, name: str, type: str, negative: bool) -> None:
        if type in ("tag", "extra"):
            type = "tag"
        self.store.record_use(name, type, negative)

    # endregion

    # region lookups and translation

    def lookup(self, tags: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Known tags of a prompt (category, posts, translation) for the chips view."""

        found: dict[str, dict[str, Any]] = {}
        index = tag_index() if registry.tags_enabled else None
        groups = _groups.dictionary(display_language()) if registry.tools_enabled else {}
        for tag in list(dict.fromkeys(tags))[:500]:
            key = tag_key(tag)
            entry = index.lookup(tag) if index is not None else None
            translation = (index.translation(tag) if index is not None else None) or groups.get(key)
            if entry is None and translation is None:
                continue
            found[tag] = {"category": entry.category if entry else None, "count": entry.count or None if entry else None, "translation": translation}
        return found

    def translate(self, texts: Sequence[str], target: str) -> list[str | None]:
        if not registry.tools_enabled:
            raise TranslationError("the prompt tools are disabled", 409)
        config = self.translation_config()
        language = display_language()
        dictionary: dict[str, str] = {}
        if registry.tags_enabled:
            dictionary.update(tag_index().translations)
        for key, value in _groups.dictionary(language if target == "en" else target).items():
            dictionary.setdefault(key, value)
        return translate(
            texts,
            target,
            config,
            dictionary,
            cached=lambda missing: self.store.cached_translations(config.cache_key, target, missing),
            store=lambda pairs: self.store.cache_translations(config.cache_key, target, pairs),
        )

    def groups(self, locale: str | None) -> list[dict[str, Any]]:
        return _groups.groups(locale) if registry.tools_enabled else []

    # endregion

    # region history recording

    def attach(self, services: Any) -> None:
        """Records the prompts of every queued job (once per services)."""

        if self._recorder is None:
            self._recorder = HistoryRecorder(self, services)
            self._recorder.start()

    def record_job(self, job: Any) -> None:
        if not registry.tools_enabled or not opt.option("prompt_tools_history"):
            return
        core = getattr(job, "core", None) or {}
        prompt, negative = core.get("prompt"), core.get("negative_prompt")
        if not isinstance(prompt, str) and not isinstance(negative, str):
            return
        self.store.record_history(prompt if isinstance(prompt, str) else "", negative if isinstance(negative, str) else "", getattr(job, "tab", "") or "", limit=_int_option("prompt_tools_history_limit"))

    # endregion


class HistoryRecorder:
    """A bus subscriber: ``job.queued`` → the job's prompts into the history
    (on its own thread; ``send_event`` must not block)."""

    def __init__(self, tools: PromptTools, services: Any) -> None:
        self.tools = tools
        self.services = services
        self._queue: queue.SimpleQueue[str | threading.Event | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._unsubscribe: Any = None

    def start(self) -> None:
        self._unsubscribe = self.services.bus.subscribe(self)
        self._thread = threading.Thread(target=self._run, name="remi-prompt-history", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._queue.put(None)

    def send_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "job.queued" and isinstance(event.get("job_id"), str):
            self._queue.put(event["job_id"])

    def send_frame(self, frame: Any) -> None:
        pass

    def wants_frames(self) -> bool:
        return False

    def flush(self, timeout: float = 5.0) -> bool:
        """Waits until the events received so far are recorded (tests)."""

        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            if isinstance(job_id, threading.Event):
                job_id.set()
                continue
            try:
                job = self.services.store.get(job_id)
                if job is not None:
                    self.tools.record_job(job)
            except Exception:
                log.exception("recording the prompt history")


_tools: dict[str, PromptTools] = {}
_tools_lock = threading.Lock()


def tools_for(services: Any) -> PromptTools:
    """The prompt tools of a server's services (one per data folder)."""

    key = str(Path(services.root).resolve())
    with _tools_lock:
        tools = _tools.get(key)
        if tools is None:
            tools = _tools[key] = PromptTools(services.root, getattr(services, "networks", None))
        return tools


def attach(services: Any) -> PromptTools:
    """Starts recording the prompt history for ``services`` and loads the
    tag list in the background, so the first completion is fast."""

    tools = tools_for(services)
    tools.attach(services)
    if registry.tags_enabled:
        threading.Thread(target=_warm_up, name="remi-tag-index", daemon=True).start()
    return tools


def forget(services: Any) -> None:
    """Closes the prompt tools of ``services`` (tests)."""

    key = str(Path(services.root).resolve())
    with _tools_lock:
        tools = _tools.pop(key, None)
    if tools is not None:
        tools.close()


def _warm_up() -> None:
    try:
        tag_index()
    except Exception:
        log.exception("loading the tag list")

