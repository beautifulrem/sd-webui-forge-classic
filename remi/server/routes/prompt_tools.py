"""Prompt tools: ``/prompt-tools/*`` — tag completion, usage ranking, tag
lookups for the chips view, translation, quick tag groups, prompt history,
favorites and collections (remi/prompt_tools)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import Field

from remi import prompt_tools as pt
from remi.server import models as m
from remi.server.deps import ServicesDep

router = APIRouter(prefix="/prompt-tools")

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}, 422: {"model": m.Detail}}

CompletionMode = Literal["tag", "lora", "embedding", "chant", "network", "wildcard", "wildcard-value"]
CompletionType = Literal["tag", "extra", "lora", "embedding", "chant", "wildcard", "wildcard-value"]
UsageType = Literal["tag", "extra", "lora", "embedding", "chant", "wildcard", "wildcard-value"]
FavoriteKind = Literal["prompt", "negative", "tags"]
HistoryField = Literal["prompt", "negative"]


def tools(services: ServicesDep) -> pt.PromptTools:
    return pt.tools_for(services)


ToolsDep = Annotated[pt.PromptTools, Depends(tools)]


def _store_call(call: Any) -> Any:
    try:
        return call()
    except pt.StoreError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error


# region Models


class PromptToolsTagSettings(m.Model):
    enabled: bool = Field(description="completion is on (the extension, the setting and the checkpoint list allow it)")
    negative: bool
    txt2img: bool = Field(True, description="complete in the txt2img form's prompt fields")
    img2img: bool = Field(True, description="complete in the img2img form's prompt fields")
    third_party: bool = Field(True, description="complete in the prompt fields of extensions (scripts and pages), not only the main prompts")
    delay: int = Field(40, description="milliseconds to wait after a keystroke before asking the server")
    only_alias: bool = Field(False, description="label a tag found by an alias with the alias only")
    source: str = Field(description="tag list family for category colours: danbooru, e621, derpibooru or merged")
    max_results: int
    loras: bool
    embeddings: bool
    chants: bool
    wildcards: bool
    wildcard_mode: str
    replace_underscores: bool
    underscore_exclusions: list[str]
    escape_parentheses: bool
    append_comma: bool
    append_space: bool
    always_space_at_end: bool
    keywords: bool
    keyword_location: str
    artist_prefix: bool
    previews: bool
    wiki_links: bool
    network_separator: str
    default_weight: float


ToolbarScope = Literal["All prompt fields", "Main prompts only"]


class PromptToolsToolSettings(m.Model):
    enabled: bool
    history: bool
    groups: bool
    translation: str | None = Field(None, description="the usable provider; null: translation is off")
    translation_provider: str = Field("none", description="the chosen provider, usable or not (a network provider may still need its API key)")
    translation_key: bool = Field(False, description="an API key is stored (it is never sent back)")
    generate: bool = Field(False, description="prompts can be written from a description (the OpenAI-compatible provider is set up)")
    language: str
    comments: bool = Field(description="prompt comments are stripped (disabled chips are written as /* … */)")
    toolbar: ToolbarScope = Field("All prompt fields", description="which prompt fields get the tools bar")


class PromptToolsConfig(m.Model):
    tags: PromptToolsTagSettings
    tools: PromptToolsToolSettings


class PromptCompletionItem(m.Model):
    type: CompletionType
    text: str = Field(description="what is inserted: tag, LoRA alias, embedding, chant content, wildcard name or value")
    label: str | None = None
    category: int | None = None
    count: int | None = None
    alias: str | None = Field(None, description="the alias the query matched")
    translation: str | None = None
    meta: str | None = None
    weight: float | None = None
    keywords: str | None = Field(None, description="a LoRA's activation text")
    preview: str | None = Field(None, description="file id of a preview image")
    description: str | None = None
    uses: int | None = None


class PromptCompletionList(m.Model):
    items: list[PromptCompletionItem]


class PromptUsageRecord(m.Model):
    name: str = Field(min_length=1, max_length=512)
    type: UsageType = "tag"
    negative: bool = False


class PromptTagLookupRequest(m.Model):
    tags: list[str] = Field(max_length=500)


class PromptTagInfo(m.Model):
    category: int | None = None
    count: int | None = None
    translation: str | None = None


class PromptTagLookup(m.Model):
    tags: dict[str, PromptTagInfo]


class PromptTranslateRequest(m.Model):
    texts: list[str] = Field(min_length=1, max_length=200)
    target: str = Field("en", pattern=r"^[A-Za-z]{2,3}(?:_[A-Za-z]{2,4})?$", description="`en`, or a display language such as `zh_CN`")


class PromptTranslation(m.Model):
    provider: str
    translations: list[str | None]


class PromptGenerateRequest(m.Model):
    description: str = Field(min_length=1, max_length=4000, description="what the image should show, in any language")
    prompt: str = Field("", max_length=4000, description="the current prompt to build on (empty: write a new one)")
    negative: bool = Field(False, description="write a negative prompt")


class PromptGenerated(m.Model):
    prompt: str


class PromptTranslationKey(m.Model):
    key: str = Field(max_length=4096, description="the provider's API key; empty: forget it")


class PromptTranslationKeyState(m.Model):
    configured: bool


class PromptGroupTag(m.Model):
    tag: str
    label: str | None = None


class PromptTagGroup(m.Model):
    name: str
    color: str | None = None
    tags: list[PromptGroupTag]


class PromptTagCategory(m.Model):
    name: str
    groups: list[PromptTagGroup]


class PromptTagGroups(m.Model):
    categories: list[PromptTagCategory]


class PromptHistoryItem(m.Model):
    id: str
    tab: str
    prompt: str
    negative_prompt: str
    name: str
    created: float
    last_used: float
    uses: int


class PromptHistoryCreate(m.Model):
    prompt: str = ""
    negative_prompt: str = ""
    tab: str = Field("", max_length=32)


class PromptHistoryPatch(m.Model):
    name: str = Field(max_length=200)
    field: HistoryField | None = Field(None, description="name every entry with this entry's prompt / negative prompt (a grouped list)")


class PromptHistoryCleared(m.Model):
    removed: int


class PromptHistoryRemoved(m.Model):
    ok: bool = True
    removed: int


class PromptCollection(m.Model):
    id: str | None = Field(description="null: the favorites outside any collection")
    name: str
    created: float
    count: int


class PromptCollectionCreate(m.Model):
    name: str = Field(min_length=1, max_length=200)


class PromptFavorite(m.Model):
    id: str
    collection_id: str | None
    kind: FavoriteKind
    text: str
    name: str
    created: float
    position: float


class PromptFavoriteCreate(m.Model):
    text: str = Field(min_length=1)
    kind: FavoriteKind = "prompt"
    name: str = Field("", max_length=200)
    collection_id: str | None = None


class PromptFavoritePatch(m.Model):
    name: str | None = Field(None, max_length=200)
    text: str | None = None
    collection_id: str | None = Field(None, description="move to this collection (null: none); only when given")
    before: str | None = Field(None, description="move in front of this favorite (null with `collection_id` given: to the end)")


class PromptFavoritesImported(m.Model):
    added: int


class PromptFavoritesExport(m.Model):
    version: int = 1
    favorites: list[dict[str, Any]]


# endregion

# region Completion


@router.get("/config", response_model=PromptToolsConfig, summary="What is enabled and how completions are inserted")
def config(tools: ToolsDep) -> dict[str, Any]:
    return tools.client_config()


@router.get("/complete", response_model=PromptCompletionList, responses=ERRORS, summary="Completions for the word being typed")
def complete(
    tools: ToolsDep,
    mode: CompletionMode = "tag",
    q: str = Query("", max_length=200),
    negative: bool = False,
    limit: int | None = Query(None, ge=1, le=100),
    wildcard: str | None = Query(None, max_length=300, description="mode wildcard-value: the wildcard's name"),
) -> dict[str, Any]:
    return {"items": tools.complete(mode, q, negative=negative, limit=limit, wildcard=wildcard)}


@router.post("/usage", response_model=m.Ok, responses=ERRORS, summary="Count an accepted completion (frequency ranking)")
def record_usage(body: PromptUsageRecord, tools: ToolsDep) -> dict[str, Any]:
    _store_call(lambda: tools.record_use(body.name, body.type, body.negative))
    return {"ok": True}


@router.delete("/usage", response_model=m.Ok, summary="Forget the usage of a completion")
def reset_usage(tools: ToolsDep, name: str = Query(min_length=1, max_length=512), type: UsageType = "tag") -> dict[str, Any]:
    return {"ok": tools.store.reset_use(name, "tag" if type == "extra" else type)}


@router.post("/lookup", response_model=PromptTagLookup, summary="Category, post count and translation of known tags")
def lookup(body: PromptTagLookupRequest, tools: ToolsDep) -> dict[str, Any]:
    return {"tags": tools.lookup(body.tags)}


@router.post("/refresh", response_model=m.Ok, summary="Reload the tag lists, wildcards and chants")
def refresh() -> dict[str, Any]:
    pt.refresh()
    return {"ok": True}


# endregion

# region Translation and tag groups


@router.post("/translate", response_model=PromptTranslation, responses={**ERRORS, 413: {"model": m.Detail}, 502: {"model": m.Detail}}, summary="Translate prompt pieces with the configured provider")
def translate(body: PromptTranslateRequest, tools: ToolsDep) -> dict[str, Any]:
    try:
        translations = tools.translate(body.texts, body.target)
    except pt.TranslationError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    return {"provider": tools.translation_config().provider, "translations": translations}


@router.post("/generate", response_model=PromptGenerated, responses={**ERRORS, 413: {"model": m.Detail}, 502: {"model": m.Detail}}, summary="Write a prompt from a description with the OpenAI-compatible provider")
def generate(body: PromptGenerateRequest, tools: ToolsDep) -> dict[str, Any]:
    try:
        return {"prompt": tools.generate(body.description, current=body.prompt, negative=body.negative)}
    except pt.TranslationError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error


@router.put("/translation-key", response_model=PromptTranslationKeyState, responses=ERRORS, summary="Store (or, empty, forget) the translation API key; it is never sent back")
def set_translation_key(body: PromptTranslationKey, tools: ToolsDep) -> dict[str, Any]:
    return {"configured": _store_call(lambda: tools.set_translation_key(body.key))}


@router.get("/groups", response_model=PromptTagGroups, summary="Quick tag groups in a language")
def groups(tools: ToolsDep, locale: str | None = Query(None, max_length=16)) -> dict[str, Any]:
    return {"categories": tools.groups(locale)}


# endregion

# region History


@router.get("/history", response_model=m.Page[PromptHistoryItem], summary="Prompt history, newest first (every term must match)")
def history(
    tools: ToolsDep,
    q: str | None = Query(None, max_length=200),
    field: HistoryField | None = Query(None, description="one item per distinct prompt / negative prompt"),
    tab: str | None = Query(None, max_length=32),
    cursor: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    items, next_cursor, total = tools.store.history(query=q, field=field, tab=tab, cursor=cursor, limit=limit)
    return {"items": items, "next_cursor": next_cursor, "total": total}


@router.post("/history", response_model=PromptHistoryItem, status_code=201, responses=ERRORS, summary="Add prompts to the history")
def add_history(body: PromptHistoryCreate, tools: ToolsDep) -> dict[str, Any]:
    item = _store_call(lambda: tools.store.record_history(body.prompt, body.negative_prompt, body.tab, limit=int(pt.opt.option("prompt_tools_history_limit"))))
    if item is None:
        raise HTTPException(status_code=422, detail="both prompts are empty")
    return item


@router.patch("/history/{entry_id}", response_model=PromptHistoryItem, responses=ERRORS, summary="Name a history entry, or its group (named entries are kept)")
def rename_history(entry_id: str, body: PromptHistoryPatch, tools: ToolsDep) -> dict[str, Any]:
    return _store_call(lambda: tools.store.rename_history(entry_id, body.name, field=body.field))


@router.delete("/history/{entry_id}", response_model=PromptHistoryRemoved, responses=ERRORS, summary="Remove a history entry, or its group")
def delete_history(
    entry_id: str,
    tools: ToolsDep,
    field: HistoryField | None = Query(None, description="remove every entry with this entry's prompt / negative prompt (a grouped list)"),
) -> dict[str, Any]:
    removed = tools.store.delete_history(entry_id, field=field)
    if not removed:
        raise HTTPException(status_code=404, detail="no such history entry")
    return {"ok": True, "removed": removed}


@router.delete("/history", response_model=PromptHistoryCleared, summary="Clear the history")
def clear_history(tools: ToolsDep, keep_named: bool = True) -> dict[str, Any]:
    return {"removed": tools.store.clear_history(keep_named=keep_named)}


# endregion

# region Favorites


@router.get("/collections", response_model=list[PromptCollection], summary="Favorite collections (the first is the unsorted favorites)")
def collections(tools: ToolsDep) -> list[dict[str, Any]]:
    return tools.store.collections()


@router.post("/collections", response_model=PromptCollection, status_code=201, responses=ERRORS, summary="Create a collection")
def create_collection(body: PromptCollectionCreate, tools: ToolsDep) -> dict[str, Any]:
    return _store_call(lambda: tools.store.create_collection(body.name))


@router.patch("/collections/{collection_id}", response_model=m.Ok, responses=ERRORS, summary="Rename a collection")
def rename_collection(collection_id: str, body: PromptCollectionCreate, tools: ToolsDep) -> dict[str, Any]:
    _store_call(lambda: tools.store.rename_collection(collection_id, body.name))
    return {"ok": True}


@router.delete("/collections/{collection_id}", response_model=m.Ok, responses=ERRORS, summary="Delete a collection and its favorites")
def delete_collection(collection_id: str, tools: ToolsDep) -> dict[str, Any]:
    if not tools.store.delete_collection(collection_id):
        raise HTTPException(status_code=404, detail="no such collection")
    return {"ok": True}


@router.get("/favorites", response_model=m.Page[PromptFavorite], summary="A page of favorites in their order (`collection` empty: unsorted; absent: all)")
def favorites(
    tools: ToolsDep,
    collection: str | None = Query(None, max_length=64),
    kind: list[FavoriteKind] | None = Query(None, description="any of these kinds (repeat the parameter); absent: every kind"),
    q: str | None = Query(None, max_length=200),
    cursor: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    items, next_cursor, total = tools.store.favorites_page(collection_id=collection or None, all_collections=collection is None, kinds=kind, query=q, cursor=cursor, limit=limit)
    return {"items": items, "next_cursor": next_cursor, "total": total}


@router.post("/favorites", response_model=PromptFavorite, status_code=201, responses=ERRORS, summary="Save a prompt or tags as a favorite")
def add_favorite(body: PromptFavoriteCreate, tools: ToolsDep) -> dict[str, Any]:
    return _store_call(lambda: tools.store.add_favorite(body.text, body.kind, name=body.name, collection_id=body.collection_id))


@router.patch("/favorites/{favorite_id}", response_model=PromptFavorite, responses=ERRORS, summary="Rename, edit, move or reorder a favorite")
def update_favorite(favorite_id: str, body: PromptFavoritePatch, tools: ToolsDep) -> dict[str, Any]:
    changes = {name: getattr(body, name) for name in body.model_fields_set}
    if changes.get("name") is None:
        changes.pop("name", None)
    if changes.get("text") is None:
        changes.pop("text", None)
    return _store_call(lambda: tools.store.update_favorite(favorite_id, changes))


@router.delete("/favorites/{favorite_id}", response_model=m.Ok, responses=ERRORS, summary="Remove a favorite")
def delete_favorite(favorite_id: str, tools: ToolsDep) -> dict[str, Any]:
    if not tools.store.delete_favorite(favorite_id):
        raise HTTPException(status_code=404, detail="no such favorite")
    return {"ok": True}


@router.get("/favorites-export", response_model=PromptFavoritesExport, summary="Every favorite with its collection name (JSON download)")
def export_favorites(tools: ToolsDep, response: Response) -> dict[str, Any]:
    response.headers["Content-Disposition"] = 'attachment; filename="prompt-favorites.json"'
    return tools.store.export_favorites()


@router.post("/favorites-import", response_model=PromptFavoritesImported, responses={**ERRORS, 413: {"model": m.Detail}}, summary="Add the favorites of an export (duplicates are skipped)")
def import_favorites(body: PromptFavoritesExport, tools: ToolsDep) -> dict[str, Any]:
    return {"added": _store_call(lambda: tools.store.import_favorites(body.model_dump()))}


# endregion
