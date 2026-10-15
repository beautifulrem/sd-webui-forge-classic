"""Pydantic models of ``/api/v2`` (request bodies and responses), so
``/openapi.json`` describes the whole API for the generated TS client.

Schema-shaped payloads (layout trees, settings items, action patches) are
documented in schema-format.md and typed loosely here (``dict[str, Any]``).
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

FormTab = Literal["txt2img", "img2img", "extras"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
T = TypeVar("T")


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Page(Model, Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    total: int | None = None


class ErrorItem(Model):
    owner: str | None = None
    param: str | None = None
    message: str


class ValidationErrorBody(Model):
    """A rejected request (schema-format.md §3)."""

    errors: list[ErrorItem]


class Detail(Model):
    detail: str


class Ok(Model):
    ok: bool = True
    message: str | None = None


# region Bootstrap and models


class PageInfo(Model):
    id: str
    title: str
    icon: str | None = None
    widget: str | None = None


class LocaleInfo(Model):
    id: str
    name: str


class ModelsSummary(Model):
    checkpoint: str | None = None
    vae: str | None = None
    text_encoders: list[str] = []
    modules: list[str] = []
    preset: str | None = None
    unet_dtype: str | None = None


class NotPorted(Model):
    """A script, Extras operation or extension tab not on ``params()`` yet."""

    kind: Literal["script", "extras", "page"]
    id: str
    title: str
    tab: str | None = None
    extension: str | None = None
    status: Literal["legacy", "disabled"] = "legacy"
    reason: str | None = None


class AuthStatus(Model):
    required: bool
    user: str | None = None


class Bootstrap(Model):
    version: str
    features: list[str]
    tabs: list[str]
    pages: list[PageInfo]
    locales: list[LocaleInfo]
    locale: str | None = None
    options: dict[str, Any]
    models: ModelsSummary
    not_ported: list[NotPorted] = []
    auth: AuthStatus
    queue: QueueStatus
    user_assets: list[str] = Field(default=[], description="files served by /user-assets/{name} (user.css, notification.mp3)")


class ModelRef(Model):
    name: str
    title: str | None = None
    hash: str | None = None
    arch: str | None = None
    size: int | None = None


class SamplerInfo(Model):
    name: str
    aliases: list[str] = []


class SchedulerInfo(Model):
    name: str
    label: str | None = None


class UpscalerInfo(Model):
    name: str


class PromptStyle(Model):
    name: str
    prompt: str = ""
    negative_prompt: str = ""


class Models(Model):
    checkpoints: list[ModelRef]
    vae: list[ModelRef]
    text_encoders: list[ModelRef]
    modules: list[ModelRef]
    samplers: list[SamplerInfo]
    schedulers: list[SchedulerInfo]
    upscalers: list[UpscalerInfo]
    styles: list[PromptStyle]
    unet_dtypes: list[str]
    presets: list[str]


class StyleBody(Model):
    prompt: str = ""
    negative_prompt: str = ""


class StylesApplyRequest(Model):
    prompt: str = ""
    negative_prompt: str = ""
    styles: list[str] = []


class StylesApplied(Model):
    prompt: str
    negative_prompt: str


class TokenCountRequest(Model):
    text: str = ""
    steps: int = Field(20, ge=1, le=10000)
    styles: list[str] = []
    negative: bool = False


class TokenCount(Model):
    tokens: int | None = None
    max_length: int | None = None


# endregion

# region Schema, actions, presets


class TabSchema(Model):
    """``GET /schema/{tab}`` (schema-format.md §4)."""

    model_config = ConfigDict(extra="allow")
    version: int
    tab: FormTab
    core: dict[str, Any]
    scripts: list[dict[str, Any]]
    conflict_params: list[str] | None = None


class PageSchema(Model):
    """``GET /pages/{id}`` (schema-format.md §8)."""

    model_config = ConfigDict(extra="allow")
    version: int
    id: str
    title: str
    icon: str | None = None
    owner: str
    widget: str | None = None
    layout: dict[str, Any] | None = None
    defaults: dict[str, Any] = {}


class ActionRequest(Model):
    """The calling owner's ``values``, the generation form (``core``) and
    the other owners' values (``scripts``); extra keys are the action's own
    arguments (``selected``, ``label``, ``preset``, ...)."""

    model_config = ConfigDict(extra="allow")
    tab: str | None = None
    values: dict[str, Any] = {}
    core: dict[str, Any] | None = None
    scripts: dict[str, dict[str, Any]] | None = None


ActionPatch = dict[str, Any]


class RulesRequest(Model):
    tab: Literal["txt2img", "img2img"] = "txt2img"
    core: dict[str, Any] = {}
    scripts: dict[str, dict[str, Any]] = {}
    selected: str | None = None


class PresetList(Model):
    anima: list[str]
    forge: list[str]
    current: str | None = None


# endregion

# region Jobs


class JobRequest(Model):
    tab: FormTab
    core: dict[str, Any] = {}
    scripts: dict[str, Any] = {}
    priority: int = Field(0, ge=-1000, le=1000)
    repeat: int = Field(1, ge=1, le=100)
    forever: bool = False


class JobCreated(Model):
    job_id: str
    job_ids: list[str]
    seq: int


class JobOutput(Model):
    id: str
    kind: Literal["image", "video", "file", "text", "json"] = "image"
    width: int | None = None
    height: int | None = None
    infotext: str | None = None


class JobProgress(Model):
    step: int
    steps: int
    eta: float | None = None
    job_no: int = 0
    job_count: int = 0
    progress: float | None = None
    text: str | None = None


class Job(Model):
    id: str
    seq: int
    tab: FormTab
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: JobProgress | None = None
    outputs: list[JobOutput] = []
    error: str | None = None
    errors: list[ErrorItem] | None = None
    summary: str | None = None
    priority: int = 0
    forever: bool = False
    retry_of: str | None = None
    infotext: str | None = None


class JobDetail(Job):
    core: dict[str, Any]
    scripts: dict[str, Any]
    layout_hash: str
    schema_version: int


class JobPatch(Model):
    priority: int | None = Field(None, ge=-1000, le=1000)
    forever: bool | None = None


MAX_REORDER = 500


class ExtensionInfo(Model):
    name: str
    enabled: bool
    builtin: bool
    remote: str | None = None
    branch: str | None = None
    version: str | None = None
    commit_date: str | None = None
    status: str | None = Field(None, description='after a check: "latest", "new commits", "behind HEAD", ...')
    can_update: bool = False


class ExtensionList(Model):
    items: list[ExtensionInfo]
    disable_all: Literal["none", "extra", "all"] | str
    access: bool = Field(description="whether this server allows installing, updating and toggling extensions")


class ExtensionsPatch(Model):
    disabled: list[str] | None = Field(None, description="names of the extensions to disable (the others are enabled)")
    disable_all: Literal["none", "extra", "all"] | None = None


class ExtensionUpdateRequest(Model):
    names: list[str] = Field(min_length=1, max_length=500)


class ExtensionInstallRequest(Model):
    url: str = Field(min_length=1, max_length=2000)
    branch: str | None = None
    dirname: str | None = Field(None, description="folder name under extensions/ (default: from the URL)")


class SaveRequest(Model):
    ids: list[str] = Field(min_length=1, max_length=500, description="output file ids")
    zip: bool = False


class SaveResult(Model):
    saved: list[str] = Field(description="saved file names, relative to the Save folder")
    zip: str | None = Field(None, description="file id of the zip archive (download with /files/{id}?download=true)")


class OpenFolderRequest(Model):
    folder: Literal["txt2img", "img2img", "extras", "grids", "save"]


class ReorderRequest(Model):
    order: list[str] = Field(max_length=MAX_REORDER, description="queued job ids in their new order; jobs keep their priority, so the order applies within each priority")


class QueueStatus(Model):
    size: int
    paused: bool
    running: str | None = None


class InterruptRequest(Model):
    clear_queue: bool = False


# endregion

# region Files, outputs, networks


class Upload(Model):
    id: str
    width: int | None = None
    height: int | None = None
    name: str
    mime: str
    size: int


class OutputItem(Model):
    id: str
    kind: Literal["image", "video", "file"] = "image"
    name: str
    width: int | None = None
    height: int | None = None
    infotext: str | None = None
    created_at: str
    job_id: str | None = None
    tab: str | None = None


class ScanResult(Model):
    added: int


class NetworkItem(Model):
    name: str
    title: str
    kind: str
    path: str
    preview: str | None = None
    description: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    search_terms: list[str] = []
    hash: str | None = None
    alias: str | None = None
    sd_version: str | None = None
    mtime: float | None = None
    size: int | None = None


class NetworkMeta(NetworkItem):
    metadata: dict[str, Any] = {}
    user_metadata: dict[str, Any] = {}


class UserMetadataPatch(Model):
    """Keys to set (``null`` removes one); ``replace`` drops the others."""

    user_metadata: dict[str, Any]
    replace: bool = False


class PreviewRequest(Model):
    file_id: str


# endregion

# region Options, infotext, i18n, server


class SettingsSchema(Model):
    """``GET/PATCH /options`` (schema-format.md §6)."""

    version: int
    categories: list[dict[str, Any]]
    quicksettings: list[str]
    values: dict[str, Any]


class InfotextRequest(Model):
    text: str = ""
    tab: FormTab | None = None


class InfotextParsed(Model):
    tab: FormTab
    core: dict[str, Any]
    scripts: dict[str, dict[str, Any]]
    params: dict[str, Any] = {}
    problems: list[ErrorItem] = []


class ImageInfoRequest(Model):
    file_id: str
    tab: FormTab | None = None


class ImageInfo(Model):
    """What PNG Info shows for an image, plus its parsed parameters."""

    infotext: str | None = None
    items: dict[str, str] = {}
    parsed: InfotextParsed | None = None


class LoginRequest(Model):
    username: str
    password: str


# endregion

Bootstrap.model_rebuild()
