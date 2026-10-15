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
    """A script file, script or Extras operation that could not be loaded
    (it failed to import, or declares A1111's ``ui()`` instead of ``params()``)."""

    kind: Literal["script", "extras"]
    id: str
    title: str
    tab: str | None = None
    extension: str | None = None
    status: Literal["disabled"] = "disabled"
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


class Note(Model):
    """A translatable note: an English message with ``{{name}}`` placeholders and their values."""

    message: str
    values: dict[str, str | int] = {}


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
    starred: bool = False
    replayable: bool = Field(True, description="false: imported with arguments that do not map to the current parameters (never run or retried as is)")
    name: str | None = Field(None, description="a name the job was given (imported tasks)")
    source: str | None = Field(None, description='the importer that brought the job in ("agent-scheduler", "prompt-queue")')
    notes: list[Note] = Field([], description="what an importer could not map exactly")


class OpaqueArgs(Model):
    count: int
    preview: list[str] = Field([], description="short descriptions of the unmapped arguments")


class JobDetail(Job):
    core: dict[str, Any]
    scripts: dict[str, Any]
    layout_hash: str
    schema_version: int
    opaque: OpaqueArgs | None = None


class JobPatch(Model):
    priority: int | None = Field(None, ge=-1000, le=1000)
    forever: bool | None = None


MAX_REORDER = 500


# region Extensions and system (remi/server/routes/extensions.py, admin.py)


class AdminErrorBody(Model):
    """A refused or failed extensions/system operation."""

    detail: str = Field(description="the message in English")
    template: str | None = Field(None, description="the message's translatable text with {{name}} placeholders")
    params: dict[str, str] = Field(default={}, description="the placeholders' values")


class ExtensionInfo(Model):
    name: str
    enabled: bool
    builtin: bool
    remote: str | None = Field(None, description="the git remote (a browsable URL when it is one, credentials removed)")
    remote_url: str | None = Field(None, description="https URL of the repository, when known")
    branch: str | None = None
    version: str | None = Field(None, description="short commit hash")
    commit_hash: str | None = None
    commit_url: str | None = Field(None, description="link to the commit on its forge (GitHub, GitLab, Codeberg)")
    commit_date: str | None = None
    status: str | None = Field(None, description='after a check: "latest", "new commits", "behind HEAD", "unknown (remote error)", ...')
    can_update: bool = False


class ExtensionList(Model):
    items: list[ExtensionInfo]
    disable_all: Literal["none", "extra", "all"] | str
    access: bool = Field(description="whether this server allows installing, updating and toggling extensions")
    restartable: bool = Field(False, description="whether /server/restart can restart the process")
    disabled_by: str | None = Field(None, description='why extensions are not loaded: "all", "extra" (the option) or a command-line flag')
    index_url: str = Field("", description="the default extension index URL")


class ExtensionsPatch(Model):
    disabled: list[str] | None = Field(None, description="names of the extensions to disable (the others are enabled)")
    disable_all: Literal["none", "extra", "all"] | None = None


class ExtensionUpdateRequest(Model):
    names: list[str] = Field(min_length=1, max_length=500)


class ExtensionCheckRequest(Model):
    names: list[str] | None = Field(None, max_length=500, description="default: every enabled extension with a remote")


class ExtensionInstallRequest(Model):
    url: str = Field(min_length=1, max_length=2000)
    branch: str | None = Field(None, max_length=200)
    dirname: str | None = Field(None, max_length=100, description="folder name under extensions/ (default: from the URL)")


class IndexEntry(Model):
    name: str
    url: str
    description: str = ""
    tags: list[str] = []
    added: str | None = None
    created_at: str | None = None
    commit_time: str | None = None
    stars: int = 0
    order: int = Field(0, description="position in the index (its own order)")
    installed: bool = False


class ExtensionIndex(Model):
    """An extension index (``index.json`` of the A1111 extensions list)."""

    url: str
    fetched_at: str | None = None
    items: list[IndexEntry]
    tags: dict[str, str] = Field(description="tag → description")


class RepoState(Model):
    remote: str | None = None
    branch: str | None = None
    commit_hash: str | None = None
    commit_url: str | None = None
    commit_date: str | None = None


class ExtensionRepoState(RepoState):
    enabled: bool = True


class ConfigStateInfo(Model):
    id: str = Field(description="the backup's file name in config_states/")
    name: str
    created_at: str | None = None
    webui_commit: str | None = None
    extensions: int = Field(0, description="number of extensions recorded")


class ConfigStateExtension(Model):
    name: str
    builtin: bool = False
    saved: ExtensionRepoState | None = Field(None, description="null: not in the backup")
    current: ExtensionRepoState | None = Field(None, description="null: not installed now")


class ConfigStateWebui(Model):
    saved: RepoState
    current: RepoState


class ConfigStateDetail(ConfigStateInfo):
    webui: ConfigStateWebui
    items: list[ConfigStateExtension]


class ConfigStateSave(Model):
    name: str = Field("", max_length=100)


class ConfigStateRestore(Model):
    what: Literal["extensions"] = Field("extensions", description="only the extensions are restored; the WebUI checkout is never reset")


class RestoreResult(Model):
    reload: bool = Field(description="apply with /server/reload (extension commits are restored while reloading)")


class LicenseEntry(Model):
    id: str
    name: str
    url: str | None = None
    description: str = ""
    license_url: str | None = Field(None, description="raw text of the license (fetched by the browser)")


class LicenseGroup(Model):
    title: str
    items: list[LicenseEntry]


class Licenses(Model):
    groups: list[LicenseGroup]


class SysinfoCheckRequest(Model):
    text: str = Field(max_length=32 * 1024 * 1024)


class SysinfoCheck(Model):
    valid: bool


class LoadedWeights(Model):
    name: str = Field(description="the model's class (UNet, VAE, text encoder, ...)")
    memory_mb: int | None = Field(None, description="memory it holds on its device, in MiB")
    device: str | None = None


class LoadedModels(Model):
    checkpoint: str | None = Field(None, description="the checkpoint in memory, if any")
    items: list[LoadedWeights]


class HashRequest(Model):
    threads: int = Field(1, ge=1, le=16, description="checkpoints hashed at once")


class HashProgress(Model):
    running: bool
    done: int = 0
    total: int = 0
    failed: list[str] = []
    current: list[str] = Field(default=[], description="checkpoints being hashed")
    finished_at: str | None = None


class ModelReloadRequest(Model):
    force: bool = Field(False, description="load again even when the selected model is already loaded")


class ServerStatus(Model):
    restartable: bool = Field(description="whether /server/restart can restart the process")
    extension_access: bool
    settings_frozen: bool = Field(description="--freeze-settings: PATCH /options refuses every change")
    model_loaded: str | None = Field(None, description="the checkpoint in memory, if any")
    version: str


# endregion


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


# endregion

# region Options, infotext, i18n, server


class SettingsSchema(Model):
    """``GET/PATCH /options`` (schema-format.md §6)."""

    version: int
    categories: list[dict[str, Any]]
    quicksettings: list[str]
    values: dict[str, Any]
    defaults: dict[str, Any] = Field(default={}, description="each stored setting's built-in default (reset to default)")


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
