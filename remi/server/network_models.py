"""Pydantic models of ``/api/v2/networks`` (see ``remi.networks``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from remi.server.models import Model


class NetworkKind(Model):
    id: str
    title: str
    """English title (the client translates it)"""
    action: Literal["prompt", "checkpoint", "module", "none"] = "prompt"
    negative: bool = False
    """cards may go into the negative prompt"""
    fields: list[str] = []
    """user-metadata keys the editor offers"""
    builtin: bool = False


class NetworkItem(Model):
    name: str
    """unique within the kind; addresses the item in the API"""
    title: str
    kind: str
    path: str
    """relative to its model folder"""
    folder: str = ""
    preview: str | None = None
    """file id (``/files``; images also ``/thumbs``)"""
    preview_type: Literal["image", "video"] | None = None
    """``video``: a short clip (``.preview.mp4`` / ``.webm``), served by ``/files`` only"""
    placeholder: str | None = None
    """a tiny WebP data URI to show blurred while the preview loads"""
    description: str | None = None
    prompt: str | None = None
    """text a card inserts (LoRA: the preferred weight and activation text applied)"""
    negative_prompt: str | None = None
    activation_text: str | None = None
    preferred_weight: float | None = None
    negative_text: str | None = None
    tags: list[str] = []
    search_terms: list[str] = []
    hash: str | None = None
    alias: str | None = None
    sd_version: str | None = None
    """the user's ``sd version`` (``Unknown`` when unset)"""
    arch: str | None = None
    """effective architecture: the user's choice, else read from the file"""
    subtype: str | None = None
    """``LoRA`` / ``LyCORIS`` / ``DoRA`` for kohya networks"""
    value: str | None = None
    """what applying the card selects (checkpoint title, module file name)"""
    mtime: float | None = None
    created: float | None = None
    """when the file was added (birth time where the system keeps one)"""
    size: int | None = None
    editable: bool = True
    """a sidecar, preview and folder exist (False for extension items without a model file)"""
    hidden: bool = False
    """in a "."-folder (listed as ``extra_networks_hidden_models`` says)"""


class DatasetTag(Model):
    tag: str
    count: int


class NetworkMeta(NetworkItem):
    metadata: dict[str, Any] = {}
    """the file's header (safetensors ``__metadata__``), embedded images left out"""
    user_metadata: dict[str, Any] = {}
    training: dict[str, Any] = {}
    """output_name, base_model, architecture, module, network_type, dim, alpha,
    clip_skip, epochs, steps, trained_at, resolutions, dataset_size, …"""
    dataset_tags: list[DatasetTag] = []
    """training captions' tags, most frequent first (at most 400)"""


class NetworkFolder(Model):
    path: str
    name: str
    count: int
    children: list[NetworkFolder] = []


class ArchCount(Model):
    id: str
    """an architecture name or ``unknown``"""
    count: int


class NetworkFacets(Model):
    kind: str
    total: int
    """items matching every filter"""
    folder_total: int = 0
    """items matching every filter but the folder (the tree's "All" row)"""
    count: int
    """items of the kind"""
    folders: list[NetworkFolder]
    archs: list[ArchCount]
    current_arch: str | None = None


class UserMetadataPatch(Model):
    """Keys to set (``null`` removes one); ``replace`` drops the others.
    Known keys are checked: ``description``, ``activation text``,
    ``negative text``, ``notes`` (text), ``preferred weight`` (number),
    ``sd version`` (``Unknown`` or an architecture), ``tags`` (list)."""

    user_metadata: dict[str, Any]
    replace: bool = False


class PreviewRequest(Model):
    file_id: str = Field(description="an uploaded or generated image")
