"""Reading model-file metadata for the network browser (no WebUI imports).

* :func:`build_tags` — the training dataset tags of a kohya-trained network
  (``ss_tag_frequency``), most frequent first (from sd_forge_lora's editor).
* :func:`training_info` — the training facts the metadata dialog shows.
* :func:`detect_arch` — the ``PresetArch`` name a network or checkpoint was
  made for, from ``modelspec.*`` / ``ss_*`` keys (None when unknown).
* :func:`normalize_user_metadata` — validates an edit of the JSON sidecar.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
import math
import re
from collections.abc import Mapping
from typing import Any

ARCHS = ("sd", "xl", "flux", "klein", "qwen", "lumina", "zit", "wan", "anima", "ernie", "pid", "krea", "qwen21")
"""``modules_forge.presets.PresetArch`` names (kept in sync by a test)."""
SD_VERSIONS = ("Unknown", *ARCHS)
"""Values of the user-metadata ``sd version`` key (sd_forge_lora's ``network.SD_VERSION``)."""

HIDDEN_METADATA_KEYS = ("ssmd_cover_images", "modelspec.thumbnail")
"""Embedded images: far too large to show as text."""

TEXT_FIELDS = ("description", "activation text", "negative text", "notes")
USER_TEXT_LIMIT = 16 * 1024
PREFERRED_WEIGHT_LIMIT = 100
"""``preferred weight`` must lie in ±this (the editor uses the same bound)."""


def parse_json(value: Any) -> Any:
    """``value`` decoded when it is a JSON object/array string (safetensors
    headers keep nested values as strings)."""

    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _is_non_comma_tagset(tags: Mapping[str, int]) -> bool:
    return sum(len(tag) for tag in tags) / len(tags) >= 16


_re_word = re.compile(r"[-_\w']+")


def build_tags(metadata: Mapping[str, Any]) -> list[tuple[str, int]]:
    """``(tag, count)`` of the training captions, most frequent first.
    Captions that are sentences rather than tag lists are split into words
    of three letters or more."""

    tags: dict[str, int] = {}
    frequency = parse_json(metadata.get("ss_tag_frequency"))
    if isinstance(frequency, Mapping):
        for folder_tags in frequency.values():
            if not isinstance(folder_tags, Mapping):
                continue
            for tag, count in folder_tags.items():
                tag = str(tag).strip()
                if not tag:
                    continue
                try:
                    tags[tag] = tags.get(tag, 0) + int(count)
                except (TypeError, ValueError):
                    continue

    if tags and _is_non_comma_tagset(tags):
        words: dict[str, int] = {}
        for text, count in tags.items():
            for word in _re_word.findall(text):
                if len(word) >= 3:
                    words[word] = words.get(word, 0) + count
        tags = words

    return sorted(tags.items(), key=lambda item: (-item[1], item[0]))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != "None" else None


def network_type(metadata: Mapping[str, Any]) -> str | None:
    """``LyCORIS`` (incl. its LoCon/LoHa/LoKr algorithms), ``DoRA`` or
    ``LoRA`` for kohya-style networks; None without training metadata."""

    module = (_text(metadata.get("ss_network_module")) or "").lower()
    args = parse_json(metadata.get("ss_network_args"))
    algo = str(args.get("algo", "")).lower() if isinstance(args, Mapping) else ""
    if "lycoris" in module or algo:
        return "LyCORIS"
    if isinstance(args, Mapping) and str(args.get("dora_wd", "")).lower() in ("true", "1"):
        return "DoRA"
    if module:
        return "LoRA"
    return None


def training_info(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The facts of a kohya / modelspec header worth showing: names,
    base model, network module and size, when it was trained, the bucket
    resolutions and the dataset size (absent keys are left out)."""

    info: dict[str, Any] = {}
    for key, name in (
        ("ss_output_name", "output_name"),
        ("ss_sd_model_name", "base_model"),
        ("ss_base_model_version", "base_model_version"),
        ("modelspec.architecture", "architecture"),
        ("modelspec.title", "title"),
        ("modelspec.author", "author"),
        ("modelspec.description", "description"),
        ("modelspec.license", "license"),
        ("ss_network_module", "module"),
        ("ss_optimizer", "optimizer"),
    ):
        if value := _text(metadata.get(key)):
            info[name] = value
    for key, name in (("ss_network_dim", "dim"), ("ss_network_alpha", "alpha"), ("ss_clip_skip", "clip_skip"), ("ss_epoch", "epochs"), ("ss_steps", "steps"), ("ss_learning_rate", "learning_rate"), ("ss_num_train_images", "train_images")):
        number = _number(metadata.get(key))
        if number is not None:
            info[name] = int(number) if number.is_integer() else number
    if kind := network_type(metadata):
        info["network_type"] = kind

    started = _number(metadata.get("ss_training_started_at")) or _number(metadata.get("ss_training_finished_at"))
    if started:
        try:
            info["trained_at"] = datetime.datetime.fromtimestamp(started, datetime.UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            pass

    buckets = parse_json(metadata.get("ss_bucket_info"))
    if isinstance(buckets, Mapping) and isinstance(buckets.get("buckets"), Mapping):
        resolutions: dict[str, int] = {}
        for bucket in buckets["buckets"].values():
            try:
                width, height = bucket["resolution"]
                resolutions[f"{int(width)}x{int(height)}"] = resolutions.get(f"{int(width)}x{int(height)}", 0) + int(bucket.get("count", 0))
            except (KeyError, TypeError, ValueError):
                continue
        if resolutions:
            info["resolutions"] = [{"resolution": name, "count": count} for name, count in sorted(resolutions.items(), key=lambda item: -item[1])]

    dirs = parse_json(metadata.get("ss_dataset_dirs"))
    if isinstance(dirs, Mapping):
        images = 0
        for params in dirs.values():
            if isinstance(params, Mapping):
                images += int(_number(params.get("img_count")) or 0)
        if images:
            info["dataset_size"] = images
    return info


_ARCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (arch, re.compile(pattern))
    for arch, pattern in (
        ("qwen21", r"qwen[-_ ]?image[-_ ]?2\.?1"),
        ("qwen", r"qwen[-_ ]?image"),
        ("klein", r"flux[-_. ]?2|klein"),
        # FLUX.1 Krea [dev] is a Flux.1 model; "krea" alone is Krea 2
        ("flux", r"flux.*krea|krea[-_. ]?flux"),
        ("krea", r"krea"),
        ("flux", r"flux"),
        ("xl", r"(?:^|[^a-z])(?:sdxl|stable-diffusion-xl|xl[-_]?v?1|sd[-_]?xl|illustrious|pony|noobai)"),
        ("lumina", r"lumina"),
        ("zit", r"z[-_]?image"),
        ("wan", r"(?:^|[^a-z])wan[-_ ]?2"),
        ("anima", r"anima"),
        ("ernie", r"ernie"),
        ("pid", r"(?:^|[^a-z])pid(?:[^a-z]|$)"),
        ("sd", r"stable-diffusion-v1|(?:^|[^a-z])sd[-_ ]?v?1(?:[-_.]?5)?(?:[^0-9]|$)"),
    )
)


def detect_arch(metadata: Mapping[str, Any]) -> str | None:
    """The ``PresetArch`` name from ``modelspec.architecture``,
    ``ss_base_model_version`` or the network module; None when the header
    does not say (SD 2.x, unknown trainers)."""

    if str(metadata.get("ss_v2", "")).lower() == "true":
        return None
    for key in ("modelspec.architecture", "ss_base_model_version", "ss_network_module"):
        value = (_text(metadata.get(key)) or "").lower()
        if not value:
            continue
        for arch, pattern in _ARCH_PATTERNS:
            if pattern.search(value):
                return arch
    return None


def cover_image(metadata: Mapping[str, Any]) -> bytes | None:
    """The first embedded cover image (``ssmd_cover_images``, base64)."""

    covers = parse_json(metadata.get("ssmd_cover_images"))
    if not isinstance(covers, list):
        return None
    for cover in covers:
        if isinstance(cover, str) and cover:
            try:
                return base64.b64decode(cover, validate=False)
            except (binascii.Error, ValueError):
                continue
    return None


def visible_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The header without embedded images, nested JSON strings decoded."""

    return {str(key): parse_json(value) for key, value in metadata.items() if key not in HIDDEN_METADATA_KEYS}


def parse_tags(value: Any) -> list[str]:
    """User tags: a list of strings or one comma-separated string."""

    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    seen: dict[str, None] = {}
    for tag in value:
        text = str(tag).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen)


def preferred_weight(user: Mapping[str, Any]) -> float | None:
    """The user's preferred weight; None when unset or 0 (= the default multiplier)."""

    weight = _number(user.get("preferred weight"))
    return weight if weight else None


def normalize_user_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    """Checks the known keys of a user-metadata edit (``None`` removes a
    key) and returns them normalised; unknown keys pass unchanged.

    Raises ValueError naming the first bad key."""

    out: dict[str, Any] = {}
    for key, value in values.items():
        key = str(key)
        if value is None:
            out[key] = None
        elif key in TEXT_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"{key!r} must be text")
            if len(value) > USER_TEXT_LIMIT:
                raise ValueError(f"{key!r} is too long")
            out[key] = value
        elif key == "preferred weight":
            number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else _number(value) if isinstance(value, str) and value.strip() else None
            if number is None or not math.isfinite(number) or not -PREFERRED_WEIGHT_LIMIT <= number <= PREFERRED_WEIGHT_LIMIT:
                raise ValueError(f"'preferred weight' must be a number between -{PREFERRED_WEIGHT_LIMIT} and {PREFERRED_WEIGHT_LIMIT}")
            out[key] = float(number)
        elif key == "sd version":
            if value not in SD_VERSIONS:
                raise ValueError(f"'sd version' must be one of {', '.join(SD_VERSIONS)}")
            out[key] = value
        elif key == "tags":
            if not isinstance(value, (str, list, tuple)):
                raise ValueError("'tags' must be a list of text")
            out[key] = parse_tags(value)
        else:
            out[key] = value
    return out
