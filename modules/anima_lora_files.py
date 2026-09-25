"""What an Anima LoRA/LoKr file trains, read from its safetensors keys.

Shared by the layer-weight and stage-scheduler extensions. Results are cached
per file version (path, mtime, size), so each file is scanned once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from modules.anima_lora_support import block_index_from_key, module_from_key, qwen_layer_index

LORA_MARKERS = (".lora_a.weight", ".lora_b.weight", ".lora_down.weight", ".lora_up.weight")
LOKR_MARKERS = (".lokr_",)


@dataclass(frozen=True)
class LoraCoverage:
    """DiT blocks, block modules and Qwen3 text-encoder layers a file trains."""

    blocks: frozenset[int] = frozenset()
    modules: frozenset[str] = frozenset()
    text_layers: frozenset[int] = frozenset()


_KIND_CACHE: dict[tuple, str | None] = {}
_COVERAGE_CACHE: dict[tuple, LoraCoverage] = {}


def lora_cache_key(filename) -> tuple | None:
    """``(path, mtime, size)`` of a .safetensors file, or None."""

    path = str(filename)
    if Path(path).suffix.lower() != ".safetensors":
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return os.path.abspath(path), stat.st_mtime, stat.st_size


def _safetensors_keys(filename) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(filename), framework="pt", device="cpu") as f:
        return list(f.keys())


def is_adapter_weight_key(key: str, adapter_kind: str) -> bool:
    markers = LOKR_MARKERS if adapter_kind == "lokr" else LORA_MARKERS
    key_lower = str(key or "").lower()
    return any(marker in key_lower for marker in markers)


def adapter_kind_for_file(filename) -> str | None:
    """"lokr", "lora", or None when the file holds neither (or cannot be read)."""

    cache_key = lora_cache_key(filename)
    if cache_key is None:
        return None
    if cache_key not in _KIND_CACHE:
        try:
            keys = _safetensors_keys(filename)
        except Exception:
            kind = None
        else:
            has_lokr = any(is_adapter_weight_key(key, "lokr") for key in keys)
            has_lora = any(is_adapter_weight_key(key, "lora") for key in keys)
            kind = "lokr" if has_lokr else "lora" if has_lora else None
        _KIND_CACHE[cache_key] = kind
    return _KIND_CACHE[cache_key]


def lora_coverage(filename, adapter_kind: str = "lora") -> LoraCoverage:
    cache_key = lora_cache_key(filename)
    if cache_key is None:
        return LoraCoverage()
    coverage_key = (cache_key, adapter_kind)
    if coverage_key not in _COVERAGE_CACHE:
        try:
            keys = _safetensors_keys(filename)
        except Exception:
            keys = []
        blocks, modules, text_layers = set(), set(), set()
        for key in keys:
            if not is_adapter_weight_key(key, adapter_kind):
                continue
            text_layer = qwen_layer_index(key)
            if text_layer is not None:
                text_layers.add(text_layer)
                continue
            block = block_index_from_key(key)
            if block is not None:
                blocks.add(block)
                module = module_from_key(key)
                if module:
                    modules.add(module)
        _COVERAGE_CACHE[coverage_key] = LoraCoverage(frozenset(blocks), frozenset(modules), frozenset(text_layers))
    return _COVERAGE_CACHE[coverage_key]


def is_anima_adapter_file(filename, adapter_kind: str = "lora") -> bool:
    """Whether the file trains Anima DiT blocks (text-encoder-only files do not count)."""

    return bool(lora_coverage(filename, adapter_kind).blocks)
