"""Synthetic fixtures for the UI performance measurements (docs/research/ui-performance.md).

    python tools/perf/fixtures.py create /tmp/remi-perf --loras 5000 --images 10000
    python tools/perf/fixtures.py remove /tmp/remi-perf
    python tools/perf/fixtures.py prompt /tmp/remi-perf-out/prompt300.txt

Creates, under the given root:

- ``Lora/<folder>/<name>.safetensors``: valid but empty safetensors files (a metadata-only
  header) spread over 50 folders, each with a small ``.preview.png`` (pass it to the server
  with ``--lora-dirs <root>/Lora``);
- ``data/``: a ``--data-dir`` with a copy of the repo's ``config.json`` and
  ``data/output/txt2img-images/<day>/*.png``: gallery images with a parameters text chunk
  (the gallery's folder scan indexes them at startup).

Everything lives under the root; ``remove`` deletes it. ``prompt`` writes the typing test's
prompt: common Danbooru general tags (tag autocomplete's ``danbooru.csv``, fixed shuffle)
until it is 300 CLIP tokens long.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = 50


def _safetensors(metadata: dict[str, str]) -> bytes:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header


def _gradient(size: tuple[int, int], seed: int):
    from PIL import Image

    rng = random.Random(seed)
    a = tuple(rng.randrange(256) for _ in range(3))
    b = tuple(rng.randrange(256) for _ in range(3))
    small = Image.new("RGB", (2, 2))
    small.putdata([a, b, b, a])
    return small.resize(size, Image.Resampling.BILINEAR)


def create_loras(root: Path, count: int) -> None:
    lora = root / "Lora"
    for index in range(count):
        folder = lora / f"pack_{index % FOLDERS:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        name = f"perf_lora_{index:05d}"
        meta = {"ss_base_model_version": "anima", "ss_output_name": name, "ss_tag_frequency": json.dumps({"set": {f"tag{index % 97}": 3}})}
        (folder / f"{name}.safetensors").write_bytes(_safetensors(meta))
        _gradient((96, 128), index).save(folder / f"{name}.preview.png", optimize=False, compress_level=1)


def create_gallery(root: Path, count: int) -> None:
    from PIL.PngImagePlugin import PngInfo

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "styles.csv"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, data / name)
    out = data / "output" / "txt2img-images"
    start = time.time() - count * 60
    for index in range(count):
        day = out / f"2026-09-{1 + index // 1000:02d}"
        day.mkdir(parents=True, exist_ok=True)
        info = PngInfo()
        info.add_text("parameters", f"1girl, perf fixture {index}\nNegative prompt: lowres\nSteps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 1, Seed: {index}, Size: 384x384, Model: anima-turbo-v1.1")
        path = day / f"{index:05d}-{index}.png"
        _gradient((384, 384), 100_000 + index).save(path, pnginfo=info, compress_level=1)
        stamp = start + index * 60
        os.utime(path, (stamp, stamp))


def write_prompt(path: Path, tokens: int = 300) -> None:
    import csv

    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(ROOT / "backend/huggingface/runwayml/stable-diffusion-inpainting/tokenizer")
    with open(ROOT / "extensions-builtin/sd-webui-tagcomplete-neo/tags/danbooru.csv", encoding="utf8") as file:
        general = [row[0].replace("_", " ") for row in csv.reader(file) if len(row) > 1 and row[1] == "0"][:600]
    random.Random(7).shuffle(general)
    tags: list[str] = []
    for tag in general:
        tags.append(tag)
        if len(tokenizer(", ".join(tags))["input_ids"]) - 2 >= tokens:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(", ".join(tags), encoding="utf8")
    print(f"{len(tags)} tags, {len(', '.join(tags))} characters -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("create", "remove", "prompt"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--loras", type=int, default=5000)
    parser.add_argument("--images", type=int, default=10_000)
    options = parser.parse_args()
    root: Path = options.root
    if options.action == "prompt":
        write_prompt(root)
        return 0
    if options.action == "remove":
        shutil.rmtree(root, ignore_errors=True)
        return 0
    if root.exists() and any(root.iterdir()):
        print(f"{root} is not empty", file=sys.stderr)
        return 1
    started = time.monotonic()
    create_loras(root, options.loras)
    create_gallery(root, options.images)
    print(f"{options.loras} LoRAs and {options.images} images in {root} ({time.monotonic() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
