"""Tiny, deterministic Anima LoRAs for exercising the LoRA features end to end.

    python tools/anima_test_loras.py models/Stable-diffusion/anima-turbo-v1.1.safetensors /tmp/remi-loras

Writes ``remi-test-a.safetensors`` and ``remi-test-b.safetensors`` (rank 4, a few MB
each): low-rank deltas on every DiT block's attention outputs and MLP of the given
checkpoint, in the ``diffusion_model.blocks.<n>.<module>.lora_down/lora_up/alpha``
layout Anima trainers write. Only the checkpoint's safetensors *header* is read (the
layer shapes); the model is not loaded. The two files differ only in their seed, so
FreeFuse can route two distinct subject LoRAs.

They change an image visibly at weight 1 without destroying it. ``tools/smoke_real.py
--features`` generates them into its temporary data folder and uses them to check Anima
FreeFuse, LoRA Layer Weight and LoRA Stage Scheduler against the real model.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

NAMES = ("remi-test-a", "remi-test-b")
RANK = 4
# The Linear layers each block gets a delta on (their output feeds the residual stream)
MODULES = ("self_attn.output_proj", "cross_attn.output_proj", "mlp.layer2")
_BLOCK_WEIGHT = re.compile(r"^(?:model\.)?(?:diffusion_model\.|net\.)?blocks\.(\d+)\.(.+)\.weight$")


def read_header(path: Path | str) -> dict:
    """The tensor table of a safetensors file (name -> {"dtype", "shape", ...})."""

    with open(path, "rb") as file:
        (size,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(size))
    header.pop("__metadata__", None)
    return header


def target_layers(header: dict) -> dict[str, tuple[int, int]]:
    """``{"blocks.<n>.<module>": (out_features, in_features)}`` of the DiT layers the test LoRAs patch."""

    layers = {}
    for key, entry in header.items():
        match = _BLOCK_WEIGHT.match(key)
        if match and match[2] in MODULES and len(entry.get("shape", ())) == 2:
            out_features, in_features = entry["shape"]
            layers[f"blocks.{int(match[1])}.{match[2]}"] = (int(out_features), int(in_features))
    return dict(sorted(layers.items(), key=lambda item: (int(item[0].split(".")[1]), item[0])))


def make_lora(layers: dict[str, tuple[int, int]], seed: int, *, rank: int = RANK, scale: float = 0.12) -> dict:
    """The LoRA tensors (fp16) for ``layers``; ``scale`` sets the up weights' spread."""

    import torch

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer, (out_features, in_features) in layers.items():
        down = torch.randn(rank, in_features, generator=generator) / in_features**0.5
        up = torch.randn(out_features, rank, generator=generator) * (scale / rank**0.5)
        prefix = f"diffusion_model.{layer}"
        tensors[f"{prefix}.lora_down.weight"] = down.to(torch.float16).contiguous()
        tensors[f"{prefix}.lora_up.weight"] = up.to(torch.float16).contiguous()
        tensors[f"{prefix}.alpha"] = torch.tensor(float(rank), dtype=torch.float16)
    return tensors


def write_test_loras(checkpoint: Path | str, out_dir: Path | str) -> list[Path]:
    """Write the test LoRAs for ``checkpoint`` into ``out_dir``; their paths."""

    from safetensors.torch import save_file

    layers = target_layers(read_header(checkpoint))
    if not layers:
        raise ValueError(f"{checkpoint}: no Anima DiT block layers ({', '.join(MODULES)}) in the header")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed, name in enumerate(NAMES, start=1):
        path = out_dir / f"{name}.safetensors"
        save_file(make_lora(layers, seed), str(path), metadata={"ss_network_dim": str(RANK), "remi_test_lora": "1"})
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="the Anima checkpoint (.safetensors) the LoRAs are for")
    parser.add_argument("out_dir", type=Path, help="where to write them (e.g. a --lora-dir)")
    options = parser.parse_args()
    for path in write_test_loras(options.checkpoint, options.out_dir):
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
