"""Tiny, deterministic Anima LoRAs for exercising the LoRA features end to end.

    python tools/anima_test_loras.py models/Stable-diffusion/anima-turbo-v1.1.safetensors /tmp/remi-loras

Writes ``remi-test-a.safetensors`` and ``remi-test-b.safetensors`` (rank 4, a few MB
each): low-rank deltas on every DiT block's attention outputs and MLP of the given
checkpoint, in the ``diffusion_model.blocks.<n>.<module>.lora_down/lora_up/alpha``
layout Anima trainers write. Only the checkpoint's safetensors *header* is read (the
layer shapes); the model is not loaded. The two files differ only in their seed, so
FreeFuse can route two distinct subject LoRAs. It also writes ``remi-test-lokr.safetensors``,
a LoKr (LyCORIS Kronecker) adapter on the same layers in the layout Anima LoKr trainers write
(``diffusion_model.blocks.<n>.<module>.lokr_w1`` with a low-rank ``lokr_w2_a`` / ``lokr_w2_b``
and ``alpha``), for the LoKR tab of LoRA Layer Weight.

They change an image visibly at weight 1 without destroying it. ``tools/smoke_real.py
--features`` generates them into its temporary data folder and uses them to check Anima
FreeFuse, LoRA Layer Weight (both its LoRA and its LoKR tab) and LoRA Stage Scheduler against
the real model.

    python tools/anima_test_loras.py --refiner CHECKPOINT OUT.safetensors

writes a second, genuinely different checkpoint instead: a copy of ``CHECKPOINT`` with a
third seeded delta of the same kind merged into its weights (every other tensor and the
metadata unchanged). ``tools/smoke_real.py --features`` uses one as the Refiner checkpoint,
so the refiner pass is seen changing the image (the weights must differ for that).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

NAMES = ("remi-test-a", "remi-test-b")
LOKR_NAME = "remi-test-lokr"
LOKR_SEED = 4
# lokr_w1 is LOKR_FACTOR x LOKR_FACTOR; lokr_w2 = lokr_w2_a @ lokr_w2_b of rank RANK, so each
# delta is kron(w1, w2) (alpha / RANK = 1), of rank LOKR_FACTOR * RANK
LOKR_FACTOR = 4
REFINER_NAME = "remi-test-refiner"
REFINER_SEED = 3
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


def make_lokr(layers: dict[str, tuple[int, int]], seed: int, *, factor: int = LOKR_FACTOR, rank: int = RANK, scale: float = 0.12) -> dict:
    """The LoKr tensors (fp16) for ``layers``: per layer ``lokr_w1`` (``factor`` x ``factor``),
    ``lokr_w2_a`` / ``lokr_w2_b`` (rank ``rank``) and ``alpha`` = ``rank``, so the delta
    ``kron(w1, w2_a @ w2_b)`` has the shape of the layer's weight and the element spread of
    ``make_lora``'s delta at the same ``scale`` (a comparable effect on the image)."""

    import torch

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer, (out_features, in_features) in layers.items():
        if out_features % factor or in_features % factor:
            raise ValueError(f"{layer}: {out_features}x{in_features} is not divisible by the LoKr factor {factor}")
        w1 = torch.randn(factor, factor, generator=generator)
        w2_a = torch.randn(out_features // factor, rank, generator=generator) * (scale / rank**0.5)
        w2_b = torch.randn(rank, in_features // factor, generator=generator) / in_features**0.5
        prefix = f"diffusion_model.{layer}"
        tensors[f"{prefix}.lokr_w1"] = w1.to(torch.float16).contiguous()
        tensors[f"{prefix}.lokr_w2_a"] = w2_a.to(torch.float16).contiguous()
        tensors[f"{prefix}.lokr_w2_b"] = w2_b.to(torch.float16).contiguous()
        tensors[f"{prefix}.alpha"] = torch.tensor(float(rank), dtype=torch.float16)
    return tensors


def write_test_lokr(checkpoint: Path | str, out_dir: Path | str) -> Path:
    """Write the test LoKr for ``checkpoint`` into ``out_dir``; its path."""

    from safetensors.torch import save_file

    layers = target_layers(read_header(checkpoint))
    if not layers:
        raise ValueError(f"{checkpoint}: no Anima DiT block layers ({', '.join(MODULES)}) in the header")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{LOKR_NAME}.safetensors"
    metadata = {"ss_network_module": "lycoris.kohya", "ss_network_args": json.dumps({"algo": "lokr", "factor": LOKR_FACTOR}), "ss_network_dim": str(RANK), "remi_test_lora": "1"}
    save_file(make_lokr(layers, LOKR_SEED), str(path), metadata=metadata)
    return path


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


def write_refiner_checkpoint(checkpoint: Path | str, out_path: Path | str, *, scale: float = 0.25) -> Path:
    """Write a copy of ``checkpoint`` with a seeded low-rank delta merged into the DiT
    layers the test LoRAs patch (``up @ down`` added to each weight, in its dtype): a
    checkpoint of the same architecture whose images differ visibly. Other tensors and
    the metadata are copied unchanged. Its path."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    header = read_header(checkpoint)
    layers = target_layers(header)
    if not layers:
        raise ValueError(f"{checkpoint}: no Anima DiT block layers ({', '.join(MODULES)}) in the header")
    delta = make_lora(layers, REFINER_SEED, scale=scale)
    by_key = {}
    for key in header:
        match = _BLOCK_WEIGHT.match(key)
        if match and f"blocks.{int(match[1])}.{match[2]}" in layers:
            by_key[key] = f"diffusion_model.blocks.{int(match[1])}.{match[2]}"
    tensors = {}
    with safe_open(str(checkpoint), framework="pt") as source:
        metadata = dict(source.metadata() or {})
        for key in source.keys():  # noqa: SIM118 - safe_open is not a mapping
            tensor = source.get_tensor(key)
            if (prefix := by_key.get(key)) is not None:
                up = delta[f"{prefix}.lora_up.weight"].float()
                down = delta[f"{prefix}.lora_down.weight"].float()
                tensor = (tensor.float() + up @ down).to(tensor.dtype)
            tensors[key] = tensor.contiguous()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={**metadata, "remi_test_refiner": "1"})
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="the Anima checkpoint (.safetensors) the LoRAs are for")
    parser.add_argument("out_dir", type=Path, help="where to write them (e.g. a --lora-dir); with --refiner the output file")
    parser.add_argument("--refiner", action="store_true", help="write a perturbed copy of the checkpoint (a second, different checkpoint) to out_dir")
    options = parser.parse_args()
    if options.refiner:
        path = write_refiner_checkpoint(options.checkpoint, options.out_dir)
        print(f"{path} ({path.stat().st_size / 1e9:.2f} GB)")
        return 0
    for path in [*write_test_loras(options.checkpoint, options.out_dir), write_test_lokr(options.checkpoint, options.out_dir)]:
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
