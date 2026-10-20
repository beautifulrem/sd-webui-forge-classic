"""A tiny, deterministic Anima checkpoint for end-to-end runs without the real model.

The real Anima (a 2B DiT, the Qwen3 0.6B text encoder and the Qwen-Image VAE)
needs gigabytes and minutes to load. This module writes a checkpoint of the same
architecture at toy size (about 40 MB, most of it the text encoder's 151,936-token
vocabulary: 28 DiT blocks of width 64, a 2-layer Qwen3 of width 32 with Anima's
tokenizers, a VAE of base width 8) with seeded random weights, and loads it as a
real ``backend.diffusion_engine.anima.Anima`` engine: the same DiT, text encoder,
LLM adapter, VAE and tokenizer classes, Forge's patchers, k-diffusion samplers and
processing pipeline. Everything an extension hooks into (``process``,
``process_before_every_sampling``, model options, attention overrides, block
patches, LoRA patches, postprocessing) runs for real, on the CPU, in seconds.
The pictures are noise-like, but deterministic: the same request gives the same
pixels, and an extension that changes the generation changes them.

The stock loader only accepts the real model sizes (``huggingface_guess``
asserts the 2048 DiT width), so ``install()`` wraps ``modules.sd_models.forge_loader``:
a file whose safetensors metadata has ``remi_tiny_anima`` is built here, any
other file goes to the stock loader. Only test and tool code calls ``install()``.

    python tools/tiny_anima.py write models/Stable-diffusion/tiny-anima.safetensors
    python tools/tiny_anima.py serve --data-dir /path/to/data [launch.py args, e.g. --port 7899 --cpu --api]

``serve`` starts the real WebUI (``launch.py``, without preparing the
environment) with ``install()`` applied and the tiny models of
:func:`write_models` in ``<data-dir>/models`` (the checkpoint, a second one for
the Refiner, test LoRAs and a LoKr, a Control-LLLite, an ADetailer detector; a
new data folder selects the checkpoint), so every page and extension can be
tried by hand, in the browser or through the APIs, against a model that loads
in a second. ``tests/test_remi_extensions_e2e.py`` uses the same models to run
every built-in extension through the server's job queue.

The layout is the one of a real Anima single-file checkpoint: the DiT under
``net.``, the text encoder under ``text_encoders.qwen3_06b.transformer.`` (the
LLM adapter under ``net.llm_adapter.``, where Anima keeps it) and the VAE under
``vae.``, so tools that read a checkpoint's header (``tools/anima_test_loras.py``)
work on it too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METADATA_KEY = "remi_tiny_anima"
FILENAME = "tiny-anima.safetensors"
REFINER_FILENAME = "tiny-anima-refiner.safetensors"
CONTROLNET_NAME = "tiny-anima-lllite"
DETECTOR_FILENAME = "tiny-face.pt"

DIT = {
    "in_channels": 16,
    "out_channels": 16,
    "patch_spatial": 2,
    "patch_temporal": 1,
    "model_channels": 64,
    "concat_padding_mask": True,
    "crossattn_emb_channels": 64,
    "adaln_lora_dim": 16,
    "num_blocks": 28,
    "num_heads": 2,
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
    "rope_t_extrapolation_ratio": 1.0,
}
"""The DiT: Anima 2B's layout (28 blocks, 16 latent channels, 2x2 patches) at width 64."""

QWEN = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16}
"""The text encoder: Qwen3 0.6B's layout (its tokenizer and vocabulary) at width 32."""

ADAPTER = {"source_dim": QWEN["hidden_size"], "target_dim": DIT["crossattn_emb_channels"], "model_dim": DIT["crossattn_emb_channels"], "num_layers": 1, "num_heads": 2}
"""The LLM adapter from the text encoder to the DiT's cross-attention context."""

VAE = {"base_dim": 8, "z_dim": 16, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1, "attn_scales": [], "temporal_downsample": [False, True, True], "dropout": 0.0}
"""The VAE: the Qwen-Image (Wan 2.1) VAE's layout at base width 8."""

PREFIXES = {"transformer": "net.", "text_encoder": "text_encoders.qwen3_06b.transformer.", "vae": "vae."}


# region Building the components


def _qwen_config():
    from backend.nn.llm.llama import Qwen3_06BConfig

    config = Qwen3_06BConfig(**{key: value for key, value in QWEN.items() if key != "head_dim"})
    config.head_dim = QWEN["head_dim"]  # a class attribute of the config, not a field
    return config


def _text_encoder():
    """A ``Qwen3_06B`` (the class Forge loads Anima's text encoder as) at toy size."""

    from torch import nn

    from backend.nn.anima import LLMAdapter
    from backend.nn.llm.llama import Llama2_, Qwen3_06B

    model = Qwen3_06B.__new__(Qwen3_06B)
    nn.Module.__init__(model)
    config = _qwen_config()
    model.num_layers = config.num_hidden_layers
    model.model = Llama2_(config)
    model.llm_adapter = LLMAdapter(**ADAPTER)
    return model


def _components(device, dtype, sd_dtype):
    """The DiT, text encoder and VAE modules, built with Forge's operations as the
    loader builds them (``sd_dtype`` equal to ``dtype`` on the CPU: on the meta device)."""

    from backend.nn.anima import Anima
    from backend.nn.wan_vae import WanVAE
    from backend.operations import using_forge_operations

    with using_forge_operations(device=device, dtype=dtype, manual_cast_enabled=False, sd_dtype=sd_dtype):
        transformer = Anima(**DIT)
    with using_forge_operations(device=device, dtype=dtype, manual_cast_enabled=True, sd_dtype=sd_dtype):
        text_encoder = _text_encoder()
    with using_forge_operations(device=device, dtype=dtype, manual_cast_enabled=False, sd_dtype=sd_dtype, extra_dtype="vae"):
        vae = WanVAE(**VAE)
    return {"transformer": transformer, "text_encoder": text_encoder, "vae": vae}


def state_dict(seed: int = 0) -> dict:
    """The checkpoint's tensors (fp32): seeded weights for every component, keyed as in a
    real Anima single-file checkpoint. Linear / conv weights are ``N(0, 1/fan_in)``,
    embeddings ``N(0, 1)``, norm scales 1 and biases 0, so activations keep their scale
    through the blocks and the sampler stays finite (:data:`CROSS_ATTENTION_GAIN`: the
    prompt's weight)."""

    import torch

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in _components(torch.device("cpu"), torch.float32, None).items():
        for key, value in module.state_dict().items():
            tensors[PREFIXES[name] + key] = _initial_value(key, value, generator)
    # Anima keeps the LLM adapter in the DiT; the loader moves it to the text encoder
    adapter = [key for key in tensors if key.startswith(PREFIXES["text_encoder"] + "llm_adapter.")]
    for key in adapter:
        tensors[PREFIXES["transformer"] + key.removeprefix(PREFIXES["text_encoder"])] = tensors.pop(key)
    return tensors


CROSS_ATTENTION_GAIN = 4.0
"""The DiT's cross-attention query/key norm scales and output projection are this many
times the plain init. A random model attends to the 512-token padded prompt evenly,
so the few prompt tokens would barely matter; sharper attention and a larger share of
the residual stream make the prompt (and so the cond/uncond difference CFG and the
guidance extensions work on) change the result by about 10 % instead of 0.1 %."""


def _initial_value(key: str, value, generator):
    import torch

    shape = tuple(value.shape)
    leaf = key.rsplit(".", 1)[-1]
    cross_attention = ".cross_attn." in key
    if leaf == "bias" or not shape:
        return torch.zeros(shape)
    if len(shape) == 1:  # norm scales (RMSNorm, LayerNorm, the VAE's RMS_norm gamma)
        return torch.full(shape, CROSS_ATTENTION_GAIN if cross_attention and ("q_norm" in key or "k_norm" in key) else 1.0)
    if cross_attention and "output_proj" in key:
        return torch.randn(shape, generator=generator) * (CROSS_ATTENTION_GAIN / math.sqrt(math.prod(shape[1:])))
    if "embed" in key and len(shape) == 2 and "proj" not in key:  # token embeddings
        return torch.randn(shape, generator=generator)
    if leaf == "gamma":
        return torch.ones(shape)
    fan_in = math.prod(shape[1:])
    return torch.randn(shape, generator=generator) / math.sqrt(fan_in)


def write(path: Path | str, seed: int = 0) -> Path:
    """Write a tiny checkpoint (``seed`` picks the weights); its path."""

    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {METADATA_KEY: json.dumps({"seed": seed, "dit": DIT, "qwen": QWEN, "adapter": ADAPTER, "vae": VAE})}
    save_file({key: value.contiguous() for key, value in state_dict(seed).items()}, str(path), metadata=metadata)
    return path


LLLITE = {"cond_emb_dim": 16, "mlp_dim": 16, "target_layers": "self_attn_q", "cond_dim": 16, "cond_resblocks": 0}
"""The Control-LLLite: Anima's ControlNet (``sd_forge_controlllite``) on every block's self-attention query."""


def write_controllllite(path: Path | str, seed: int = 2) -> Path:
    """Write a Control-LLLite for the tiny DiT (seeded, every weight non-zero, so a
    control image changes the result), in the saved layout ``sd_forge_controlllite``
    loads (``lllite_conditioning1.*``, ``lllite_dit_blocks_<n>_...``); its path."""

    import torch
    from safetensors.torch import save_file

    folder = str(ROOT / "extensions-builtin" / "sd_forge_controlllite")
    if folder not in sys.path:
        sys.path.append(folder)
    from lib_controllllite.lib_controllllite_anima import ControlNetLLLiteDiT

    generator = torch.Generator().manual_seed(seed)
    lllite = ControlNetLLLiteDiT(_components(torch.device("cpu"), torch.float32, None)["transformer"], **LLLITE)
    tensors = {}
    for key, value in lllite.conditioning1.state_dict().items():
        tensors[f"lllite_conditioning1.{key}"] = _initial_value(key, value, generator)
    for module in lllite.lllite_modules:
        for key, value in module.state_dict().items():
            # the zero-initialised FiLM and up projections too: an untrained LLLite does nothing
            tensors[f"{module.lllite_name}.{key}"] = torch.randn(value.shape, generator=generator) / math.sqrt(max(1, math.prod(value.shape[1:])))
        tensors[f"{module.lllite_name}.depth_embed"] = torch.randn(lllite.depth_embeds.shape[1:], generator=generator) * 0.1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    return path


def write_detector(path: Path | str, seed: int = 0) -> Path:
    """Write an ADetailer detector: Ultralytics' YOLOv8n built from its bundled
    config with seeded random weights (no download). It finds low-confidence
    boxes anywhere, which is all an end-to-end run of ADetailer needs; its path."""

    import torch
    from ultralytics import YOLO

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        YOLO("yolov8n.yaml").save(str(path))
    finally:
        torch.random.set_rng_state(state)
    return path


def is_tiny(path: Any) -> bool:
    """Whether ``path`` is a checkpoint this module wrote."""

    if not isinstance(path, (str, os.PathLike)) or not str(path).endswith(".safetensors") or not os.path.isfile(path):
        return False
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as file:
        return METADATA_KEY in (file.metadata() or {})


# endregion

# region Loading


def load(path: Path | str):
    """The ``Anima`` engine of the tiny checkpoint at ``path``, as ``forge_loader`` returns one."""

    import torch
    from huggingface_guess import model_list
    from transformers import Qwen2Tokenizer, T5TokenizerFast

    import backend.args
    from backend import memory_management
    from backend.diffusion_engine.anima import Anima
    from backend.state_dict import load_state_dict
    from backend.utils import load_torch_file

    tensors = load_torch_file(str(path))
    parts: dict[str, dict] = {name: {} for name in PREFIXES}
    for key, value in tensors.items():
        if key.startswith(PREFIXES["transformer"] + "llm_adapter."):
            parts["text_encoder"][key.removeprefix(PREFIXES["transformer"])] = value
            continue
        name = next(name for name, prefix in PREFIXES.items() if key.startswith(prefix))
        parts[name][key.removeprefix(PREFIXES[name])] = value

    backend.args.dynamic_args.reset()
    backend.args.dynamic_args.anima = True
    unet_config = dict(DIT)
    # as huggingface_guess.guess and the loader's split_state_dict prepare it
    guess = model_list.Anima({"image_model": "anima", **unet_config})
    guess.unet_config.pop("image_model", None)
    guess.unet_key_prefix = [PREFIXES["transformer"]]
    guess.clip_target = guess.clip_target({})
    guess.model_type = guess.model_type({})
    guess.ztsnr = False

    components = _components(memory_management.cpu, torch.float32, torch.float32)
    for name, module in components.items():
        load_state_dict(module, parts[name], log_name=f"tiny Anima {name}")
        module.eval()
    transformer = components["transformer"]
    transformer.config = unet_config
    transformer.storage_dtype = transformer.computation_dtype = torch.float32
    transformer.load_device = memory_management.get_torch_device()
    transformer.initial_device = memory_management.cpu
    transformer.offload_device = memory_management.unet_offload_device()

    repo = ROOT / "backend" / "huggingface" / "circlestone-labs" / "Anima"
    huggingface_components = {
        "transformer": transformer,
        "text_encoder": components["text_encoder"],
        "vae": components["vae"],
        "tokenizer": Qwen2Tokenizer.from_pretrained(str(repo / "tokenizer")),
        "tokenizer_2": T5TokenizerFast.from_pretrained(str(repo / "tokenizer_2")),
    }
    for name in ("tokenizer", "tokenizer_2"):
        huggingface_components[name]._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
    return Anima(estimated_config=guess, huggingface_components=huggingface_components)


def install() -> None:
    """Load tiny checkpoints with :func:`load` (idempotent); other files with the stock loader."""

    from modules import sd_models

    stock = sd_models.forge_loader
    if getattr(stock, "remi_tiny_anima", False):
        return

    def forge_loader(sd, additional_state_dicts=None):
        if is_tiny(sd):
            return load(sd)
        return stock(sd, additional_state_dicts=additional_state_dicts)

    forge_loader.remi_tiny_anima = True
    sd_models.forge_loader = forge_loader


# endregion


def _serve(data_dir: Path, launch_args: list[str]) -> None:
    """``launch.py`` (without preparing the environment) with :func:`install` applied
    before the first model load, and the tiny models in ``data_dir/models``."""

    sys.argv = ["launch.py", "--skip-prepare-environment", "--data-dir", str(data_dir), *launch_args]
    sys.path[:0] = [str(ROOT), str(ROOT / "modules_forge" / "packages")]
    os.chdir(ROOT)
    write_models(data_dir / "models")
    config = data_dir / "config.json"
    if not config.is_file():  # a new data folder: the model, not the Refiner's checkpoint (first by name)
        config.write_text(json.dumps({"sd_model_checkpoint": FILENAME}), encoding="utf-8")
    from modules import launch_utils, logging_config

    logging_config.setup_logging(launch_utils.args.loglevel)
    import webui  # initialises the WebUI; no model is loaded yet

    install()
    webui.webui()
    from modules_forge import main_thread

    main_thread.loop()


def write_models(models: Path) -> dict[str, Path]:
    """The tiny models an end-to-end run uses, written into the ``models`` folder
    when missing: the checkpoint, a second one (the Refiner's), test LoRAs and a
    LoKr for it (``tools/anima_test_loras.py``), a Control-LLLite and an ADetailer
    detector. Their paths by role."""

    paths = {
        "checkpoint": models / "Stable-diffusion" / FILENAME,
        "refiner": models / "Stable-diffusion" / REFINER_FILENAME,
        "controlnet": models / "ControlNet" / f"{CONTROLNET_NAME}.safetensors",
        "detector": models / "adetailer" / DETECTOR_FILENAME,
    }
    if not paths["checkpoint"].is_file():
        write(paths["checkpoint"], seed=0)
    if not paths["refiner"].is_file():
        write(paths["refiner"], seed=1)
    if str(ROOT / "tools") not in sys.path:
        sys.path.append(str(ROOT / "tools"))
    import anima_test_loras

    if not (models / "Lora" / f"{anima_test_loras.NAMES[0]}.safetensors").is_file():
        anima_test_loras.write_test_loras(paths["checkpoint"], models / "Lora")
        anima_test_loras.write_test_lokr(paths["checkpoint"], models / "Lora")
    if not paths["controlnet"].is_file():
        write_controllllite(paths["controlnet"])
    if not paths["detector"].is_file():
        write_detector(paths["detector"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write_command = commands.add_parser("write", help="write a tiny checkpoint")
    write_command.add_argument("path", type=Path)
    write_command.add_argument("--seed", type=int, default=0)
    serve_command = commands.add_parser("serve", help="start the WebUI with a tiny checkpoint (extra arguments go to launch.py)")
    serve_command.add_argument("--data-dir", type=Path, required=True)
    args, rest = parser.parse_known_args()
    if args.command == "write":
        if rest:
            parser.error(f"unrecognized arguments: {' '.join(rest)}")
        sys.path[:0] = [str(ROOT), str(ROOT / "modules_forge" / "packages")]
        print(write(args.path, args.seed))
        return 0
    _serve(args.data_dir.resolve(), rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
