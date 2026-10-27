"""Moss-Image and Moss-Image-Edit (upstream be6fa5d4 "Moss-Image", c6c2b43d
"Moss-Image-Edit") as Remi sees them without loading the 9B text encoder or
the DiT: detection and the loader's engine list, the converted
``moss_do_reference`` setting, the engine's reference routing and its prompt
token count (the SPA token counter calls ``get_prompt_lengths_on_ui``)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "backend" / "huggingface" / "OpenMOSS-Team" / "MOSS-Image-SFT"
IMAGE_PAD = 248056  # "<|image_pad|>", replaced by the reference images
PAD = 248044  # "<|endoftext|>"


def _dit_state_dict(dim=256, ffn=512, layers=3, prefix="model.diffusion_model."):
    sd = {f"{prefix}shared_block_modulation.weight": torch.empty(6 * dim, 4)}
    for index in range(layers):
        sd[f"{prefix}layers.{index}.proj.ffn_up_gate.weight"] = torch.empty(2 * ffn, 4)
        sd[f"{prefix}layers.{index}.proj.qkv.weight"] = torch.empty(3 * dim, 4)
    return sd


def test_a_moss_dit_is_detected_and_has_an_engine():
    from huggingface_guess import guess, model_list

    from backend.diffusion_engine.moss_image import MOSSImage
    from backend.loader import possible_models

    config = guess(_dit_state_dict())
    assert isinstance(config, model_list.MOSSImage) and config.huggingface_repo == "OpenMOSS-Team/MOSS-Image-SFT"
    assert config.unet_config == {"dim": 256, "ffn_dim": 512, "num_heads": 2, "num_layers": 3, "fused_qkv": True}
    assert MOSSImage in possible_models and model_list.MOSSImage in MOSSImage.matched_guesses
    # The diffusers layout the loader builds the components from ships with Remi.
    for component in ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "transformer/config.json", "vae/config.json", "tokenizer/tokenizer_config.json"):
        assert (REPO / component).is_file(), component


def test_the_reference_setting_offers_every_method_the_engine_knows():
    from backend.diffusion_engine.moss_image import REFERENCE_METHODS
    from modules import shared_options
    from remi import ui

    info = shared_options.options_templates["moss_do_reference"]
    args = info.component_args() if callable(info.component_args) else info.component_args
    assert info.component is ui.Choice and info.default == "Disable"
    assert tuple(args["choices"]) == ("Disable", *REFERENCE_METHODS)


class _TextEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, images=()):
        self.calls.append((list(prompt), list(images)))
        return ["cond"]


@pytest.fixture
def engine(monkeypatch):
    from backend import memory_management
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(memory_management, "load_model_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dynamic_args, "ref_latents", [])
    monkeypatch.setattr(dynamic_args, "moss_reference_method", None)
    monkeypatch.setattr(dynamic_args, "moss_ref_kv", None)
    model = moss_image.MOSSImage.__new__(moss_image.MOSSImage)
    model.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=object()))
    model.text_processing_engine_qwen = _TextEngine()
    model.ref_latents = []
    model.ini_latent = None
    # (vision input, reference latent) per image, without the VAE
    model.encode_vision = lambda image: (("vision", image), ("latent", image))
    return model


def _prompt(*texts, negative=False):
    from modules.prompt_parser import SdConditioning

    return SdConditioning(list(texts), is_negative_prompt=negative)


@pytest.mark.parametrize(
    "mode,method,vision",
    [("ai-toolkit", "index_timestep_zero_isolated", True), ("Moss-Image-Edit", "index_timestep_zero_text_start", False)],
)
def test_reference_methods_route_the_img2img_image_and_stitched_references(engine, monkeypatch, mode, method, vision):
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference=mode))
    engine.ini_latent, engine.ref_latents = "init", ["stitched"]

    engine.get_learned_conditioning(_prompt("make it blue"))
    (prompt, images), = engine.text_processing_engine_qwen.calls
    assert prompt == ["make it blue"] and images == ([("vision", "init"), ("vision", "stitched")] if vision else [])
    assert dynamic_args.moss_reference_method == method and dynamic_args.ref_latents == [("latent", "init"), ("latent", "stitched")]
    assert engine.ini_latent is None  # consumed by the positive prompt

    engine.get_learned_conditioning(_prompt("", negative=True))  # the negative prompt keeps them
    assert engine.text_processing_engine_qwen.calls[-1] == ([""], [])
    assert dynamic_args.moss_reference_method == method and len(dynamic_args.ref_latents) == 2


def test_disabled_references_clear_the_previous_ones(engine, monkeypatch):
    from backend.args import dynamic_args
    from backend.diffusion_engine import moss_image

    monkeypatch.setattr(moss_image, "opts", SimpleNamespace(moss_do_reference="Disable"))
    dynamic_args.ref_latents.append("stale")
    dynamic_args.moss_reference_method, dynamic_args.moss_ref_kv = "index_timestep_zero_isolated", "kv"
    engine.ref_latents = ["stitched"]

    engine.get_learned_conditioning(_prompt("a cat"))
    assert engine.text_processing_engine_qwen.calls == [(["a cat"], [])]
    assert dynamic_args.ref_latents == [] and dynamic_args.moss_reference_method is None and dynamic_args.moss_ref_kv is None


@pytest.fixture(scope="module")
def text_engine(tmp_path_factory):
    import shutil

    from transformers import Qwen2TokenizerFast

    from backend.huggingface import Token, compare_sha256, decompress, sha256
    from backend.text_processing.moss_image_engine import Qwen359BEngine

    tokenizer_dir = REPO / "tokenizer"
    if not (tokenizer_dir / "tokenizer.json").is_file():
        # Not decompressed yet (the launcher does it): the shipped archive, in a
        # scratch folder rather than the checkout.
        scratch = tmp_path_factory.mktemp("moss-tokenizer")
        shutil.copy(tokenizer_dir / "tokenizer_config.json", scratch)
        decompress(Token.moss_compress, str(scratch / "tokenizer.json"))
        compare_sha256(str(scratch / "tokenizer.json"), sha256.moss)
        tokenizer_dir = scratch
    encoder = torch.nn.Module()
    encoder.num_layers = 32
    return Qwen359BEngine(text_encoder=encoder, tokenizer=Qwen2TokenizerFast.from_pretrained(str(tokenizer_dir)))


def test_prompt_lengths_count_the_chat_template(text_engine):
    from backend.diffusion_engine.moss_image import MOSSImage

    model = MOSSImage.__new__(MOSSImage)
    model.text_processing_engine_qwen = text_engine
    empty, _ = model.get_prompt_lengths_on_ui("")
    count, limit = model.get_prompt_lengths_on_ui("a red bicycle by the sea")
    assert empty > 0 and count > empty and limit == 999  # no 75-token chunks: one sequence
    long = " ".join(["cat"] * 1200)
    assert model.get_prompt_lengths_on_ui(long)[1] == model.get_prompt_lengths_on_ui(long)[0] > 999


def test_reference_images_fill_the_image_pad_tokens(text_engine):
    images = [torch.zeros(1, 8, 8, 3), torch.ones(1, 8, 8, 3)]
    (tokens,) = text_engine._tokenize_with_weights("swap the colours", images)
    filled = [token for token, *_ in tokens if isinstance(token, dict)]
    assert [item["data"] for item in filled] == images and all(item["type"] == "image" for item in filled)
    assert IMAGE_PAD not in [token for token, *_ in tokens]
    (plain,) = text_engine._tokenize_with_weights("swap the colours", [])
    assert not any(isinstance(token, dict) for token, *_ in plain) and PAD not in [token for token, *_ in plain]
