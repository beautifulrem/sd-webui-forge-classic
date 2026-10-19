"""Infotext paste round-trips (``POST /api/v2/infotext/parse``) for keys the
generic infotext mapping cannot restore on its own: the refiner checkpoint and
the img2img "Inpaint area"."""

from types import SimpleNamespace

import pytest

from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


def _infotext(prompt, params):
    from modules.infotext_utils import quote

    return f"{prompt}\nNegative prompt: \n" + ", ".join(f"{key}: {quote(value)}" for key, value in params.items())


# region Refiner


@pytest.fixture
def checkpoints(webui, monkeypatch):
    """Two registered checkpoints, as sd_models.list_models() registers them."""
    from modules import sd_models

    monkeypatch.setattr(sd_models, "checkpoints_list", {})
    monkeypatch.setattr(sd_models, "checkpoint_aliases", {})
    infos = []
    for name, shorthash in (("anima-copy.safetensors", "fba1195327"), ("sub/anima-turbo-v1.1.safetensors", None)):
        stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        info = SimpleNamespace(
            name=name,
            name_for_extra=stem,
            model_name=name.rsplit(".", 1)[0].replace("/", "_"),
            title=name if shorthash is None else f"{name} [{shorthash}]",
            short_title=stem if shorthash is None else f"{stem} [{shorthash}]",
            filename=f"/models/{name}",
        )
        info.ids = [info.title, info.name, info.name_for_extra, info.model_name] + ([shorthash, info.short_title] if shorthash else [])
        sd_models.checkpoints_list[info.title] = info
        for alias in info.ids:
            sd_models.checkpoint_aliases[alias] = info
        infos.append(info)
    return infos


@pytest.mark.parametrize("use_short", [False, True])
def test_refiner_checkpoint_pastes_as_one_of_the_choices(webui, checkpoints, use_short):
    from modules.processing_scripts.refiner import checkpoint_choices
    from remi.core.infotext import parse_infotext

    webui.opts.data["sd_checkpoint_dropdown_use_short"] = use_short
    try:
        choices = checkpoint_choices()
        for info in checkpoints:
            # sd_samplers_common.apply_refiner writes the short title
            text = _infotext("cat", {"Steps": 20, "Sampler": "Euler", "Refiner": info.short_title, "Refiner switch at": 0.5})
            result = parse_infotext(text, "txt2img")
            assert result["problems"] == []
            refiner = result["scripts"]["refiner"]
            assert refiner["enable"] is True and refiner["switch_at"] == 0.5
            assert refiner["checkpoint"] in choices
            assert refiner["checkpoint"] == (info.short_title if use_short else info.name)
            # ...and the pasted values validate for a new job
            from remi.core.tabs import processing_args

            processing_args("txt2img", dict(result["core"]), result["scripts"])
    finally:
        webui.opts.data.pop("sd_checkpoint_dropdown_use_short", None)


def test_unknown_refiner_checkpoint_is_left_alone(webui, checkpoints):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("cat", {"Steps": 20, "Refiner": "missing-model [0123456789]", "Refiner switch at": 0.5}), "txt2img")
    assert "checkpoint" not in result["scripts"]["refiner"] and result["problems"] == []


# endregion

# region Inpaint area


@pytest.mark.parametrize(
    ("params", "area"),
    [
        # written by processing since the round-trip fix
        ({"Mask blur": 4, "Inpaint area": "Whole picture"}, "Whole picture"),
        ({"Mask blur": 4, "Inpaint area": "Only masked", "Masked area padding": 32}, "Only masked"),
        # inpaint infotexts written before: no key means the whole picture
        ({"Mask blur": 4}, "Whole picture"),
        ({"Masked content": "latent noise"}, "Whole picture"),
        ({"Mask mode": "Inpaint not masked"}, "Whole picture"),
    ],
)
def test_inpaint_area_round_trips(webui, params, area):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("dog", {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.75, **params}), "img2img")
    assert result["problems"] == [] and result["core"]["inpaint_full_res"] == area


def test_plain_img2img_infotext_leaves_inpaint_area_alone(webui):
    from remi.core.infotext import parse_infotext

    result = parse_infotext(_infotext("dog", {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.75}), "img2img")
    assert "inpaint_full_res" not in result["core"]


def test_whole_picture_inpaint_writes_inpaint_area():
    """processing writes the key for "Whole picture" (the form's default is "Only masked")."""
    import inspect

    from modules.processing import StableDiffusionProcessingImg2Img

    src = inspect.getsource(StableDiffusionProcessingImg2Img.init)
    assert 'self.extra_generation_params["Inpaint area"] = "Whole picture"' in src
    assert 'self.extra_generation_params["Inpaint area"] = "Only masked"' in src


# endregion
