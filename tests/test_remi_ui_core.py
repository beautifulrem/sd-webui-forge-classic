"""remi.core: the generation forms, named-value entry points, infotext
parsing, conflict locks and presets."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from remi import ui
from remi.testing import webui_state

TXT2IMG_LEGACY_DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "styles": [],
    "batch_count": 1,
    "batch_size": 1,
    "cfg_scale": 6.0,
    "distilled_cfg_scale": 3.0,
    "height": 1024,
    "width": 1024,
    "enable_hr": False,
    "denoising_strength": 0.6,
    "hr_scale": 2.0,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 32,  # the default UI preset's ("sd") Hires steps
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_checkpoint_name": "Use same checkpoint",
    "hr_additional_modules": ["Use same choices"],
    "hr_sampler_name": "Use same sampler",
    "hr_scheduler": "Use same scheduler",
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "hr_cfg": 6.0,
    "hr_distilled_cfg": 3.0,
    "override_settings": [],
}


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def anima_checkpoint(monkeypatch):
    from modules import sd_models

    monkeypatch.setattr(sd_models.model_data, "forge_loading_parameters", {"checkpoint_info": SimpleNamespace(filename="anima.safetensors")}, raising=False)


# region Forms


def test_txt2img_form_has_the_legacy_inputs_and_defaults(webui):
    from modules.txt2img import LEGACY_ARGS
    from remi.core.tabs import core_layout

    layout = core_layout("txt2img")
    assert set(layout.names) == {*LEGACY_ARGS, "script"}
    defaults = layout.defaults()
    assert {name: defaults[name] for name in LEGACY_ARGS} == TXT2IMG_LEGACY_DEFAULTS
    assert defaults["script"] == "None"


def test_img2img_and_extras_forms(webui):
    from remi.core.tabs import core_layout

    img2img = core_layout("img2img")
    defaults = img2img.defaults()
    assert defaults["mode"] == "img2img" and defaults["resize_mode"] == "Crop and Resize" and defaults["inpaint_full_res"] == "Only masked"
    args = dict(zip(img2img.names, img2img.values_to_args({"mode": "Inpaint upload", "resize_tab": "Resize by", "inpainting_fill": "latent noise"})))
    assert (args["mode"], args["resize_tab"], args["inpainting_fill"], args["resize_mode"], args["inpaint_full_res"]) == (4, 1, 2, 1, 1)
    assert core_layout("extras").defaults() == {"mode": "Single Image", "image": None, "image_batch": [], "input_dir": "", "output_dir": "", "show_extras_results": True, "video_input": ""}


@pytest.mark.parametrize("tab", ["txt2img", "img2img", "extras"])
def test_tab_schema_is_strict_json(webui, tab):
    from remi.core.tabs import tab_schema

    schema = tab_schema(tab)
    text = json.dumps(schema, allow_nan=False)
    assert schema["tab"] == tab and schema["core"]["owner"] == "core"
    slots = {node["section"] for node in _walk(schema["core"]["layout"]) if node["type"] == "slot"}
    assert "scripts" in slots
    if tab != "extras":
        assert {"seed", "sampler", "accordions", "cfg"} <= slots
        assert {entry["section"] for entry in schema["scripts"]} - {None} <= slots
        assert "sampler.sampler_name" in schema["conflict_params"]
    assert "NaN" not in text


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_size_and_accordion_blocks_stack(webui, tab):
    """Width/height (and the swap button) share a row, the batch sliders get
    their own, and the accordions (Hires. fix, script panels) stack."""
    from remi.core.tabs import tab_schema

    blocks = tab_schema(tab)["core"]["layout"]["children"]

    def holding(names):
        return next(block for block in blocks if names <= {node.get("name") for node in _walk(block)})

    def row_of(block, name):
        return next(node for node in _walk(block) if node["type"] == "row" and name in {child.get("name") for child in node["children"]})

    size = holding({"width", "batch_count"})
    assert size["type"] == "column"
    assert row_of(size, "width") is row_of(size, "height") is not row_of(size, "batch_count")
    assert row_of(size, "batch_count") is row_of(size, "batch_size")
    accordions = next(block for block in blocks if any(node.get("section") == "accordions" for node in _walk(block) if node["type"] == "slot"))
    assert accordions["type"] == "column"


def test_categories_follow_the_user_order(webui):
    from remi.core.forms import txt2img_form

    webui.opts.data["ui_reorder_list"] = ["seed", "prompt"]
    try:
        first = [node for node in txt2img_form().items]
        assert first[0].type == "slot" and first[0].section == "seed"
    finally:
        webui.opts.data["ui_reorder_list"] = []


def _walk(node):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)
    if "enable" in node:
        yield node["enable"]


# endregion

# region Entry points


def test_named_txt2img_builds_the_same_processing_as_the_legacy_inputs(webui):
    from modules import txt2img

    p = txt2img.prepare_txt2img({"prompt": "cat", "enable_hr": True, "hr_scale": 1.5, "hr_sampler_name": "Use same sampler"}, {"seed": {"seed": 3}})
    legacy_values = {**TXT2IMG_LEGACY_DEFAULTS, "prompt": "cat", "enable_hr": True, "hr_scale": 1.5}
    q = txt2img.txt2img_create_processing("", None, *[legacy_values[name] for name in txt2img.LEGACY_ARGS], *p.script_args)
    for key in vars(p):
        if not key.startswith("_"):
            assert repr(getattr(p, key)) == repr(getattr(q, key)) or key == "script_args_value", key
    # Assigning script_args runs the scripts' setup (the Seed script sets p.seed)
    assert p.hr_sampler_name is None and p.enable_hr and p.seed == 3
    assert p.script_args[webui.txt2img.by_id["seed"].args_from] == 3


def test_entry_points_reject_bad_values_with_every_error(webui):
    from modules import txt2img

    with pytest.raises(ui.ValidationError) as caught:
        txt2img.prepare_txt2img({"width": "wide", "cfg_scale": float("inf"), "script": "Ghost"}, {"seed": {"seed": None}})
    assert {(error["owner"], error["param"]) for error in caught.value.to_json()["errors"]} == {("core", "width"), ("core", "cfg_scale"), ("core", "script"), ("seed", "seed")}


def test_selected_script_is_checked_when_core_values_fail(webui):
    from modules import scripts
    from remi.core.tabs import processing_args

    class Picky(scripts.Script):
        def title(self):
            return "Picky"

        def params(self, is_img2img):
            return ui.Slider("loops", 1, 8, 2, 1)

    with webui_state(script_classes=[Picky]):
        with pytest.raises(ui.ValidationError) as caught:
            processing_args("txt2img", {"width": "wide", "script": "Picky"}, {"picky": {"loops": "many"}})
        assert {(error["owner"], error["param"]) for error in caught.value.to_json()["errors"]} == {("core", "width"), ("picky", "loops")}


@pytest.mark.parametrize("tab", ["txt2img", "img2img", "extras"])
def test_badly_shaped_request_bodies_are_validation_errors(webui, tab):
    from remi.core.tabs import processing_args

    for core, scripts in (("abc", None), ([1], {}), ({}, "abc"), ({}, {"seed": "x"}), ({1: 2}, None)):
        with pytest.raises(ui.ValidationError):
            processing_args(tab, core, scripts)


def test_named_img2img_resolves_uploads_and_mode(webui, anima_checkpoint):
    from modules import img2img
    from remi.core import JobContext

    image = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    mask = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    mask.paste((255, 255, 255, 255), (0, 0, 32, 64))
    uploads = {"img": image, "mask": mask}
    job = JobContext(user="me", resolve_upload=lambda node, upload_id: uploads[upload_id], extra={"remi_job": 1})
    p, batch = img2img.prepare_img2img({"mode": "Inpaint", "init_img": "img", "mask": "mask", "width": 512}, {}, job)
    assert not batch["is_batch"] and p.user == "me" and p.remi_job == 1 and p.width == 512
    assert p.init_images[0].size == (64, 64)
    assert p.image_mask.getpixel((0, 0))[0] == 255 and p.image_mask.getpixel((63, 0))[0] == 0
    with pytest.raises(ui.ValidationError):
        img2img.prepare_img2img({"mode": "Painting"}, {}, job)


def test_img2img_masks_and_required_images(webui, anima_checkpoint):
    from modules import img2img
    from remi.core import JobContext

    image = Image.new("RGB", (64, 64), (10, 20, 30))
    # An uploaded white-on-black L mask: read by luminance though the form opens it as RGBA
    plain = Image.new("L", (64, 64), 0)
    plain.paste(255, (32, 0, 64, 64))
    uploads = {"img": image, "plain": plain}
    job = JobContext(resolve_upload=lambda node, upload_id: uploads[upload_id])
    p, _batch = img2img.prepare_img2img({"mode": "Inpaint upload", "init_img": "img", "mask_upload": "plain"}, {}, job)
    assert p.image_mask.getpixel((40, 10))[0] == 255 and p.image_mask.getpixel((10, 10))[0] == 0

    def errors(core):
        with pytest.raises(ui.ValidationError) as caught:
            img2img.prepare_img2img(core, {}, job)
        return {(error["param"], error["message"]) for error in caught.value.to_json()["errors"]}

    assert errors({"mode": "Inpaint", "init_img": "img"}) == {("mask", "paint a mask over the image")}
    assert errors({"mode": "Inpaint upload", "init_img": "img", "mask": "plain"}) == {("mask_upload", "a mask image is required")}
    assert errors({"mode": "Inpaint sketch"}) == {("init_img", "an image is required"), ("sketch", "paint the area to inpaint")}
    assert errors({"mode": "img2img"}) == {("init_img", "an image is required")}
    assert errors({"mode": "Batch"}) == {("batch_upload", "upload the images to process")}


def test_compose_inputs_matches_each_legacy_mode():
    from modules.img2img import compose_inputs

    base = Image.new("RGBA", (100, 100), (0, 0, 255, 255))
    sketch = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    sketch.paste((255, 0, 0, 255), (40, 40, 60, 60))
    drawn = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    drawn.paste((255, 255, 255, 200), (0, 0, 50, 100))
    uploaded = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    uploaded.paste((255, 255, 255, 255), (50, 0, 100, 100))

    image, mask = compose_inputs(0, base, sketch, drawn)
    assert image.getpixel((50, 50)) == (0, 0, 255, 255) and mask is None
    image, mask = compose_inputs(1, base, sketch, None)
    assert image.getpixel((50, 50)) == (255, 0, 0, 255) and mask is None
    image, mask = compose_inputs(2, base, None, drawn)
    assert mask.getpixel((10, 10))[0] == 255 and mask.getpixel((90, 10))[0] == 0
    image, mask = compose_inputs(3, base, sketch, None)
    assert image.getpixel((50, 50))[:3] == (255, 0, 0)
    assert mask.getpixel((50, 50))[0] == 255 and mask.getpixel((39, 50))[0] == 255 and mask.getpixel((10, 10))[0] == 0  # dilated
    image, mask = compose_inputs(4, base, None, uploaded)
    assert mask.getpixel((75, 50))[0] == 255 and mask.getpixel((25, 50))[0] == 0
    assert compose_inputs(5, None, None, None) == (None, None)


def test_legacy_img2img_inputs_map_to_the_shared_canvas():
    from modules.img2img import _legacy_core

    names = ("init_img", "sketch", "sketch_fg", "init_img_with_mask", "init_img_with_mask_fg", "inpaint_color_sketch", "inpaint_color_sketch_fg", "init_img_inpaint", "init_mask_inpaint")
    values = {name: name for name in names}
    values.update({name: 0 for name in ("prompt", "negative_prompt", "mask_blur", "inpainting_fill", "batch_size", "cfg_scale", "distilled_cfg_scale", "image_cfg_scale", "denoising_strength", "height", "width", "scale_by", "resize_mode", "inpaint_full_res", "inpaint_full_res_padding", "inpainting_mask_invert", "selected_scale_tab", "n_iter", "prompt_styles", "override_settings_texts")})
    values.update({name: None for name in ("img2img_batch_input_dir", "img2img_batch_output_dir", "img2img_batch_inpaint_mask_dir", "img2img_batch_use_png_info", "img2img_batch_png_info_props", "img2img_batch_png_info_dir", "img2img_batch_source_type", "img2img_batch_upload")})
    layers = {mode: tuple(_legacy_core(mode, values)[key] for key in ("init_img", "sketch", "mask", "mask_upload")) for mode in range(6)}
    assert layers == {
        0: ("init_img", None, None, None),
        1: ("sketch", "sketch_fg", None, None),
        2: ("init_img_with_mask", None, "init_img_with_mask_fg", None),
        3: ("inpaint_color_sketch", "inpaint_color_sketch_fg", None, None),
        4: ("init_img_inpaint", None, None, "init_mask_inpaint"),
        5: (None, None, None, None),
    }


def test_core_actions(webui):
    from remi.core.actions import core_actions

    ctx = ui.ActionContext(tab="txt2img")
    assert ui.call_action(core_actions, "swap_dimensions", ctx, {"width": 832, "height": 1216}) == {"width": {"value": 1216}, "height": {"value": 832}}
    patch = ui.call_action(core_actions, "hr_resolution", ctx, {"enable_hr": True, "width": 832, "height": 1216, "hr_scale": 1.5})
    assert "832x1216" in patch["__outputs__"]["hr_final_resolution"] and "1280x1856" in patch["__outputs__"]["hr_final_resolution"]
    assert ui.call_action(core_actions, "hr_resolution", ctx, {"enable_hr": False})["__outputs__"]["hr_final_resolution"] == ""
    # Missing values are the form's defaults; bad ones are rejected
    assert ui.call_action(core_actions, "swap_dimensions", ctx, {"width": 512}) == {"width": {"value": 1024}, "height": {"value": 512}}
    for action, values in (("swap_dimensions", {"width": None}), ("hr_resolution", {"hr_scale": "big"})):
        with pytest.raises(ui.ValidationError):
            ui.call_action(core_actions, action, ctx, values)
    with pytest.raises(ui.ValidationError):
        ui.call_action(core_actions, "__preset__", ctx, {})


# endregion

# region Infotext


def _write_infotext(prompt, negative, params):
    """An infotext as processing.create_infotext writes one."""

    from modules.infotext_utils import quote

    return f"{prompt}\nNegative prompt: {negative}\n" + ", ".join(f"{key}: {quote(value)}" for key, value in params.items())


def test_infotext_round_trip(webui):
    from remi.core.infotext import parse_infotext

    text = _write_infotext(
        "a cat, (sitting:1.2)",
        "blurry",
        {
            "Steps": 30,
            "Sampler": "Euler",
            "Schedule type": "Simple",
            "CFG scale": 4.5,
            "Seed": 1234,
            "Size": "832x1216",
            "Denoising strength": 0.35,
            "Hires upscale": 1.5,
            "Hires upscaler": "Latent",
            "Hires steps": 12,
            "Variation seed": 99,
            "Variation seed strength": 0.25,
            "Refiner switch at": 0.8,
            "Clip skip": 2,
            "Emphasis": "No norm",
        },
    )
    result = parse_infotext(text, "txt2img")
    json.dumps(result, allow_nan=False)
    core, scripts = result["core"], result["scripts"]
    assert core["prompt"] == "a cat, (sitting:1.2)" and core["negative_prompt"] == "blurry"
    assert (core["width"], core["height"], core["cfg_scale"]) == (832, 1216, 4.5)
    assert core["enable_hr"] is True and core["hr_scale"] == 1.5 and core["hr_second_pass_steps"] == 12 and core["denoising_strength"] == 0.35
    assert scripts["seed"] == {"seed": 1234, "subseed_show": True, "subseed": 99, "subseed_strength": 0.25}
    assert scripts["sampler"] == {"steps": 30, "sampler_name": "Euler", "scheduler": "Simple"}
    assert scripts["refiner"]["switch_at"] == 0.8 and scripts["refiner"]["enable"] is False
    # Settings no param reads become override settings (Clip skip is never pasted)
    assert core["override_settings"] == ["Emphasis: No norm"]
    assert result["params"]["Steps"] == "30" and result["problems"] == []

    # The parsed values validate and produce the same infotext keys again
    from remi.core.tabs import processing_args

    core_args, script_args = processing_args("txt2img", {key: value for key, value in core.items()}, scripts)
    assert core_args["width"] == 832 and script_args[webui.txt2img.by_id["seed"].args_from] == 1234


def test_infotext_for_img2img_and_both_tabs(webui):
    from remi.core.infotext import parse_infotext

    text = _write_infotext("dog", "", {"Steps": 20, "Sampler": "Euler", "Mask blur": 8, "Mask mode": "Inpaint not masked", "Masked content": "fill", "Inpaint area": "Only masked", "Masked area padding": 64, "Denoising strength": 0.5})
    result = parse_infotext(text, "img2img")
    assert {key: result["core"][key] for key in ("mask_blur", "inpainting_mask_invert", "inpainting_fill", "inpaint_full_res", "inpaint_full_res_padding", "denoising_strength")} == {"mask_blur": 8, "inpainting_mask_invert": "Inpaint not masked", "inpainting_fill": "fill", "inpaint_full_res": "Only masked", "inpaint_full_res_padding": 64, "denoising_strength": 0.5}
    # Without a tab, the img2img-only keys select img2img
    detected = parse_infotext(text)
    assert detected["tab"] == "img2img" and detected["core"]["mask_blur"] == 8 and detected["scripts"]["sampler"]["steps"] == 20
    hires = _write_infotext("dog", "", {"Steps": 20, "Denoising strength": 0.5, "Hires upscale": 2})
    assert parse_infotext(hires)["tab"] == "txt2img" and parse_infotext(hires)["core"]["enable_hr"] is True
    assert parse_infotext(_write_infotext("dog", "", {"Steps": 20, "Sampler": "Euler", "Denoising strength": 0.5}))["tab"] == "img2img"
    assert parse_infotext("cat\nSteps: 20")["tab"] == "txt2img"


def test_infotext_pasted_callbacks_run(webui, monkeypatch):
    from modules import script_callbacks
    from remi.core.infotext import parse_infotext

    def rewrite(text, params):
        params["Seed"] = "77"

    monkeypatch.setattr(script_callbacks, "infotext_pasted_callback", lambda text, params: rewrite(text, params))
    assert parse_infotext("x\nSteps: 20, Sampler: Euler, Seed: 1", "txt2img")["scripts"]["seed"]["seed"] == 77


# endregion

# region Conflicts


def test_resolve_conflict_locks_mirrors_the_panels():
    from modules import anima_feature_conflicts as conflicts

    locks = conflicts.resolve_conflict_locks(
        {
            conflicts.ROLE_SAMPLER: "Anima Flow Euler",
            conflicts.ROLE_SCHEDULER: "Beta",
            conflicts.ROLE_DYNAMIC_SHIFT: True,
            conflicts.ROLE_GUIDANCE_MODE: "APG",
            conflicts.ROLE_MOMENTUM: True,
            conflicts.ROLE_DCW: False,
        },
        selected=conflicts.ROLE_MOMENTUM,
    )
    assert locks.values[conflicts.ROLE_GUIDANCE_MODE] == conflicts.STANDARD_GUIDANCE and locks.values[conflicts.ROLE_MOMENTUM] is True
    assert locks.choices[conflicts.ROLE_GUIDANCE_MODE] == (conflicts.STANDARD_GUIDANCE,)
    assert locks.values[conflicts.ROLE_DYNAMIC_SHIFT] is False and locks.enabled[conflicts.ROLE_SCHEDULER] is False
    assert any("Dynamic Shift disabled" in message for message in locks.messages)

    # Without the guidance roles only the present rules apply
    locks = conflicts.resolve_conflict_locks({conflicts.ROLE_SAMPLER: "Euler", conflicts.ROLE_SCHEDULER: "Karras", conflicts.ROLE_DYNAMIC_SHIFT: True}, selected=conflicts.ROLE_SCHEDULER)
    assert locks.values == {conflicts.ROLE_SCHEDULER: "Karras", conflicts.ROLE_DYNAMIC_SHIFT: False}

    locks = conflicts.resolve_conflict_locks({conflicts.ROLE_FREEFUSE: True, conflicts.ROLE_REFINER: True, conflicts.ROLE_REFINER_CHECKPOINT: "low.safetensors"}, selected=conflicts.ROLE_REFINER)
    assert locks.values == {conflicts.ROLE_FREEFUSE: False, conflicts.ROLE_REFINER: True, conflicts.ROLE_REFINER_CHECKPOINT: "low.safetensors"}
    assert locks.enabled[conflicts.ROLE_FREEFUSE] is False
    assert conflicts.resolve_conflict_locks({}) == conflicts.ConflictLocks()
    assert conflicts.exclusive_group_order(conflicts.SPATIAL_GROUP, ["artist", "custom", "freefuse"]) == ("freefuse", "artist", "custom")


def _anima_layouts():
    """Layouts of ported Anima panels as they declare their params."""

    from modules.anima_feature_conflicts import SPATIAL_GROUP

    return {
        "anima_guidance": ui.ScriptLayout(
            ui.Group(
                ui.Choice("mode", ["Standard / preserve existing", "APG", "FDG (experimental)"], preset="guidance.mode"),
                ui.Toggle("momentum", preset="guidance.momentum_enabled"),
                ui.Toggle("dcw", preset="guidance.dcw_enabled"),
            ),
            owner="anima_guidance",
        ),
        "anima_freefuse": ui.ScriptLayout(ui.Section("FreeFuse", enable=ui.Toggle("enable", preset="freefuse.enabled"), exclusive=(SPATIAL_GROUP, "freefuse")), owner="anima_freefuse"),
        "anima_regional": ui.ScriptLayout(ui.Section("Regional", enable=ui.Toggle("enable", preset="regional.enabled"), exclusive=(SPATIAL_GROUP, "regional")), owner="anima_regional"),
        "artist_mixer": ui.ScriptLayout(ui.Group(ui.Toggle("enable", preset="artist.enabled", exclusive=(SPATIAL_GROUP, "artist")), ui.Choice("optimization", ["平衡", "Quality"], preset=("artist.optimization", "平衡"))), owner="artist_mixer"),
    }


def _tab_layouts(webui):
    from remi.core.tabs import core_layout

    layouts = {"core": core_layout("txt2img")}
    layouts.update({script.script_id: script.layout for script in webui.txt2img.scripts})
    layouts.update(_anima_layouts())
    return layouts


def test_conflict_patch(webui):
    from remi.core.conflicts import conflict_patch

    layouts = _tab_layouts(webui)
    values = {
        "sampler": {"sampler_name": "Euler"},
        "anima_guidance": {"mode": "APG", "momentum": True},
        "anima_freefuse": {"enable": True},
        "anima_regional": {"enable": True},
        "refiner": {"enable": True, "checkpoint": "None"},
    }
    patch = conflict_patch("txt2img", values, "anima_regional.enable", layouts=layouts)
    assert patch["anima_regional.enable"] == {"value": True, "enabled": True}
    assert patch["anima_freefuse.enable"] == {"value": False, "enabled": False}
    assert patch["artist_mixer.enable"] == {"value": False, "enabled": False}
    # The newest switch wins: Momentum was not the last change, so APG stays
    assert patch["anima_guidance.momentum"]["value"] is False
    assert patch["anima_guidance.mode"]["value"] == "APG"
    assert "Momentum disabled" in patch["__toast__"]["message"]
    # Regional won the group, so FreeFuse (now off) no longer clears the Refiner
    assert patch["refiner.enable"] == {"value": True, "enabled": True}
    assert patch["refiner.checkpoint"] == {"value": "None", "enabled": True}
    assert "Refiner cleared" not in patch["__toast__"]["message"]

    patch = conflict_patch("txt2img", {**values, "anima_regional": {"enable": False}}, "anima_guidance.momentum", layouts=layouts)
    assert patch["anima_guidance.mode"]["value"] == "Standard / preserve existing" and patch["anima_guidance.momentum"]["value"] is True
    assert patch["anima_freefuse.enable"] == {"value": True, "enabled": True} and patch["anima_regional.enable"]["enabled"] is False
    # FreeFuse (on) wins over a Refiner switched on without a checkpoint, and locks it
    assert patch["refiner.enable"] == {"value": False, "enabled": False}
    assert patch["refiner.checkpoint"] == {"value": "None", "enabled": False}
    assert "Refiner cleared" in patch["__toast__"]["message"]


def test_role_rules_and_exclusive_groups_agree(webui, monkeypatch):
    """A role rule that switches a group member off frees its group: one entry
    per control, values in order, ``enabled`` from every lock."""

    from modules import shared
    from remi.core.conflicts import conflict_patch

    monkeypatch.setattr(shared, "list_checkpoint_tiles", lambda use_short=False: ["X"])
    layouts = _tab_layouts(webui)
    values = {"anima_freefuse": {"enable": True}, "refiner": {"enable": True, "checkpoint": "X"}}
    patch = conflict_patch("txt2img", values, "refiner.checkpoint", layouts=layouts)
    assert patch["anima_freefuse.enable"] == {"value": False, "enabled": False}
    assert patch["refiner.enable"] == {"value": True, "enabled": True} and patch["refiner.checkpoint"] == {"value": "X", "enabled": True}
    # FreeFuse is off now: the spatial group is free again
    assert patch["anima_regional.enable"] == {"value": False, "enabled": True} and patch["artist_mixer.enable"] == {"value": False, "enabled": True}
    assert "FreeFuse disabled" in patch["__toast__"]["message"]

    # The next member in line takes over when the Refiner pushes FreeFuse out
    values["anima_regional"] = {"enable": True}
    patch = conflict_patch("txt2img", values, "refiner.checkpoint", layouts=layouts)
    assert patch["anima_freefuse.enable"] == {"value": False, "enabled": False}
    assert patch["anima_regional.enable"] == {"value": True, "enabled": True} and patch["artist_mixer.enable"] == {"value": False, "enabled": False}

    # Turning FreeFuse on wins over the Refiner, which is cleared and locked
    patch = conflict_patch("txt2img", {**values, "anima_regional": {"enable": False}}, "anima_freefuse.enable", layouts=layouts)
    assert patch["anima_freefuse.enable"] == {"value": True, "enabled": True}
    assert patch["refiner.enable"] == {"value": False, "enabled": False} and patch["refiner.checkpoint"] == {"value": "None", "enabled": False}


def test_conflicts_action_reads_the_other_scripts_values(webui, monkeypatch):
    from remi.core import conflicts
    from remi.core.actions import core_actions

    seen = {}
    monkeypatch.setattr(conflicts, "conflict_patch", lambda tab, values, selected: seen.update(tab=tab, values=values, selected=selected) or {})
    ctx = ui.ActionContext(tab="img2img", owner="core", scripts={"sampler": {"sampler_name": "Euler"}}, extra={"selected": "sampler.sampler_name"})
    ui.call_action(core_actions, "__conflicts__", ctx, {"width": 512})
    assert seen == {"tab": "img2img", "values": {"sampler": {"sampler_name": "Euler"}, "core": {"width": 512}}, "selected": "sampler.sampler_name"}
    with pytest.raises(ui.ValidationError):
        ui.call_action(core_actions, "__conflicts__", ui.ActionContext(tab="extras"), {})
    with pytest.raises(ui.ValidationError):
        ui.call_action(core_actions, "__conflicts__", ctx, ["not", "named"])


# endregion

# region Presets


def test_anima_preset_patch(webui):
    from modules.anima_presets import TURBO_NAG_LABEL, preset_named_values
    from remi.core.presets import anima_preset_patch

    patch = anima_preset_patch("txt2img", TURBO_NAG_LABEL, layouts=_tab_layouts(webui))
    assert patch["sampler.sampler_name"] == {"value": "Euler", "enabled": True}
    assert patch["sampler.steps"] == {"value": 10} and patch["cfg_scale"] == {"value": 1.0}
    assert patch["enable_hr"] == {"value": False, "enabled": True} and patch["refiner.enable"] == {"value": False, "enabled": True}
    assert patch["anima_guidance.momentum"] == {"value": False, "enabled": True}
    # A localized own value is kept, the table's value otherwise
    assert patch["artist_mixer.optimization"] == {"value": "平衡"}
    assert patch["__toast__"]["level"] == "success"
    with pytest.raises(KeyError):
        preset_named_values(TURBO_NAG_LABEL, [("prompt.positive", "core", "prompt", None)])


def test_forge_preset_patch():
    from remi.core.presets import forge_preset_patch, forge_preset_settings

    opts = SimpleNamespace(
        anima_t2i_step=30,
        anima_t2i_hr_step=0,
        anima_t2i_sampler="ER SDE",
        anima_t2i_scheduler="Beta",
        anima_t2i_width=832,
        anima_t2i_height=-1,
        anima_t2i_cfg=4.5,
        anima_t2i_hr_cfg=4.0,
        anima_t2i_dcfg=3.0,
        anima_t2i_hr_dcfg=3.0,
        anima_t2i_batch_size=2,
        forge_checkpoint_anima="anima.safetensors",
        forge_additional_modules_anima=["/models/vae/anima_vae.safetensors"],
        forge_unet_storage_dtype_anima="Automatic",
        sd_modules_dropdown_use_short=True,
        sd_model_checkpoint="other.safetensors",
    )
    patch = forge_preset_patch("txt2img", "anima", opts)
    assert patch["sampler.steps"] == {"value": 30} and "hr_second_pass_steps" not in patch
    assert patch["sampler.sampler_name"] == {"value": "ER SDE"} and patch["width"] == {"value": 832} and "height" not in patch
    assert patch["batch_size"] == {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size", "value": 2}
    assert patch["distilled_cfg_scale"]["value"] == 3.0 and "visible" in patch["distilled_cfg_scale"]
    assert not any(key.startswith("__") for key in patch)
    assert forge_preset_settings("anima", opts) == {"forge_preset": "anima", "sd_model_checkpoint": "anima.safetensors", "forge_additional_modules": ["anima_vae.safetensors"], "forge_unet_storage_dtype": "Automatic"}
    img2img = forge_preset_patch("img2img", "wan", SimpleNamespace(wan_i2i_batch_size=17))
    assert img2img["batch_size"]["label"] == "Frames" and img2img["batch_size"]["step"] > 1 and "hr_cfg" not in img2img


def _find(node, name):
    return next(item for item in _walk(node) if item.get("name") == name)


def test_tab_schema_and_validation_follow_the_ui_preset(webui):
    from remi.core.tabs import processing_args, tab_schema

    saved = dict(webui.opts.data)
    try:
        webui.opts.data.update(forge_preset="anima", anima_t2i_width=832, anima_t2i_height=1216)
        schema = tab_schema("txt2img")
        core = schema["core"]["defaults"]
        assert (core["width"], core["height"], core["cfg_scale"], core["hr_second_pass_steps"]) == (832, 1216, 4.0, 32)
        distilled = _find(schema["core"]["layout"], "distilled_cfg_scale")
        assert distilled["label"] == "Shift" and "visible_if" not in distilled
        sampler = next(entry for entry in schema["scripts"] if entry["id"] == "sampler")
        assert sampler["defaults"]["steps"] == 32 and sampler["defaults"]["scheduler"] == "Beta"
        # Missing names take the same defaults
        core_args, script_args = processing_args("txt2img", {}, {})
        assert core_args["width"] == 832 and script_args[webui.txt2img.by_id["sampler"].args_from] == 32

        webui.opts.data.update(forge_preset="wan")
        batch = _find(tab_schema("img2img")["core"]["layout"], "batch_size")
        assert (batch["label"], batch["maximum"], batch["step"]) == ("Frames", 241, 16)
        assert processing_args("img2img", {"batch_size": 33, "init_img": Image.new("RGB", (8, 8))}, {})[0]["batch_size"] == 33

        webui.opts.data.update(forge_preset="sd")
        assert _find(tab_schema("txt2img")["core"]["layout"], "distilled_cfg_scale")["visible_if"] is False
    finally:
        webui.opts.data.clear()
        webui.opts.data.update(saved)


def test_forge_preset_action_switches_the_model_on_the_server(webui, monkeypatch):
    import sys

    from remi.core.actions import core_actions

    calls = []
    fake = SimpleNamespace(
        module_map={},
        dtype_change=lambda dtype, preset, save, refresh: calls.append(("dtype", dtype, save, refresh)) or True,
        modules_change=lambda modules, preset, save, refresh: calls.append(("modules", modules, save, refresh)) or False,
        checkpoint_change=lambda name, preset, save, refresh: calls.append(("checkpoint", name, save, refresh)) or True,
        refresh_model_loading_parameters=lambda: calls.append(("refresh",)),
    )
    monkeypatch.setitem(sys.modules, "modules_forge.main_entry", fake)
    monkeypatch.setattr("modules_forge.main_entry", fake, raising=False)
    monkeypatch.setattr(webui.opts, "save", lambda filename: calls.append(("save",)))
    saved = dict(webui.opts.data)
    try:
        webui.opts.data.update(forge_checkpoint_anima="anima.safetensors", forge_unet_storage_dtype_anima="Automatic")
        patch = ui.call_action(core_actions, "__forge_preset__", ui.ActionContext(tab="txt2img", extra={"preset": "anima"}), {})
        assert calls == [("dtype", "Automatic", False, False), ("modules", [], False, False), ("checkpoint", "anima.safetensors", False, False), ("save",), ("refresh",)]
        assert patch["__settings__"]["forge_preset"] == "anima" and webui.opts.forge_preset == "anima"
        assert patch["cfg_scale"] == {"value": 4.0} and "sampler.steps" in patch
        for extra in ({}, {"preset": "nope"}):
            with pytest.raises(ui.ValidationError):
                ui.call_action(core_actions, "__forge_preset__", ui.ActionContext(tab="txt2img", extra=extra), {})
    finally:
        webui.opts.data.clear()
        webui.opts.data.update(saved)


# endregion

# region Declarative Anima helpers


def test_table_params_binds_infotext_and_presets_in_table_order():
    from modules.anima_support import PasteKey, table_params

    root = ui.Section("DAVE", enable=ui.Toggle("enabled"), children=[ui.Row(ui.Slider("tau", 0, 1, 0.1, 0.01), ui.Slider("strength", 0, 1, 0.3, 0.05)), ui.Text("blocks", "8-18")])
    key = PasteKey("DAVE blocks", lambda params: params.get("DAVE blocks", "").upper())
    order = table_params([("enabled", "DAVE", "dave.enabled"), ("strength", "DAVE strength", "dave.strength"), ("tau", "DAVE tau", "dave.tau"), ("blocks", key, "dave.blocks")], root)
    layout = ui.ScriptLayout(root, order=order)
    assert layout.names == ["enabled", "strength", "tau", "blocks"]
    assert layout.infotext == {"DAVE": "enabled", "DAVE strength": "strength", "DAVE tau": "tau", "DAVE blocks": "blocks"}
    assert set(layout.presets) == {"dave.enabled", "dave.strength", "dave.tau", "dave.blocks"}
    assert layout.paste({"DAVE blocks": "a-b"})[0] == {"blocks": "A-B"}
    with pytest.raises(KeyError):
        table_params([("enabled", None, None)], root)
    with pytest.raises(KeyError):
        table_params([("enabled", None, "dave.nope"), ("strength", None, None), ("tau", None, None), ("blocks", None, None)], ui.Section("X", enable=ui.Toggle("enabled"), children=[ui.Toggle("strength"), ui.Toggle("tau"), ui.Toggle("blocks")]))


def test_anima_patch_panel_params():
    from modules import scripts
    from modules.anima_support import AnimaPatchPanel

    class Panel(AnimaPatchPanel, scripts.Script):
        INFOTEXT_KEYS = ("XB", "XB strength")
        PRESET_FIELDS = ("xattn_boost.enabled", "xattn_boost.strength")

        def title(self):
            return "Boost"

        def params(self, is_img2img):
            return self.panel_params(ui.Section("Boost", ui.Slider("strength", 0, 4, 2, 0.1), enable=ui.Toggle("enabled")))

    panel = Panel()
    root = panel.params(False)
    layout = ui.ScriptLayout(root, order=panel.param_order)
    assert Panel.panel_param_names() == ("enabled", "strength") and layout.names == ["enabled", "strength"]
    assert layout.infotext == {"XB": "enabled", "XB strength": "strength"}


# endregion


# region Jobs


@pytest.fixture
def job_runtime(monkeypatch):
    """A fresh ``shared.state`` and the progress bookkeeping the entry points use."""

    from modules import shared, shared_state

    monkeypatch.setattr(shared, "state", shared_state.State())
    monkeypatch.setattr(shared, "total_tqdm", SimpleNamespace(clear=lambda: None))
    return shared.state


def test_entry_points_set_up_the_job_state(webui, job_runtime, monkeypatch):
    from modules import processing, progress, txt2img
    from remi.core import JobContext

    seen = {}

    def fake_process(p):
        seen.update(interrupted=job_runtime.interrupted, job=job_runtime.job, task=progress.current_task)
        job_runtime.interrupted = True  # interrupted while it runs
        return SimpleNamespace(js=lambda: "{}")

    monkeypatch.setattr(processing, "process_images", fake_process)
    job_runtime.interrupted = job_runtime.skipped = True  # left over from an earlier job
    txt2img.run_txt2img({"prompt": "cat"}, {}, JobContext(id_task="task(remi-1)"))
    assert seen == {"interrupted": False, "job": "task(remi-1)", "task": "task(remi-1)"}
    assert not job_runtime.interrupted and not job_runtime.skipped and progress.current_task is None
    assert "task(remi-1)" in progress.finished_tasks


def _extras_operations():
    gr = pytest.importorskip("gradio")
    from modules import scripts_postprocessing

    class Scale(scripts_postprocessing.ScriptPostprocessing):
        name, order = "Scale", 1

        def ui(self):
            return {"factor": gr.Slider(1, 4, value=2, step=1, label="Factor"), "method": gr.Dropdown(["nearest", "bilinear"], value="bilinear", type="index", label="Method")}

        def process(self, pp, factor, method):
            pp.image = pp.image.resize((pp.image.width * factor, pp.image.height * factor))
            pp.info["Scale"] = f"{factor}x method {method}"

    class Tint(scripts_postprocessing.ScriptPostprocessing):
        name, order = "Tint", 2

        def params(self):
            return ui.Section("Tint", enable=ui.Toggle("enable", False), children=[ui.Color("color", "#ff0000")])

        def process(self, pp, enable, color):
            if enable:
                pp.info["Tint"] = color

    class Broken(scripts_postprocessing.ScriptPostprocessing):
        name, order = "Broken", 3

        def ui(self):
            return {"x": [][1]}

    return [Scale, Tint, Broken]


def test_extras_with_operations(job_runtime, monkeypatch):
    from modules import images, postprocessing
    from remi.core.tabs import tab_schema

    with webui_state({"use_original_name_batch": True}, postprocessing=_extras_operations()) as state:
        schema = tab_schema("extras")
        entries = {entry["id"]: entry for entry in schema["scripts"]}
        # The failing operation is dropped; the others keep working
        assert list(entries) == ["scale", "tint"] and [script.name for script in state.postproc.scripts] == ["Scale", "Tint"]
        assert entries["scale"]["legacy"] and entries["scale"]["args"] == ["factor", "method"] and entries["scale"]["defaults"] == {"factor": 2, "method": "bilinear"}
        assert not entries["tint"]["legacy"] and entries["tint"]["args"] == ["enable", "color"]

        image = Image.new("RGB", (8, 8))
        outputs, info = postprocessing.run_extras_named({"image": image}, {"scale": {"factor": 3, "method": "nearest"}, "tint": {"enable": True}}, save_output=False)
        assert outputs[0].size == (24, 24) and "3x method 0" in info and "#ff0000" in info

        saved = []
        monkeypatch.setattr(images, "save_image", lambda image, **kwargs: saved.append(kwargs["forced_filename"]) or ("", None))
        named, unnamed = Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))
        named.filename = "/uploads/holiday.png"
        postprocessing.run_extras_named({"mode": "Batch Process", "image_batch": [named, unnamed]}, {})
        assert saved == ["holiday", None]

        with pytest.raises(ui.ValidationError) as caught:
            postprocessing.run_extras_named({"mode": "Single Image"}, {"scale": {"method": "cubic"}, "ghost": {}})
        assert {(error["owner"], error["param"]) for error in caught.value.to_json()["errors"]} == {("core", "image"), ("scale", "method"), ("ghost", None)}


# endregion
