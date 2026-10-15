"""The ported ControlNet panel: ``params()`` declares the processing
arguments the Gradio ``ui()`` did (one ``ControlNetUnit`` per unit), the
unit JSON of the ``controlnet-unit`` widget becomes the same ``ControlNetUnit``
the Gradio controls built, infotext still round-trips, and the widget's
actions (preview, pose, region canvas, model refresh) work without Gradio.

Legacy record: ``tests/fixtures/legacy_ui_<tab>.json`` (``args``: the
positional arguments; ``unit_controls``: the controls of one unit), taken from
the Gradio ``ui()`` before the port (and before the Gradio UI was removed).

Intentional differences from the Gradio panel (the widget's behaviour,
webui/src/widgets/controlnet-unit, tested there):

- a blank canvas ("Create New Canvas") is the shared image editor's: white
  (it was black, ``np.zeros``) and the generation size (no size sliders);
- any new control image switches the unit on, a blank canvas included (the
  old JS clicked "Enable" on a file upload only); "Generate Canvas" does not;
- "Preview as Input" is a visible switch whenever there is a preview (it was
  hidden and only set by applying a pose, which still sets it);
- switching the control type to "Region" keeps the control image (it was
  cleared; "Generate Canvas" replaces it);
- the enabled units show as an "Nx Unit" summary in the section header (the
  old accordion badge) and as a line of buttons inside the unit (the old
  highlighted tab headers);
- the batch directory is ignored under ``--hide-ui-dir-config`` (it was read
  from a hidden textbox).
"""

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import remi_port_harness as harness
from PIL import Image

EXTENSION = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION / "scripts" / "controlnet.py"
CONTRACT = harness.REPO / "webui" / "src" / "widgets" / "controlnet-unit" / "contract.json"
LEGACY_PREPROCESSORS = harness.REPO / "extensions-builtin" / "forge_legacy_preprocessors"
MODELS = ["None", "control_v11p_sd15_canny", "diffusers_xl_depth_full", "t2i_union_promax"]


@pytest.fixture(scope="module")
def cn():
    from remi.testing import webui_state

    if str(LEGACY_PREPROCESSORS) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(LEGACY_PREPROCESSORS))  # annotator (pose drawing)
    with webui_state():
        import modules_forge.supported_preprocessor  # noqa: F401 - None and canny

        module = harness.load_script(SCRIPT, "controlnet_params_under_test")
        yield module


@pytest.fixture
def registry(cn, monkeypatch):
    """A known preprocessor registry and model list (restored afterwards)."""

    from lib_controlnet import global_state
    from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter

    class NoModel(Preprocessor):
        def __init__(self):
            super().__init__()
            self.name = "reference_test"
            self.tags = ["Reference"]
            self.do_not_need_model = True
            self.show_control_mode = False
            self.slider_resolution = PreprocessorParameter(visible=False)
            self.slider_1 = PreprocessorParameter(label="Style Fidelity", value=0.5, minimum=0.0, maximum=1.0, step=0.01, visible=True)

    saved = dict(global_state.supported_preprocessors)
    base = {name: saved[name] for name in ("None", "canny")}
    global_state.supported_preprocessors.clear()
    global_state.supported_preprocessors.update(base, reference_test=NoModel())
    global_state.get_sorted_preprocessors.cache_clear()
    monkeypatch.setattr(global_state, "controlnet_names", list(MODELS))
    monkeypatch.setattr(global_state, "controlnet_filename_dict", {name: None if name == "None" else f"/models/{name}.safetensors" for name in MODELS})
    yield global_state
    global_state.supported_preprocessors.clear()
    global_state.supported_preprocessors.update(saved)
    global_state.get_sorted_preprocessors.cache_clear()


def layout(cn, tab="txt2img"):
    return harness.script_layout(cn.ControlNetForForgeOfficial(), tab)


def plain(value):
    return value.value if hasattr(value, "value") and not isinstance(value, (bool, int, float, str)) else value


def settings_of(unit):
    from lib_controlnet.unit_param import SETTINGS

    return {name: plain(getattr(unit, name)) for name in SETTINGS}


def png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def files(tmp_path):
    from remi.files import FileStore

    store = FileStore(tmp_path / "files")
    yield store
    store.close()


def resolver(files):
    return lambda node, file_id: files.resolve_upload(node, file_id)


def call(script, action, values, files, tab="txt2img", core=None, **extra):
    from remi import ui

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=SimpleNamespace(files=files))))
    ctx = ui.ActionContext(tab=tab, owner="controlnet", core=core or {"width": 512, "height": 512}, request=request, extra=extra)
    return ui.call_action(script, action, ctx, values)


# region Parity with the Gradio ui()


@pytest.mark.parametrize("tab", harness.TABS)
def test_arguments_match_the_legacy_ui(cn, tab):
    """Each unit is still one argument; it is now the unit JSON (a custom
    ``controlnet-unit`` param, read from its infotext entry ``ControlNet
    <i>``) instead of a ``gr.State``. img2img adds the batch directory,
    formerly a Gradio textbox outside the script's arguments."""

    from lib_controlnet.unit_param import default_value

    snapshot = harness.load_snapshot(SCRIPT, tab)
    count = len(snapshot["args"])
    changes = {i: {"kind": "custom", "default": default_value(tab == "img2img"), "infotext": f"ControlNet {i}", "widget": "controlnet-unit", "props": ...} for i in range(count)}
    if tab == "img2img":
        changes[count] = {
            "kind": "text",
            "label": "Controlnet input directory",
            "default": "",
            "infotext": None,
            "paste": False,
            "preset": None,
            "exclusive": None,
            "visible_if": {"param": "core.mode", "eq": "Batch"},
            "enabled_if": None,
        }
    harness.assert_matches_snapshot(layout(cn, tab), snapshot["args"], changes)

    # The unit's settings start where the Gradio unit's did (images: none)
    for legacy in snapshot["args"]:
        default = default_value(tab == "img2img")
        assert {key: default[key] for key in legacy["default"] if key in default} == {key: value for key, value in legacy["default"].items() if key != "_idx"}


LEGACY_FIELDS = {
    "upload_independent_img_in_img2img": "upload_independent",
    "region_num": "region_subjects",
    "region_height": "region_vertical",
}
# Dropped: the preview shows whenever there is one ("Allow Preview"); a blank
# canvas takes the generation size (the "New Canvas" sliders); the hidden
# detect-map switch stays a unit field (save_detected_map, default on).
DROPPED = {"preprocessor_preview", "new_canvas_width", "new_canvas_height", "save_detected_map"}
# Bounds that follow the preprocessor (catalog sliders), not the control.
DYNAMIC = {"processor_res", "threshold_a", "threshold_b"}
# Choices that follow the registry and model folders (catalog).
CATALOG_CHOICES = {"type_filter", "module", "model"}


@pytest.mark.parametrize("tab", harness.TABS)
def test_unit_controls_match_the_legacy_ui(cn, tab):
    from lib_controlnet.external_code import ControlNetUnit
    from lib_controlnet.unit_param import CONTROLS, default_value

    snapshot = harness.load_snapshot(SCRIPT, tab)
    default = default_value(tab == "img2img")
    seen = set()
    for legacy in snapshot["unit_controls"]:
        field = legacy["field"]
        if field in DROPPED:
            continue
        key = LEGACY_FIELDS.get(field, field)
        control = CONTROLS[key]
        seen.add(key)
        assert control["label"] == legacy["label"], field
        assert control.get("info") == legacy.get("info"), field
        if "minimum" in legacy and key not in DYNAMIC:
            assert (control["minimum"], control["maximum"], control["step"]) == (legacy["minimum"], legacy["maximum"], legacy["step"]), field
        if "choices" in legacy and key not in CATALOG_CHOICES:
            assert control["choices"] == legacy["choices"], field
        if key in default:
            assert default[key] == legacy["default"], field
        elif key == "timestep_range":
            assert [default["guidance_start"], default["guidance_end"]] == legacy["default"]
        elif key.startswith("region_"):
            assert control["default"] == legacy["default"], field
    expected = {key for key in CONTROLS if tab == "img2img" or key != "upload_independent"}
    assert seen == expected
    # The infotext fields the Gradio controls were registered under are what a paste reads
    fields = {key.split(" ", 2)[2] for key in snapshot["infotext_keys"]}
    assert fields == set(ControlNetUnit.infotext_fields())


@pytest.mark.parametrize(("count", "tab", "names"), [(1, "txt2img", ["unit_0"]), (5, "img2img", ["unit_0", "unit_1", "unit_2", "unit_3", "unit_4", "batch_input_dir"])])
def test_the_unit_count_follows_the_setting(cn, monkeypatch, count, tab, names):
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "control_net_unit_count", count)
    assert layout(cn, tab).names == names


def test_default_arguments_are_the_legacy_default_units(cn):
    from lib_controlnet.external_code import ControlNetUnit

    legacy = ControlNetUnit(enabled=False, module="None", model="None")
    args = layout(cn, "img2img").values_to_args({})
    assert [settings_of(unit) for unit in args[:3]] == [settings_of(legacy)] * 3
    assert all(unit.image is None and unit.generated_image is None for unit in args[:3])
    assert args[3] == ""


def test_schema_export(cn, registry):
    schema = layout(cn, "img2img").schema()
    json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["title"] == "ControlNet Integrated" and schema["open"] is False
    tabs, batch_dir = schema["children"]
    assert [tab["title"] for tab in tabs["children"]] == ["ControlNet Unit 1", "ControlNet Unit 2", "ControlNet Unit 3"]
    first, unit = (tabs["children"][i]["children"][0] for i in (0, 1))
    assert unit["type"] == "custom" and unit["widget"] == "controlnet-unit" and unit["name"] == "unit_1" and unit["label"] == ""
    assert unit["infotext"] == "ControlNet 1"
    assert unit["props"]["unit"] == 1 and unit["props"]["img2img"] is True and unit["props"]["tabs"] == [tab["title"] for tab in tabs["children"]]
    # The catalog ships once per tab, with the first unit (the widget of every unit reads it there)
    assert "catalog" not in unit["props"] and unit["props"]["controls"] == first["props"]["controls"]
    props = first["props"]
    assert [p["name"] for p in props["catalog"]["preprocessors"]] == ["None", "canny", "reference_test"]
    assert props["catalog"]["models"] == MODELS
    types = {entry["name"]: entry for entry in props["catalog"]["types"]}
    assert list(types) == ["All", "Canny", "Reference", "Region"]
    # The old "Control Type" switch: the second entry of the filtered lists
    # Models by name: the type, its preprocessors' filters, and union models
    assert types["Canny"] == {"name": "Canny", "modules": ["None", "canny"], "models": ["None", "control_v11p_sd15_canny", "t2i_union_promax"], "default_module": "canny", "default_model": "control_v11p_sd15_canny"}
    assert types["All"]["default_module"] == "None" and types["All"]["default_model"] == "None"
    canny = props["catalog"]["preprocessors"][1]
    assert canny["slider_1"] == {"label": "Low Threshold", "minimum": 0, "maximum": 256, "step": 1, "value": 100, "visible": True}
    assert canny["model"] is True and props["catalog"]["preprocessors"][2]["model"] is False
    assert batch_dir["name"] == "batch_input_dir" and batch_dir["visible_if"] == {"param": "core.mode", "eq": "Batch"}


def test_the_widget_contract_is_current(cn, registry):
    """webui/src/widgets/controlnet-unit/contract.json is what the server
    exports (the widget's tests read it). ``REMI_UPDATE_CONTRACT=1`` rewrites it."""

    from lib_controlnet.unit_param import CONTROLS, PASTE_MARKER, default_value

    schema = layout(cn, "txt2img").schema()
    props = schema["children"][0]["children"][0]["children"][0]["props"]
    contract = {
        "paste_marker": PASTE_MARKER,
        "default": {"txt2img": default_value(False), "img2img": default_value(True)},
        "controls": CONTROLS,
        "props": props,
    }
    text = json.dumps(contract, indent=1, ensure_ascii=False) + "\n"
    if os.environ.get("REMI_UPDATE_CONTRACT"):
        CONTRACT.write_text(text, encoding="utf8")
    assert CONTRACT.read_text(encoding="utf8") == text, "stale widget contract: rerun with REMI_UPDATE_CONTRACT=1"


# endregion

# region Values


def test_the_unit_value_is_normalized(cn):
    from lib_controlnet.unit_param import normalize

    from remi import ui

    value = normalize({"enabled": "true", "weight": 3, "guidance_start": 0.9, "guidance_end": 0.2, "processor_res": 511.6, "threshold_a": -5, "resize_mode": "Inner Fit (Scale to Fit)", "control_mode": 2, "hr_option": "HiResFixOption.LOW_RES_ONLY", "__paste__": True, "unknown": 1}, False)
    assert value["enabled"] is True and value["weight"] == 2.0
    assert (value["guidance_start"], value["guidance_end"]) == (0.2, 0.9)
    assert value["processor_res"] == 512 and value["threshold_a"] == -1
    assert (value["resize_mode"], value["control_mode"], value["hr_option"]) == ("Crop and Resize", "ControlNet is more important", "Low res only")
    assert "__paste__" not in value and "unknown" not in value and "upload_independent" not in value
    assert normalize({}, True)["upload_independent"] is False
    for bad in ({"weight": "heavy"}, {"enabled": "yes"}, {"module": 3}, {"image": ""}, {"control_mode": "Loud"}, {"pose": "x"}, {"weight": float("nan")}, []):
        with pytest.raises(ui.ParamError):
            normalize(bad, False)


def test_processing_args_are_what_the_legacy_controls_built(cn, registry, files):
    """The same settings and pictures, once as the Gradio unit group passed
    them (``UiControlNetUnit(*unit_args)``, the canvases' RGBA arrays) and
    once as the widget's JSON with uploads, give the same ``ControlNetUnit``.
    Painted masks only carry their alpha (the brush colour was never read)."""

    from lib_controlnet.external_code import UiControlNetUnit

    rng = np.random.default_rng(1)
    picture = Image.fromarray(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8))
    alpha = np.zeros((12, 16), dtype=np.uint8)
    alpha[2:6, 3:9] = 255
    painted = np.zeros((12, 16, 4), dtype=np.uint8)
    painted[..., 3] = alpha
    mask_picture = Image.fromarray(np.where(rng.random((12, 16)) > 0.5, 255, 0).astype(np.uint8))
    preview = Image.fromarray(rng.integers(0, 255, (12, 16, 3), dtype=np.uint8))

    def upload(image):
        return files.add_upload_bytes(png(image), "x.png").id

    value = {
        "enabled": True,
        "module": "canny",
        "model": "control_v11p_sd15_canny",
        "weight": 0.85,
        "resize_mode": "Resize and Fill",
        "processor_res": 640,
        "threshold_a": 90,
        "threshold_b": 210,
        "guidance_start": 0.1,
        "guidance_end": 0.8,
        "pixel_perfect": True,
        "control_mode": "My prompt is more important",
        "hr_option": "Low res only",
        "image": upload(picture),
        "image_fg": upload(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))),
        "mask_upload": True,
        "mask_image": upload(mask_picture),
        "mask_image_fg": upload(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))),
        "generated_image": upload(preview),
        "use_preview_as_input": True,
    }
    from remi import ui

    new = layout(cn).values_to_args({"unit_0": value}, ui.Context(resolve_upload=resolver(files)))[0]

    rgba = lambda image: np.asarray(image.convert("RGBA"))  # noqa: E731 - what LogicalImage gave
    unit_args = (
        True,  # use_preview_as_input
        rgba(preview),  # generated_image.background
        rgba(mask_picture),  # mask_image.background
        painted,  # mask_image.foreground
        "Low res only",  # hr_option
        True,  # enabled
        "canny",
        "control_v11p_sd15_canny",
        0.85,
        rgba(picture),  # image.background
        painted,  # image.foreground
        "Resize and Fill",
        640,
        90,
        210,
        0.1,
        0.8,
        True,  # pixel_perfect
        "My prompt is more important",
    )
    old = UiControlNetUnit(*unit_args)

    assert settings_of(new) == settings_of(old)
    for name in ("image", "generated_image", "mask_image"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name), err_msg=name)
    for name in ("image_fg", "mask_image_fg"):
        np.testing.assert_array_equal(getattr(new, name)[:, :, 3], getattr(old, name)[:, :, 3], err_msg=name)


def test_a_sketch_is_painted_into_the_control_image(cn, registry, files):
    from remi import ui

    base = files.add_upload_bytes(png(Image.new("RGB", (4, 4), "white")), "base.png").id
    stroke = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    stroke.putpixel((1, 2), (255, 0, 0, 255))
    sketch = files.add_upload_bytes(png(stroke), "sketch.png").id
    value = {"enabled": True, "module": "None", "model": "control_v11p_sd15_canny", "image": base, "sketch": sketch}
    unit = layout(cn).values_to_args({"unit_0": value}, ui.Context(resolve_upload=resolver(files)))[0]
    assert tuple(unit.image[2, 1]) == (255, 0, 0, 255) and tuple(unit.image[0, 0]) == (255, 255, 255, 255)


def test_img2img_units_read_the_img2img_image_unless_independent(cn, registry, files):
    from remi import ui

    image = files.add_upload_bytes(png(Image.new("RGB", (4, 4))), "a.png").id
    ctx = ui.Context(resolve_upload=resolver(files))
    value = {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": image, "generated_image": image, "use_preview_as_input": True}
    shared_input = layout(cn, "img2img").values_to_args({"unit_0": value}, ctx)[0]
    assert shared_input.image is None and shared_input.generated_image is None and shared_input.use_preview_as_input is False
    own = layout(cn, "img2img").values_to_args({"unit_0": {**value, "upload_independent": True}}, ctx)[0]
    assert own.image.shape == (4, 4, 4) and own.use_preview_as_input is True


def test_enabled_units_are_checked_before_the_job(cn, registry, files):
    from remi import ui

    lay = layout(cn)
    ctx = ui.Context(resolve_upload=files.check_upload)
    image = files.add_upload_bytes(png(Image.new("RGB", (4, 4))), "a.png").id
    good = {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": image}
    lay.values_to_args({"unit_0": good}, ctx)
    for change, message in (
        ({"model": "None"}, "select a ControlNet model"),
        ({"model": "gone"}, "was not found"),
        ({"module": "nope"}, "unknown preprocessor"),
        ({"image": None}, "upload a control image"),
        ({"image": "missing-upload-id"}, "unknown upload"),
    ):
        with pytest.raises(ui.ValidationError) as error:
            lay.values_to_args({"unit_0": {**good, **change}}, ctx)
        assert message in str(error.value), change
    # A model-free preprocessor, a disabled unit with stale uploads, a preview used as input
    lay.values_to_args({"unit_0": {**good, "module": "reference_test", "model": "None"}}, ctx)
    lay.values_to_args({"unit_0": {**good, "enabled": False, "image": "missing-upload-id", "model": "gone"}}, ctx)
    lay.values_to_args({"unit_0": {**good, "image": None, "generated_image": image, "use_preview_as_input": True}}, ctx)


def test_api_units_and_the_batch_directory_reach_processing(cn, monkeypatch):
    """``/sdapi`` sends unit dicts: one per unit slot, then (img2img) the
    batch directory; arguments past the slots are ignored, as before."""

    from lib_controlnet.external_code import ControlNetUnit

    from modules import shared

    script = cn.ControlNetForForgeOfficial()
    layout(cn, "img2img")  # three units
    on = {"enabled": True, "module": "canny", "weight": 0.5}
    args = [on, ControlNetUnit(enabled=False), "not a unit", "/data/control", {**on, "weight": 0.25}]
    units = script.get_enabled_units(args)
    assert len(units) == 1 and isinstance(units[0], ControlNetUnit) and units[0].weight == 0.5
    assert script.get_batch_input_dir(args) == "/data/control"
    assert script.get_batch_input_dir(args[:3]) == "" and script.get_batch_input_dir([on, on, on, on]) == ""
    # ADetailer runs ControlNet with one unit argument
    assert len(script.get_enabled_units([ControlNetUnit(enabled=True)])) == 1
    # With directory settings hidden no request may point ControlNet at a server directory
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert script.get_batch_input_dir(args) == ""


def test_api_units_decode_the_image_mask_pair():
    """The A1111 API also sends ``"image": {"image": b64, "mask": b64}``."""

    import base64

    from lib_controlnet.external_code import ControlNetUnit

    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (200, 10, 10)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    unit = ControlNetUnit.from_dict({"enabled": True, "image": {"image": encoded, "mask": None}})
    assert unit.image["image"].shape == (4, 6, 3) and unit.image["image"][0, 0, 0] == 200 and unit.image["mask"] is None


def test_the_batch_directory_supplies_control_images(cn, registry, tmp_path):
    from lib_controlnet.external_code import ControlNetUnit

    from modules.processing import StableDiffusionProcessingImg2Img

    for name, colour in (("a", (10, 0, 0)), ("b", (20, 0, 0))):
        Image.new("RGB", (8, 8), colour).save(tmp_path / f"{name}.png")
    source = Image.new("RGB", (8, 8))
    source.filename = str(tmp_path / "b.png")

    p = StableDiffusionProcessingImg2Img.__new__(StableDiffusionProcessingImg2Img)
    p.init_images, p.image_mask, p.resize_mode, p.inpaint_full_res = [source], None, 0, False
    script = cn.ControlNetForForgeOfficial()
    script.batch_input_dir = str(tmp_path)
    unit = ControlNetUnit(enabled=True, module="canny")
    unit._idx = 0
    preprocessor = registry.get_preprocessor("canny")
    [(image, mask)], _ = script.get_input_data(p, unit, preprocessor, 8, 8)
    assert image[0, 0, 0] == 20 and mask is None  # the file of the same name
    source.filename = "elsewhere.png"
    [(image, _)], _ = script.get_input_data(p, unit, preprocessor, 8, 8)
    assert image[0, 0, 0] == 10  # else the next file


# endregion

# region Infotext


def test_infotext_round_trips_through_the_unit(cn, registry):
    from lib_controlnet.external_code import ControlNetUnit
    from lib_controlnet.infotext import Infotext

    source = ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", weight=0.75, resize_mode="Resize and Fill", processor_res=768, threshold_a=64, threshold_b=128, guidance_start=0.05, guidance_end=0.6, pixel_perfect=True, control_mode="Balanced", hr_option="High res only")
    p = SimpleNamespace(extra_generation_params={})
    Infotext.write_infotext([source], p)
    params = dict(p.extra_generation_params)

    lay = layout(cn)
    values, problems = lay.paste(params)
    assert problems == []
    pasted = values["unit_0"]
    assert pasted["__paste__"] is True and "enabled" not in pasted and "image" not in pasted
    assert set(pasted) == {"__paste__", *ControlNetUnit.infotext_fields()}
    unit = lay.values_to_args({"unit_0": {**pasted, "enabled": True, "image": "id"}})[0]
    assert {field: plain(getattr(unit, field)) for field in ControlNetUnit.infotext_fields()} == {field: plain(getattr(source, field)) for field in ControlNetUnit.infotext_fields()}


def test_api_written_infotext_and_bad_fields_paste(cn):
    """Every infotext field is pasted, as the old paste of a whole
    ``ControlNetUnit`` did: the ones the entry leaves out (the infotext
    omits an unset -1 resolution or threshold) take the unit defaults, so a
    paste does not keep the current unit's values. A field that does not
    read keeps the current value."""

    lay = layout(cn)
    values, problems = lay.paste({"ControlNet 1": "Module: canny, Weight: heavy, Resize Mode: ResizeMode.OUTER_FIT, Hr Option: HiResFixOption.BOTH, Control Mode: ControlMode.CONTROL"})
    assert problems == []
    assert values == {
        "unit_1": {
            "__paste__": True,
            "module": "canny",
            "model": "None",
            "resize_mode": "Resize and Fill",
            "processor_res": -1,
            "threshold_a": -1,
            "threshold_b": -1,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "pixel_perfect": False,
            "control_mode": "ControlNet is more important",
            "hr_option": "Both",
        }
    }
    _, problems = lay.paste({"ControlNet 0": "garbage"})
    assert [problem.param for problem in problems] == ["unit_0"]


def test_reading_units_from_infotext_can_be_switched_off(cn, monkeypatch):
    from modules import shared

    monkeypatch.setitem(shared.opts.data, "control_net_sync_field_args", False)
    assert layout(cn).infotext == {}


def test_units_from_positional_arguments(cn):
    """Old positional arguments (imported queue items, ``args_to_values``)."""

    from lib_controlnet.external_code import ControlNetUnit

    values = layout(cn).args_to_values([ControlNetUnit(enabled=True, module="canny", weight=0.4, image=np.zeros((2, 2, 3), np.uint8)), {"enabled": False, "resize_mode": 0}])
    assert values["unit_0"]["enabled"] is True and values["unit_0"]["weight"] == 0.4 and values["unit_0"]["image"] is None
    assert values["unit_1"]["resize_mode"] == "Just Resize"
    json.dumps(values)


@pytest.mark.parametrize("tab", harness.TABS)
def test_imported_units_keep_their_pictures(cn, registry, files, tab):
    """A job importer that stores images (``Context.store_upload``) gets a
    unit's pictures as uploads, and the unit then gives processing the same
    pictures: arrays, API base64 text and the old ``{image, mask}`` pair."""

    import base64

    from lib_controlnet.external_code import ControlNetUnit

    from remi import ui

    rng = np.random.default_rng(2)
    picture = rng.integers(0, 255, (6, 8, 4), dtype=np.uint8)
    alpha = np.zeros((6, 8), np.uint8)
    alpha[1:3] = 255
    painted = np.dstack([rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), alpha])  # brush colour never read
    preview = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    mask_picture = np.dstack([np.where(rng.random((6, 8)) > 0.5, 255, 0).astype(np.uint8)] * 3 + [np.full((6, 8), 255, np.uint8)])
    old = ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", image=picture, image_fg=painted, mask_image=mask_picture, mask_image_fg=painted, generated_image=preview, use_preview_as_input=True)

    lay = layout(cn, tab)
    store = lambda image: files.add_upload_bytes(png(image), "imported.png").id  # noqa: E731
    values = lay.args_to_values([old, {"enabled": True, "module": "canny", "model": "control_v11p_sd15_canny", "image": base64.b64encode(png(Image.fromarray(preview))).decode()}, ControlNetUnit(enabled=True, module="canny", model="control_v11p_sd15_canny", image={"image": preview, "mask": mask_picture})], ui.Context(store_upload=store))
    json.dumps(values)
    assert values["unit_0"]["mask_upload"] is True and values["unit_0"].get("upload_independent", True) is True
    new = lay.values_to_args(values, ui.Context(resolve_upload=resolver(files)))

    assert settings_of(new[0]) == settings_of(old)
    for name in ("image", "mask_image"):
        np.testing.assert_array_equal(getattr(new[0], name), getattr(old, name), err_msg=name)
    np.testing.assert_array_equal(new[0].generated_image[:, :, :3], preview)
    for name in ("image_fg", "mask_image_fg"):
        np.testing.assert_array_equal(getattr(new[0], name)[:, :, 3], alpha, err_msg=name)
    np.testing.assert_array_equal(new[1].image[:, :, :3], preview)
    np.testing.assert_array_equal(new[2].image[:, :, :3], preview)
    np.testing.assert_array_equal(new[2].mask_image, mask_picture)
    # Without an image store the pictures are dropped (and an enabled txt2img unit then asks for one)
    assert lay.args_to_values([old])["unit_0"]["image"] is None


# endregion

# region Actions


def test_preview_runs_the_preprocessor(cn, registry, files):
    script = cn.ControlNetForForgeOfficial()
    picture = Image.new("RGB", (64, 48), "black")
    picture.paste((255, 255, 255), (16, 8, 48, 40))
    image = files.add_upload_bytes(png(picture), "in.png").id
    values = {"unit_1": {"module": "canny", "image": image, "processor_res": 64, "threshold_a": 100, "threshold_b": 200}}
    patch = call(script, "preview", values, files, unit=1)
    preview = patch["__outputs__"]["preview"]
    assert preview["unit"] == 1 and preview["pose"] is None
    edges = np.asarray(Image.open(files.get(preview["image"]).path))
    # Canny works at the preprocessor resolution (the short side), as before
    assert edges.shape[:2] == (64, 85) and edges.max() == 255 and edges[0, 0].max() == 0

    assert call(script, "preview", {"unit_0": {"module": "canny"}}, files, unit=0)["__toast__"]["level"] == "warning"
    assert call(script, "preview", {"unit_0": {"module": "nope", "image": image}}, files, unit=0)["__toast__"]["level"] == "error"


def test_preview_passes_pixel_perfect_resolution_and_the_painted_mask(cn, registry, files, monkeypatch):
    seen = {}
    canny = registry.get_preprocessor("canny")

    def fake(input_image, resolution, slider_1=None, slider_2=None, input_mask=None, json_pose_callback=None, **kwargs):
        seen.update(resolution=resolution, slider_1=slider_1, mask=input_mask, shape=input_image.shape)
        return input_image

    monkeypatch.setattr(type(canny), "__call__", lambda self, *args, **kwargs: fake(*args, **kwargs))
    image = files.add_upload_bytes(png(Image.new("RGB", (100, 50))), "in.png").id
    alpha = np.zeros((50, 100), np.uint8)
    alpha[:10] = 255
    mask = files.add_upload_bytes(png(Image.fromarray(np.dstack([np.full_like(alpha, 255)] * 3 + [alpha]))), "m.png").id
    script = cn.ControlNetForForgeOfficial()
    call(script, "preview", {"unit_0": {"module": "canny", "image": image, "image_fg": mask, "pixel_perfect": True, "resize_mode": "Just Resize"}}, files, core={"width": 1024, "height": 768}, unit=0)
    assert seen["resolution"] == 768  # max(768/50, 1024/100) * 50
    assert seen["slider_1"] == 100  # an unset (-1) threshold is the preprocessor's default
    assert seen["mask"].shape == (50, 100, 3) and seen["mask"][0, 0, 0] == 255 and seen["mask"][20, 0, 0] == 0


def test_pose_rendering(cn, files):
    script = cn.ControlNetForForgeOfficial()
    pose = {"canvas_width": 64, "canvas_height": 32, "people": [{"pose_keypoints_2d": [10, 10, 1, 20, 20, 1] + [0, 0, 0] * 16}]}
    patch = call(script, "render_pose", {}, files, unit=2, pose=json.dumps(pose))
    out = patch["__outputs__"]["pose"]
    assert out["unit"] == 2 and out["pose"] == pose
    assert Image.open(files.get(out["image"]).path).size == (64, 32)
    assert call(script, "render_pose", {}, files, unit=0, pose="{")["__toast__"]["level"] == "error"
    assert call(script, "render_pose", {}, files, unit=0, pose={"people": []})["__toast__"]["level"] == "error"


def test_region_canvas(cn, files):
    from remi import ui

    script = cn.ControlNetForForgeOfficial()
    out = call(script, "region_canvas", {}, files, core={"width": 96, "height": 64}, unit=0, subjects=3, shape="Oval")["__outputs__"]["canvas"]
    picture = np.asarray(Image.open(files.get(out["image"]).path).convert("RGB"))
    assert picture.shape == (64, 96, 3) and (picture != 255).any()
    blank = call(script, "region_canvas", {}, files, unit=0, subjects=0)["__outputs__"]["canvas"]
    assert (np.asarray(Image.open(files.get(blank["image"]).path)) == 255).all()
    for extra in ({"unit": 9}, {"unit": 0, "shape": "Star"}, {"unit": 0, "subjects": "many"}):
        with pytest.raises(ui.ValidationError):
            call(script, "region_canvas", {}, files, **extra)


def test_refresh_models_rescans_and_returns_the_catalog(cn, registry, files, monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "update_controlnet_filenames", lambda: calls.append(1))
    catalog = call(cn.ControlNetForForgeOfficial(), "refresh_models", {}, files)["__outputs__"]["catalog"]
    assert calls == [1] and catalog["models"] == MODELS


# endregion


def test_zh_cn_covers_the_panel(cn, registry):
    from lib_controlnet.unit_param import CONTROLS, INVERT_HINT

    from modules import shared

    cn.on_ui_settings()
    schemas = [layout(cn, tab).schema() for tab in harness.TABS]
    props = schemas[0]["children"][0]["children"][0]["children"][0]["props"]
    texts = {INVERT_HINT, *(f"ControlNet Unit {i}" for i in range(1, 6))}
    for control in CONTROLS.values():
        texts |= {control["label"], *control.get("choices", ()), *([control["info"]] if control.get("info") else [])}
    texts |= {entry["name"] for entry in props["catalog"]["types"]}
    texts |= {spec["label"] for p in props["catalog"]["preprocessors"] for spec in (p["resolution"], p["slider_1"], p["slider_2"]) if spec["visible"] and p["name"] != "reference_test"}
    texts |= {"Upload a control image first", "Unknown preprocessor {{name}}", "The preprocessor failed: {{error}}", "The pose file is too large", "The pose is not valid JSON", "Not an OpenPose pose: {{error}}"}
    texts |= {info.label for key, info in shared.opts.data_labels.items() if key.startswith("control_net_")} | {"ControlNet", "e.g. training output directory"}
    harness.assert_localized(EXTENSION, schemas, extra=texts)
