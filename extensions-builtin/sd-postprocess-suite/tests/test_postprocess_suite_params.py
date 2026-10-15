"""Post Processing Suite on ``params()``: the legacy ``ui()`` contract, the
pipeline arguments process hooks receive, the collapsed (lazily rendered)
panel, the preset actions (with the LUT file) and "Apply PP → Send to img2img"."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from pp_suite_testlib import EXT, SCRIPT, contract, load_suite
from PIL import Image

from remi import ui
from remi.testing import webui_state

CUBE = "LUT_3D_SIZE 2\n" + "\n".join(f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1)) + "\n"
"""An identity 3D LUT."""


@pytest.fixture(scope="module")
def pp():
    with webui_state():
        yield load_suite()


@pytest.fixture
def files(tmp_path):
    from remi.files import FileStore

    store = FileStore(tmp_path / "files")
    yield store
    store.close()


@pytest.fixture
def presets(pp, tmp_path, monkeypatch):
    directory = tmp_path / "presets"
    monkeypatch.setattr(pp, "_PRESET_DIR", str(directory))
    monkeypatch.setattr(pp, "_LEGACY_PRESET_DIR", str(tmp_path / "legacy"))
    return directory


def make(module, tab="txt2img"):
    script = module.PostProcessSuite()
    script.is_img2img = tab == "img2img"
    script.layout = ui.ScriptLayout(script.params(script.is_img2img), owner="post_processing_suite", order=script.param_order)
    return script


def call(script, action, values, store, **extra):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=SimpleNamespace(files=store))))
    return ui.call_action(script, action, ui.ActionContext(tab="txt2img", owner="post_processing_suite", request=request, extra=extra), values)


@pytest.mark.parametrize(("tab", "extra"), [("txt2img", ("send", "preset")), ("img2img", ("preset",))])
def test_params_keep_the_legacy_contract(pp, tab, extra):
    script = make(pp, tab)
    snapshot = contract.load_snapshot(EXT / "tests" / "fixtures" / f"legacy_ui_{tab}.json")
    # The .cube upload was a gr.File (a JSON param on the legacy path); it is a file upload now
    lut = script.layout.names.index("lut_lut_file")
    expected = {lut: {"kind": "file", "default": None}}
    assert contract.differences(script.layout, snapshot, expected=expected, extra=extra) == []
    assert len(pp.pipeline_names()) == pp.P.PIPELINE_ARG_COUNT


def test_positional_arguments_are_the_pipeline_layout(pp):
    """Named values reach the hooks as the flat list ``P.unflatten`` reads."""

    preset = {"master": True, "stages": {"tone": {"enabled": True, "order": 7, "params": {"exposure": 0.5}}, "noise": {"enabled": True, "params": {"kind": "uniform", "seed": 3}}}}
    flat = pp.P.deserialize_to_values(preset)
    script = make(pp, "img2img")
    values = dict(zip(pp.pipeline_names(), flat))
    args = script.layout.values_to_args({**values, "apply_mode": "Both", "preset": "mine"})
    assert args[: len(flat)] == flat and args[len(flat):] == ["Both", "mine"]
    master, states = pp.P.unflatten(args)
    tone = next(s for s in states if s["key"] == "tone")
    assert master and tone["enabled"] and tone["order"] == 7 and tone["vals"]["exposure"] == 0.5
    assert script._apply_mode(args) == "Both"
    # txt2img: the trailing UI-only params never read as an apply mode
    txt = make(pp, "txt2img")
    assert txt._apply_mode(txt.layout.values_to_args({"enable": True})) == "Final output"


def test_the_pipeline_runs_in_the_last_hook(pp):
    script = make(pp)
    args = script.layout.values_to_args({"enable": True, "sepia_enabled": True, "sepia_strength": 1.0})
    p = SimpleNamespace(extra_generation_params={})
    pp_image = SimpleNamespace(image=Image.new("RGB", (8, 8), (0, 0, 255)))
    script.postprocess_image_after_composite(p, pp_image, *args)
    r, g, b = pp_image.image.getpixel((0, 0))
    assert r > b  # sepia
    assert json.loads(p.extra_generation_params["PP Suite"])["sepia"]["strength"] == 1.0


def test_the_panel_is_collapsed_and_lazy(pp):
    """Every category and stage starts closed: the SPA mounts a section's
    controls only when it is opened."""

    script = make(pp)
    sections = [node for node in script.layout.root.walk() if isinstance(node, ui.Section)]
    categories = {category for category, _ in pp.P.grouped_stages()}
    assert len(sections) == 1 + len(categories) + len(pp.P.STAGES)
    assert not any(section.open for section in sections)
    stage = next(section for section in sections if section.title == "Tone (exposure / contrast / levels)")
    assert stage.enable.name == "tone_enabled"
    json.dumps(script.layout.schema(ui.Context()), allow_nan=False)


def test_presets_save_load_and_delete_with_the_lut(pp, files, presets):
    script = make(pp)
    lut = files.add_upload_bytes(CUBE.encode(), "identity.cube")
    values = {"enable": True, "tone_enabled": True, "tone_exposure": 1.25, "lut_enabled": True, "lut_lut_file": lut.id, "preset": "My Look"}

    patch = call(script, "preset_save", values, files)
    assert patch["__toast__"]["level"] == "success"
    assert patch["preset"]["value"] == "My Look" and {"value": "My Look", "label": "My Look"} in patch["preset"]["choices"]
    saved = json.loads((presets / "My Look.json").read_text(encoding="utf-8"))
    kept = saved["stages"]["lut"]["params"]["lut_file"]
    assert kept.startswith(str(presets / "luts")) and open(kept, encoding="utf-8").read() == CUBE  # a copy owned by the preset

    patch = call(script, "preset_load", {"preset": "My Look"}, files)
    assert patch["tone_exposure"] == {"value": 1.25} and patch["enable"] == {"value": True} and patch["sepia_enabled"] == {"value": False}
    reloaded = files.get(patch["lut_lut_file"]["value"])
    # The preset's copy becomes an upload again (the same content: the store keeps one record)
    assert reloaded is not None and reloaded.path.endswith(".cube") and open(reloaded.path, encoding="utf-8").read() == CUBE

    assert call(script, "preset_save", {"preset": "  "}, files)["__toast__"]["level"] == "warning"
    assert call(script, "preset_load", {"preset": "Nope"}, files)["__toast__"]["level"] == "warning"
    patch = call(script, "preset_delete", {"preset": "My Look"}, files)
    assert patch["preset"] == {"choices": [], "value": ""} and not (presets / "My Look.json").exists()
    assert not os.path.exists(kept)  # the preset's LUT copy goes with it
    assert call(script, "preset_refresh", {}, files) == {"preset": {"choices": []}}


def test_overwriting_a_preset_replaces_its_lut_copy(pp, files, presets):
    script = make(pp)
    first = files.add_upload_bytes(CUBE.encode(), "first.cube")
    second = files.add_upload_bytes(CUBE.encode() + b"\n", "second.cube")
    values = {"lut_enabled": True, "preset": "Look"}

    def lut_of(name):
        return json.loads((presets / f"{name}.json").read_text(encoding="utf-8"))["stages"]["lut"]["params"]["lut_file"]

    call(script, "preset_save", {**values, "lut_lut_file": first.id}, files)
    old = lut_of("Look")
    call(script, "preset_save", {**values, "preset": "Other", "lut_lut_file": first.id}, files)
    call(script, "preset_save", {**values, "lut_lut_file": second.id}, files)
    assert not os.path.exists(old) and os.path.exists(lut_of("Look")) and os.path.exists(lut_of("Other"))
    # A preset without a LUT drops its copy too; a LUT outside presets/luts is never removed
    call(script, "preset_save", values, files)
    assert sorted(os.listdir(presets / "luts")) == [os.path.basename(lut_of("Other"))]
    outside = files.get(first.id).path
    (presets / "Legacy.json").write_text(json.dumps({"stages": {"lut": {"params": {"lut_file": outside}}}}), encoding="utf-8")
    call(script, "preset_delete", {"preset": "Legacy"}, files)
    assert os.path.exists(outside)


def test_a_preset_whose_lut_is_gone_loads_without_it(pp, files, presets):
    presets.mkdir(parents=True)
    (presets / "Old.json").write_text(json.dumps({"v": 1, "master": True, "stages": {"lut": {"enabled": True, "params": {"lut_file": "/tmp/uploads/missing.cube"}}}}), encoding="utf-8")
    patch = call(make(pp), "preset_load", {"preset": "Old"}, files)
    assert patch["lut_enabled"] == {"value": True} and patch["lut_lut_file"] == {"value": None}


def test_apply_and_send_post_processes_the_selected_result(pp, files):
    script = make(pp)
    buffer = __import__("io").BytesIO()
    Image.new("RGB", (16, 16), (40, 90, 200)).save(buffer, format="PNG")
    source = files.add_upload_bytes(buffer.getvalue(), "result.png")

    patch = call(script, "apply_send", {"enable": True, "sepia_enabled": True, "sepia_strength": 1.0}, files, file_id=source.id)
    sent = files.get(patch["__outputs__"]["sent"])
    assert sent is not None and sent.is_image and sent.id != source.id
    pixels = np.asarray(Image.open(sent.path).convert("RGB"))
    assert pixels[0, 0, 0] > pixels[0, 0, 2]  # sepia applied

    assert call(script, "apply_send", {"enable": True}, files)["__toast__"]["level"] == "warning"
    with pytest.raises(ui.ValidationError):
        call(script, "apply_send", {"sepia_strength": "strong"}, files, file_id=source.id)


def test_ported_without_javascript():
    contract.assert_ported(SCRIPT)
    assert not (EXT / "javascript").exists()


def test_zh_cn_covers_every_string(pp):
    import re

    strings = {"Post Processing Suite"} | set(re.findall(r'ui\.toast\("([^"]+)"', SCRIPT.read_text(encoding="utf-8")))
    for tab in ("txt2img", "img2img"):
        strings |= contract.schema_strings(make(pp, tab).layout.schema(ui.Context()))
    assert contract.missing_translations(strings, EXT / "localizations" / "zh_CN.json") == []
