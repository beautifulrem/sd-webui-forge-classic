"""The JoyCaption page (remi.pages): layout parity with the former Gradio tab,
schema export, actions and tasks (with a fake engine), and zh_CN coverage."""

import importlib.util
import json
import threading
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

EXT = Path(__file__).resolve().parents[1]
REPO = EXT.parents[1]
SCRIPT = EXT / "scripts" / "joycaption.py"

_spec = importlib.util.spec_from_file_location("remi_page_harness", REPO / "tests" / "remi_page_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def jc(webui, tmp_path, monkeypatch):
    from remi import pages

    pages.clear()
    module = harness.load_page_script(SCRIPT, "joycaption_under_test")
    # user caption types live in a temporary models folder
    monkeypatch.setattr(module.jcp, "_user_types_path", lambda: tmp_path / "user_caption_types.json")
    yield module
    pages.clear()


@pytest.fixture
def page(jc):
    from remi import pages

    return pages.get("joycaption")


class FakeEngine:
    def __init__(self):
        self.loaded_quant = None
        self.unloads = 0
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def is_loaded(self):
        return self.loaded_quant is not None

    def load(self, key):
        self.loaded_quant = key

    def unload(self):
        self.unloads += 1
        self.loaded_quant = None

    def generate(self, image, prompt, system_prompt, max_new_tokens, temperature, top_p, top_k):
        self.gate.wait(5)
        self.calls.append({"size": image.size, "prompt": prompt, "system": system_prompt, "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k})
        return f"caption {len(self.calls)}"


@pytest.fixture
def engine(jc, monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(jc.jcm, "get_engine", lambda: fake)
    monkeypatch.setattr(jc.jcm, "is_model_downloaded", lambda: True)
    return fake


def _png_with_prompt(text, size=(64, 48)):
    image = Image.new("RGB", size, "blue")
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", text)
    image._pnginfo = info
    return image


# region Layout


# Legacy record index -> param (None: dropped on purpose)
LEGACY_MAP = {
    0: "quant", 1: "keep_loaded", 2: "caption_type", 3: "caption_length", 4: "name_input",
    5: "use_custom", 6: "custom_prompt", 7: "use_system_prompt", 8: "system_prompt",
    9: "max_new_tokens", 10: "temperature", 11: "top_p", 12: "top_k", 13: "extra_options",
    14: "edit_select", 15: "edit_name", 16: "edit_any", 17: "edit_word", 18: "edit_length",
    19: "image", 20: "caption_out", 21: "prompt_out", 22: "combined",
    23: "folder", 24: "recursive", 25: "overwrite", 26: "prefix", 27: "suffix", 28: "save_ext",
    29: None,  # the batch log is an output now (it never was an input)
}


# The precisions the old tab offered with bitsandbytes (the snapshot was
# recorded without it: bf16 only)
LEGACY_QUANTS_WITH_BNB = ["bf16 (best quality, ~17 GB VRAM)", "int8 (~10 GB VRAM)", "nf4 (~6 GB VRAM)"]


@pytest.mark.parametrize("bitsandbytes", [False, True])
def test_layout_matches_the_gradio_tab(page, jc, monkeypatch, bitsandbytes):
    snapshot = harness.load_snapshot(SCRIPT)
    assert snapshot["title"] == page.title == "JoyCaption"
    assert snapshot["inputs"][0]["choices"] == LEGACY_QUANTS_WITH_BNB[:1]
    monkeypatch.setattr(jc, "_has_bitsandbytes", lambda: bitsandbytes)
    legacy_types = snapshot["inputs"][14]["choices"]
    harness.assert_page_matches(
        page.layout(),
        snapshot,
        LEGACY_MAP,
        {
            **({0: {"choices": LEGACY_QUANTS_WITH_BNB}} if bitsandbytes else {}),
            14: {"default": "", "choices": ["", *legacy_types]},  # "Select a caption type" instead of no selection
        },
    )
    # the recorded inputs keep their order
    names = page.layout().names
    order = [names.index(name) for name in LEGACY_MAP.values() if name]
    assert order == sorted(order)


def test_schema_export(page):
    data = page.schema()
    json.dumps(data, allow_nan=False)
    assert data["owner"] == "page:joycaption" and data["icon"] == "camera" and data["on_load"] == "status"
    assert data["send_to"] == [{"kind": "image", "param": "image", "label": "JoyCaption"}]
    widgets = {node["name"]: node for node in _walk(data["layout"]) if node.get("type") == "custom"}
    assert {name: node["widget"] for name, node in widgets.items()} == {
        "load": "task-runner", "caption": "task-runner", "batch": "task-runner", "send_append": "send-button", "send_replace": "send-button"
    }
    assert widgets["send_append"]["props"] == {"to": "prompt", "source": "combined", "target": "txt2img", "mode": "append", "label": "→ Append to txt2img Prompt (does not replace)", "variant": "primary"}
    assert widgets["send_replace"]["props"]["mode"] == "replace"
    # buttons and widgets are never submitted nor drafted
    assert all(node.get("submit") is False and node.get("persist") is False for node in widgets.values())
    defaults = data["defaults"]
    assert defaults["caption_type"] == "Descriptive (Formal)" and defaults["caption_length"] == "long" and defaults["max_new_tokens"] == 512


def _walk(node):
    if isinstance(node, dict):
        yield node
        for child in node.get("children", []):
            yield from _walk(child)


def test_no_gradio_left():
    harness.assert_page_gradio_free(EXT, [SCRIPT])


# endregion

# region Actions


def test_status_unload_and_system_prompt(page, engine):
    ctx = harness.fake_ctx(page)
    assert harness.run_action(page, "status", ctx)["__outputs__"]["status"] == {"text": "⚪ Weights present, model not loaded"}
    engine.load("bf16")
    assert harness.run_action(page, "status", ctx)["__outputs__"]["status"] == {"text": "✅ Model loaded ({{quant}}): ready", "params": {"quant": "bf16"}}
    harness.run_action(page, "unload", ctx)
    assert not engine.is_loaded()
    assert harness.run_action(page, "reset_system_prompt", ctx)["system_prompt"]["value"].startswith("You are an expert image captioner.")


def test_unload_waits_for_the_running_tasks(page, engine, tmp_path):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    other = harness.fake_ctx(page, files, user="someone")
    engine.gate.clear()
    harness.run_action(page, "caption", ctx, {"image": files.add_image(Image.new("RGB", (8, 8)))})
    refused = harness.run_action(page, "unload", other)  # the model is shared by every user
    assert page.busy("caption") and refused["__toast__"]["level"] == "warning" and engine.unloads == 0
    engine.gate.set()
    harness.wait_task(page, ctx, "caption")
    engine.load("bf16")
    assert "__toast__" not in harness.run_action(page, "unload", other) and engine.unloads == 1


def test_caption_type_editor(page, jc):
    ctx = harness.fake_ctx(page)
    loaded = harness.run_action(page, "load_type", ctx, {"edit_select": "Straightforward"})
    assert loaded["edit_name"] == {"value": "Straightforward"} and loaded["edit_any"]["value"].startswith("Write a straightforward caption")
    assert harness.run_action(page, "save_type", ctx, {"edit_name": " "})["__outputs__"]["edit_status"] == {"text": "❗ Name is required"}
    saved = harness.run_action(page, "save_type", ctx, {"edit_name": "Mine", "edit_any": "A", "edit_word": "W {word_count}", "edit_length": "L {length}"})
    assert saved["caption_type"]["value"] == "Mine" and {"value": "Mine", "label": "Mine"} in saved["caption_type"]["choices"]
    assert saved["edit_select"]["choices"][0] == {"value": "", "label": "Select a caption type"}
    assert jc.jcp.load_user_caption_types() == {"Mine": ["A", "W {word_count}", "L {length}"]}
    # the new type is offered by the next schema too
    assert any(choice["value"] == "Mine" for choice in _choice(page.schema(), "caption_type"))
    deleted = harness.run_action(page, "delete_type", ctx, {"edit_name": "Mine"})
    assert deleted["__outputs__"]["edit_status"] == {"text": "✅ Deleted user override '{{name}}'", "params": {"name": "Mine"}}
    assert deleted["caption_type"]["value"] == "Descriptive (Formal)" and deleted["edit_select"]["value"] == ""
    builtin = harness.run_action(page, "delete_type", ctx, {"edit_name": "Straightforward"})
    assert builtin["__outputs__"]["edit_status"]["text"] == "❗ '{{name}}' is built-in. Delete only removes user overrides."


def _choice(schema, name):
    return next(node for node in _walk(schema["layout"]) if node.get("name") == name)["choices"]


def test_a_new_image_fills_the_combined_box_with_its_prompt(page, tmp_path):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    image_id = files.add_image(_png_with_prompt("1girl, smile\nNegative prompt: bad\nSteps: 20, Sampler: Euler"))
    assert harness.run_action(page, "image_changed", ctx, {"image": image_id}) == {"combined": {"value": "1girl, smile"}}
    plain = files.add_image(Image.new("RGB", (8, 8)), "plain.png")
    assert harness.run_action(page, "image_changed", ctx, {"image": plain}) == {"combined": {"value": ""}}
    assert harness.run_action(page, "image_changed", ctx, {"image": None}) == {"combined": {"value": ""}}


def test_caption_task(page, engine, tmp_path):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    assert harness.run_action(page, "caption", ctx, {})["__toast__"]["level"] == "warning"  # no image
    image_id = files.add_image(_png_with_prompt("1girl,\nSteps: 20, Sampler: Euler", size=(2048, 1024)))
    values = {"image": image_id, "caption_type": "Straightforward", "caption_length": "any", "keep_loaded": False, "temperature": 0.3, "top_k": 5, "use_system_prompt": False}
    started = harness.run_action(page, "caption", ctx, values)
    assert started["__outputs__"]["task:caption"]["state"] in ("queued", "running", "completed")
    done = harness.wait_task(page, ctx, "caption")
    assert done["caption_out"] == {"value": "caption 1"}
    assert done["prompt_out"]["value"].startswith("Write a straightforward caption for this image.")
    assert done["combined"] == {"value": "1girl, caption 1"}  # the image's prompt, then the caption
    call = engine.calls[0]
    assert call["size"][0] * call["size"][1] <= 1024 * 1024 and call["system"] == "" and call["temperature"] == 0.3 and call["top_k"] == 5
    assert not engine.is_loaded()  # keep_loaded off


def test_caption_with_custom_prompt_and_extras(page, engine, tmp_path):
    files = harness.FakeFiles(tmp_path)
    ctx = harness.fake_ctx(page, files)
    image_id = files.add_image(Image.new("RGB", (32, 32)))
    harness.run_action(page, "caption", ctx, {"image": image_id, "use_custom": True, "custom_prompt": "  Describe it.  "})
    done = harness.wait_task(page, ctx, "caption")
    assert done["prompt_out"] == {"value": "Describe it."} and done["combined"] == {"value": "caption 1"}
    assert engine.is_loaded() and engine.calls[0]["system"].startswith("You are an expert image captioner.")


def test_batch_task(page, engine, tmp_path, monkeypatch):
    folder = tmp_path / "set"
    folder.mkdir()
    for name in ("a.png", "b.jpg"):
        Image.new("RGB", (16, 16)).save(folder / name)
    (folder / "b.caption").write_text("old", encoding="utf-8")
    ctx = harness.fake_ctx(page)
    assert harness.run_action(page, "batch", ctx, {"folder": ""})["__toast__"]["level"] == "warning"
    assert harness.run_action(page, "batch", ctx, {"folder": str(tmp_path / "missing")})["__toast__"]["message"]["text"] == "❗ No images found in: {{folder}}"
    harness.run_action(page, "batch", ctx, {"folder": str(folder), "prefix": "p, ", "suffix": " .s", "save_ext": ".caption"})
    done = harness.wait_task(page, ctx, "batch")
    assert (folder / "a.caption").read_text(encoding="utf-8") == "p, caption 1 .s"
    assert (folder / "b.caption").read_text(encoding="utf-8") == "old"  # not overwritten
    assert "Done. Captioned 1, skipped 1, errors 0." in done["__outputs__"]["log"]
    assert done["__toast__"]["message"] == {"text": "Done. Captioned {{done}}, skipped {{skipped}}, errors {{errors}}.", "params": {"done": 1, "skipped": 1, "errors": 0}}


def test_batch_can_be_cancelled_and_is_refused_with_hidden_folders(page, engine, tmp_path, monkeypatch, jc):
    folder = tmp_path / "set"
    folder.mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8)).save(folder / f"{index}.png")
    ctx = harness.fake_ctx(page)
    engine.gate.clear()
    harness.run_action(page, "batch", ctx, {"folder": str(folder)})
    harness.run_action(page, "__cancel__", ctx, task="batch")
    engine.gate.set()
    done = harness.wait_task(page, ctx, "batch")
    assert "Cancelled." in done["__outputs__"]["log"] and len(engine.calls) <= 1
    monkeypatch.setattr(jc, "_folders_hidden", lambda: True)
    assert harness.run_action(page, "batch", ctx, {"folder": str(folder)})["__toast__"]["level"] == "error"


def test_load_task_reports_the_download_phase(page, engine, jc, monkeypatch):
    monkeypatch.setattr(jc.jcm, "is_model_downloaded", lambda: False)
    seen = []
    original = jc._ensure_loaded

    def spy(task, quant):
        original(task, quant)
        seen.append(task.text)

    monkeypatch.setattr(jc, "_ensure_loaded", spy)
    ctx = harness.fake_ctx(page)
    harness.run_action(page, "load", ctx, {"quant": "bf16 (best quality, ~17 GB VRAM)"})
    done = harness.wait_task(page, ctx, "load")
    assert seen == ["Downloading the model weights (~17 GB, first use only)…"] and engine.loaded_quant == "bf16"
    assert done["__toast__"] == {"level": "success", "message": "JoyCaption model loaded"}


# endregion


def test_zh_cn_covers_the_page(page):
    port_spec = importlib.util.spec_from_file_location("remi_port_harness", REPO / "tests" / "remi_port_harness.py")
    port = importlib.util.module_from_spec(port_spec)
    port_spec.loader.exec_module(port)
    port.assert_localized(EXT, [], extra=harness.page_strings(page, SCRIPT))
