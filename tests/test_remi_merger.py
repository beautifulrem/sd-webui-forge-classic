"""The Checkpoint Merger page (``remi/core/merger.py``, registered by
``scripts/checkpoint_merger.py``) and the merge it runs
(``modules.extras.run_modelmerger``): the legacy tab's controls, a real
merge of small checkpoints through the page's task, progress, cancel-free
validation and output paths that stay inside the checkpoint folder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from remi import pages, ui
from remi.server.testing import server_app
from remi.testing import webui_state

ROOT = Path(__file__).resolve().parents[1]
OWNER = "page:modelmerger"


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture(autouse=True)
def registry():
    pages.clear()
    yield
    pages.clear()


@pytest.fixture
def models(webui, tmp_path, monkeypatch):
    """Three tiny checkpoints A, B, C in a temporary checkpoint folder."""

    from modules import sd_models, shared, shared_state

    monkeypatch.setattr(shared, "state", shared_state.State())  # what the merge reports through
    folder = tmp_path / "Stable-diffusion"
    folder.mkdir()
    monkeypatch.setattr(sd_models, "model_path", str(folder))
    infos = {}
    for name, value in (("a", 1.0), ("b", 3.0), ("c", 2.0)):
        path = folder / f"{name}.safetensors"
        # more than five UNet keys: the merge recognises the ldm prefix (huggingface_guess)
        weights = {f"model.diffusion_model.k{index}": torch.full((2,), value) for index in range(6)}
        weights.update({"model.diffusion_model.w": torch.full((2, 2), value), "model.diffusion_model.v": torch.full((3,), value)})
        save_file(weights, str(path), metadata={"origin": name})
        infos[f"{name}.safetensors"] = SimpleNamespace(filename=str(path), name=f"{name}.safetensors", model_name=name, title=f"{name}.safetensors [{name * 10}]", metadata={"origin": name}, sha256=f"sha-{name}", hash=f"h-{name}", calculate_shorthash=lambda: None)
    monkeypatch.setattr(sd_models, "checkpoint_aliases", dict(infos))
    monkeypatch.setattr(sd_models, "checkpoints_list", {info.title: info for info in infos.values()})
    listed = []
    monkeypatch.setattr(sd_models, "list_models", lambda: listed.append(1))
    return SimpleNamespace(folder=folder, listed=listed)


def _layout():
    from remi.core import merger

    return merger.CheckpointMerger(merger.PAGE_ID, merger.TITLE).layout()


def test_the_page_keeps_the_legacy_tab_controls_in_order(models):
    """modules/ui_checkpoint_merger.py of the last Gradio revision: Save
    Current Model, A / B / C, method and multiplier, filename, discard,
    Save Metadata with its sources, recipe and preview, Merge."""

    layout = _layout()
    names = [node.name for node in layout.inputs]
    assert names == [
        "save_filename",
        "save_unet",
        "save_checkpoint",
        "primary_model_name",
        "secondary_model_name",
        "tertiary_model_name",
        "interp_method",
        "multiplier",
        "custom_name",
        "discard_weights",
        "save_metadata",
        "config_source",
        "add_merge_recipe",
        "merge",
    ]
    defaults = layout.defaults(ui.Context())
    assert defaults["interp_method"] == "Weighted Sum" and defaults["multiplier"] == 0.5 and defaults["save_filename"] == "flux.safetensors"
    assert defaults["config_source"] == ["A", "B", "C"] and defaults["save_metadata"] is True and defaults["add_merge_recipe"] is True
    by_name = layout.by_name
    choices = by_name["primary_model_name"].schema(ui.Context())["choices"]
    # labelled by title, valued by file name (stable when the title gains its hash)
    assert [choice["value"] for choice in choices] == ["", "a.safetensors", "b.safetensors", "c.safetensors"]
    assert choices[1]["label"] == "a.safetensors [aaaaaaaaaa]"
    # B and the multiplier hide without interpolation, C shows for Add Difference only (the Gradio update_method)
    assert by_name["secondary_model_name"].visible_if == {"param": "interp_method", "ne": "No Interpolation"}
    assert by_name["multiplier"].visible_if == {"param": "interp_method", "ne": "No Interpolation"}
    assert by_name["tertiary_model_name"].visible_if == {"param": "interp_method", "eq": "Add Difference"}


def test_the_method_titles_match_the_merge(webui):
    from modules.extras import InterpolationMethod
    from remi.core import merger

    assert list(merger.METHODS) == InterpolationMethod.titles()


def test_the_script_registers_the_page_and_follows_the_life_cycle(webui):
    from modules import script_callbacks, script_loading

    script_loading.load_module(str(ROOT / "scripts" / "checkpoint_merger.py"))
    page = pages.get("modelmerger")
    assert page is not None and page.title == "Checkpoint Merger" and page.icon == "git-merge"
    script_callbacks.script_unloaded_callback()
    assert pages.get("modelmerger") is None


def test_output_paths_stay_in_the_checkpoint_folder(models):
    from modules import extras

    assert extras.merge_output_path("merged") == str((models.folder / "merged.safetensors").resolve())
    assert extras.merge_output_path("sub/merged.safetensors") == str((models.folder / "sub" / "merged.safetensors").resolve())
    for bad in ("../escape", "/tmp/escape", "sub/../../escape", "", "  "):
        with pytest.raises(ValueError):
            extras.merge_output_path(bad)


def test_weighted_sum_and_add_difference_merge_with_progress(models):
    from modules import extras, shared

    seen = []
    result = extras.run_modelmerger(None, "a.safetensors", "b.safetensors", "", "Weighted Sum", 0.25, "ws", "", True, ["B"], False, progress=lambda text, fraction: seen.append((text, fraction)))
    out = models.folder / "ws.safetensors"
    assert result == f"Checkpoint saved to {out.resolve()}"
    weights = load_file(str(out))
    assert torch.allclose(weights["model.diffusion_model.w"], torch.full((2, 2), 1.5))  # 1 × 0.75 + 3 × 0.25
    texts = [text for text, _ in seen]
    assert texts[0] == "Loading B" and "Loading A" in texts and "Merging A and B" in texts and texts[-1] == "Saving"
    fractions = [fraction for _, fraction in seen if fraction is not None]
    assert fractions == sorted(fractions) and fractions[0] >= 0 and fractions[-1] < 1
    assert shared.state.job == ""  # ended

    extras.run_modelmerger(None, "a.safetensors", "b.safetensors", "c.safetensors", "Add Difference", 0.5, "ad", "^model\\.diffusion_model\\.v$", False, [], False)
    weights = load_file(str(models.folder / "ad.safetensors"))
    assert torch.allclose(weights["model.diffusion_model.w"], torch.full((2, 2), 1.5))  # 1 + (3 − 2) × 0.5
    assert "model.diffusion_model.v" not in weights  # discarded


def test_a_failing_progress_stops_the_merge_and_ends_the_state(models):
    from modules import extras, shared

    class Stop(Exception):
        pass

    def stop(text, fraction):
        if text == "Loading A":
            raise Stop

    with pytest.raises(Stop):
        extras.run_modelmerger(None, "a.safetensors", "b.safetensors", "", "Weighted Sum", 0.5, "never", "", False, [], False, progress=stop)
    assert not (models.folder / "never.safetensors").exists()
    assert shared.state.job == ""
    assert extras.run_modelmerger(None, "", "", "", "Weighted Sum", 0.5, "", "", False, [], False) == "Failed: Missing Primary Model"


@pytest.fixture
def server(models, tmp_path):
    from remi.core import merger

    merger.register()
    with server_app(tmp_path / "data") as server:
        yield server


def _act(server, action, values, **extra):
    answer = server.client.post(f"/api/v2/actions/{OWNER}/{action}", json={"tab": OWNER, "values": values, **extra})
    assert answer.status_code == 200, answer.text
    return answer.json()


def _wait(server, task, task_id):
    deadline = time.time() + 20
    while time.time() < deadline:
        answer = _act(server, "__task__", {}, task=task, id=task_id)
        if answer["__outputs__"][f"task:{task}"]["done"]:
            return answer
        time.sleep(0.05)
    raise AssertionError("the task did not end")


def test_the_page_merges_through_a_task_and_announces_the_new_checkpoint(server, models, monkeypatch):
    import remi.networks

    announced = []
    monkeypatch.setattr(remi.networks, "notify_changed", lambda *kinds: announced.append(kinds))
    assert server.client.get("/api/v2/pages/modelmerger").json()["title"] == "Checkpoint Merger"
    values = {"primary_model_name": "a.safetensors", "secondary_model_name": "b.safetensors", "interp_method": "Weighted Sum", "multiplier": 0.5, "custom_name": "page-merge", "save_metadata": True, "config_source": ["A"], "add_merge_recipe": False}
    started = _act(server, "merge", values)
    status = started["__outputs__"]["task:merge"]
    final = _wait(server, "merge", status["id"])
    assert final["__outputs__"]["task:merge"]["state"] == "completed", final
    assert final["__outputs__"]["result"] == {"text": "Checkpoint saved to {{path}}", "params": {"path": str((models.folder / "page-merge.safetensors").resolve())}}
    assert final["__toast__"]["level"] == "success"
    merged = models.folder / "page-merge.safetensors"
    assert torch.allclose(load_file(str(merged))["model.diffusion_model.w"], torch.full((2, 2), 2.0))
    from safetensors import safe_open

    with safe_open(str(merged), "pt") as handle:
        assert handle.metadata() == {"origin": "a"}  # metadata copied from A only
    assert announced == [("checkpoints",)] and models.listed


def test_the_page_checks_its_inputs_before_starting(server, models):
    base = {"primary_model_name": "a.safetensors", "secondary_model_name": "b.safetensors", "interp_method": "Weighted Sum"}
    cases = [
        ({**base, "primary_model_name": ""}, "Choose the primary model (A)"),
        ({**base, "secondary_model_name": ""}, "Choose the secondary model (B)"),
        ({**base, "interp_method": "Add Difference"}, "Choose the tertiary model (C)"),
        ({**base, "discard_weights": "(["}, "Layers to Discard is not a valid regular expression"),
        ({**base, "custom_name": "../outside"}, "The filename must stay inside the checkpoint folder"),
        ({**base, "custom_name": "b"}, "The filename is one of the models being merged; choose another"),
    ]
    for values, message in cases:
        answer = _act(server, "merge", values)
        assert answer == {"__toast__": {"level": "warning", "message": message}}, values
    # no interpolation needs A only (a format conversion)
    started = _act(server, "merge", {"primary_model_name": "a.safetensors", "interp_method": "No Interpolation", "custom_name": "copy"})
    final = _wait(server, "merge", started["__outputs__"]["task:merge"]["id"])
    assert final["__outputs__"]["task:merge"]["state"] == "completed" and (models.folder / "copy.safetensors").is_file()
    from safetensors import safe_open

    with safe_open(str(models.folder / "copy.safetensors"), "pt") as handle:
        recipe = json.loads(handle.metadata()["sd_merge_recipe"])  # the defaults: metadata and the merge recipe
    assert recipe["interp_method"] == "No Interpolation" and recipe["primary_model_hash"] == "sha-a"


def test_refresh_and_metadata_preview(server, models):
    refreshed = _act(server, "refresh", {})
    assert set(refreshed) == {"primary_model_name", "secondary_model_name", "tertiary_model_name"}
    assert [choice["value"] for choice in refreshed["tertiary_model_name"]["choices"]][1:] == ["a.safetensors", "b.safetensors", "c.safetensors"]
    preview = _act(server, "read_metadata", {"primary_model_name": "a.safetensors", "secondary_model_name": "c.safetensors"})
    assert preview == {"__outputs__": {"metadata_json": {"a.safetensors": {"origin": "a"}, "c.safetensors": {"origin": "c"}}}}
    assert _act(server, "read_metadata", {})["__toast__"]["level"] == "warning"


def test_save_current_model_rejects_paths_outside_the_folder(server):
    assert _act(server, "save_unet", {"save_filename": "../../x"}) == {"__toast__": {"level": "warning", "message": "The filename must stay inside the checkpoint folder"}}


def test_the_page_strings_are_translated(models):
    """Every label, note, button and message of the page has a zh-CN entry."""

    from remi.core import merger

    catalog = json.loads((ROOT / "localizations" / "zh_CN.json").read_text(encoding="utf-8"))
    schema = merger.CheckpointMerger(merger.PAGE_ID, merger.TITLE).schema()
    found: set[str] = {merger.TITLE, *merger.STAGES}

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        for key in ("label", "info", "title", "text"):
            if isinstance(node.get(key), str) and node[key].strip() and node.get("type") != "html":
                found.add(node[key])
        props = node.get("props")
        if isinstance(props, dict) and isinstance(props.get("label"), str):
            found.add(props["label"])
        for choice in () if str(node.get("name", "")).endswith("_model_name") else node.get("choices") or ():
            if isinstance(choice, dict) and isinstance(choice.get("label"), str):
                found.add(choice["label"])
        for key in ("children", "enable", "layout"):
            if key in node:
                walk(node[key])

    walk(schema["layout"])
    source = (ROOT / "remi" / "core" / "merger.py").read_text(encoding="utf-8")
    import re

    found.add("Select a model")  # the model lists' empty entry (the file names are data)
    found |= set(re.findall(r'(?:toast|message)\(\s*"([^"]+)"', source))
    found |= set(re.findall(r'text = "([^"]+)" if unet else "([^"]+)"', source)[0])
    found |= set(re.findall(r'task\.progress\(None, "([^"]+)"\)', source))
    found |= set(re.findall(r'(?:return|problem =) "([^"]+)"', source))  # _problem's toasts
    found |= set(merger.METHOD_NOTES.values())
    found |= set(merger.METHODS)
    missing = sorted(text for text in found if text not in {"A", "B", "C"} and not catalog.get(text))  # model letters are data
    assert len(found) > 40 and missing == [], "\n".join(missing)
