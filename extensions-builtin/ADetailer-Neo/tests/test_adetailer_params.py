"""ADetailer on ``params()``: the legacy ``ui()`` contract (recorded in
``fixtures/legacy_ui_<tab>.json``), the schema, the processing arguments
(form scalars packed into the dicts the legacy ``gr.State`` held, A1111 API
dicts unchanged), infotext paste, the Refresh action, settings, X/Y/Z axes,
routes, no Gradio and the zh_CN catalog.

Documented changes from the legacy ``ui()``:

- The 4 ``gr.State`` arguments (one dict per tab, filled by ``.change``
  listeners and a listener on the core Generate button) became each tab's
  params, named ``<attr><suffix>`` (``ad_model``, ``ad_model_2nd``, ...) and
  passed positionally in ``ALL_ARGS`` order; ``pack_args`` rebuilds the dicts.
- Labels inside a tab lost their ordinal (the tab title shows it); the
  prompts, unlabelled before, are labelled with their former placeholder's
  first line ("ADetailer Prompt"), the placeholder keeping the hint.
- Show/hide listeners became ``visible_if`` on the "Use separate ..." toggles;
  Detector Classes shows for YOLO-World detectors (``visible_if`` on a hidden
  per-tab flag the ``ad_model_changed`` action sets; the flags are the last
  positional args, never processed), and is disabled without CLIP
  (``enabled_if``).
- The ControlNet guidance RangeSlider is ``ui.RangeSlider`` (``range-slider``
  widget) on its infotext key; its paste reads ``"(a, b)"`` as before.
- Pasting switches ADetailer (and a tab) on when the infotext has that tab's
  ``ADetailer model`` key and the tabs it lacks off; the legacy paste never
  did (nothing writes "ADetailer enable" / "ADetailer tab enable").
- A "Refresh models" button rescans the detector models.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from adetailer_testlib import EXT, FIXTURES, REPO, SCRIPT, contract, kit, load_adetailer
from lib_adetailer import ui as adui

from remi import ui
from remi.testing import webui_state

TABS = ("txt2img", "img2img")


# Detector files the tests select (models/adetailer is gitignored)
FAKE_MODELS = {name: f"/fake/adetailer/{name}" for name in ("face_yolov8n.pt", "hand_yolov8n.pt", "hand_yolov8s.pt", "yolov8x-worldv2.pt")}


@pytest.fixture(scope="module")
def ad():
    with webui_state():
        module = load_adetailer()
        saved = module.model_mapping
        module.model_mapping = dict(FAKE_MODELS)
        try:
            yield module
        finally:
            module.model_mapping = saved


@pytest.fixture
def models(ad, monkeypatch):
    """``ad_max_models`` for the layouts built in a test."""

    from modules import shared

    def use(count):
        monkeypatch.setitem(shared.opts.data, "ad_max_models", count)

    use(4)
    return use


def make(ad, tab="txt2img"):
    script, layout = kit.layout_of(ad.AfterDetailerScript, tab, filename=str(SCRIPT))
    script.layout = layout
    return script


def snapshot(tab):
    return json.loads((FIXTURES / f"legacy_ui_{tab}.json").read_text(encoding="utf-8"))


def conditions(root):
    """``{param name: (visible_if conditions, enabled_if conditions)}`` along
    each input's path (its own and its containers')."""

    found = {}

    def visit(node, visible, enabled):
        visible = visible + ([node.visible_if] if node.visible_if is not None else [])
        enabled = enabled + ([node.enabled_if] if node.enabled_if is not None else [])
        if isinstance(node, ui.Input):
            found[node.name] = (visible, enabled)
        for child in node.children:
            visit(child, visible, enabled)

    visit(root, [], [])
    return found


# region Legacy contract

# attr -> label now (the tab title shows the ordinal)
LABELS = {"ad_model": "ADetailer Detector", "ad_model_classes": "Detector Classes", "ad_prompt": "ADetailer Prompt", "ad_negative_prompt": "ADetailer Negative Prompt"}
# attr -> the toggle that shows it (the legacy .change visibility listeners)
SHOWN_BY = {
    "ad_inpaint_only_masked_padding": "ad_inpaint_only_masked",
    "ad_noise_multiplier": "ad_use_noise_multiplier",
    "ad_sampler": "ad_use_sampler",
    "ad_scheduler": "ad_use_sampler",
    "ad_inpaint_width": "ad_use_inpaint_width_height",
    "ad_inpaint_height": "ad_use_inpaint_width_height",
    "ad_steps": "ad_use_steps",
    "ad_cfg_scale": "ad_use_cfg_scale",
    "ad_checkpoint": "ad_use_checkpoint",
    "ad_vae": "ad_use_vae",
}
# Server-state lists: (fixed leading choices, the WebuiInfo source or callable)
DYNAMIC = {"ad_model", "ad_sampler", "ad_scheduler", "ad_checkpoint", "ad_vae", "ad_controlnet_model", "ad_controlnet_module"}


@pytest.mark.parametrize("tab", TABS)
def test_main_toggles_keep_the_legacy_contract(ad, models, tab):
    script = make(ad, tab)
    legacy = snapshot(tab)
    assert legacy["options"]["ad_max_models"] == 4 and legacy["count"] == 2 + 4
    entries = contract.describe(script.layout)
    enable, skip = entries[:2]
    want = dict(legacy["inputs"][0], paste=True)  # documented: the paste infers "on" from the model keys
    assert {key: enable.get(key) for key in want} == want
    assert {key: skip.get(key) for key in legacy["inputs"][1]} == legacy["inputs"][1]
    # The legacy checkbox was hidden on txt2img
    assert script.layout.by_name["ad_skip_img2img"].hidden is (tab == "txt2img")
    # One gr.State per tab before; now every tab's params, in ALL_ARGS order
    assert [entry["kind"] for entry in legacy["inputs"][2:]] == ["state"] * 4
    assert script.layout.names == list(ad.param_order(4)) and len(script.layout) == 2 + 4 * (len(adui.ALL_ARGS) + 1)
    assert script.layout.names[-4:] == ["ad_model_is_world", "ad_model_is_world_2nd", "ad_model_is_world_3rd", "ad_model_is_world_4th"]


@pytest.mark.parametrize("tab", TABS)
def test_tab_params_keep_the_legacy_fields(ad, models, tab):
    script = make(ad, tab)
    layout, legacy = script.layout, snapshot(tab)
    ctx = ui.Context()
    paths = conditions(layout.root)
    for n, fields in enumerate(legacy["fields"]):
        assert [field["attr"] for field in fields] == list(adui.ALL_ARGS.attrs)
        defaults = {node.name: node.resolve_default(ctx) for node in layout.inputs}
        for field in fields:
            attr = field["attr"]
            name = ad.param_name(attr, n)
            node = layout.by_name[name]
            schema = node.schema(ctx)
            where = f"{tab} tab {n + 1} {attr}"

            label = LABELS.get(attr, field["label"])
            assert schema["label"] == label, where
            assert schema.get("info") == field.get("info"), where
            assert node.infotext == field["infotext"].removeprefix("<paste function: ").removesuffix(">"), where
            assert node.preset_key is None, where

            if field["kind"] == "RangeSlider":
                assert isinstance(node, ui.RangeSlider) and schema["widget"] == "range-slider", where
                assert schema["props"] == {key: field[key] for key in ("minimum", "maximum", "step")}, where
                assert schema["default"] == field["default"], where
                continue
            assert schema["type"] == field["kind"], where
            for key in ("minimum", "maximum", "step", "integer", "lines", "max_lines", "style"):
                assert schema.get(key) == field.get(key), (where, key)
            if attr == "ad_checkpoint":
                assert schema["default"] == next(iter(ad.WEBUI_INFO.checkpoints_list()), None), where
            else:
                assert schema["default"] == field["default"], where

            choices = [choice["value"] for choice in schema.get("choices", ())]
            if attr == "ad_model":
                assert choices == ["None", *ad.model_mapping], where
            elif attr == "ad_sampler":
                assert choices == ["Use same sampler", *ad.WEBUI_INFO.sampler_names()], where
            elif attr == "ad_scheduler":
                assert choices == ["Use same scheduler", *ad.WEBUI_INFO.scheduler_names()], where
            elif attr in DYNAMIC:
                assert choices[:1] == field["choices"][:1] or not field["choices"], where
            elif "choices" in field:
                assert choices == field["choices"], where

            if attr in ("ad_prompt", "ad_negative_prompt"):
                assert schema["prompt"] is True and schema["placeholder"].startswith("If blank, the original"), where

            visible, enabled = paths[name]
            own = [c for c in visible if c is not False]
            if attr in SHOWN_BY:
                assert own == [ui.param(ad.param_name(SHOWN_BY[attr], n))], where
            elif attr == "ad_model_classes":
                assert own == [ui.param(ad.param_name(adui.WORLD_FLAG, n))], where
            elif attr.startswith("ad_controlnet"):
                assert visible == ([] if adui.CNET_AVAILABLE else [False]), where
            else:
                assert own == [], where
            # Shown at first exactly as the legacy control was
            shown = all(ui.evaluate(c, defaults) for c in visible)
            if not attr.startswith("ad_controlnet"):
                assert shown == field["visible"], where
            if attr != "ad_model_classes":
                assert enabled == [], where


def test_tab_count_defaults_to_the_setting_default(ad, monkeypatch):
    from modules import shared

    monkeypatch.delitem(shared.opts.data, "ad_max_models", raising=False)
    assert ad.num_models() == ad.DEFAULT_MAX_MODELS == 4
    monkeypatch.setitem(shared.opts.data, "ad_max_models", 99)
    assert ad.num_models() == ad.MAX_MODELS


def test_tabs_and_their_titles(ad, models):
    models(3)
    script = make(ad)
    tabs = next(node for node in script.layout.root.walk() if isinstance(node, ui.Tabs))
    assert [tab.title for tab in tabs.children] == ["1st", "2nd", "3rd"]
    assert [script.layout.by_name[ad.param_name("ad_tab_enable", n)].default for n in range(3)] == [True, False, False]
    groups = [node.title for node in tabs.children[1].walk() if isinstance(node, ui.Group)]
    assert groups == ["Detection", "Mask Preprocessing", "Inpaint Parameters", "ControlNet"]
    button = next(node for node in tabs.children[0].walk() if isinstance(node, ui.Button))
    assert (button.label, button.action) == ("Refresh models", "refresh_models")


# endregion

# region Schema and processing arguments


@pytest.mark.parametrize("tab", TABS)
def test_schema_export(ad, models, tab):
    with webui_state({"ad_max_models": 2}, script_classes=[ad.AfterDetailerScript]) as state:
        runner = state.img2img if tab == "img2img" else state.txt2img
        entry = next(item for item in runner.scripts_schema() if item["title"] == "ADetailer")
        assert entry["id"] == "adetailer" and entry["kind"] == "alwayson"
        assert entry["args"] == list(ad.param_order(2))
        assert json.loads(json.dumps(entry)) == entry  # plain JSON
        assert entry["defaults"]["ad_controlnet_guidance_start_end_2nd"] == [0.0, 1.0]
        script = runner.by_id["adetailer"]
        assert script.args_to - script.args_from == 2 + 2 * (len(adui.ALL_ARGS) + 1)
        assert set(ui.actions_of(script)) == {"refresh_models", "ad_model_changed"}


def legacy_unit(ad, tab="txt2img", **values):
    """A tab's dict as the legacy gr.State held it (its widgets' values)."""

    # The checkpoint dropdown started on the first checkpoint (none in tests)
    return {**snapshot(tab)["inputs"][2]["default"], "ad_checkpoint": ad.WEBUI_INFO.checkpoint_default, **values}


def process_args(ad, script, values):
    return ad.pack_args(script.layout.values_to_args(values))


def test_form_values_pack_into_the_legacy_dicts(ad, models):
    script = make(ad)
    tab2 = {"ad_model": "face_yolov8n.pt", "ad_prompt": "detailed face", "ad_use_steps": True, "ad_steps": 12, "ad_controlnet_guidance_start_end": (0.2, 0.6), "ad_tab_enable": True}
    values = {"ad_enable": True, **{ad.param_name(attr, 1): value for attr, value in tab2.items()}}
    args = process_args(ad, script, values)
    assert args[:2] == (True, False) and len(args) == 2 + 4
    assert all(isinstance(unit, dict) and list(unit) == list(adui.ALL_ARGS.attrs) for unit in args[2:])
    # The same ADetailerArgs the legacy UI's states gave process()
    default = ad.ADetailerArgs(**legacy_unit(ad))
    assert ad.ADetailerArgs(**args[2]) == default
    assert ad.ADetailerArgs(**args[3]) == ad.ADetailerArgs(**legacy_unit(ad, **tab2))
    assert ad.ADetailerArgs(**args[3]).ad_controlnet_guidance_start_end == (0.2, 0.6)
    # The UI-only flags are dropped, and scalars without them pack the same
    flat = script.layout.values_to_args(values)
    assert flat[-4:] == [False] * 4
    assert ad.pack_args(flat[:-4]) == args


def test_checkpoint_title_from_before_its_hash_was_known_is_accepted(ad, models, monkeypatch):
    """A page loaded before the checkpoint's hash was computed holds
    ``anima`` as every tab's default; once the first generation hashed it the
    choice reads ``anima [fba1195327]``. A second Generate from that page must
    not be rejected (422 ``'anima' is not one of the choices``)."""

    script = make(ad)
    monkeypatch.setattr(ad.WEBUI_INFO, "checkpoints_list", lambda: ["anima"])
    stale = script.layout.coerce({})
    assert {stale[ad.param_name("ad_checkpoint", n)] for n in range(4)} == {"anima"}
    monkeypatch.setattr(ad.WEBUI_INFO, "checkpoints_list", lambda: ["anima [fba1195327]", "other [0123abcd]"])
    values = script.layout.coerce(stale)
    assert {values[ad.param_name("ad_checkpoint", n)] for n in range(4)} == {"anima [fba1195327]"}
    # and the hashed title of an infotext from another machine before this one hashed it
    monkeypatch.setattr(ad.WEBUI_INFO, "checkpoints_list", lambda: ["anima", "other"])
    assert script.layout.coerce({"ad_checkpoint": "other [0123abcd]"})["ad_checkpoint"] == "other"


def test_world_flag_follows_the_detector(ad, models):
    script = make(ad)
    layout = script.layout
    assert layout.by_name["ad_model"].on_change == "ad_model_changed"
    flag = layout.by_name["ad_model_is_world_2nd"]
    assert isinstance(flag, ui.State) and flag.default is False and flag.submit is False
    ctx = ui.ActionContext(tab="txt2img", owner="adetailer")
    values = {"ad_model": "face_yolov8n.pt", "ad_model_2nd": "yolov8x-worldv2.pt", "ad_model_3rd": "None"}
    patch = ui.call_action(script, "ad_model_changed", ctx, values)
    assert patch == {"ad_model_is_world": {"value": False}, "ad_model_is_world_2nd": {"value": True}, "ad_model_is_world_3rd": {"value": False}}
    # Detector Classes shows exactly while the flag is set
    classes = conditions(layout.root)["ad_model_classes_2nd"][0]
    assert all(ui.evaluate(c, {"ad_model_is_world_2nd": True}) for c in classes)
    assert not all(ui.evaluate(c, {"ad_model_is_world_2nd": False}) for c in classes)


class FakeP(SimpleNamespace):
    def __init__(self):
        super().__init__(extra_generation_params={})


def run_process(ad, args):
    script = ad.AfterDetailerScript()
    p = FakeP()
    script.process(p, *args)
    return p


def test_process_gets_the_same_arguments_from_the_form_and_the_legacy_dicts(ad, models):
    script = make(ad)
    tab1 = {"ad_model": "face_yolov8n.pt", "ad_negative_prompt": "blurry", "ad_mask_k": 2, "ad_x_offset": 8, "ad_use_sampler": True, "ad_sampler": "Euler", "ad_controlnet_guidance_start_end": (0.0, 0.5)}
    tab3 = {"ad_model": "hand_yolov8n.pt", "ad_tab_enable": True, "ad_denoising_strength": 0.35}
    values = {"ad_enable": True, **tab1, **{ad.param_name(attr, 2): value for attr, value in tab3.items()}}
    form = run_process(ad, script.layout.values_to_args(values))
    legacy = run_process(ad, (True, False, legacy_unit(ad, **tab1), legacy_unit(ad, ad_tab_enable=False), legacy_unit(ad, **tab3), legacy_unit(ad, ad_tab_enable=False)))
    assert form.extra_generation_params == legacy.extra_generation_params
    params = form.extra_generation_params
    assert params["ADetailer model"] == "face_yolov8n.pt" and params["ADetailer model 3rd"] == "hand_yolov8n.pt"
    assert not hasattr(form, "_ad_disabled")

    disabled = run_process(ad, script.layout.values_to_args({"ad_model": "face_yolov8n.pt"}))
    assert disabled._ad_disabled is True and disabled.extra_generation_params == {}


def test_api_dicts_reach_process_as_before(ad, models):
    """``alwayson_scripts: {"ADetailer": {"args": [true, false, {...}]}}``
    fills the first slots; the form params after them keep their defaults
    and are not read."""

    script = make(ad)
    defaults = script.layout.values_to_args({})
    unit = {"ad_model": "face_yolov8n.pt", "ad_confidence": 0.5}
    for sent in ([True, False, unit], [unit]):
        args = list(defaults)
        args[: len(sent)] = sent
        assert ad.pack_args(args) == tuple(args)
        p = run_process(ad, args)
        assert p.extra_generation_params["ADetailer model"] == "face_yolov8n.pt" and p.extra_generation_params["ADetailer confidence"] == 0.5
        assert "ADetailer model 2nd" not in p.extra_generation_params
    p = run_process(ad, [False, False, unit, *defaults[3:]])
    assert p._ad_disabled is True


def test_xyz_overrides_apply_to_the_first_tab(ad, models):
    script = make(ad)
    p = FakeP()
    p._ad_xyz = {"ad_model": "hand_yolov8s.pt", "ad_denoising_strength": 0.25}
    ad.AfterDetailerScript().process(p, *script.layout.values_to_args({"ad_enable": True, "ad_model": "face_yolov8n.pt"}))
    assert p.extra_generation_params["ADetailer model"] == "hand_yolov8s.pt"
    assert p.extra_generation_params["ADetailer denoising strength"] == 0.25


# endregion

# region Infotext


def paste(script, params):
    values, problems = script.layout.paste({key: str(value) for key, value in params.items()})
    assert problems == []
    return values


def test_infotext_round_trip(ad, models):
    script = make(ad)
    tab1 = {"ad_model": "face_yolov8n.pt", "ad_prompt": "a, b [SEP] c", "ad_confidence": 0.45, "ad_mask_merge_invert": "Merge", "ad_inpaint_only_masked": False, "ad_use_cfg_scale": True, "ad_cfg_scale": 6.5, "ad_restore_face": True}
    tab2 = {"ad_model": "hand_yolov8n.pt", "ad_tab_enable": True, "ad_use_inpaint_width_height": True, "ad_inpaint_width": 768, "ad_inpaint_height": 640, "ad_use_noise_multiplier": True, "ad_noise_multiplier": 1.1}
    values = {"ad_enable": True, **tab1, **{ad.param_name(attr, 1): value for attr, value in tab2.items()}}
    p = run_process(ad, script.layout.values_to_args(values))

    pasted = paste(script, p.extra_generation_params)
    assert pasted["ad_enable"] is True and pasted["ad_tab_enable"] is True and pasted["ad_tab_enable_2nd"] is True
    assert "ad_model_3rd" not in pasted
    args = process_args(ad, script, pasted)
    for n, tab in enumerate((tab1, tab2)):
        unit = ad.ADetailerArgs(**args[2 + n])
        for attr, value in tab.items():
            assert getattr(unit, attr) == value, (n, attr)


def test_controlnet_guidance_pastes_its_tuple_text(ad, models):
    script = make(ad)
    values = paste(script, {"ADetailer ControlNet guidance start/end 2nd": "(0.1, 0.75)", "ADetailer ControlNet guidance start/end": "[0.3, 0.2]"})
    assert values["ad_controlnet_guidance_start_end_2nd"] == [0.1, 0.75]
    assert values["ad_controlnet_guidance_start_end"] == [0.2, 0.3]


def test_paste_switches_off_the_tabs_the_infotext_lacks(ad, models):
    """An infotext of a 2nd-tab-only run enables that tab alone: a tab enabled
    in the form must not add a pass the pasted generation never had."""

    script = make(ad)
    p = run_process(ad, script.layout.values_to_args({"ad_enable": True, "ad_tab_enable": False, "ad_tab_enable_2nd": True, "ad_model_2nd": "hand_yolov8n.pt"}))
    assert "ADetailer model" not in p.extra_generation_params and p.extra_generation_params["ADetailer model 2nd"] == "hand_yolov8n.pt"
    pasted = paste(script, p.extra_generation_params)
    assert pasted["ad_enable"] is True
    assert [pasted[ad.param_name("ad_tab_enable", n)] for n in range(4)] == [False, True, False, False]
    # Form: 1st and 3rd tabs on with models; after the paste only the 2nd runs
    form = {"ad_model": "face_yolov8n.pt", "ad_tab_enable_3rd": True, "ad_model_3rd": "hand_yolov8s.pt"}
    replayed = run_process(ad, script.layout.values_to_args({**form, **pasted}))
    assert {key for key in replayed.extra_generation_params if key.startswith("ADetailer model")} == {"ADetailer model 2nd"}


def test_paste_without_adetailer_keys_changes_nothing(ad, models):
    assert paste(make(ad), {"Steps": 20, "Sampler": "Euler"}) == {}
    assert paste(make(ad), {"ADetailer enable": False, "ADetailer model": "face_yolov8n.pt"})["ad_enable"] is False


# endregion

# region Refresh, settings, X/Y/Z, routes


def test_refresh_rescans_the_detectors(ad, models, monkeypatch):
    saved = dict(ad.model_mapping)
    calls = []

    def fake_get_models(*dirs, download=True):
        calls.append(download)
        return {"face_yolov8n.pt": "/m/face_yolov8n.pt", "new-worldv2.pt": "/m/new-worldv2.pt"}

    monkeypatch.setattr(ad, "get_models", fake_get_models)
    script = make(ad)
    try:
        values = {"ad_model": "face_yolov8n.pt", "ad_model_2nd": "gone.pt", "ad_vae": "None", "ad_model_3rd": "new-worldv2.pt"}
        patch = ui.call_action(script, "refresh_models", ui.ActionContext(tab="txt2img", owner="adetailer"), values)
        assert calls == [False]  # scan only, never a download
        assert list(ad.model_mapping) == ["face_yolov8n.pt", "new-worldv2.pt"]
        assert [choice["value"] for choice in patch["ad_model"]["choices"]] == ["None", "face_yolov8n.pt", "new-worldv2.pt"]
        assert "value" not in patch["ad_model"] and patch["ad_model_2nd"]["value"] == "None"
        # The World flags follow the (possibly reset) selections
        assert [patch[f"ad_model_is_world{ad.suffix(n, '_')}"] for n in range(3)] == [{"value": False}, {"value": False}, {"value": True}]
        assert "ad_model_is_world_4th" not in patch
        assert {f"ad_model{ad.suffix(n, '_')}" for n in range(4)} <= set(patch)
        assert "ad_controlnet_model_4th" in patch and "ad_vae" in patch and "value" not in patch["ad_vae"]
        assert patch["__toast__"]["level"] == "success" and patch["__toast__"]["message"] == {"text": "{{count}} ADetailer models", "params": {"count": 2}}
        # Jobs validate against the rescanned list
        assert process_args(ad, script, {"ad_model": "new-worldv2.pt"})[2]["ad_model"] == "new-worldv2.pt"
    finally:
        ad.model_mapping = saved


@pytest.fixture
def downloads(monkeypatch):
    """Stock model downloads allowed (no --ad-no-huggingface); every
    download recorded and written as a stub file instead of fetched."""

    from lib_adetailer.detection import common

    from modules import shared

    fetched = []

    def fake_download(url, dst, progress=True):
        fetched.append((url, dst))
        with open(dst, "wb") as file:
            file.write(b"stub")

    monkeypatch.setattr(shared.cmd_opts, "ad_no_huggingface", False)
    monkeypatch.setattr(common, "download_url_to_file", fake_download)
    return common, fetched


def test_scanning_never_downloads_and_lists_the_stock_models(downloads, tmp_path):
    common, fetched = downloads
    (tmp_path / "own.pt").write_bytes(b"x")
    (tmp_path / "face_yolov8n.pt").write_bytes(b"x")
    models = common.get_models(str(tmp_path), "", str(tmp_path / "missing"))
    assert fetched == []  # startup and Refresh only scan
    assert models["own.pt"] == str(tmp_path / "own.pt") and models["face_yolov8n.pt"] == str(tmp_path / "face_yolov8n.pt")
    # the stock detectors not downloaded yet are listed where they will go
    assert set(common.DOWNLOADS) <= set(models)
    assert models["hand_yolov8n.pt"] == str(tmp_path / "hand_yolov8n.pt") and not (tmp_path / "hand_yolov8n.pt").exists()


def test_without_downloads_only_the_files_are_listed(downloads, tmp_path, monkeypatch):
    common, fetched = downloads
    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "ad_no_huggingface", True)
    (tmp_path / "own.pt").write_bytes(b"x")
    assert common.get_models(str(tmp_path), download=True) == {"own.pt": str(tmp_path / "own.pt")}
    with pytest.raises(FileNotFoundError, match="--ad-no-huggingface"):
        common.download_model("hand_yolov8n.pt", tmp_path / "hand_yolov8n.pt")
    assert fetched == []


def test_a_stock_detector_is_downloaded_when_a_unit_first_uses_it(ad, downloads, tmp_path, monkeypatch):
    common, fetched = downloads
    monkeypatch.setattr(ad, "model_mapping", common.get_models(str(tmp_path)))
    script = make(ad)
    path = script.get_ad_model("hand_yolov8n.pt")
    assert path == str(tmp_path / "hand_yolov8n.pt") and (tmp_path / "hand_yolov8n.pt").read_bytes() == b"stub"
    assert fetched == [(common.DOWNLOADS["hand_yolov8n.pt"], path)]
    # once: the next run uses the file
    assert script.get_ad_model("hand_yolov8n.pt") == path and len(fetched) == 1
    # a listed file that is gone and has no download source is an error, not a fetch
    monkeypatch.setitem(ad.model_mapping, "own.pt", str(tmp_path / "own.pt"))
    with pytest.raises(FileNotFoundError):
        script.get_ad_model("own.pt")
    assert len(fetched) == 1


def test_loading_the_script_makes_no_download(monkeypatch, tmp_path):
    # The module-level scan at import (startup) with downloads allowed
    from lib_adetailer.detection import common

    from modules import shared

    monkeypatch.setattr(shared.cmd_opts, "ad_no_huggingface", False)
    monkeypatch.setattr(common, "download_url_to_file", lambda *args, **kwargs: pytest.fail("downloaded at startup"))
    with webui_state():
        module = kit.load_script(SCRIPT, "adetailer_startup_under_test")
        try:
            assert set(common.DOWNLOADS) <= set(module.model_mapping)
        finally:
            sys.modules.pop("adetailer_startup_under_test", None)


@pytest.fixture
def settings(ad):
    from modules import shared

    saved = dict(shared.opts.data_labels)
    try:
        ad.on_ui_settings()
        yield {key: info for key, info in shared.opts.data_labels.items() if key not in saved}
    finally:
        shared.opts.data_labels.clear()
        shared.opts.data_labels.update(saved)


def test_settings_keep_their_keys_and_use_remi_widgets(ad, settings):
    legacy = {entry["key"]: entry for entry in snapshot("txt2img")["settings"]}
    assert list(settings) == list(legacy)
    for key, info in settings.items():
        assert (info.label, info.default) == (legacy[key]["label"], legacy[key]["default"]), key
        assert info.component is None or info.component.__module__.startswith("remi.ui"), key
    assert settings["ad_max_models"].component is ui.Slider and settings["ad_max_models"].requires_reload_ui
    assert settings["ad_bbox_sortby"].component is ui.Choice and settings["ad_bbox_sortby"].component_args["style"] == "radio"
    assert settings["ad_match_inpaint_bbox_size"].component_args == {"choices": ["Off", "Strict", "Free"], "style": "radio"}
    # the mode descriptions are choice hints (translated text), not an HTML list
    assert list(settings["ad_match_inpaint_bbox_size"].choice_hints) == ["Off", "Strict", "Free"] and settings["ad_match_inpaint_bbox_size"].html_blocks == []
    assert settings["ad_extra_models_dir"].component is ui.Text


def test_registers_its_xyz_axes(ad, monkeypatch):
    import importlib.util

    from modules import scripts

    path = REPO / "scripts" / "xyz_grid.py"
    spec = importlib.util.spec_from_file_location("xyz_grid_for_adetailer", path)
    xyz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xyz)
    monkeypatch.setattr(xyz.Script, "__module__", "scripts.xyz_grid")
    monkeypatch.setattr(scripts, "scripts_data", [scripts.ScriptClassData(xyz.Script, str(path), str(REPO), xyz)])
    ad.on_before_ui()
    axes = {axis.label: axis for axis in xyz.axis_options if axis.label.startswith("[ADetailer]")}
    assert len(axes) == 14
    assert axes["[ADetailer] ADetailer Detector"].choices() == ["None", *ad.model_mapping]
    p = SimpleNamespace(prompt="a cat", negative_prompt="")
    axes["[ADetailer] Inpaint Denoising Strength"].apply(p, 0.3, [0.3])
    axes["[ADetailer] Prompt S/R"].apply(p, "dog", ["cat", "dog"])
    assert p._ad_xyz == {"ad_denoising_strength": 0.3} and p._ad_xyz_prompt_sr == [("cat", "dog")]
    ad.on_before_ui()  # idempotent
    assert sum(axis.label.startswith("[ADetailer]") for axis in xyz.axis_options) == 14


def test_routes(ad):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    ad.add_api_endpoints(None, app)
    client = TestClient(app)
    assert client.get("/adetailer/v1/version").json() == {"version": "Neo"}
    assert client.get("/adetailer/v1/ad_model").json() == {"ad_model": list(ad.model_mapping)}
    assert "ad_model" in client.get("/adetailer/v1/schema").json()["properties"]


# endregion

# region Ported and localization


def test_ported():
    sources = [path for path in sorted(EXT.rglob("*.py")) if "tests" not in path.relative_to(EXT).parts]
    for path in sources:
        contract.assert_ported(path)
    assert not list(EXT.glob("javascript/*")) and not (EXT / "style.css").exists()

    # Imports and builds both tabs' layouts in a fresh interpreter
    code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(EXT / 'tests')!r})",
            "import adetailer_testlib as lib",
            "from remi.testing import webui_state",
            "with webui_state({'ad_max_models': 2}):",
            "    ad = lib.load_adetailer()",
            "    for tab in ('txt2img', 'img2img'):",
            "        lib.kit.layout_of(ad.AfterDetailerScript, tab)",
            "print('standalone')",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("standalone"), result.stderr[-4000:]


def test_zh_cn_covers_every_string(ad, models, settings):
    from modules import shared
    from remi.ui.settings import settings_schema

    models(ad.MAX_MODELS)
    strings = set()
    for tab in TABS:
        strings |= kit.schema_strings(make(ad, tab).layout.schema(ui.Context()))
    section = next(section for category in settings_schema(shared.opts)["categories"] for section in category["sections"] if section["id"] == "ADetailer")
    strings |= kit.schema_strings(section["items"])
    strings |= {note for item in section["items"] for note in item.get("notes", ())}
    strings -= {"path/to/models|another/path/to/models"}  # an example path
    # Server lists (file and sampler names) are not translated here
    info = ad.WEBUI_INFO
    strings -= {*ad.model_mapping, *info.sampler_names(), *info.scheduler_names(), *info.checkpoints_list(), *info.vae_list(), *ad.get_cn_models(), *ad.get_cn_modules()} - {"None"}
    extra = {
        # Both placeholders (with / without CLIP) and the Refresh toast
        'Comma-separated class names to detect (e.g. "person,cat" ; default: COCO 80 classes)',
        "Detector Classes require Clip to be installed (https://github.com/ultralytics/CLIP.git)",
        "{{count}} ADetailer models",
    }
    kit.assert_localized(EXT, strings, extra=extra)


# endregion
