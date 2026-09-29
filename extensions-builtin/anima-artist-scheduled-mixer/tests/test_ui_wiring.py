import gradio as gr

from modules import scripts
from test_runtime_dispatch import mixer


def _build_ui(monkeypatch, tmp_path, tabname="txt2img"):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    script = mixer.Script()
    script.tabname, script.is_txt2img, script.is_img2img = tabname, tabname == "txt2img", tabname == "img2img"
    callbacks = {}
    for elem_id, callback in script.on_after_component_elem_id:
        callbacks.setdefault(elem_id, []).append(callback)
    with gr.Blocks() as demo:
        # Built by the WebUI before the script panels
        base = gr.Slider(1.0, 24.0, value=3.0, elem_id=f"{tabname}_distilled_cfg_scale")
        hires = gr.Slider(1.0, 24.0, value=3.0, elem_id="txt2img_hr_distilled_cfg")
        for component in (base, hires):
            for callback in callbacks.get(component.elem_id, []):
                callback(scripts.OnComponent(component=component))
        controls = script.ui(False)
    return script, demo, base, hires, controls


def test_generation_shift_sliders_drive_the_stage_timings(monkeypatch, tmp_path):
    script, demo, base, hires, controls = _build_ui(monkeypatch, tmp_path)
    assert script.base_shift_component is base and script.hires_shift_component is hires
    # Gradio renders gr.HTML with {@html}: script tags there never run
    assert not any("<script" in str(block.value) for block in demo.blocks.values() if isinstance(block, gr.HTML))

    runtime_base_shift, runtime_hires_shift = controls[5], controls[6]
    for slider, runtime_shift in ((base, runtime_base_shift), (hires, runtime_hires_shift)):
        listeners = [dep for dep in demo.config["dependencies"] if (slider._id, "change") in dep["targets"]]
        outputs = sorted(len(dep["outputs"]) for dep in listeners)
        # The slider re-times Start/End/Peak of every row and updates the script arg
        assert outputs == [1, mixer.MAX_ARTIST_ROWS * 3]
        assert any(dep["outputs"] == [runtime_shift._id] for dep in listeners)
    assert mixer._same_value(5.5) == 5.5


def _panel_values(enable=True, artist="wlop"):
    """What the autosave posts: the header with the template fields, then every row."""
    values = [enable, "", mixer.APPLY_BASE, 2, 1, False, False, 4.0, 3.0, 0.7, mixer.OPT_BALANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, True]
    rows = [[True, artist, 1.0, "0-51", 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]] * (mixer.MAX_ARTIST_ROWS * 2)
    return values + [value for row in rows for value in row]


def _element(demo, elem_id):
    return next(block for block in demo.blocks.values() if getattr(block, "elem_id", None) == elem_id)


def test_panel_edits_are_remembered(monkeypatch, tmp_path):
    script, demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    dependencies = demo.config["dependencies"]
    autosave = [dep for dep in dependencies if len(dep["targets"]) > mixer.MAX_ARTIST_ROWS * 20]
    assert len(autosave) == 1 and autosave[0]["trigger_mode"] == "always_last"
    events = {}
    for target, event in autosave[0]["targets"]:
        events.setdefault(target, set()).add(event)
    assert {controls[0]._id, controls[-1]._id} <= set(events)  # the enable box, the last hires row

    # A textbox saves when its edit is done, not per keystroke; the rest on .change
    artist, weight, blocks, curve = controls[14], controls[15], controls[16], controls[20]
    assert isinstance(artist, gr.Textbox) and isinstance(blocks, gr.Textbox)
    assert events[artist._id] == events[blocks._id] == {"blur", "submit"}
    assert events[weight._id] == events[curve._id] == events[controls[0]._id] == {"change"}
    # Handlers that write the textboxes (applying a template, the optimization
    # preset's Blocks) save once they finish
    hires_blocks = controls[13 + mixer.MAX_ARTIST_ROWS * mixer.ROW_WIDTH + 3]
    writers = {
        dep["id"]
        for dep in dependencies
        if (_element(demo, script.elem_id("template_apply"))._id, "click") in dep["targets"]
        or ((controls[8]._id, "input") in dep["targets"] and {blocks._id, hires_blocks._id} & set(dep["outputs"]))
    }
    followers = [dep for dep in dependencies if dep["trigger_after"] in writers]
    assert len(writers) == 3 and len(followers) == 3
    assert all(dep["inputs"] == autosave[0]["inputs"] for dep in followers)

    writes = []
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", {})
    monkeypatch.setattr(mixer, "write_json_atomic", lambda path, data: writes.append(data))
    values = _panel_values()
    mixer._save_current_settings_ui("txt2img", *values)
    assert len(writes) == 1 and writes[0]["enable"] and writes[0]["base_shift"] == 4.0
    mixer._save_current_settings_ui("txt2img", *values)  # unchanged: no second write
    assert len(writes) == 1


def test_each_tab_remembers_its_own_panel(monkeypatch, tmp_path):
    import json

    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", {})
    folder = tmp_path / "extension-data" / mixer._DATA_NAME
    folder.mkdir(parents=True)
    # Earlier versions kept one file for both tabs: each tab starts from it
    shared = mixer._current_settings_payload(*_panel_values(artist="shared artist"))
    (folder / "artist_mixer_current_settings.json").write_text(json.dumps(shared))
    for tab in mixer.SETTINGS_TABS:
        assert mixer._current_settings_data(tab)["base_rows"][0][1] == "shared artist"

    mixer._save_current_settings_ui("txt2img", *_panel_values(artist="txt2img artist"))
    mixer._save_current_settings_ui("img2img", *_panel_values(enable=False, artist="img2img artist"))
    txt2img, img2img = (mixer._current_settings_data(tab) for tab in mixer.SETTINGS_TABS)
    assert txt2img["enable"] and txt2img["base_rows"][0][1] == "txt2img artist"
    assert not img2img["enable"] and img2img["base_rows"][0][1] == "img2img artist"
    # Saving one tab again does not touch the other
    mixer._save_current_settings_ui("img2img", *_panel_values(artist="img2img again"))
    assert mixer._current_settings_data("txt2img")["base_rows"][0][1] == "txt2img artist"

    # A reset restores the defaults instead of seeding the old shared panel again
    mixer._reset_all_defaults_ui("txt2img")
    assert mixer._current_settings_data("txt2img") is None
    assert mixer._current_settings_data("img2img")["base_rows"][0][1] == "img2img again"
    lang = mixer._language()
    assert mixer._initial_ui_defaults("txt2img", lang) == mixer._builtin_default_ui_state(lang)

    # A generation remembers the panel of the tab it runs in
    from types import SimpleNamespace

    from test_runtime_dispatch import _run_args

    script = mixer.Script()
    script.tabname = "img2img"
    script.process_before_every_sampling(SimpleNamespace(is_api=False, extra_generation_params={}), *_run_args())
    assert mixer._current_settings_data("img2img")["base_rows"][0][1] == "(base artist:-1)"
    assert mixer._current_settings_data("txt2img") is None


def test_the_script_args_are_read_back_in_the_order_ui_returns_them(monkeypatch, tmp_path):
    for tabname in mixer.SETTINGS_TABS:
        script, _demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path, tabname)
        args = mixer._MixerArgs.from_args(controls)
        for name in mixer.HEADER_ARGS:
            assert getattr(args, name).elem_id == script.elem_id(name)
        assert [component.elem_id for component in args.row_values(False)[: mixer.ROW_WIDTH]][:2] == [script.elem_id("base_enabled_0"), script.elem_id("base_artist_0")]
        assert args.row_values(True)[-1].elem_id == script.elem_id(f"hires_auto_shift_{mixer.MAX_ARTIST_ROWS - 1}")


def test_only_user_edits_rewrite_the_rows(monkeypatch, tmp_path):
    # Applying a template writes the optimization preset and every row;
    # handlers bound with .change would then overwrite its Blocks and timings.
    _script, demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    optimization = controls[8]
    base_rows = controls[13 : 13 + mixer.MAX_ARTIST_ROWS * mixer.ROW_WIDTH]
    dependencies = demo.config["dependencies"]

    def rewrites(component):
        """Events of ``component`` that write row components (not the autosave)."""
        row_ids = {row._id for row in controls[13:]}
        return sorted(
            event
            for dep in dependencies
            if len(dep["targets"]) < mixer.MAX_ARTIST_ROWS * 20 and row_ids & set(dep["outputs"])
            for target, event in dep["targets"]
            if target == component._id
        )

    assert rewrites(optimization) == ["input", "input"]  # the base and the hires rows
    for row in range(mixer.MAX_ARTIST_ROWS):
        stage, auto_shift = base_rows[row * mixer.ROW_WIDTH + 8], base_rows[row * mixer.ROW_WIDTH + 9]
        assert rewrites(stage) == ["input"] and rewrites(auto_shift) == ["input"]
