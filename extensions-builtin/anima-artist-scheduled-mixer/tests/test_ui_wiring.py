import gradio as gr

from modules import scripts
from test_runtime_dispatch import mixer


def _build_ui(monkeypatch, tmp_path):
    from modules import paths_internal

    monkeypatch.setattr(paths_internal, "data_path", str(tmp_path))
    script = mixer.Script()
    script.tabname, script.is_txt2img, script.is_img2img = "txt2img", True, False
    callbacks = {}
    for elem_id, callback in script.on_after_component_elem_id:
        callbacks.setdefault(elem_id, []).append(callback)
    with gr.Blocks() as demo:
        # Built by the WebUI before the script panels
        base = gr.Slider(1.0, 24.0, value=3.0, elem_id="txt2img_distilled_cfg_scale")
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


def test_panel_edits_are_remembered(monkeypatch, tmp_path):
    _script, demo, _base, _hires, controls = _build_ui(monkeypatch, tmp_path)
    autosave = [dep for dep in demo.config["dependencies"] if len(dep["targets"]) > mixer.MAX_ARTIST_ROWS * 20]
    assert len(autosave) == 1 and autosave[0]["trigger_mode"] == "always_last"
    targets = {target for target, _event in autosave[0]["targets"]}
    assert {controls[0]._id, controls[-1]._id} <= targets  # the enable box, the last hires row

    writes = []
    monkeypatch.setattr(mixer, "_LAST_REMEMBERED_SETTINGS", None)
    monkeypatch.setattr(mixer, "write_json_atomic", lambda path, data: writes.append(data))
    values = [True, "", mixer.APPLY_BASE, 2, 1, False, False, 4.0, 3.0, 0.7, mixer.OPT_BALANCE, mixer.COMBINE_OUTPUT_AVG, mixer.FUSION_INTERPOLATE, False, True]
    rows = [[True, "wlop", 1.0, "0-51", 0.0, 1.0, 0.5, mixer.CURVE_HOLD, mixer.PRESET_CUSTOM, False]] * (mixer.MAX_ARTIST_ROWS * 2)
    values += [value for row in rows for value in row]
    mixer._save_current_settings_ui(*values)
    assert len(writes) == 1 and writes[0]["enable"] and writes[0]["base_shift"] == 4.0
    mixer._save_current_settings_ui(*values)  # unchanged: no second write
    assert len(writes) == 1
