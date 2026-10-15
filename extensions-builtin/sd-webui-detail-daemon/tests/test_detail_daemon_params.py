"""Detail Daemon on ``params()``: the legacy ``ui()`` contract, infotext paste
(current, multi-daemon and the old ``DD_*`` format) into the same process()
arguments, the settings, the XYZ axes and the schedule the ``sigma-curve``
widget mirrors."""

import json

import pytest
from detail_daemon_testlib import SCRIPT, contract, load_detail_daemon
from schedule_cases import FIXTURE, cases

from remi import ui
from remi.testing import webui_state

FIXTURES = SCRIPT.parents[1] / "tests" / "fixtures"


@pytest.fixture(scope="module")
def dd():
    with webui_state():
        yield load_detail_daemon()


def make(module, tab="txt2img"):
    script = module.Script()
    script.is_img2img = tab == "img2img"
    script.layout = ui.ScriptLayout(script.params(script.is_img2img), owner="detail_daemon", order=script.param_order)
    return script


@pytest.mark.parametrize("tab", ["txt2img", "img2img"])
def test_params_keep_the_legacy_contract(dd, tab):
    script = make(dd, tab)
    snapshot = contract.load_snapshot(FIXTURES / f"legacy_ui_{tab}.json")
    # The bound tab choice and the curve preview are UI only: after the daemons
    assert contract.differences(script.layout, snapshot, extra=("daemon", "curves")) == []


def test_the_daemon_count_setting_sizes_the_panel(dd):
    from modules import shared

    shared.opts.data["detail_daemon_count"] = 3
    try:
        script = make(dd)
        assert len(script.layout) == 1 + 3 * 12 + 2
        curves = script.layout.by_name["curves"]
        assert [d["title"] for d in curves.schema(ui.Context())["props"]["daemons"]] == ["I", "II", "III"]
    finally:
        del shared.opts.data["detail_daemon_count"]


def paste(script, infotext_params):
    from modules import script_callbacks

    params = dict(infotext_params)
    script_callbacks.infotext_pasted_callback("", params)
    values, problems = script.layout.paste(params)
    assert problems == []
    return values


def test_infotext_round_trip_reaches_process_unchanged(dd):
    script = make(dd)
    infotext = "D1:1,0,cond,0.25,0.1,0.9,0.4,1.5,0.05,-0.1,0.2,1;D3:1,1,uncond,-0.3,0.3,0.7,0.5,2.0,0.0,0.0,0.0,0"
    values = paste(script, {"Detail Daemon": infotext})
    args = script.layout.values_to_args(values)
    assert args[0] is True
    daemons = [args[1 + 12 * i : 13 + 12 * i] for i in range(6)]
    assert daemons[0] == [True, False, "cond", 0.1, 0.9, 0.4, 0.25, 1.5, 0.05, -0.1, 0.2, True]
    # As before, the parser numbers the entries by position, not by their tag (D3 fills daemon II)
    assert daemons[1] == [True, True, "uncond", 0.3, 0.7, 0.5, -0.3, 2.0, 0.0, 0.0, 0.0, False]
    assert daemons[2][0] is False

    # process() gets the positional arguments and writes the infotext back
    p = type("P", (), {})()
    p.sampler_name, p.cfg_scale, p.batch_size, p.extra_generation_params = "Euler", 7.0, 1, {}
    try:
        script.process(p, *args)
        assert p.extra_generation_params["Detail Daemon"] == "D1:1,0,cond,0.25,0.1,0.9,0.4,1.5,0.05,-0.1,0.2,1;D2:1,1,uncond,-0.3,0.3,0.7,0.5,2.0,0.0,0.0,0.0,0"
        assert [d["name"] for d in script.daemon_data] == ["Daemon 1", "Daemon 2"]
    finally:
        script.cleanup(p)


def test_old_single_daemon_infotext(dd):
    script = make(dd)
    values = paste(script, {"DD_enabled": "True", "DD_mode": "both", "DD_amount": "0.3", "DD_start": "0.1", "DD_end": "0.6", "DD_smooth": "False"})
    assert values["enabled"] is True and values["d1_active"] is True
    assert values["d1_amount"] == 0.3 and values["d1_start"] == 0.1 and values["d1_end"] == 0.6 and values["d1_smooth"] is False
    assert values["d2_active"] is False
    # An infotext without Detail Daemon switches it off, as the old paste did
    assert paste(script, {"Steps": "20"})["enabled"] is False


def test_settings_use_remi_widgets(dd):
    from modules import shared

    saved = dict(shared.opts.data_labels)
    try:
        dd.add_settings()
        assert shared.opts.data_labels["detail_daemon_count"].component is ui.Slider
        assert shared.opts.data_labels["detail_daemon_verbose"].component is ui.Toggle
    finally:
        shared.opts.data_labels.clear()
        shared.opts.data_labels.update(saved)


def test_registers_its_xyz_axes(dd, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    from modules import scripts

    path = SCRIPT.parents[3] / "scripts" / "xyz_grid.py"
    spec = importlib.util.spec_from_file_location("xyz_grid_for_detail_daemon", path)
    xyz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xyz)
    monkeypatch.setattr(scripts, "scripts_data", [SimpleNamespace(path=str(path), module=xyz)])
    dd.xyz_support()
    labels = [x.label for x in xyz.axis_options if x.label.startswith("[Detail Daemon]")]
    assert labels[:3] == ["[Detail Daemon] Mode", "[Detail Daemon] Amount", "[Detail Daemon] Start"] and len(labels) == 10


def test_the_widget_schedule_fixture_is_current(dd):
    """webui/src/widgets/sigma-curve/schedule.fixture.json is this function's output."""

    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == json.loads(json.dumps(cases(dd.Script().make_schedule)))


def test_ported_without_matplotlib():
    contract.assert_ported(SCRIPT)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "matplotlib" not in source and not (SCRIPT.parents[1] / "style.css").exists()


def test_zh_cn_covers_every_string(dd):
    import re

    from modules import shared

    strings = contract.schema_strings(make(dd).layout.schema(ui.Context()))
    saved = dict(shared.opts.data_labels)
    try:
        dd.add_settings()
        strings |= {shared.opts.data_labels[key].label for key in ("detail_daemon_count", "detail_daemon_verbose")}
    finally:
        shared.opts.data_labels.clear()
        shared.opts.data_labels.update(saved)
    roman = re.compile(r"^[IVX]+$")  # the daemon tabs
    assert contract.missing_translations(strings, SCRIPT.parents[1] / "localizations" / "zh_CN.json", ignore=roman.match) == []
