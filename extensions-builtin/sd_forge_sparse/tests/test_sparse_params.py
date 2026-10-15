"""The ported panel: params() declares what the Gradio ui() did. The
"Timestep Range" RangeSlider is ``ui.RangeSlider`` (the shared
``range-slider`` widget): one ``[low, high]`` argument, as Gradio passed it."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sparse.py"
# Gradio's RangeSlider had no remi.ui equivalent (a json param in the record)
RANGE_SLIDER = {2: {"kind": "custom", "widget": "range-slider", "props": {"minimum": 0.0, "maximum": 1.0, "step": 0.05}}}


@pytest.fixture(scope="module")
def sparse():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "sparse_params_under_test")


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(sparse, tab):
    layout = harness.script_layout(sparse.SparseAttentionForForge(), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab), RANGE_SLIDER)


def test_arguments_are_the_processing_signature(sparse):
    layout = harness.script_layout(sparse.SparseAttentionForForge())
    hook = inspect.signature(sparse.SparseAttentionForForge.process_before_every_sampling)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p", "kwargs")]
    defaults = [parameter.default for name, parameter in hook.parameters.items() if name not in ("self", "p", "kwargs", "enable")]
    assert layout.values_to_args({})[1:] == [list(value) if isinstance(value, tuple) else value for value in defaults]


@pytest.mark.parametrize(
    ("given", "expected"),
    [([0.2, 0.6], [0.2, 0.6]), ([0.9, 0.1], [0.1, 0.9]), ([-1, 2], [0.0, 1.0]), ((0, 1), [0.0, 1.0]), (["0.2", 0.5], [0.2, 0.5]), (["x", 0.5], None), ([0.5], None), ("0.2,0.5", None), ([True, 0.5], None), (None, None)],
)
def test_the_range_is_two_ordered_numbers_inside_the_bounds(sparse, given, expected):
    from remi import ui

    layout = harness.script_layout(sparse.SparseAttentionForForge())
    if expected is None:
        with pytest.raises(ui.ValidationError):
            layout.values_to_args({"percent": given})
    else:
        assert layout.values_to_args({"percent": given})[2] == expected


def test_the_range_pastes_from_infotext_text(sparse):
    from remi import ui

    node = harness.script_layout(sparse.SparseAttentionForForge()).by_name["percent"]
    assert isinstance(node, ui.RangeSlider)
    assert node.from_paste("0.1, 0.9", ui.Context()) == node.from_paste("(0.9, 0.1)", ui.Context()) == [0.1, 0.9]


def test_the_range_reaches_the_patch(sparse):
    seen = {}

    class Predictor:
        def percent_to_sigma(self, percent):
            seen.setdefault("percents", []).append(percent)
            return 1.0 - percent

    class Unet:
        model = SimpleNamespace(predictor=Predictor())
        model_options = {"transformer_options": {}}

        def clone(self):
            return self

    script = sparse.SparseAttentionForForge()
    args = harness.script_layout(script).values_to_args({"enable": True, "percent": [0.25, 0.75], "dense_blocks": "1, 3-4"})
    p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=Unet())))
    script.process_before_every_sampling(p, *args)
    assert seen["percents"] == [0.25, 0.75]
    assert "optimized_attention_override" in p.sd_model.forge_objects.unet.model_options["transformer_options"]


def test_schema_export(sparse):
    schema = harness.script_layout(sparse.SparseAttentionForForge()).schema()
    json.dumps(schema, allow_nan=False)
    timestep_range = schema["children"][0]["children"][1]
    assert timestep_range == {
        "type": "custom",
        "name": "percent",
        "label": "Timestep Range",
        "default": [0.15, 0.85],
        "widget": "range-slider",
        "props": {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
    }


def test_zh_cn_covers_the_panel(sparse):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(sparse.SparseAttentionForForge(), tab).schema() for tab in harness.TABS])


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT])
