"""The ported panel: params() declares what the Gradio ui() did."""

import inspect
import json
from pathlib import Path

import pytest
import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compile.py"


@pytest.fixture(scope="module")
def compile_script():
    from remi.testing import webui_state

    with webui_state():
        yield harness.load_script(SCRIPT, "compile_params_under_test")


def _script(module):
    # __init__ only sets global dynamo options; keep them untouched for other tests.
    return module.TorchCompileForForge.__new__(module.TorchCompileForForge)


@pytest.mark.parametrize("tab", harness.TABS)
def test_params_match_the_legacy_ui(compile_script, tab):
    layout = harness.script_layout(_script(compile_script), tab)
    harness.assert_matches_snapshot(layout, harness.load_snapshot(SCRIPT, tab))


def test_arguments_are_the_processing_signature(compile_script):
    layout = harness.script_layout(_script(compile_script))
    hook = inspect.signature(compile_script.TorchCompileForForge.process_batch)
    assert layout.names == [name for name in hook.parameters if name not in ("self", "p", "kwargs")]
    assert layout.values_to_args({"preset": "dynamic"}) == ["dynamic"]


def test_schema_export(compile_script):
    from remi import ui

    schema = harness.script_layout(_script(compile_script)).schema(ui.Context())
    text = json.dumps(schema, allow_nan=False)
    assert schema["type"] == "section" and schema["title"] == "Torch Compile Integrated" and not schema["open"]
    assert schema["children"][1]["type"] == "markdown" and "**torch.compile**" in text


def test_zh_cn_covers_the_panel(compile_script):
    harness.assert_localized(SCRIPT.parents[1], [harness.script_layout(_script(compile_script), tab).schema() for tab in harness.TABS])


def test_imports_without_gradio():
    harness.assert_gradio_free([SCRIPT])
