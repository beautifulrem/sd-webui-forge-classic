"""Callback registration (``modules/script_callbacks.py``): upstream f5292b0f
finds the registering file by walking frames (``caller_filename()``) instead
of ``inspect.stack()``. The callback's file, its name and
``remove_current_script_callbacks()`` must still point at the script that
registered it, not at ``script_callbacks.py`` or a helper in it."""

import importlib.util
import os

import pytest

SCRIPT = """
from modules import script_callbacks


def on_model_loaded(sd_model):
    pass


def on_image_saved(params):
    pass


def register():
    script_callbacks.on_model_loaded(on_model_loaded)
    script_callbacks.on_model_loaded(on_model_loaded, name="second")
    script_callbacks.on_image_saved(on_image_saved)


def unregister():
    script_callbacks.remove_current_script_callbacks()
"""


@pytest.fixture
def script(tmp_path):
    from modules import script_callbacks

    path = tmp_path / "my_script.py"
    path.write_text(SCRIPT, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("my_script_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for function in (module.on_model_loaded, module.on_image_saved):
        script_callbacks.remove_callbacks_for_function(function)


def _mine(category, module):
    from modules import script_callbacks

    return [c for c in script_callbacks.callback_map[f"callbacks_{category}"] if c.callback in (module.on_model_loaded, module.on_image_saved)]


def test_caller_filename_is_the_first_frame_outside_the_module():
    from modules import script_callbacks

    assert os.path.samefile(script_callbacks.caller_filename(), __file__)


def test_callbacks_record_the_registering_script(script):
    script.register()

    loaded = _mine("model_loaded", script)
    assert [os.path.basename(c.script) for c in loaded] == ["my_script.py", "my_script.py"]
    assert os.path.samefile(loaded[0].script, script.__file__)
    assert [c.name for c in loaded] == ["base/my_script.py/model_loaded", "base/my_script.py/model_loaded/second"]
    script.register()  # the same names again get a suffix
    assert [c.name for c in _mine("model_loaded", script)][2:] == ["base/my_script.py/model_loaded-1", "base/my_script.py/model_loaded/second-1"]


def test_remove_current_script_callbacks_removes_only_the_callers(script):
    from modules import script_callbacks

    def other(sd_model):
        pass

    script.register()
    script_callbacks.on_model_loaded(other)  # registered from this test file
    try:
        script_callbacks.ordered_callbacks("model_loaded")  # fills the ordered cache too
        script.unregister()
        assert _mine("model_loaded", script) == [] and _mine("image_saved", script) == []
        assert all(c.callback not in (script.on_model_loaded, script.on_image_saved) for c in script_callbacks.ordered_callbacks("model_loaded"))
        assert [c.callback for c in script_callbacks.callback_map["callbacks_model_loaded"] if c.callback is other] == [other]
    finally:
        script_callbacks.remove_callbacks_for_function(other)
