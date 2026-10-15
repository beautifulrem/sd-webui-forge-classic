"""Records ControlNet's ``tests/fixtures/legacy_ui_<tab>.json`` from its Gradio panel.

The port removed that panel, so it is taken from a revision before the port
(``--rev``, by default the last one with the Gradio UI), exported to a
temporary folder. Needs Gradio 4.40 and ``gradio_rangeslider``::

    /tmp/remi-testenv/bin/python tests/record_controlnet_legacy_ui.py [--rev REV] [--out DIR]

Each file holds ``args`` (the script's positional arguments, as
``remi_port_harness.legacy_snapshot`` describes them), ``unit_args`` (the
``UiControlNetUnit`` fields the unit group's controls fed, in order),
``infotext_keys`` (the registered paste keys) and ``unit_controls`` (the
controls of the first unit). Not a test module (and kept out of the
extension, which must not import Gradio).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from enum import Enum
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTENSION = "extensions-builtin/sd_forge_controlnet"
LEGACY_REV = "80cfa274ff872a7f3dd0f93666268107c960ff1c"
"""The last revision with the Gradio panel (``lib_controlnet/controlnet_ui``)."""

ABOUT = "Recorded from the Gradio ui() before the port (gradio 4.40, preprocessors: modules_forge.supported_preprocessor only, no ControlNet models)."

# Fixture field -> attribute of the unit group (``canvas_editor.<name>`` for the Region canvas)
UNIT_CONTROLS = {
    "enabled": "enabled",
    "pixel_perfect": "pixel_perfect",
    "preprocessor_preview": "preprocessor_preview",
    "mask_upload": "mask_upload",
    "use_preview_as_input": "use_preview_as_input",
    "upload_independent_img_in_img2img": "upload_independent_img_in_img2img",
    "type_filter": "type_filter",
    "module": "module",
    "model": "model",
    "weight": "weight",
    "timestep_range": "timestep_range",
    "processor_res": "processor_res",
    "threshold_a": "threshold_a",
    "threshold_b": "threshold_b",
    "control_mode": "control_mode",
    "resize_mode": "resize_mode",
    "hr_option": "hr_option",
    "save_detected_map": "save_detected_map",
    "region_num": "canvas_editor.num",
    "region_height": "canvas_editor.height",
    "region_randomize": "canvas_editor.randomize",
    "region_shape": "canvas_editor.shape",
    "new_canvas_width": "canvas_width",
    "new_canvas_height": "canvas_height",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def _export(rev: str, target: Path) -> Path:
    archive = subprocess.run(["git", "-C", str(REPO), "archive", rev, EXTENSION], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return target / EXTENSION / "scripts" / "controlnet.py"


def _plain(value):
    return value.value if isinstance(value, Enum) else value


def _control(field: str, component, infotext: dict[int, str]) -> dict:
    entry = {"field": field, "component": type(component).__name__, "label": component.label, "default": component.value}
    if hasattr(component, "minimum") and hasattr(component, "maximum"):
        entry.update(minimum=component.minimum, maximum=component.maximum, step=component.step)
    entry.update(visible=component.visible, info=getattr(component, "info", None))
    choices = getattr(component, "choices", None)
    if choices is not None:
        entry["choices"] = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in choices]
    entry["infotext"] = infotext.get(id(component))
    return entry


def _stand_ins(context, gr) -> None:
    """The native WebUI components the unit group wires itself to (its
    ``A1111Context``, filled by ``on_after_component`` in the WebUI)."""

    for name in [name for name in dir(type(context)) if not name.startswith("_")]:
        if isinstance(getattr(type(context), name), property) or getattr(context, name) is not None:
            continue
        if name.endswith("_button"):
            component = gr.Button()
        elif name.endswith("_slider"):
            component = gr.Slider(64, 2048, 512, step=8)
        elif name == "txt2img_enable_hr":
            component = gr.Checkbox()
        else:
            component = gr.Textbox(visible=False)
        setattr(context, name, component)


def record(rev: str, out: Path) -> list[Path]:
    port = _load(REPO / "tests" / "remi_port_harness.py", "remi_port_harness")
    port.prepare_imports()
    sys.path.insert(0, str(REPO / "extensions-builtin" / "forge_legacy_preprocessors"))  # annotator (the pose editor)
    _load(REPO / "tests" / "extension_script_harness.py", "extension_script_harness").prepare_webui_imports()

    import gradio as gr

    from modules.anima_presets import preset_controls, reset_preset_controls
    from remi.testing import webui_state
    from remi.ui.legacy import legacy_layout

    written = []
    with tempfile.TemporaryDirectory() as tmp, webui_state():
        import modules_forge.supported_preprocessor  # noqa: F401 - None and canny

        script_path = _export(rev, Path(tmp))
        module = port.load_script(script_path, "legacy_controlnet")
        from lib_controlnet import global_state

        global_state.controlnet_names = ["None"]
        global_state.controlnet_filename_dict = {"None": None}

        groups = []
        original = module.ControlNetUiGroup

        def collect(*args, **kwargs):
            groups.append(original(*args, **kwargs))
            return groups[-1]

        module.ControlNetUiGroup = collect
        for tab in port.TABS:
            groups.clear()
            script = module.ControlNetForForgeOfficial()
            script.filename = str(script_path)
            script.tabname, script.is_img2img, script.is_txt2img = tab, tab == "img2img", tab != "img2img"
            script.infotext_fields = []
            reset_preset_controls(tab)
            with gr.Blocks():
                _stand_ins(original.a1111_context, gr)
                inputs = list(script.ui(script.is_img2img) or [])
            # As remi_port_harness.legacy_snapshot describes the arguments (recorded before the wiring keys)
            names, components, _values = preset_controls(tab)
            presets = {id(component): name for name, component in zip(names, components)}
            entries = port.describe(legacy_layout("legacy", inputs, script.infotext_fields))
            for entry, control in zip(entries, inputs):
                entry["preset"] = presets.get(id(control))
                if isinstance(getattr(control, "value", None), module.ControlNetUnit):  # the unit's gr.State: its fields
                    entry.update(label="", default={name: _plain(value) for name, value in vars(control.value).items()})
                for key in port.WIRING_KEYS:
                    if entry.get(key) is None:
                        entry.pop(key, None)

            locators = {id(component): locator for component, locator in script.infotext_fields}
            controls = []
            for field, attribute in UNIT_CONTROLS.items():
                component = groups[0]
                for part in attribute.split("."):
                    component = getattr(component, part, None)
                if component is not None:
                    controls.append(_control(field, component, locators))
            fields = [field.name for field in dataclasses.fields(module.ControlNetUnit)]
            snapshot = {
                "_about": ABOUT,
                "args": entries,
                # The group built UiControlNetUnit(*controls): the fields up to "control_mode"
                "unit_args": fields[: fields.index("control_mode") + 1],
                "infotext_keys": sorted(locators.values()),
                "unit_controls": controls,
            }
            target = out / f"legacy_ui_{tab}.json"
            target.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
            written.append(target)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default=LEGACY_REV, help="a revision with the Gradio panel")
    parser.add_argument("--out", type=Path, default=REPO / EXTENSION / "tests" / "fixtures")
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    for path in record(options.rev, options.out.resolve()):
        print(path)  # noqa: T201 - the CLI lists what it wrote
