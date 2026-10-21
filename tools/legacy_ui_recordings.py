"""Every committed legacy ``ui()`` fixture, and how ``tools/record_legacy_ui.py`` records it again.

The ports' tests compare each ``params()`` panel with a record of the Gradio
``ui()`` it replaced. Those records were taken in several formats, one per test
harness that reads them (a bare list for ``tests/remi_port_harness.py``, the
``{script, tab, count, infotext_fields, inputs}`` contract of
``tests/legacy_ui_contract.py``, the Anima panels' formats, ControlNet's unit
record, the extension pages, the processing scripts). :data:`RECORDINGS` lists
each fixture with the script and revision it was recorded from, its format and
the WebUI context the ``ui()`` needs (the core components a script wires to,
the upscalers, ControlNet's preprocessors, ...), so that

    python tools/record_legacy_ui.py --fixtures --check

records all of them again in the legacy revision and compares every value with
the committed files (``tests/test_remi_legacy_ui_fixtures.py`` keeps the list
complete and the comparison working on every format, without Gradio).

Nothing here imports Gradio: the recording functions run in the recorder's
Gradio 4.40 interpreter, inside an export of the legacy revision
(``record_legacy_ui.LEGACY_REV``), and use that revision's Gradio module
(:func:`legacy_gradio`) to build the panels in. The format
functions describe the legacy controls with this checkout's harnesses (the code
the tests compare with), as ``record_legacy_ui`` does for the default format.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABS = ("txt2img", "img2img")
PRE_PORT = "80cfa274ff872a7f3dd0f93666268107c960ff1c"
"""The parent of a7baa5ad, which ported the scripts, the extensions and the pages to ``params()``."""
GRADIO_REV = "9c96c45ee561487d2c4baa6900a885d3a71affe5"
"""The last revision with the Gradio UI (``record_legacy_ui.LEGACY_REV``): ``modules/processing_scripts`` still had their ``ui()``."""

WIRING_KEYS = ("exclusive", "visible_if", "enabled_if")
"""Keys ``remi_port_harness.describe()`` added after the bare-list fixtures were
recorded: absent there means none (as ``assert_matches_snapshot`` reads them)."""


@dataclasses.dataclass(frozen=True)
class Recording:
    """A committed fixture (one file per tab, or one file) and how it was recorded."""

    name: str
    script: str
    """The recorded script, repository-relative."""
    format: str
    """A key of :data:`FORMATS`."""
    files: Mapping[str, str]
    """``{tab: fixture path}`` (repository-relative); a single-file format uses the key ``"all"``."""
    rev: str = PRE_PORT
    """The script's revision before its port (its extension folder is exported from it)."""
    class_name: str | None = None
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    preimport: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()
    """Keys of :data:`CONTEXTS`, applied in order before the ``ui()`` is built."""
    shown_without: str = ""
    """Why the tabs are recorded whatever ``show()`` says (an optional package or
    a model the panel is gated on: ``record_legacy_ui.py --tabs``); empty: ``show()`` decides."""

    @property
    def extension(self) -> str | None:
        parts = Path(self.script).parts
        return "/".join(parts[:2]) if parts[0] == "extensions-builtin" else None

    @property
    def export_paths(self) -> tuple[str, ...]:
        """What the recorder exports from :attr:`rev`: the extension folder, or the script file."""

        return (self.extension or self.script,)


def _ext(name: str, script: str, fmt: str, tabs: Iterable[str] = TABS, **kwargs: Any) -> Recording:
    folder = f"extensions-builtin/{name}"
    return Recording(name, f"{folder}/scripts/{script}", fmt, {tab: f"{folder}/tests/fixtures/legacy_ui_{tab}.json" for tab in tabs}, **kwargs)


def _core(name: str, cls: str, tabs: Iterable[str] = TABS, fmt: str = "contract", **kwargs: Any) -> Recording:
    return Recording(f"scripts/{name}", f"scripts/{name}.py", fmt, {tab: f"tests/fixtures/legacy_ui/{name}_{tab}.json" for tab in tabs}, class_name=cls, **kwargs)


RECORDINGS: tuple[Recording, ...] = (
    # tests/remi_port_harness.py (a bare list per tab)
    _ext("sd_forge_spectrum", "spectrum.py", "port"),
    _ext("sd_forge_compile", "compile.py", "port", shown_without="Triton"),
    _ext("sd_forge_image_stitch", "image_stitch.py", "port"),
    _ext("sd_forge_multidiffusion", "forge_multidiffusion.py", "port", tabs=("img2img",), preimport=("backend.nn.cnets.cldm",)),
    _ext("sd_forge_neveroom", "forge_never_oom.py", "port"),
    _ext("sd_forge_pid", "pid.py", "port", shown_without="a PiD model in models/"),
    _ext("sd_forge_radial", "forge_radial.py", "port", shown_without="RadialAttention's CUDA kernels"),
    _ext("sd_forge_sparse", "sparse.py", "port"),
    _ext("sd-forge-dynamic-shift", "dynamic_shift.py", "port"),
    _ext("sd-forge-prompt-blacklist", "prompt_blacklist.py", "port"),
    _ext("sd-lock-img2img-steps", "lock_steps.py", "port", tabs=("img2img",)),
    _ext("sd-webui-prompt-anchor", "prompt_anchor.py", "port"),
    _ext("soft-inpainting", "soft_inpainting.py", "port", tabs=("img2img",)),
    _ext(
        "extra-options-section",
        "extra_options_section.py",
        "port",
        # tests/test_extra_options_params.py: CLIP_stop_at_last_layers and sd_vae are refused by the panel
        options={
            "extra_options_txt2img": ["CLIP_stop_at_last_layers", "randn_source", "eta_ancestral", "tiling", "eta_noise_seed_delta", "emphasis"],
            "extra_options_img2img": ["initial_noise_multiplier", "img2img_color_correction", "sd_vae", "img2img_extra_noise"],
            "extra_options_cols": 2,
        },
    ),
    # tests/legacy_ui_contract.py ({script, tab, count, infotext_fields, inputs})
    _ext("sd-webui-detail-daemon", "detail_daemon.py", "contract"),
    _ext("sd-postprocess-suite", "postprocess_suite.py", "contract"),
    _ext("ADetailer-Neo", "adetailer.py", "adetailer", options={"ad_max_models": 4}, contexts=("adetailer-detectors",)),
    _core("loopback", "Loopback", tabs=("img2img",)),
    _core("prompt_matrix", "PromptMatrix"),
    _core("prompts_from_file", "PromptsFromTexts"),
    _core("sd_upscale", "SDUpscale", tabs=("img2img",), contexts=("upscalers",)),
    _core("xyz_grid", "Script"),
    Recording("scripts/postprocessing_upscale", "scripts/postprocessing_upscale.py", "extras", {"extras": "tests/fixtures/legacy_ui/postprocessing_upscale_extras.json"}, class_name="ScriptPostprocessingUpscale", contexts=("upscalers",)),
    Recording("scripts/degrid", "scripts/degrid.py", "extras", {"extras": "tests/fixtures/legacy_ui/degrid_extras.json"}, class_name="ScriptPostprocessingDegrid"),
    # tests/remi_panel_harness.py ({shown, script, inputs})
    _ext("sd_forge_anima_dave", "anima_dave.py", "panel"),
    _ext("sd_forge_anima_freefuse", "anima_freefuse.py", "panel"),
    _ext("sd_forge_anima_regional", "anima_regional.py", "panel"),
    _ext("sd_forge_anima_xattn_boost", "anima_xattn_boost.py", "panel"),
    _ext("sd-forge-anima-resolutions", "anima_resolutions.py", "panel"),
    # tests/remi_script_harness.py ({script, title, tab, shown, inputs, non_arg_controls, paste_field_names})
    _ext("sd_forge_anima_guidance", "anima_guidance.py", "script"),
    _ext("sd_forge_anima_hires_guard", "anima_hires_guard.py", "script"),
    _ext("sd_forge_anima_presets", "anima_remi_presets.py", "script"),
    _ext("sd-forge-anima-prompt-rescale", "anima_prompt_rescale.py", "script"),
    # tests/remi_anima_lora_harness.py ({extension, tab, count, paste_field_names, inputs})
    # recorded with the panels in English (their own language setting, "zh" by default)
    _ext("anima-artist-scheduled-mixer", "anima_artist_scheduled_mixer.py", "lora", options={"anima_artist_scheduled_mixer_language": "en"}),
    _ext("anima-lora-layer-weight", "anima_lora_layer_weight.py", "lora", options={"anima_lora_layer_weight_language": "en"}),
    _ext("anima-lora-stage-scheduler", "anima_lora_stage_scheduler.py", "lora", options={"anima_lora_stage_scheduler_language": "en"}),
    # ControlNet's unit record (args, unit_args, infotext_keys, unit_controls)
    _ext("sd_forge_controlnet", "controlnet.py", "controlnet", contexts=("controlnet",)),
    # tests/remi_page_harness.py (an on_ui_tabs page: inputs, buttons, title)
    Recording("sd-forge-prompt-workshop", "extensions-builtin/sd-forge-prompt-workshop/scripts/prompt_workshop.py", "page", {"page": "extensions-builtin/sd-forge-prompt-workshop/tests/fixtures/legacy_ui_page.json"}),
    Recording("sd-webui-joycaption-modified", "extensions-builtin/sd-webui-joycaption-modified/scripts/joycaption.py", "page", {"page": "extensions-builtin/sd-webui-joycaption-modified/tests/fixtures/legacy_ui_page.json"}),
    # tests/test_remi_ui_scripts.py (the always-on processing scripts, recorded at the last Gradio revision)
    Recording("modules/processing_scripts", "modules/processing_scripts", "processing_scripts", {"all": "tests/fixtures/legacy_ui/processing_scripts.json"}, rev=GRADIO_REV),
)

BY_NAME = {recording.name: recording for recording in RECORDINGS}

NOT_RECORDINGS = {
    "tests/fixtures/legacy_ui/extension_settings.json": "the settings each extension registered, read from the sources by tools/record_legacy_settings.py (no ui())",
    "extensions-builtin/sd-forge-anima-resolutions/tests/fixtures/legacy_upscale_targets.json": "values of _compute_upscale_target, not a ui() record",
}
"""Files next to the fixtures that are not ``ui()`` records, and what they are."""


def recording_of(script: Path | str) -> Recording | None:
    """The recording of a committed fixture made from ``script`` (a path in this checkout), if any."""

    path = Path(script)
    if path.is_absolute():
        if not path.resolve().is_relative_to(REPO):
            return None
        path = path.resolve().relative_to(REPO)
    return next((recording for recording in RECORDINGS if recording.script == path.as_posix()), None)


def with_options(recording: Recording, options: Any) -> Recording:
    """``recording`` with ``record_legacy_ui.py``'s per-script options: ``--tabs`` (recorded
    whatever ``show()`` says), ``--options`` and ``--preimport`` added, ``--class``."""

    files = dict(recording.files)
    if options.tabs:
        files = {tab: recording.files.get(tab, f"legacy_ui_{tab}.json") for tab in options.tabs}
    return dataclasses.replace(
        recording,
        files=files,
        options={**recording.options, **options.options},
        preimport=(*recording.preimport, *options.preimport),
        class_name=options.class_name or recording.class_name,
        shown_without=recording.shown_without or ("--tabs" if options.tabs else ""),
    )


def fixture_files() -> list[str]:
    """Every committed legacy fixture of the tree (repository-relative), records or not."""

    found = [*REPO.glob("extensions-builtin/*/tests/fixtures/legacy_*.json"), *REPO.glob("tests/fixtures/legacy_ui/*.json")]
    return sorted(path.relative_to(REPO).as_posix() for path in found)


# region Comparison


def _same(got: Any, want: Any) -> bool:
    if isinstance(got, bool) or isinstance(want, bool):
        return type(got) is type(want) and got == want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        return got == want
    return type(got) is type(want) and got == want


def differences(got: Any, want: Any, where: str = "") -> list[str]:
    """Every place where the new recording ``got`` differs from the committed ``want``, recursively."""

    if isinstance(want, dict) and isinstance(got, dict):
        problems = []
        for key in [*want, *(key for key in got if key not in want)]:
            here = f"{where}.{key}" if where else str(key)
            if key not in got:
                problems.append(f"{here}: the fixture has {want[key]!r}, the recording has no such key")
            elif key not in want:
                problems.append(f"{here}: the recording has {got[key]!r}, the fixture has no such key")
            else:
                problems += differences(got[key], want[key], here)
        return problems
    if isinstance(want, list) and isinstance(got, list):
        problems = [f"{where or 'the record'}: {len(got)} items recorded, the fixture has {len(want)}"] if len(got) != len(want) else []
        for index, (item, expected) in enumerate(zip(got, want, strict=False)):
            label = expected.get("label") if isinstance(expected, dict) else None
            problems += differences(item, expected, f"{where}[{index}]" + (f" ({label!r})" if label else ""))
        return problems
    return [] if _same(got, want) else [f"{where or 'the record'}: recorded {got!r}, the fixture has {want!r}"]


def normalized(data: Any, fmt: str) -> Any:
    """``data`` as compared: a bare-list (``port``) record lists every wiring key (absent = none)."""

    if fmt == "port" and isinstance(data, list):
        return [{**dict.fromkeys(WIRING_KEYS), **entry} if isinstance(entry, dict) else entry for entry in data]
    return data


SHAPES: dict[str, tuple[frozenset[str], ...]] = {
    "contract": (frozenset({"script", "tab", "count", "infotext_fields", "inputs"}),),
    "extras": (frozenset({"script", "tab", "count", "inputs"}),),
    "adetailer": (frozenset({"script", "tab", "options", "count", "infotext_fields", "infotext_keys", "inputs", "fields", "settings"}),),
    "panel": (frozenset({"shown", "script", "inputs"}), frozenset({"shown", "inputs"})),
    "script": (frozenset({"script", "title", "tab", "shown", "inputs", "non_arg_controls", "paste_field_names"}),),
    "lora": (frozenset({"extension", "tab", "count", "paste_field_names", "inputs"}),),
    "controlnet": (frozenset({"_about", "args", "unit_args", "infotext_keys", "unit_controls"}),),
    "page": (frozenset({"inputs", "buttons", "title"}),),
    "processing_scripts": (frozenset({"_about", "samples", "txt2img", "img2img"}),),
}
"""The top-level keys of each object format (``port`` is a bare list)."""


def format_of(data: Any) -> str:
    """The format of a record, from its shape: a key of :data:`FORMATS`, or ``"unknown"``."""

    if isinstance(data, list):
        return "port"
    keys = frozenset(data) if isinstance(data, dict) else None
    return next((fmt for fmt, shapes in SHAPES.items() if keys in shapes), "unknown")


def compare_data(got: Any, want: Any, fmt: str, name: str = "") -> list[str]:
    prefix = f"{name}: " if name else ""
    return [prefix + problem for problem in differences(normalized(got, fmt), normalized(want, fmt))]


# endregion

# region Recording (runs in the Gradio 4.40 interpreter, in the legacy revision)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _harness(name: str):
    """One of this checkout's test harnesses (they describe what the tests compare)."""

    return sys.modules.get(f"current_{name}") or _load(REPO / "tests" / f"{name}.py", f"current_{name}")


def legacy_gradio():
    """The Gradio of the legacy revision: the module its ``modules/ui_components.py``
    imported (sys.path has the exported tree first). This checkout imports no Gradio:
    the recorder runs only in its own Gradio 4.40 interpreter, inside that export."""

    from modules import ui_components  # the legacy tree's: every Gradio-era UI module imports Gradio as ``gr``

    return ui_components.gr


def _jsonable(value: Any) -> Any:
    from remi.ui.schema import jsonable

    return jsonable(value.value if isinstance(value, Enum) else value)


@dataclasses.dataclass
class Built:
    """A script's Gradio ``ui()`` built for one tab, with what the formats read."""

    script: Any
    tab: str
    shown: bool
    controls: list[Any]
    blocks: Any = None
    stand_ins: frozenset[int] = frozenset()
    presets: dict[int, tuple[str, Any]] = dataclasses.field(default_factory=dict)

    @property
    def fields(self) -> list[Any]:
        return list(getattr(self.script, "infotext_fields", None) or [])

    @property
    def layout(self):
        from remi.ui.legacy import legacy_layout

        return legacy_layout("legacy", self.controls, self.fields)

    def preset(self, control) -> str | None:
        return self.presets.get(id(control), (None, None))[0]


# The core components the WebUI built before the scripts' ui() (its txt2img / img2img tabs),
# with the elem_ids the scripts look up through on_after_component
CORE_COMPONENTS = (
    ("txt2img_prompt", "Textbox", {}),
    ("txt2img_generate", "Button", {}),
    ("txt2img_width", "Slider", {"minimum": 64, "maximum": 2048, "value": 512, "step": 8}),
    ("txt2img_height", "Slider", {"minimum": 64, "maximum": 2048, "value": 512, "step": 8}),
    ("img2img_prompt", "Textbox", {}),
    ("img2img_generate", "Button", {}),
    ("img2img_width", "Slider", {"minimum": 64, "maximum": 2048, "value": 512, "step": 8}),
    ("img2img_height", "Slider", {"minimum": 64, "maximum": 2048, "value": 512, "step": 8}),
)


def core_components() -> list[Any]:
    """Stand-ins for :data:`CORE_COMPONENTS`, created in the current ``gr.Blocks``
    (their ``after_component`` callbacks run, as in the WebUI)."""

    gr = legacy_gradio()

    return [getattr(gr, kind)(elem_id=elem_id, visible=False, **kwargs) for elem_id, kind, kwargs in CORE_COMPONENTS]


def build(script, tab: str, *, check_shown: bool = True, prepare: Callable[[], Any] | None = None) -> Built:
    """``script``'s ``ui()`` on ``tab`` as the WebUI built it: a fresh preset
    registry, the core components first, then the panel in one ``gr.Blocks``."""

    gr = legacy_gradio()

    from modules.anima_presets import preset_controls, reset_preset_controls

    script.tabname = tab
    script.is_img2img = tab == "img2img"
    script.is_txt2img = not script.is_img2img
    script.infotext_fields = []
    shown = bool(script.show(script.is_img2img)) if check_shown else True
    if not shown:
        return Built(script, tab, False, [])
    reset_preset_controls(tab)
    with gr.Blocks() as blocks:
        stand_ins = frozenset(id(component) for component in core_components())
        if prepare is not None:
            prepare()
        controls = list(script.ui(script.is_img2img) or [])
    names, components, values = preset_controls(tab)
    presets = {id(component): (name, value) for name, component, value in zip(names, components, values, strict=True)}
    return Built(script, tab, True, controls, blocks, stand_ins, presets)


def _script_class(module, recording: Recording):
    from modules import scripts

    classes = [value for value in vars(module).values() if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__]
    if recording.class_name:
        classes = [cls for cls in classes if cls.__name__ == recording.class_name]
    if len(classes) != 1:
        raise SystemExit(f"expected one script class in {recording.script}, found {[cls.__name__ for cls in classes]} (use --class)")
    if classes[0].ui is scripts.Script.ui:
        raise SystemExit(f"{classes[0].__name__} has no Gradio ui() in {module.__file__} (already ported to params()?): record the revision before its port ({recording.rev[:10]}: --script-rev {recording.rev[:10]})")
    return classes[0]


def _tab_scripts(recording: Recording, script_path: Path, harness) -> dict[str, Built]:
    module = harness.load_script(script_path, f"legacy_snapshot_{script_path.stem}")
    sys.modules.setdefault(module.__name__, module)
    cls = _script_class(module, recording)
    built = {}
    for tab in recording.files:
        script = cls()
        script.filename = str(script_path)
        built[tab] = build(script, tab, check_shown=not recording.shown_without)
    return built


# region Formats


def _port_entries(built: Built) -> list[dict[str, Any]]:
    entries = _harness("remi_port_harness").describe(built.layout)
    for entry, control in zip(entries, built.controls, strict=True):
        entry["preset"] = built.preset(control)
    return entries


def format_port(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """``remi_port_harness.describe()`` of each tab's arguments (the default format)."""

    out = {}
    for tab, built in _tab_scripts(recording, script_path, harness).items():
        if not built.shown:
            raise SystemExit(f"{recording.script} does not show on {tab}")
        out[tab] = _port_entries(built)
    return out


def _contract_entries(built: Built, *, names: list[str] | None = None) -> list[dict[str, Any]]:
    from remi.ui.legacy import legacy_layout

    layout = legacy_layout("legacy", built.controls, built.fields, names=names)
    entries = _harness("legacy_ui_contract").describe(layout)
    for entry, control in zip(entries, built.controls, strict=True):
        entry["info"] = _info(control)
        entry["preset"] = built.preset(control)
        if names is None:
            entry.pop("name")
    return entries


def _contract(built: Built) -> dict[str, Any]:
    return {"script": built.script.title(), "tab": built.tab, "count": len(built.controls), "infotext_fields": len(built.fields), "inputs": _contract_entries(built)}


def format_contract(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """``legacy_ui_contract.describe()`` of each tab's arguments, with the panel's counts."""

    return {tab: _contract(built) for tab, built in _tab_scripts(recording, script_path, harness).items()}


def format_panel(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """``remi_panel_harness.layout_contract()`` of each tab's arguments; ``{"shown": False, "inputs": []}`` off the tab."""

    contract = _harness("remi_panel_harness").layout_contract
    out = {}
    for tab, built in _tab_scripts(recording, script_path, harness).items():
        if not built.shown:
            out[tab] = {"shown": False, "inputs": []}
            continue
        entries = contract(built.layout)
        for entry, control in zip(entries, built.controls, strict=True):
            entry.pop("name")
            entry["preset"] = built.preset(control)
        out[tab] = {"shown": True, "script": type(built.script).__name__, "inputs": entries}
    return out


def _gradio_class(control) -> str:
    """The Gradio class of a control, as the bridge maps it (Forge's ``InputAccordion`` is a ``Checkbox``)."""

    from remi.ui.legacy import gradio_kind

    return gradio_kind(control) or type(control).__name__


SCRIPT_KEYS = ("type", "label", "default", "minimum", "maximum", "step", "integer", "choices", "style", "lines")


def _non_arguments(built: Built) -> list[dict[str, Any]]:
    """The components the panel built but did not return (texts, buttons, tools), in creation order."""

    gr = legacy_gradio()

    arguments = {id(control) for control in built.controls} | built.stand_ins
    found = []
    for block in built.blocks.blocks.values():
        if id(block) in arguments or not isinstance(block, gr.components.Component) or isinstance(block, gr.State):
            continue
        found.append({"legacy_control": _gradio_class(block), "label": block.label, "value": _jsonable(block.value)})
    return found


def format_script(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """The Anima panels' record: each argument's schema (``remi_script_harness`` keys),
    its Gradio class and visibility, the other components and the paste field names."""

    from remi import ui

    out = {}
    for tab, built in _tab_scripts(recording, script_path, harness).items():
        record = {"script": type(built.script).__name__, "title": built.script.title(), "tab": tab, "shown": built.shown, "inputs": [], "non_arg_controls": [], "paste_field_names": []}
        if built.shown:
            ctx = ui.Context()
            for node, control in zip(built.layout.inputs, built.controls, strict=True):
                schema = node.schema(ctx)
                entry = {key: schema[key] for key in SCRIPT_KEYS if key in schema}
                if "choices" in entry:
                    entry["choices"] = [choice["value"] for choice in entry["choices"]]
                entry.update(infotext=node.infotext, preset=built.preset(control), info=_info(control), legacy_control=_gradio_class(control), visible=bool(control.visible))
                record["inputs"].append(entry)
            record["non_arg_controls"] = _non_arguments(built)
            record["paste_field_names"] = list(getattr(built.script, "paste_field_names", None) or [])
        out[tab] = record
    return out


def _lora_entry(built: Built, node, control, targets: dict[int, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"kind": node.type, "legacy_control": _gradio_class(control), "label": control.label, "default": _jsonable(control.value)}
    if control.visible is False:
        entry["visible"] = False
    for key in ("minimum", "maximum", "step", "precision"):
        if getattr(control, key, None) is not None:
            entry[key] = getattr(control, key)
    if getattr(control, "choices", None) is not None:
        entry["choices"] = _jsonable([list(choice) if isinstance(choice, tuple) else choice for choice in control.choices])
    if id(control) in targets:
        target = targets[id(control)]
        if isinstance(target, str):  # a key (a PasteKey is also the paste function)
            entry["infotext"] = str(target)
        if callable(target):
            entry["paste"] = True
    name, value = built.presets.get(id(control), (None, None))
    if name is not None:
        entry.update(preset=name, preset_value=_jsonable(value))
    return entry


def format_lora(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """The LoRA panels' record: each argument's Gradio attributes, paste key and preset field."""

    out = {}
    for tab, built in _tab_scripts(recording, script_path, harness).items():
        targets: dict[int, Any] = {}
        for field in built.fields:
            targets.setdefault(id(field[0]), field[1])
        inputs = [_lora_entry(built, node, control, targets) for node, control in zip(built.layout.inputs, built.controls, strict=True)]
        out[tab] = {"extension": Path(recording.extension).name, "tab": tab, "count": len(built.controls), "paste_field_names": list(getattr(built.script, "paste_field_names", None) or []), "inputs": inputs}
    return out


class _ReadKeys(dict):
    """An empty infotext that notes the keys a paste function reads."""

    def __init__(self):
        super().__init__()
        self.read: list[str] = []

    def get(self, key, default=None):
        self.read.append(key)
        return default

    def __getitem__(self, key):
        self.read.append(key)
        raise KeyError(key)

    def __contains__(self, key):
        self.read.append(key)
        return False


def _paste_function(function) -> str:
    probe = _ReadKeys()
    try:
        function(probe)
    except Exception:  # only the keys it looked up matter
        pass
    return f"<paste function: {', '.join(dict.fromkeys(probe.read))}>"


def _info(control) -> str | None:
    """The help text of a control: Gradio's ``info=`` or Forge's ``tooltip=``."""

    return getattr(control, "info", None) or getattr(control, "webui_tooltip", None)


def _adetailer_field(attr: str, control, target) -> dict[str, Any]:
    from remi import ui
    from remi.ui.legacy import gradio_kind, legacy_node

    entry: dict[str, Any] = {"attr": attr}
    if gradio_kind(control) is None:  # gradio_rangeslider's RangeSlider: no remi.ui equivalent in the bridge
        entry.update(kind=type(control).__name__, label=control.label, default=_jsonable(control.value), minimum=control.minimum, maximum=control.maximum, step=control.step)
    else:
        schema = legacy_node(attr, control, label=str(control.label or attr)).schema(ui.Context())
        entry.update(kind=schema["type"], label=control.label, info=_info(control), default=schema.get("default"))
        for key in ("style", "choices", "lines", "max_lines", "placeholder", "minimum", "maximum", "step", "integer"):
            if key in schema:
                entry[key] = [choice["value"] for choice in schema[key]] if key == "choices" else schema[key]
        if control.show_label is False:
            entry["show_label"] = False
    entry.update(visible=control.visible, interactive=control.interactive, infotext=target if isinstance(target, str) else _paste_function(target))
    return entry


def _option_record(key: str, info) -> dict[str, Any]:
    component = info.component
    return {
        "key": key,
        "label": info.label,
        "default": _jsonable(info.default),
        "component": getattr(component, "__name__", None),
        "component_args": _jsonable(info.component_args) if not callable(info.component_args) else "<function>",
        "comment_after": info.comment_after,
        "needs_reload_ui": info.requires_reload_ui,
    }


def format_adetailer(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """ADetailer's record: the contract of its arguments (two toggles, one ``gr.State`` per
    tab), every tab's widgets (``fields``, by ``ALL_ARGS`` attribute) and its settings."""

    from modules import shared

    out = {}
    built_tabs = _tab_scripts(recording, script_path, harness)
    from lib_adetailer.args import ALL_ARGS  # the exported extension's (on sys.path once loaded)

    module = sys.modules[f"legacy_snapshot_{script_path.stem}"]  # loaded by _tab_scripts
    saved = dict(shared.opts.data_labels)
    try:
        module.on_ui_settings()
        settings = [_option_record(key, info) for key, info in shared.opts.data_labels.items() if key not in saved]
    finally:
        shared.opts.data_labels.clear()
        shared.opts.data_labels.update(saved)
    attrs = list(ALL_ARGS.attrs)
    for tab, built in built_tabs.items():
        record = _contract(built)
        fields = built.fields
        units = [fields[2 + n * len(attrs) : 2 + (n + 1) * len(attrs)] for n in range(len(built.controls) - 2)]
        out[tab] = {
            "script": record["script"],
            "tab": tab,
            "options": dict(recording.options),
            "count": record["count"],
            "infotext_fields": record["infotext_fields"],
            "infotext_keys": [target if isinstance(target, str) else "<paste function>" for _component, target in fields],
            "inputs": record["inputs"],
            "fields": [[_adetailer_field(attr, component, target) for attr, (component, target) in zip(attrs, unit, strict=True)] for unit in units],
            "settings": settings,
        }
    return out


def format_extras(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """An Extras operation (``ScriptPostprocessing``): its controls by key, in ``legacy_ui_contract`` terms."""

    gr = legacy_gradio()

    from modules import scripts_postprocessing

    module = harness.load_script(script_path, f"legacy_snapshot_{script_path.stem}")
    classes = [value for value in vars(module).values() if isinstance(value, type) and issubclass(value, scripts_postprocessing.ScriptPostprocessing) and value.__module__ == module.__name__]
    classes = [cls for cls in classes if cls.__name__ == recording.class_name]
    operation = classes[0]()
    with gr.Blocks():
        core_components()
        controls = operation.ui()
    built = Built(operation, "extras", True, list(controls.values()))
    return {"extras": {"script": operation.name, "tab": "extras", "count": len(controls), "inputs": _contract_entries(built, names=list(controls))}}


CONTROLNET_ABOUT = "Recorded from the Gradio ui() before the port (Gradio 4.40, preprocessors: modules_forge.supported_preprocessor only, no ControlNet models)."

# Fixture field -> attribute of the unit group (``canvas_editor.<name>`` for the Region canvas)
CONTROLNET_UNIT_CONTROLS = {
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


def _controlnet_control(field: str, component, infotext: dict[int, str]) -> dict[str, Any]:
    entry = {"field": field, "component": type(component).__name__, "label": component.label, "default": _jsonable(component.value)}
    if hasattr(component, "minimum") and hasattr(component, "maximum"):
        entry.update(minimum=component.minimum, maximum=component.maximum, step=component.step)
    entry.update(visible=component.visible, info=getattr(component, "info", None))
    choices = getattr(component, "choices", None)
    if choices is not None:
        entry["choices"] = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in choices]
    entry["infotext"] = infotext.get(id(component))
    return entry


def _controlnet_stand_ins(context) -> None:
    """The native WebUI components the unit group wires itself to (its
    ``A1111Context``, which ``on_after_component`` fills in the WebUI)."""

    gr = legacy_gradio()

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


def format_controlnet(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """ControlNet's record: the script's arguments (one ``gr.State`` per unit, its
    ``ControlNetUnit`` fields as the default), the unit fields the group's controls
    fed, the paste keys and the first unit's controls."""

    import modules_forge.supported_preprocessor  # noqa: F401 - None and the built-in preprocessors

    module = harness.load_script(script_path, f"legacy_snapshot_{script_path.stem}")
    from lib_controlnet import global_state

    global_state.controlnet_names = ["None"]
    global_state.controlnet_filename_dict = {"None": None}
    groups = []
    original = module.ControlNetUiGroup

    def collect(*args, **kwargs):
        groups.append(original(*args, **kwargs))
        return groups[-1]

    module.ControlNetUiGroup = collect
    out = {}
    try:
        for tab in recording.files:
            groups.clear()
            script = module.ControlNetForForgeOfficial()
            script.filename = str(script_path)
            built = build(script, tab, prepare=lambda: _controlnet_stand_ins(original.a1111_context))
            entries = _port_entries(built)
            for entry, control in zip(entries, built.controls, strict=True):
                if isinstance(getattr(control, "value", None), module.ControlNetUnit):  # the unit's gr.State: its fields
                    entry.update(label="", default={name: _jsonable(value) for name, value in vars(control.value).items()})
                for key in WIRING_KEYS:  # recorded before the wiring keys
                    if entry.get(key) is None:
                        entry.pop(key, None)
            locators = {id(component): locator for component, locator in built.fields}
            controls = []
            for field, attribute in CONTROLNET_UNIT_CONTROLS.items():
                component = groups[0]
                for part in attribute.split("."):
                    component = getattr(component, part, None)
                if component is not None:
                    controls.append(_controlnet_control(field, component, locators))
            fields = [field.name for field in dataclasses.fields(module.ControlNetUnit)]
            out[tab] = {
                "_about": CONTROLNET_ABOUT,
                "args": entries,
                # The group built UiControlNetUnit(*controls): the fields up to "control_mode"
                "unit_args": fields[: fields.index("control_mode") + 1],
                "infotext_keys": sorted(locators.values()),
                "unit_controls": controls,
            }
    finally:
        module.ControlNetUiGroup = original
    return out


def format_page(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """An ``on_ui_tabs`` page: its input controls (``remi_page_harness.describe_node``)
    and buttons in creation order, and its title."""

    gr = legacy_gradio()

    from modules import script_callbacks
    from remi import ui
    from remi.ui.legacy import gradio_kind, legacy_layout

    describe_node = _harness("remi_page_harness").describe_node
    script_callbacks.clear_callbacks()
    harness.load_script(script_path, f"legacy_page_{script_path.stem}")
    tabs = [tab for callback in script_callbacks.callback_map["callbacks_ui_tabs"] for tab in callback.callback() or []]
    if len(tabs) != 1:
        raise SystemExit(f"expected one tab from {recording.script}, got {len(tabs)}")
    blocks, title = tabs[0][0], tabs[0][1]
    controls = [block for block in blocks.blocks.values() if gradio_kind(block) is not None]
    entries = []
    for node, control in zip(legacy_layout("legacy", controls, []).inputs, controls, strict=True):
        entry = describe_node(node, ui.Context())
        if not entry["label"].strip() or entry["label"].startswith("#"):
            entry["label"] = str(getattr(control, "label", "") or "")
        entry["elem_id"] = getattr(control, "elem_id", None)
        entries.append(entry)
    buttons = [str(block.value) for block in blocks.blocks.values() if isinstance(block, gr.Button)]
    return {"page": {"inputs": entries, "buttons": buttons, "title": title}}


PROCESSING_ABOUT = "Recorded from the Gradio ui() of modules/processing_scripts before it was removed; paste: the paste function's result for each of the test's PARAMS_SAMPLES"
PROCESSING_SAMPLES = (
    {},
    {"Variation seed": "5", "Seed resize from-1": "512", "Refiner": "x", "Sampler": "Euler", "Schedule type": "Karras"},
    {"Sampler": "DPM++ 2M Karras", "Hires sampler": "Euler"},
)
"""Infotexts the paste functions are recorded on (``tests/test_remi_ui_scripts.py`` replays them)."""


def _node_class(node) -> str:
    return next(cls.__name__ for cls in type(node).__mro__ if cls.__module__ == "remi.ui.nodes")


def format_processing_scripts(recording: Recording, script_path: Path, harness) -> dict[str, Any]:
    """The always-on processing scripts of the legacy revision (``modules/processing_scripts``):
    per tab and script, each argument's ``remi.ui`` class, label, default, bounds or
    choices, and its infotext key or what its paste function makes of each sample."""

    from remi import ui
    from remi.testing import processing_script_classes

    ctx = ui.Context()
    out: dict[str, Any] = {"_about": PROCESSING_ABOUT, "samples": [dict(sample) for sample in PROCESSING_SAMPLES]}
    for tab in TABS:
        out[tab] = []
        for data in processing_script_classes():
            script = data.script_class()
            script.filename = data.path
            built = build(script, tab, check_shown=False)
            inputs = []
            for node, control in zip(built.layout.inputs, built.controls, strict=True):
                kind = _node_class(node)
                entry: dict[str, Any] = {"kind": kind, "label": str(control.label or ""), "default": _jsonable(node.resolve_default(ctx))}
                if kind == "Slider":
                    entry.update(minimum=node.minimum, maximum=node.maximum, step=node.step)
                if kind == "Choice":
                    entry["choices"] = _jsonable(node.values(ctx))
                if node.infotext:
                    entry["infotext"] = node.infotext
                elif node.paste is not None:
                    entry["paste"] = [_jsonable(node.paste(dict(sample))) for sample in PROCESSING_SAMPLES]
                inputs.append(entry)
            out[tab].append({"script": type(script).__name__, "inputs": inputs})
    return {"all": out}


FORMATS: dict[str, Callable[[Recording, Path, Any], dict[str, Any]]] = {
    "port": format_port,
    "contract": format_contract,
    "adetailer": format_adetailer,
    "extras": format_extras,
    "panel": format_panel,
    "script": format_script,
    "lora": format_lora,
    "controlnet": format_controlnet,
    "page": format_page,
    "processing_scripts": format_processing_scripts,
}

# endregion

# region Contexts


def context_upscalers(tree: Path) -> None:
    """The upscalers the WebUI lists with no model in ``models/ESRGAN`` (``shared.sd_upscalers``)."""

    from modules import modelloader, shared

    saved = shared.cmd_opts.esrgan_models_path
    with tempfile.TemporaryDirectory(prefix="remi-legacy-esrgan-") as empty:
        shared.cmd_opts.esrgan_models_path = empty
        try:
            modelloader.load_upscalers()
        finally:
            shared.cmd_opts.esrgan_models_path = saved


def context_controlnet(tree: Path) -> None:
    """ControlNet's pose editor imports ``annotator`` from the legacy preprocessors extension
    (on sys.path in the WebUI, which loads every extension folder first)."""

    sys.path.insert(0, str(tree / "extensions-builtin" / "forge_legacy_preprocessors"))


ADETAILER_DETECTORS = (
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "hand_yolov8n.pt",
    "hand_yolov8s.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
    "yolov8x-worldv2.pt",
    "mediapipe_face_short.tflite",
    "mediapipe_face_full.tflite",
    "face_landmarker.task",
)
"""The detectors ADetailer downloads on its first start (``get_models``' ``TO_DOWNLOAD``)."""


def context_adetailer_detectors(tree: Path) -> None:
    """ADetailer as installed: its downloaded detectors, as empty stand-in files in a
    folder of ``ad_extra_models_dir`` (no download: ``ad_no_huggingface``). It lists
    them in the order ``Path.rglob`` finds them (the fixture's is APFS's, macOS)."""

    import atexit
    import shutil

    from modules import shared

    folder = Path(tempfile.mkdtemp(prefix="remi-legacy-adetailer-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    for name in ADETAILER_DETECTORS:
        (folder / name).touch()
    shared.cmd_opts.ad_no_huggingface = True
    shared.opts.data["ad_extra_models_dir"] = str(folder)


CONTEXTS: dict[str, Callable[[Path], None]] = {
    "adetailer-detectors": context_adetailer_detectors,
    "upscalers": context_upscalers,
    "controlnet": context_controlnet,
}

# endregion


def record(recording: Recording, tree: Path, scratch: Path, out: Path, export: Callable[..., None], *, script_path: Path | None = None) -> list[Path]:
    """Record ``recording`` in the Gradio-era ``tree`` into ``out`` (one file per
    fixture, named as the committed one); ``scratch`` receives the script's export
    from :attr:`Recording.rev` (``export``: ``record_legacy_ui.export``), unless
    ``script_path`` names the script to record (already exported, or on disk)."""

    if script_path is None:
        export(recording.rev, scratch, *recording.export_paths)
        script_path = scratch / recording.script
    sys.argv = sys.argv[:1]
    for path in (str(tree / "modules_forge" / "packages"), str(tree)):
        sys.path.insert(0, path)
    for name in recording.preimport:
        importlib.import_module(name)

    from remi.testing import webui_state

    with webui_state():  # the WebUI's import order first, so modules.infotext_utils imports for real
        pass
    harness = _load(tree / "tests" / "extension_script_harness.py", "extension_script_harness")
    harness.prepare_webui_imports()
    with webui_state(dict(recording.options)):
        for name in recording.contexts:
            CONTEXTS[name](tree)
        data = FORMATS[recording.format](recording, script_path, harness)
    written = []
    for key, path in recording.files.items():
        target = out / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data[key], indent=1, ensure_ascii=False) + "\n", encoding="utf8")
        written.append(target)
    return written


# endregion
