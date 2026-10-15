"""The extensions that register into ControlNet (ControlNet-LLLite, IP-Adapter
and the forge_* preprocessor folders) load without Gradio, as the WebUI loads
them, and what they register is offered by the unit widget's catalog; their
slider labels are translated (each extension's ``localizations/zh_CN.json``,
the control types in ControlNet's)."""

import json
import os
import subprocess
import sys

import pytest
import remi_port_harness as harness

EXTENSIONS = harness.REPO / "extensions-builtin"
SCRIPTS = [
    # Forge's order: extension folders alphabetically, ControlNet itself last
    "forge_legacy_preprocessors/scripts/legacy_preprocessors.py",
    "forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py",
    "forge_preprocessor_reference/scripts/forge_reference.py",
    "forge_preprocessor_tile/scripts/preprocessor_tile.py",
    "sd_forge_controlllite/scripts/forge_controllllite.py",
    "sd_forge_ipadapter/scripts/forge_ipadapter.py",
    "sd_forge_controlnet/scripts/controlnet.py",
]

PROBE = """
import importlib.util, json, sys
from remi.testing import webui_state
with webui_state():
    for index, script in enumerate({scripts!r}):
        folder = script.rsplit("/scripts/", 1)[0]
        sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location(f"cn_ext_{{index}}", script)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    from lib_controlnet.unit_param import catalog
    from modules_forge.shared import supported_control_models, supported_preprocessors
    origin = {{name: type(p).__module__ for name, p in supported_preprocessors.items()}}
    result = {{
        "catalog": catalog(),
        "origin": origin,
        "control_models": [model.__name__ for model in supported_control_models],
        "gradio": sorted(name for name in sys.modules if name.partition(".")[0] in ("gradio", "gradio_client", "gradio_rangeslider")),
    }}
print("RESULT " + json.dumps(result))
"""


@pytest.fixture(scope="module")
def loaded():
    scripts = [str(EXTENSIONS / script) for script in SCRIPTS]
    code = harness.GRADIO_BLOCKER + "\nsys.argv = sys.argv[:1]\n" + PROBE.format(scripts=scripts)
    paths = [str(harness.REPO), str(harness.REPO / "modules_forge" / "packages"), str(EXTENSIONS / "forge_legacy_preprocessors")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    result = subprocess.run([sys.executable, "-c", code], cwd=harness.REPO, env=env, capture_output=True, text=True, timeout=900)
    line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
    assert result.returncode == 0 and line, result.stdout[-2000:] + result.stderr[-4000:]
    return json.loads(line[len("RESULT ") :])


def test_they_load_without_gradio(loaded):
    assert loaded["gradio"] == []


def test_their_preprocessors_and_patchers_are_offered(loaded):
    catalog = loaded["catalog"]
    names = [p["name"] for p in catalog["preprocessors"]]
    assert names[0] == "None" and names[1:] == sorted(names[1:])
    for name in ("canny", "depth_midas", "dw_openpose_full", "lineart_anime", "inpaint_only+lama", "reference_adain+attn", "tile_colorfix+sharp", "CLIP-ViT-H (IPAdapter)", "InsightFace (InstantID)"):
        assert name in names, name
    types = {entry["name"]: entry for entry in catalog["types"]}
    for tag in ("All", "Canny", "Depth", "OpenPose", "Inpaint", "Reference", "Tile", "IP-Adapter", "Instant-ID", "Region"):
        assert tag in types, tag
    assert "inpaint_only" in types["Inpaint"]["modules"] and types["Inpaint"]["default_module"] != "None"
    assert {"ControlLLLiteAnimaPatcher"} <= set(loaded["control_models"])
    assert any("IPAdapter" in name for name in loaded["control_models"])
    json.dumps(catalog, allow_nan=False)


CATALOGS = ("forge_legacy_preprocessors", "forge_preprocessor_reference", "forge_preprocessor_tile")


def folder_of(module: str) -> str | None:
    """The extension folder of a preprocessor class (the probe loads ``SCRIPTS[i]`` as ``cn_ext_<i>``)."""

    if not module.startswith("cn_ext_"):
        return None  # modules_forge (None, canny)
    return SCRIPTS[int(module.rpartition("_")[2])].partition("/")[0]


def test_their_slider_labels_are_translated(loaded):
    """Each extension's own labels in its catalog (the shared ones, e.g.
    "Resolution", in ControlNet's); no entry contradicts a shared catalog."""

    controlnet = json.loads((EXTENSIONS / "sd_forge_controlnet" / "localizations" / "zh_CN.json").read_text(encoding="utf8"))
    for p in loaded["catalog"]["preprocessors"]:
        folder = folder_of(loaded["origin"][p["name"]])
        path = EXTENSIONS / str(folder) / "localizations" / "zh_CN.json"
        own = json.loads(path.read_text(encoding="utf8")) if folder and path.is_file() else {}
        for spec in (p["resolution"], p["slider_1"], p["slider_2"]):
            if spec["visible"]:
                assert spec["label"] in own or spec["label"] in controlnet, (p["name"], spec["label"])
    for folder in CATALOGS:
        assert harness.catalog_conflicts(EXTENSIONS / folder) == [], folder


def test_no_gradio_in_their_sources():
    folders = {script.partition("/")[0] for script in SCRIPTS}
    offenders = []
    for folder in sorted(folders):
        for path in (EXTENSIONS / folder).rglob("*.py"):
            text = path.read_text(encoding="utf8", errors="replace")
            if any(line.lstrip().startswith(("import gradio", "from gradio")) for line in text.splitlines()):
                offenders.append(str(path.relative_to(EXTENSIONS)))
    assert offenders == []
    assert not (EXTENSIONS / "sd_forge_controlnet" / "javascript").exists() and not (EXTENSIONS / "sd_forge_controlnet" / "style.css").exists()
