"""zh_CN with every built-in extension loaded, as a WebUI start loads them:
texts that only appear once the extensions are in place (the X/Y/Z plot axes
ADetailer, Detail Daemon, Dynamic Shift, ... add to the core script) are
translated by the catalog ``/api/v2/i18n/zh_CN`` serves (core and extension
catalogs merged) or by the SPA's own catalog."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import json, sys
sys.argv = ["launch.py", "--skip-prepare-environment", "--skip-version-check", "--skip-torch-cuda-test", "--data-dir", sys.argv[1]]
import webui
from remi.core.tabs import tab_schema
from modules import script_callbacks
from remi.server import i18n, runtime
script_callbacks.before_ui_callback()  # as remi.server.launch starts the app
runtime.prepare()

def find(node, name):
    if isinstance(node, list):
        return next((hit for item in node if (hit := find(item, name)) is not None), None)
    if not isinstance(node, dict):
        return None
    if node.get("name") == name:
        return node
    return next((hit for key in ("children", "layout", "enable") if key in node and (hit := find(node[key], name)) is not None), None)

axes = {}
for tab in ("txt2img", "img2img"):
    script = next(script for script in tab_schema(tab)["scripts"] if script["title"] == "X/Y/Z plot")
    for axis in "xyz":
        node = find(script["layout"], f"{axis}_type")
        axes[f"{tab} {axis}"] = [choice["label"] for choice in node["choices"]]
catalog, _ = i18n.catalog("zh_CN")
print("REMI-PROBE " + json.dumps({"axes": axes, "catalog": catalog}))
"""


def test_every_xyz_axis_is_translated(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "modules_forge" / "packages")])}
    result = subprocess.run([sys.executable, "-c", PROBE, str(tmp_path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    line = next((line for line in result.stdout.splitlines() if line.startswith("REMI-PROBE ")), None)
    assert result.returncode == 0 and line, result.stdout[-2000:] + result.stderr[-4000:]
    probe = json.loads(line.removeprefix("REMI-PROBE "))
    axes, catalog = probe["axes"], probe["catalog"]
    spa = json.loads((ROOT / "webui" / "src" / "locales" / "zh-CN.json").read_text(encoding="utf-8"))

    labels = {label for listed in axes.values() for label in listed}
    # the extensions' axes are there (the probe saw the WebUI as users do)
    assert {"[ADetailer] ADetailer Prompt", "[Detail Daemon] Mode", "[DynShift] Start"} <= labels
    assert {"Seed", "Sampler", "Prompt S/R"} <= labels
    # "Nothing" … "Prompt S/R": every label shown in the axis dropdowns has a translation
    missing = sorted(label for label in labels if not (catalog.get(label) or spa.get(label) or "").strip())
    assert missing == [], f"zh_CN lacks {len(missing)} X/Y/Z axis labels: {missing}"
