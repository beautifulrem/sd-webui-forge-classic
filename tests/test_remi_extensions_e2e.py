"""Every built-in extension (and every script of ``scripts/`` and
``modules/processing_scripts/``) runs end to end, through the real server.

``tests/remi_extensions_e2e.py`` starts the WebUI in a fresh interpreter on the
CPU with every built-in extension and a tiny Anima model of the real
architecture (``tools/tiny_anima.py``), and queues each case through
``POST /api/v2/jobs`` twice: with the extension off and on. These tests read
its record: the job completes without warnings or script errors, the
extension's hooks ran with the arguments the schema converted, the infotext
gains the extension's keys, the image changes (or, where the extension must not
change it, stays the same pixels), and pasting the infotext into a fresh form
reproduces the same pixels and infotext.

An extension or script that is not a case is in :data:`NOT_RUN` with the
reason; ``test_every_extension_and_script_is_run_or_explained`` keeps both in
step with the tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("remi_extensions_e2e", ROOT / "tests" / "remi_extensions_e2e.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("remi_extensions_e2e", module)
    spec.loader.exec_module(module)
    return module


driver = _driver()
CASES = {case.name: case for case in driver.CASES}

NOT_RUN: dict[str, str] = {
    # no processing: UI, pages, prompt tools, the queue
    "extensions-builtin/agent-scheduler": "superseded by the server's job queue (tests/test_remi_server_jobs.py); it has no script",
    "extensions-builtin/extra-options-section": "only places settings in the generation forms (tests in its folder); no processing hook",
    "extensions-builtin/sd-lock-img2img-steps": "a form lock on img2img Steps (the SPA's form store: webui/src/ui/locks.ts and its tests); its script has no processing hook",
    "extensions-builtin/sd-forge-prompt-workshop": "a page (tests in its folder); no generation hook",
    "extensions-builtin/sd-webui-joycaption-modified": "a captioning page; its model is about 17 GB (tests in its folder run it with stubs)",
    "extensions-builtin/sd-webui-language-diffusion": "UI translations (tests/test_remi_core_zh_cn.py, its own tests)",
    "extensions-builtin/sd-webui-prompt-all-in-one-neo": "prompt tools of the editor (tests/test_remi_prompt_tools.py); no generation hook",
    "extensions-builtin/sd-webui-tagcomplete-neo": "tag completion in the prompt editor (tests/test_remi_prompt_tools.py); no generation hook",
    "extensions-builtin/sd_forge_anima_presets": "an action that fills the form from a preset (tests in its folder, tests/test_anima_presets.py); its process() only logs Turbo advice",
    # ControlNet preprocessors and models (canny, tile_colorfix, inpaint_only and the Anima Control-LLLite run as cases)
    "extensions-builtin/forge_preprocessor_reference": "reference_only / reference_adain record and patch the SD 1.5 / SDXL UNet's attn1 blocks, which the Anima DiT does not have",
    "extensions-builtin/sd_forge_ipadapter": "IP-Adapter models exist for SD 1.5 / SDXL only; it replaces the UNet's attn2 (patches_replace), which the Anima DiT does not have",
    # shown only where they can run
    "extensions-builtin/sd_forge_compile": "shown only with Triton (CUDA): torch.compile of the DiT",
    "extensions-builtin/sd_forge_radial": "shown only with the RadialAttention CUDA kernels (Wan video)",
    "extensions-builtin/sd_forge_pid": "shown only when PiD models are installed (a separate pixel-space model)",
    "scripts/checkpoint_merger.py": "the Checkpoint Merger page (tests/test_remi_merger.py); no script",
    "modules/processing_scripts/sampler.py": "sets the sampler, scheduler and steps of every case (the round trips restore them)",
    "modules/processing_scripts/seed.py": "sets the seed of every case (the round trips restore it; xyz_plot varies it)",
}
"""Built-in extensions and scripts without a case, with the reason."""


@pytest.fixture(scope="module")
def record(tmp_path_factory) -> dict:
    data = tmp_path_factory.mktemp("e2e")
    out = data / "record.json"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "modules_forge" / "packages")]), "YOLO_OFFLINE": "1"}
    result = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "tests" / "remi_extensions_e2e.py"), str(data / "data"), str(out)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0 and out.is_file(), result.stdout[-4000:] + result.stderr[-8000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _has_key(infotext: str, key: str) -> bool:
    return infotext.startswith(f"{key}: ") or f", {key}: " in infotext or f"\n{key}: " in infotext


@pytest.mark.parametrize("name", list(CASES))
def test_extension_runs_end_to_end(record, name):
    case = CASES[name]
    entry = record["cases"][name]
    assert entry["status"] == "completed", entry["error"]
    assert entry["status_off"] == "completed"
    assert entry["script_errors"] == []
    # only the expected warnings, each once
    assert len(entry["warnings"]) == len(case.warnings) and all(any(expected in text for text in entry["warnings"]) for expected in case.warnings), entry["warnings"]
    assert entry["outputs"] == case.outputs
    for key in case.infotext:
        assert _has_key(entry["infotext"], key) and not _has_key(entry["infotext_off"], key), key
    for text in case.contains:
        assert text in entry["infotext"], text
    changed = entry["size"] != entry["size_off"] or entry["difference"] > 0
    assert changed == case.changes, (entry["difference"], entry["size"], entry["size_off"])
    if case.same_as is not None:
        assert entry["same_as_difference"] == 0.0
    if case.differs_from is not None:
        assert entry["differs_from_difference"] > 0
    for hook in case.hooks:
        assert hook in entry["calls"], (hook, sorted(entry["calls"]))
        received = entry["calls"][hook]
        # the arguments the hook got, named by the script's params, are the requested values
        for param, value in {**case.params, **case.received}.items():
            if isinstance(value, (bool, int, float)) or (isinstance(value, str) and not value.startswith("@")):
                assert received.get(param) == value, (hook, param, received.get(param))
    if case.round_trip:
        trip = entry["round_trip"]
        assert trip["problems"] == [] and trip["status"] == "completed", trip
        assert trip["same_pixels"], trip["difference"]
        assert trip["same_infotext"], (trip["infotext"], entry["infotext"])


def test_the_runs_are_deterministic(record):
    # the cases that must not change the image give the very same pixels: any
    # difference elsewhere is the extension's
    unchanged = [name for name, case in CASES.items() if not case.changes]
    assert unchanged and all(record["cases"][name]["difference"] == 0.0 for name in unchanged)


def test_every_extension_and_script_is_run_or_explained(record):
    sources = {source for owners in record["owners"].values() for source in owners.values()}
    run_sources = {record["owners"][case.tab][case.owner] for case in CASES.values()}
    covered = {_unit(source) for source in run_sources} | {unit for case in CASES.values() for unit in case.covers}
    tree = {f"extensions-builtin/{path.name}" for path in (ROOT / "extensions-builtin").iterdir() if path.is_dir() and not path.name.startswith((".", "__"))}
    tree |= {f"scripts/{path.name}" for path in (ROOT / "scripts").glob("*.py")}
    tree |= {f"modules/processing_scripts/{path.name}" for path in (ROOT / "modules" / "processing_scripts").glob("*.py") if path.name != "__init__.py"}
    assert sorted(tree - covered - set(NOT_RUN)) == [], "neither a case nor in NOT_RUN"
    assert sorted(covered & set(NOT_RUN)) == [], "a case runs it: drop it from NOT_RUN"
    assert sorted(set(NOT_RUN) - tree) == [], "NOT_RUN names what is not in the tree"
    assert sorted(covered - tree) == [], "a case covers what is not in the tree"
    # every loaded script and Extras operation that has a case target is run
    assert sorted({_unit(source) for source in sources} - covered - set(NOT_RUN)) == []
    assert all(len(reason) > 30 for reason in NOT_RUN.values())


def _unit(source: str) -> str:
    parts = source.split("/")
    return "/".join(parts[:2]) if parts[0] == "extensions-builtin" else source
