"""Guards shared by the Forge panel ports (compile, multidiffusion, neveroom,
pid, radial, sparse, spectrum, negpip, lora_schedule, comments)."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("remi_port_harness", REPO / "tests" / "remi_port_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

PANELS = ("sd_forge_compile", "sd_forge_multidiffusion", "sd_forge_neveroom", "sd_forge_pid", "sd_forge_radial", "sd_forge_sparse", "sd_forge_spectrum")


@pytest.mark.parametrize("extension", PANELS)
def test_zh_cn_agrees_with_the_shared_catalogs(extension):
    # Catalogs merge into one flat map: a differing entry for core's (or the
    # language pack's) English text would rename that text everywhere.
    assert harness.catalog_conflicts(REPO / "extensions-builtin" / extension) == []


def test_comments_script_imports_standalone():
    harness.assert_imports_standalone([], modules=["modules.processing_scripts.comments"])
