"""``/sdapi/v1`` stays compatible with Forge Neo's (``tools/sdapi_contract.py``).

Existing A1111 / Forge clients call Remi's ``/sdapi/v1``: every route and
model field of the last absorbed upstream commit must still be there, with
the same defaults and a type that accepts what upstream's did.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("tool_sdapi_contract", REPO / "tools" / "sdapi_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def has_revision(rev: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=REPO, capture_output=True, check=False).returncode == 0


def test_sdapi_matches_the_last_absorbed_upstream():
    rev = contract.default_rev()
    if not has_revision(rev):
        pytest.skip(f"{rev[:10]} is not in this clone (shallow?)")
    assert contract.check(rev) == []


def test_last_absorbed_is_the_merge_base_with_upstream():
    found = subprocess.run(["git", "merge-base", "HEAD", contract.UPSTREAM], cwd=REPO, capture_output=True, text=True, check=False)
    if found.returncode != 0:
        pytest.skip("no upstream/neo remote in this clone")
    assert found.stdout.strip() == contract.LAST_ABSORBED, "a sync moved the merge base: update LAST_ABSORBED in tools/sdapi_contract.py"


API = """
class Api:
    def __init__(self):
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=models.TextToImageResponse)
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        if flag:
            self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        pass

    def get_config(self):
        pass

    def set_config(self, req: dict[str, Any]):
        pass

    def stop_webui(self):
        pass
"""


def test_routes_are_read_with_their_models_and_parameters():
    found = contract.routes(API)
    assert found[("POST", "/sdapi/v1/txt2img")] == {"response_model": "models.TextToImageResponse", "params": (("txt2imgreq", "models.StableDiffusionTxt2ImgProcessingAPI"),)}
    assert found[("GET", "/sdapi/v1/options")]["response_model"] == "models.OptionsModel"
    assert found[("POST", "/sdapi/v1/options")] == {"response_model": None, "params": (("req", "dict[str, Any]"),)}
    assert ("POST", "/sdapi/v1/server-stop") in found  # routes behind a flag count too
    assert contract.compare_routes(API, API) == []


def test_route_differences_are_reported():
    ours = API.replace('self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])', "pass").replace("response_model=models.OptionsModel", "response_model=models.Other").replace("txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI", "txt2imgreq: dict").replace("def get_config(self):", "def get_config(self):\n        self.add_api_route('/sdapi/v1/remi-extra', self.get_config)\n    def unused(self):")
    assert contract.compare_routes(API, ours) == [
        "route GET /sdapi/v1/options: response_model is 'models.Other', upstream has 'models.OptionsModel'",
        "route POST /sdapi/v1/server-stop: missing in Remi",
        "route POST /sdapi/v1/txt2img: params is (('txt2imgreq', 'dict'),), upstream has (('txt2imgreq', 'models.StableDiffusionTxt2ImgProcessingAPI'),)",
    ]


def models_module(name: str, **models: type[BaseModel]):
    import types

    module = types.ModuleType(name)
    for key, model in models.items():
        model.__module__ = name
        setattr(module, key, model)
    return module


def test_model_fields_defaults_and_types_are_compared():
    class Upstream(BaseModel):
        steps: int = Field(default=20, description="steps")
        name: str
        port: str | None = None

    class Widened(BaseModel):  # optional where upstream was not, a new optional field: compatible
        steps: int | None = Field(default=20, description="other text")
        name: str
        port: str | None = None
        extra: bool = False

    class Broken(BaseModel):
        steps: str = "20"
        port: int | None = None
        required: int

    theirs = models_module("upstream_models", Item=Upstream, Gone=Upstream)
    assert contract.compare_models(theirs, models_module("ok_models", Item=Widened, Gone=Widened)) == []
    problems = contract.compare_models(theirs, models_module("broken_models", Item=Broken))
    assert problems == [
        "model Gone: missing in Remi",
        "model Item: fields ['name'] missing in Remi",
        "model Item: Remi requires ['required'], upstream does not",
        "model Item.port: Remi's schema {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': None} does not accept upstream's {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None}",
        "model Item.steps: default '20', upstream 20",
    ]


def test_allowed_models_still_compare_names_and_defaults(monkeypatch):
    class Upstream(BaseModel):
        port: str | None = None
        debug: bool = False

    class Remi(BaseModel):
        port: int | None = None
        debug: bool | None = True

    monkeypatch.setattr(contract, "ALLOWED", {"Flags": "test"})
    problems = contract.compare_models(models_module("up_flags", Flags=Upstream), models_module("remi_flags", Flags=Remi))
    assert problems == ["model Flags.debug: default True, upstream False"]


def test_unresolved_merge_conflicts_stop_the_check_with_a_clear_message(monkeypatch, capsys):
    # mid-merge the WebUI cannot be imported (conflict markers): say so instead of a SyntaxError traceback
    monkeypatch.setattr(contract, "unmerged_files", lambda: ["modules/launch_utils.py", "scripts/xyz_grid.py"])
    monkeypatch.setattr(contract, "check", lambda rev: pytest.fail("check() must not run while conflicts are unresolved"))

    assert contract.main(["--rev", "HEAD"]) == 2
    err = capsys.readouterr().err
    assert "2 file(s) with unresolved merge conflicts: modules/launch_utils.py, scripts/xyz_grid.py" in err
    assert "upstream-absorption.md" in err


def test_unmerged_files_is_empty_outside_a_merge():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    assert contract.unmerged_files() == []
