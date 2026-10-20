"""Check that Remi's ``/sdapi/v1`` API still matches Forge Neo's.

Remi keeps the A1111 / Forge Neo API (``modules/api/api.py``,
``modules/api/models.py``) for existing clients. This compares the working
tree with upstream's two files at a revision:

- routes: every ``add_api_route`` of upstream's ``Api`` (method, path,
  response model, the endpoint's parameters and their annotations) exists in
  Remi's with the same values; Remi may add routes;
- models: every pydantic model of upstream's ``models.py`` exists in Remi's
  with the same fields, defaults and required set, and each field accepts
  what upstream's accepts (equal, or the same type made optional). Both
  modules are built in this runtime, so the models that are generated from
  ``modules.processing`` and the settings compare like for like.

Documented exceptions are in ``ALLOWED``. Run it with the WebUI's
interpreter (no model is loaded)::

    $PYTHON tools/sdapi_contract.py                     # against the last absorbed upstream commit
    $PYTHON tools/sdapi_contract.py --rev upstream/neo  # during a sync, after the merge

The default revision is ``git merge-base HEAD upstream/neo``, or
``LAST_ABSORBED`` in a clone without the ``upstream`` remote.
``tests/test_remi_sdapi_contract.py`` runs the default check (``tools/check.sh``
"pytest:tests").
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import types
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = "upstream/neo"
LAST_ABSORBED = "46af880c003d31846bbea984ce00fc7be247ce5b"
"""The last Forge Neo commit merged into Remi (docs/upstream-sync-2026-10-14.md);
update it with each sync (tests/test_remi_sdapi_contract.py checks it against
the merge base when the ``upstream`` remote is there)."""
API = "modules/api/api.py"
MODELS = "modules/api/models.py"

ALLOWED: dict[str, str] = {
    "FlagsModel": ("field types follow each flag's argparse type and nargs (Remi's _flag_type): upstream typed a flag without a default as str and others as their default's type, so /sdapi/v1/cmd-flags answered 500 for --port 7860 or a repeated flag; names and defaults are unchanged"),
}
"""Models whose field *types* may differ from upstream's (names, defaults and
required fields are still compared), with the reason."""

IGNORED_KEYS = ("title", "description")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def default_rev() -> str:
    found = subprocess.run(["git", "merge-base", "HEAD", UPSTREAM], cwd=ROOT, capture_output=True, text=True, check=False)
    return found.stdout.strip() if found.returncode == 0 and found.stdout.strip() else LAST_ABSORBED


def show(rev: str, path: str) -> str:
    return git("show", f"{rev}:{path}")


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as file:
        return file.read()


# region Routes


def routes(source: str) -> dict[tuple[str, str], dict[str, Any]]:
    """``{(METHOD, path): {"response_model", "params"}}`` of the ``add_api_route`` calls of ``class Api``."""

    tree = ast.parse(source)
    api = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    methods = {node.name: node for node in api.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found = {}
    for call in ast.walk(api):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_api_route"):
            continue
        if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        endpoint = call.args[1] if len(call.args) > 1 else keywords.get("endpoint")
        params: tuple = ()
        if isinstance(endpoint, ast.Attribute) and endpoint.attr in methods:
            arguments = methods[endpoint.attr].args
            params = tuple((arg.arg, ast.unparse(arg.annotation) if arg.annotation else None) for arg in [*arguments.args[1:], *arguments.kwonlyargs])
        response = keywords.get("response_model")
        verbs = ast.literal_eval(keywords["methods"]) if "methods" in keywords else ["GET"]
        for verb in verbs:
            found[(verb.upper(), call.args[0].value)] = {"response_model": ast.unparse(response) if response is not None else None, "params": params}
    return found


def compare_routes(upstream: str, remi: str) -> list[str]:
    theirs, ours = routes(upstream), routes(remi)
    problems = []
    for key, want in sorted(theirs.items()):
        got = ours.get(key)
        if got is None:
            problems.append(f"route {key[0]} {key[1]}: missing in Remi")
            continue
        for field in ("response_model", "params"):
            if got[field] != want[field]:
                problems.append(f"route {key[0]} {key[1]}: {field} is {got[field]!r}, upstream has {want[field]!r}")
    return problems


# endregion

# region Models


def build_models(source: str, name: str) -> types.ModuleType:
    """Execute a ``models.py`` source as module ``name`` in this runtime."""

    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - the repository's own models.py at a revision
    return module


def model_classes(module: types.ModuleType) -> dict[str, Any]:
    from pydantic import BaseModel

    return {name: value for name, value in vars(module).items() if isinstance(value, type) and issubclass(value, BaseModel) and value is not BaseModel and value.__module__ in (module.__name__, "pydantic.main")}


def core(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {key: core(value) for key, value in schema.items() if key not in IGNORED_KEYS}
    if isinstance(schema, list):
        return [core(item) for item in schema]
    return schema


def accepts(ours: dict, theirs: dict) -> bool:
    """Our field schema accepts what theirs does: equal, or theirs made optional."""

    if ours == theirs:
        return True
    options = ours.get("anyOf")
    if not options or {"type": "null"} not in options:
        return False
    rest = {key: value for key, value in ours.items() if key != "anyOf"}
    wanted = {key: value for key, value in theirs.items() if key != "default"}
    if rest.get("default") != theirs.get("default"):
        return False
    if wanted in options:
        return True
    return bool(wanted.get("anyOf")) and all(option in options for option in wanted["anyOf"])


def compare_models(upstream: types.ModuleType, remi: types.ModuleType) -> list[str]:
    theirs, ours = model_classes(upstream), model_classes(remi)
    problems = []
    for name, model in sorted(theirs.items()):
        if name not in ours:
            problems.append(f"model {name}: missing in Remi")
            continue
        want, got = core(model.model_json_schema()), core(ours[name].model_json_schema())
        want_fields, got_fields = want.get("properties", {}), got.get("properties", {})
        missing = sorted(set(want_fields) - set(got_fields))
        if missing:
            problems.append(f"model {name}: fields {missing} missing in Remi")
        required = sorted(set(got.get("required", [])) - set(want.get("required", [])))
        if required:
            problems.append(f"model {name}: Remi requires {required}, upstream does not")
        for field in sorted(set(want_fields) & set(got_fields)):
            a, b = want_fields[field], got_fields[field]
            if a.get("default") != b.get("default"):
                problems.append(f"model {name}.{field}: default {b.get('default')!r}, upstream {a.get('default')!r}")
            elif name not in ALLOWED and not accepts(b, a):
                problems.append(f"model {name}.{field}: Remi's schema {b} does not accept upstream's {a}")
    return problems


# endregion


def check(rev: str) -> list[str]:
    problems = compare_routes(show(rev, API), read(API))
    sys.argv = sys.argv[:1]  # modules.shared parses the command line
    packages = os.path.join(ROOT, "modules_forge", "packages")
    for path in (packages, ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from remi.testing import webui_state

    with webui_state():
        theirs = build_models(show(rev, MODELS), "sdapi_contract_upstream_models")
        ours = build_models(read(MODELS), "sdapi_contract_remi_models")
        problems += compare_models(theirs, ours)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rev", help=f"upstream revision to compare with (default: merge base with {UPSTREAM}, else {LAST_ABSORBED[:10]})")
    options = parser.parse_args(argv)
    rev = options.rev or default_rev()
    label = f"{rev} ({git('rev-parse', '--short=10', rev).strip()})" if options.rev else rev[:10]
    problems = check(rev)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"/sdapi/v1 differs from upstream {label} in {len(problems)} place(s): restore compatibility, or document the difference in ALLOWED", file=sys.stderr)
        return 1
    print(f"/sdapi/v1 routes and models match upstream {label} ({', '.join(sorted(ALLOWED))}: field types differ as documented)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
