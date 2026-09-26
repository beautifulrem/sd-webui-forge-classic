"""Settings > Callback order lists ``ScriptRunner.callback_names``: every hook
ScriptRunner runs through ``ordered_scripts`` must be there to be reorderable."""

import ast
from pathlib import Path

from modules import scripts


def _ordered_hooks():
    tree = ast.parse((Path(__file__).parents[1] / "modules" / "scripts.py").read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScriptRunner")
    return {
        node.args[0].value
        for node in ast.walk(runner)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "ordered_scripts" and node.args and isinstance(node.args[0], ast.Constant)
    }


def test_every_ordered_hook_can_be_reordered_in_the_settings():
    hooks = _ordered_hooks()
    assert {"after_model_load", "cleanup", "process_before_every_sampling"} <= hooks
    names = scripts.ScriptRunner().callback_names
    assert hooks <= set(names), sorted(hooks - set(names))
    assert len(names) == len(set(names))
