import ast
import fnmatch
import re
from pathlib import Path

import pytest

# The script imports gradio/WebUI modules at import time; load only the pure
# prompt-cleaning helpers.
SCRIPT = Path(__file__).parents[1] / "scripts" / "prompt_blacklist.py"
_HELPERS = {"_strip_emphasis", "_normalize", "_parse_blacklist", "_is_blacklisted", "_bracket_residue", "clean_prompt"}
_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
_nodes = [
    node
    for node in _tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in _HELPERS)
    or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets))
]
_namespace = {"re": re, "fnmatch": fnmatch}
exec(compile(ast.Module(_nodes, []), str(SCRIPT), "exec"), _namespace)
clean_prompt = _namespace["clean_prompt"]


@pytest.mark.parametrize(
    ("prompt", "blacklist", "expected"),
    [
        ("(red hair, blue eyes:1.2), 1girl", "red hair", "(blue eyes:1.2), 1girl"),
        ("(red hair, blue eyes:1.2), 1girl", "blue eyes", "(red hair:1.2), 1girl"),
        ("(red hair, blue eyes:1.2), 1girl", "red hair, blue eyes", "1girl"),
        ("((a, (x, y))), z", "x, y", "((a)), z"),
        ("(x, (a, b:1.2))", "a, b", "(x)"),
        ("(x, (a, b:1.2):1.1)", "a, b", "(x:1.1)"),
        ("(x, (a, b:1.2):1.1)", "b", "(x, (a:1.2):1.1)"),
        ("a, \\(b\\), c", "b", "a, c"),
        ("1girl, (watermark:-1.0), solo", "watermark", "1girl, solo"),
        ("x, BREAK, y", "x", "BREAK, y"),
    ],
)
def test_removal_keeps_emphasis_groups_balanced(prompt, blacklist, expected):
    assert clean_prompt(prompt, blacklist, False)[0] == expected


def test_colon_tags_are_not_weights():
    assert clean_prompt("1girl, :3, smile", ":3", False)[0] == "1girl, smile"
    assert clean_prompt("aspect ratio 16:9, aspect ratio 4:3", "", True)[0] == "aspect ratio 16:9, aspect ratio 4:3"
    assert (
        clean_prompt("1girl, (aspect ratio 16:9:1.1), (aspect ratio 16:10:1.1)", "", True)[0]
        == "1girl, (aspect ratio 16:9:1.1), (aspect ratio 16:10:1.1)"
    )
    assert clean_prompt("1girl, (:3:1.2), smile", ":3", False)[0] == "1girl, smile"


def test_duplicates_are_removed():
    assert clean_prompt("a, (a:1.2), b", "", True)[0] == "a, b"
