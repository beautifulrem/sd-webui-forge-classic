import ast
from pathlib import Path


def test_sampling_callback_has_one_ordered_runner_implementation():
    source_path = Path(__file__).parents[1] / "modules" / "scripts.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ScriptRunner"
    )
    implementations = [
        node
        for node in runner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "process_before_every_sampling"
    ]

    assert len(implementations) == 1
    assert "self.ordered_scripts" in ast.unparse(implementations[0])
