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


def test_batch_hook_propagates_explicit_script_abort():
    source_path = Path(__file__).parents[1] / "modules" / "scripts.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ScriptRunner"
    )
    hook = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "before_process_batch"
    )
    handlers = [
        handler
        for node in ast.walk(hook)
        if isinstance(node, ast.Try)
        for handler in node.handlers
    ]

    abort_handler = next(
        handler
        for handler in handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "ScriptAbort"
    )
    assert any(isinstance(node, ast.Raise) for node in abort_handler.body)


def test_after_model_load_hook_runs_after_checkpoint_reload():
    root = Path(__file__).parents[1]
    scripts_module = ast.parse(
        (root / "modules" / "scripts.py").read_text(encoding="utf-8")
    )
    runner = next(
        node
        for node in scripts_module.body
        if isinstance(node, ast.ClassDef) and node.name == "ScriptRunner"
    )
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "after_model_load"
        for node in runner.body
    )

    processing_module = ast.parse(
        (root / "modules" / "processing.py").read_text(encoding="utf-8")
    )
    process_images = next(
        node
        for node in processing_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "process_images"
    )
    reload_line = next(
        node.lineno
        for node in ast.walk(process_images)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "manage_model_and_prompt_cache"
    )
    hook_line = next(
        node.lineno
        for node in ast.walk(process_images)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "after_model_load"
    )

    assert hook_line > reload_line
