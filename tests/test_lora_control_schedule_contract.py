"""``publish_lora_control_schedule`` keeps a reference to LoRA Control's
``LoRAControl.mapping``, published once on import. That only stays live while
control.py mutates the dict in place (``clear()`` / item assignment) and never
rebinds ``mapping``; a rebinding would leave the Anima LoRA controls reading a
stale, empty dict."""

import ast
from pathlib import Path

from modules import anima_lora_support

_CONTROL = Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_lora_schedule" / "scripts" / "control.py"
_tree = ast.parse(_CONTROL.read_text(encoding="utf-8"))


def _targets(node):
    if isinstance(node, (ast.Assign, ast.Delete)):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        return [node.target]
    return []


def _names(target):
    for node in ast.walk(target):
        if isinstance(node, ast.Attribute) and node.attr == "mapping":
            yield node
        elif isinstance(node, ast.Name) and node.id == "mapping":
            yield node


def test_control_publishes_its_class_mapping_once_at_import():
    calls = [
        node.value
        for node in _tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "publish_lora_control_schedule"
    ]
    assert [ast.unparse(call.args[0]) for call in calls] == ["LoRAControl.mapping"]


def test_control_never_rebinds_mapping():
    lora_control = next(node for node in _tree.body if getattr(node, "name", None) == "LoRAControl")
    # the one binding: the class attribute itself
    definition = next(node for node in lora_control.body if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "mapping")
    rebinds = [
        ast.unparse(node)
        for node in ast.walk(_tree)
        if node is not definition
        and any(
            # ``x.mapping[...] = ...`` mutates; ``x.mapping = ...`` / ``del x.mapping`` rebind
            not isinstance(target, ast.Subscript) and any(True for _ in _names(target))
            for target in _targets(node)
        )
    ]
    assert rebinds == []
    assert not any(isinstance(node, (ast.Global, ast.Nonlocal)) and "mapping" in node.names for node in ast.walk(_tree))


def test_published_mapping_is_read_live(monkeypatch):
    monkeypatch.setattr(anima_lora_support, "_lora_control_schedule", {})
    mapping = {}
    anima_lora_support.publish_lora_control_schedule(mapping)

    mapping["a.safetensors"] = [(1.0, 0.0), (0.0, 1.0)]
    assert anima_lora_support.lora_control_files() == {"a.safetensors"}
    mapping.clear()
    assert anima_lora_support.lora_control_files() == set()
