"""The remi hunk in ``StableDiffusionProcessingTxt2Img.sample`` that switches
the Hires pass's VAE / text encoder choices (``hr_additional_modules``).

modules.processing needs the initialised WebUI, so the hunk (the
``hr_additional_modules = getattr(...)`` line and the ``if`` after it) is run
from its source against stubs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "modules" / "processing.py"


def _load_hunk():
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == "StableDiffusionProcessingTxt2Img")
    body = next(node for node in cls.body if getattr(node, "name", None) == "sample").body
    index = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["hr_additional_modules"]
    )
    hunk = body[index : index + 2]
    assert isinstance(hunk[1], ast.If) and "hr_additional_modules" in ast.unparse(hunk[1].test)
    function = ast.parse("def hunk(self, main_entry):\n    reload = False\n    return reload").body[0]
    function.body[1:1] = hunk
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module([function], [])), str(_SOURCE), "exec"), namespace)  # noqa: S102 - our own source file
    return namespace["hunk"]


hunk = _load_hunk()


class _MainEntry:
    def __init__(self, changed=True):
        self.calls, self.changed = [], changed

    def modules_change(self, modules, **kwargs):
        self.calls.append(modules)
        return self.changed


def test_a_missing_attribute_keeps_the_modules():
    """API payloads / old scripts never set hr_additional_modules (upstream's
    ``getattr(..., []) or []`` then called modules_change(self.hr_additional_modules)
    and raised AttributeError)."""
    main_entry = _MainEntry()

    assert hunk(SimpleNamespace(), main_entry) is False
    assert main_entry.calls == []


@pytest.mark.parametrize("value", [None, ["Use same choices"], ["Use same choices", "vae.safetensors"]])
def test_none_or_use_same_choices_keeps_the_modules(value):
    main_entry = _MainEntry()

    assert hunk(SimpleNamespace(hr_additional_modules=value), main_entry) is False
    assert main_entry.calls == []


@pytest.mark.parametrize("value", [[], ["vae.safetensors"]])
def test_an_explicit_list_changes_the_modules(value):
    """``[]`` is an explicit "no additional modules" choice, not "same as the
    first pass"."""
    main_entry = _MainEntry()

    assert hunk(SimpleNamespace(hr_additional_modules=value), main_entry) is True
    assert main_entry.calls == [value]


def test_an_unchanged_choice_does_not_reload():
    main_entry = _MainEntry(changed=False)

    assert hunk(SimpleNamespace(hr_additional_modules=["vae.safetensors"]), main_entry) is False
