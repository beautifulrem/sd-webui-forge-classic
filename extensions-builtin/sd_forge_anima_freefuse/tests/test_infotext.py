"""A FreeFuse run's infotext replays into the panel it was run from, through
the WebUI's paste and through the API's infotext replay."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the other panels' suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

PROMPT = "alice and bob and carol in a forest at dusk"
# Every setting after the subjects away from its default
SETTINGS = [5, 20, 0.2, 500, "forest at dusk", 1.2, 20, 2, 0.8, 0.9, 8.0, 2.0, "0-9, 18-27"]
ALICE, BOB, CAROL = ("alice_lora", "alice"), ("bob_lora", "bob"), ("carol_lora", "carol")


class _Engine:
    """Character tokens, one chunk with the appended EOS (as AnimaTextProcessingEngine)."""

    t5_tokenizer = staticmethod(lambda texts, **kwargs: {"input_ids": [[ord(character) for character in text] for text in texts]})

    @staticmethod
    def tokenize_line(prompt):
        return [SimpleNamespace(t5_tokens=[ord(character) for character in prompt] + [1])]


@pytest.fixture
def freefuse(monkeypatch):
    # Another suite of this folder may leave a PasteField stub behind
    stub = types.ModuleType("modules.infotext_utils")
    stub.PasteField = harness.forge_infotext().PasteField
    monkeypatch.setitem(sys.modules, "modules.infotext_utils", stub)
    module = harness.load_script(ROOT / "scripts" / "anima_freefuse.py", "anima_freefuse_infotext_under_test")
    monkeypatch.setattr(module, "apply_freefuse_patch", lambda unet, state: unet)
    return module


def _recorder(freefuse):
    def record(args):
        """The infotext a FreeFuse pass with the script args ``args`` records
        (process() marks FreeFuse active when it is enabled)."""
        p, _ = harness.make_process()
        p.sd_model.text_processing_engine_anima = _Engine()
        p.prompts, p.negative_prompts, p.firstpass_steps = [PROMPT], [""], 10
        if args[0]:
            freefuse.set_feature_active(p, freefuse.FREEFUSE)
        freefuse.AnimaFreeFuseScript().process_before_every_sampling(p, *args)
        return p.extra_generation_params

    return record


@pytest.mark.parametrize(
    "subjects, off",
    [
        # Every control away from its default: subject 1 off, subject 3 on
        ([(False, *ALICE), (True, *BOB), (True, *CAROL)], 1),
        # The default layout: subject 3 off, and no background phrase
        ([(True, *ALICE), (True, *BOB), (False, *CAROL)], 3),
    ],
    ids=["subject-1-off", "subject-3-off"],
)
def test_the_infotext_replays_into_the_same_script_args(freefuse, subjects, off):
    script = freefuse.AnimaFreeFuseScript()
    controls = harness.build_ui(script)
    defaults = [control.value for control in controls]
    settings = SETTINGS if off == 1 else [*SETTINGS[:4], "", *SETTINGS[5:]]
    args = [True, *(value for subject in subjects for value in subject), *settings]
    # An inactive subject records only its switch: the replay leaves its texts at their defaults
    expected = list(args)
    expected[3 * off - 1 : 3 * off + 1] = defaults[3 * off - 1 : 3 * off + 1]
    assert all(value != default for value, default in zip(SETTINGS, defaults[10:]))

    params = harness.assert_infotext_round_trip(script, controls, args, _recorder(freefuse), expected)
    assert params["Anima FreeFuse"] == "2 subjects, two-pass same-noise"
    # The recorded "False" of the inactive subject replays as False, not bool("False")
    assert params[f"Anima FreeFuse subject {off} active"] is False
    assert f"Anima FreeFuse subject {off} LoRA" not in params
    assert ("Anima FreeFuse background" in params) is (off == 1)


def test_an_off_panel_records_nothing_and_replays_its_defaults(freefuse):
    script = freefuse.AnimaFreeFuseScript()
    controls = harness.build_ui(script)
    args = [False, True, *ALICE, True, *BOB, True, *CAROL, *SETTINGS]
    harness.assert_infotext_round_trip(script, controls, args, _recorder(freefuse), [control.value for control in controls])
