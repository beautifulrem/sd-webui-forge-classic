import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_regional.py"
# Shared with the other suites of this folder, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("remi_panel_harness", ROOT.parents[1] / "tests" / "remi_panel_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def _load():
    import modules

    try:
        from modules import sd_models  # noqa: F401
    except ImportError:
        # The model loader needs the full WebUI install; the encode cache key reads model_data only
        modules.sd_models = sys.modules["modules.sd_models"] = types.ModuleType("modules.sd_models")
    module = harness.load_script(SCRIPT, "anima_regional_under_test")
    module.is_anima_engine = lambda model: True
    return module


regional = _load()
REGION = (True, "a red cat", 0.0, 0.0, 0.5, 1.0, 1.0)
UNUSED_REGION = (False, "", 0.0, 0.0, 0.5, 1.0, 1.0)


def _process(p, *region):
    """process() with ``region`` as region 1 and the other regions unused."""
    regional.AnimaRegionalScript().process(p, True, "0-3", 0.0, 0.65, 0.03, 0.15, *region, *UNUSED_REGION, *UNUSED_REGION)


@pytest.mark.parametrize("artist_mixer_first", [True, False])
def test_regional_marks_itself_active_and_takes_priority_over_artist_mixer(artist_mixer_first):
    from modules.anima_feature_conflicts import ARTIST_MIXER, REGIONAL, is_feature_active
    from modules.anima_support import negpip_extra_prompts, register_negpip_prompts

    p = SimpleNamespace(sd_model=object())
    if artist_mixer_first:
        register_negpip_prompts(p, ARTIST_MIXER, ["artist"])
    _process(p, *REGION)
    if not artist_mixer_first:  # the callback order was re-sorted
        register_negpip_prompts(p, ARTIST_MIXER, ["artist"])

    assert is_feature_active(p, REGIONAL)
    # NegPiP only looks at the prompts that will be encoded
    assert negpip_extra_prompts(p) == ["a red cat"]
    regional.AnimaRegionalScript().postprocess(p, None)
    assert not is_feature_active(p, REGIONAL)


def test_regional_stays_inactive_without_a_used_region_or_under_freefuse():
    from modules.anima_feature_conflicts import FREEFUSE, REGIONAL, is_feature_active, set_feature_active

    p = SimpleNamespace(sd_model=object())
    _process(p, False, *REGION[1:])
    assert not is_feature_active(p, REGIONAL)

    set_feature_active(p, FREEFUSE)
    _process(p, *REGION)
    assert not is_feature_active(p, REGIONAL)


def test_region_prompts_are_encoded_once_until_the_text_encoder_changes():
    import torch

    calls = []

    class Model:
        current_lora_hash = "[]"

        def __init__(self):
            self.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=SimpleNamespace(patches_uuid="te-1")))

        def get_learned_conditioning(self, prompt):
            calls.append(list(prompt))
            return [torch.full((1, 3, 2), float(len(calls)))]

    p = SimpleNamespace(sd_model=Model(), width=64, height=64)
    first, _ = regional._encode_cached(p, "a red cat")
    again, _ = regional._encode_cached(p, "a red cat")  # next batch / hires pass
    assert again is first and calls == [["a red cat"]]

    regional._encode_cached(p, "a blue dog")
    assert len(calls) == 2

    # Qwen layer weights re-apply the TE LoRA patches without changing its hash
    p.sd_model.forge_objects.clip.patcher.patches_uuid = "te-2"
    assert regional._encode_cached(p, "a red cat")[0] is not first
    p.sd_model.current_lora_hash = "['other.safetensors']"
    regional._encode_cached(p, "a red cat")
    # NegPiP's conditioning hook changes the output format
    p.sd_model.get_learned_conditioning = p.sd_model.get_learned_conditioning
    regional._encode_cached(p, "a red cat")
    assert len(calls) == 5


def _recording_process(steps=10):
    import torch

    calls = []

    class Model:
        current_lora_hash = "[]"
        forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=SimpleNamespace(patches_uuid="te")))

        def get_learned_conditioning(self, prompt):
            calls.append(list(prompt))
            return [torch.full((1, 3, 2), float(len(calls)))]

    # firstpass_steps: the base pass's total steps, as setup_conds sets them
    return SimpleNamespace(sd_model=Model(), width=64, height=64, firstpass_steps=steps), calls


@pytest.mark.parametrize(
    "prompt, encoded, entry_texts, switched",
    [
        # Forge's schedule over 10 steps: "cat" to step 5, "dog" after it
        ("a [cat:dog:5]", ["a cat", "a dog"], ["a cat", "a dog"], 6),
        # Alternation switches every step: each distinct text is encoded once
        ("[red|blue] hair", ["red hair", "blue hair"], ["red hair", "blue hair"] * 5, 2),
    ],
)
def test_region_prompt_editing_is_resolved_like_the_main_prompt(prompt, encoded, entry_texts, switched):
    p, calls = _recording_process()
    region = regional._region(p, prompt, 0.0, 0.0, 0.5, 1.0, 1.0)

    assert calls == [[text] for text in encoded]
    by_text = {text: float(index + 1) for index, text in enumerate(encoded)}
    assert [float(cond.flatten()[0]) for cond, _mask in region.entries] == [by_text[text] for text in entry_texts]
    assert region.at(0) is region.entries[0] and region.conditioning is region.entries[0][0]
    assert region.at(switched) is region.entries[1]  # the first step of the second text


def test_a_plain_region_prompt_is_encoded_as_written():
    p, calls = _recording_process()
    region = regional._region(p, "a red (cat:1.2)", 0.0, 0.0, 0.5, 1.0, 1.0)
    assert calls == [["a red (cat:1.2)"]]
    assert region.ends == () and region.entries == ()


def test_block_field_is_read_in_the_2b_layout():
    from modules.anima_lora_support import model_block_set

    for total in (28, 40, 52):
        assert model_block_set("0-27", total, feature="x") == set(range(total))  # the default keeps every block
        assert model_block_set("0-51", total, feature="x") == set(range(total))  # as does the old default's infotext
    deep = model_block_set("27", 52, feature="x")
    assert max(deep) == 51 and min(deep) > 27  # the last 2B block is the model's deep end


def test_block_field_defaults_to_the_2b_range():
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    defaults = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "Text"
        and any(keyword.arg == "label" and getattr(keyword.value, "value", None) == "Anima blocks" for keyword in node.keywords)
    ]
    assert harness.script_layout(regional.AnimaRegionalScript()).defaults()["blocks"] == "0-27"
    assert defaults == ["0-27"]
