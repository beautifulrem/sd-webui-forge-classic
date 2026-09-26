import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).parents[1] / "scripts" / "anima_regional.py"


def _load():
    import modules

    try:
        from modules import infotext_utils  # noqa: F401
    except ImportError:
        # Its image helpers need the full WebUI install; the hooks only need PasteField
        stub = types.ModuleType("modules.infotext_utils")
        stub.PasteField = tuple
        modules.infotext_utils = sys.modules["modules.infotext_utils"] = stub
    try:
        from modules import sd_models  # noqa: F401
    except ImportError:
        # The model loader needs the full WebUI install; the encode cache key reads model_data only
        modules.sd_models = sys.modules["modules.sd_models"] = types.ModuleType("modules.sd_models")
    spec = importlib.util.spec_from_file_location("anima_regional_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.is_anima_engine = lambda model: True
    return module


regional = _load()
REGION = (True, "a red cat", 0.0, 0.0, 0.5, 1.0, 1.0)


def _process(p, *region):
    regional.AnimaRegionalScript().process(p, True, "0-3", 0.0, 0.65, 0.03, 0.15, *region)


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


def test_block_field_is_read_in_the_2b_layout():
    from modules.anima_lora_support import model_block_set

    assert model_block_set("0-51", 28, feature="x") == set(range(28))
    assert model_block_set("0-51", 52, feature="x") == set(range(52))  # the default keeps every block
    deep = model_block_set("27", 52, feature="x")
    assert max(deep) == 51 and min(deep) > 27  # the last 2B block is the model's deep end
