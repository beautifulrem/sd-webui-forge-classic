import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    spec = importlib.util.spec_from_file_location("anima_regional_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.is_anima_engine = lambda model: True
    return module


regional = _load()
REGION = (True, "a red cat", 0.0, 0.0, 0.5, 1.0, 1.0)


def _process(p, *region):
    regional.AnimaRegionalScript().process(p, True, "0-3", 0.0, 0.65, 0.03, 0.15, *region)


def test_regional_marks_itself_active_and_takes_priority_over_artist_mixer():
    from modules.anima_feature_conflicts import REGIONAL, is_feature_active
    from modules.anima_support import NEGPIP_EXTRA_PROMPTS, register_negpip_prompts

    p = SimpleNamespace(sd_model=object())
    register_negpip_prompts(p, "anima_artist_mixer", ["artist"])  # Artist Mixer's process() runs first
    _process(p, *REGION)

    assert is_feature_active(p, REGIONAL)
    assert getattr(p, NEGPIP_EXTRA_PROMPTS) == {"anima_artist_mixer": [], "anima_regional": ["a red cat"]}
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
