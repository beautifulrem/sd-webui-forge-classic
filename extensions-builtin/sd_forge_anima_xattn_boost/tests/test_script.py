import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Shared with the DAVE suite, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", ROOT.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

boost = harness.load_script(ROOT / "scripts" / "anima_xattn_boost.py", "anima_xattn_boost_under_test")


def test_written_infotext_restores_the_panel_controls():
    harness.assert_infotext_restores_controls(
        boost.AnimaXattnBoostScript(),
        (True, 2.0, 0.85, 0.5),
        {"Anima xattn boost enabled": True, "Anima xattn boost strength": 2.0, "Anima xattn boost band": 0.85, "Anima xattn boost renorm": 0.5},
    )


def test_apply_preset_turns_the_panel_off_at_its_defaults():
    harness.assert_registers_preset(boost.AnimaXattnBoostScript(), "xattn_boost")


def test_process_patches_the_pass_and_writes_infotext():
    p, unet = harness.make_process()
    boost.AnimaXattnBoostScript().process_before_every_sampling(p, True, 2.0, 0.85, 0.5)
    patched = p.sd_model.forge_objects.unet
    assert patched is not unet and "model_function_wrapper" in patched.model_options
    assert "model_function_wrapper" not in unet.model_options
    assert p.extra_generation_params == boost.generation_params(2.0, 0.85, 0.5)


def test_disabled_unit_strength_and_non_anima_are_no_ops():
    for enable, strength, anima in ((False, 2.0, True), (True, 1.0, True), (True, 2.0, False)):
        p, unet = harness.make_process()
        p.sd_model.is_anima_engine = anima
        boost.AnimaXattnBoostScript().process_before_every_sampling(p, enable, strength, 0.85, 0.5)
        assert p.sd_model.forge_objects.unet is unet
        assert p.extra_generation_params == {}
