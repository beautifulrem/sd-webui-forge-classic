from __future__ import annotations

import unittest

from modules.anima_presets import (
    PRESET_CONFLICT_CONTROLS,
    PRESET_DESCRIPTIONS,
    PRESETS,
    SAFE_BASE_AESTHETIC_LABEL,
    SAFE_BASE_AESTHETIC_PRESET,
    TURBO_LABEL,
    preset_value,
    preset_controls,
    turbo_settings_advice,
    register_preset_control,
    reset_preset_controls,
)


class SafeBaseAestheticPresetTests(unittest.TestCase):
    def test_native_generation_baseline_matches_recommended_profile(self):
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["native.sampler"], "ER SDE")
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["native.scheduler"], "Automatic")
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["native.steps"], 36)
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["native.cfg"], 4.5)

    def test_every_optional_generation_stack_starts_disabled(self):
        optional_switches = (
            "native.hires",
            "native.refiner",
            "guidance.enabled",
            "dynamic_shift.enabled",
            "spectrum.enabled",
            "regional.enabled",
            "artist.enabled",
            "freefuse.enabled",
            "pid.enabled",
            "hires_guard.enabled",
        )

        self.assertTrue(
            all(SAFE_BASE_AESTHETIC_PRESET[name] is False for name in optional_switches)
        )

    def test_inactive_algorithms_still_receive_reproducible_defaults(self):
        expected = {
            "guidance.mode": "Standard / preserve existing",
            "guidance.nag_scale": 2.0,
            "guidance.momentum_strength": 0.5,
            "guidance.dcw_lambda": -0.015,
            "guidance.pc3_gamma": 1.0,
            "guidance.pc3_tolerance": 0.005,
            "dynamic_shift.start": 5.0,
            "dynamic_shift.end": 1.5,
            "spectrum.policy": "Conservative",
            "hires_guard.preset": "Balanced",
        }

        for name, value in expected.items():
            self.assertEqual(SAFE_BASE_AESTHETIC_PRESET[name], value)

    def test_spatial_defaults_cover_both_28_and_40_block_anima_models(self):
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["regional.blocks"], "0-51")
        self.assertEqual(SAFE_BASE_AESTHETIC_PRESET["freefuse.bias_blocks"], "0-51")

    def test_registry_keeps_component_order_and_resolves_values(self):
        first = object()
        second = object()
        reset_preset_controls("test")
        register_preset_control("test", "native.steps", first)
        register_preset_control("test", "artist.optimization", second, "平衡")

        names, components, values = preset_controls("test")

        self.assertEqual(names, ("native.steps", "artist.optimization"))
        self.assertEqual(components, (first, second))
        self.assertEqual(values, (36, "平衡"))

    def test_registry_rejects_unowned_fields(self):
        reset_preset_controls("test")
        with self.assertRaises(KeyError):
            register_preset_control("test", "prompt.positive", object())

    def test_every_unlockable_conflict_control_has_a_preset_value(self):
        self.assertLessEqual(PRESET_CONFLICT_CONTROLS, SAFE_BASE_AESTHETIC_PRESET.keys())


class PresetMenuTests(unittest.TestCase):
    def test_every_preset_overrides_known_fields_and_is_described(self):
        for label, overrides in PRESETS.items():
            self.assertLessEqual(overrides.keys(), SAFE_BASE_AESTHETIC_PRESET.keys(), label)
            self.assertIn(label, PRESET_DESCRIPTIONS)

    def test_turbo_follows_the_distilled_model_card(self):
        self.assertEqual(preset_value(TURBO_LABEL, "native.cfg", 4.5), 1.0)
        self.assertTrue(8 <= preset_value(TURBO_LABEL, "native.steps", 36) <= 12)
        # Everything else comes from the (possibly localized) base value.
        self.assertEqual(preset_value(TURBO_LABEL, "artist.optimization", "平衡"), "平衡")
        self.assertEqual(preset_value(SAFE_BASE_AESTHETIC_LABEL, "native.cfg", 4.5), 4.5)


class TurboAdviceTests(unittest.TestCase):
    def test_turbo_with_base_settings_is_flagged(self):
        self.assertIn("Turbo", turbo_settings_advice("anima-turbo-v1.1.safetensors", 4.5, 36))
        self.assertIsNone(turbo_settings_advice("anima-turbo-v1.1.safetensors", 1.0, 10))

    def test_base_with_turbo_settings_is_flagged(self):
        self.assertIn("Non-Turbo", turbo_settings_advice("anima-base-v1.0", 1.0, 10))
        self.assertIsNone(turbo_settings_advice("anima-aesthetic-v1.1", 4.5, 36))
        self.assertIsNone(turbo_settings_advice("my-merge", 1.0, 10))  # unknown: no guess


if __name__ == "__main__":
    unittest.main()
