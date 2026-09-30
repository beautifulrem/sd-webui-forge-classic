from __future__ import annotations

import unittest
from pathlib import Path

from modules.anima_feature_conflicts import (
    ARTIST_MIXER,
    FREEFUSE,
    GUIDANCE_MODES,
    REGIONAL,
    resolve_exclusive_features,
    resolve_freefuse_refiner_conflicts,
    resolve_guidance_conflicts,
    resolve_schedule_conflicts,
)
from modules.anima_lora_support import ANIMA_BLOCK_MAPPINGS, model_block_set
from modules.anima_presets import (
    PRESET_CONFLICT_CONTROLS,
    PRESET_DESCRIPTIONS,
    PRESETS,
    SAFE_BASE_AESTHETIC_LABEL,
    SAFE_BASE_AESTHETIC_PRESET,
    TURBO_LABEL,
    TURBO_NAG_LABEL,
    preset_value,
    preset_controls,
    turbo_settings_advice,
    register_preset_control,
    reset_preset_controls,
)
from modules.anima_support import parse_block_list

BUILTIN = Path(__file__).resolve().parents[1] / "extensions-builtin"
# Block bands written in the 28-block (2B) layout, not whole-model selections
REFERENCE_LAYOUT_BANDS = frozenset({"dave.blocks"})


def _resolved(label: str) -> dict[str, object]:
    """Every control's value after applying ``label``."""
    return {name: preset_value(label, name, value) for name, value in SAFE_BASE_AESTHETIC_PRESET.items()}


class PresetConsistencyTests(unittest.TestCase):
    def test_every_preset_is_already_conflict_free(self):
        # The preset panel unlocks the conflict controls after applying a
        # preset; the processing-time resolvers must then leave it unchanged.
        for label in PRESETS:
            values = _resolved(label)
            sampler = values["native.sampler"]
            with self.subTest(label):
                guidance = resolve_guidance_conflicts(
                    sampler=sampler,
                    mode=values["guidance.mode"],
                    momentum=values["guidance.momentum_enabled"],
                    dcw=values["guidance.dcw_enabled"],
                )
                self.assertIn(values["guidance.mode"], GUIDANCE_MODES)
                self.assertEqual(
                    (guidance.mode, guidance.momentum, guidance.dcw, guidance.messages),
                    (values["guidance.mode"], values["guidance.momentum_enabled"], values["guidance.dcw_enabled"], ()),
                )

                schedule = resolve_schedule_conflicts(
                    sampler=sampler,
                    scheduler=values["native.scheduler"],
                    dynamic_shift=values["dynamic_shift.enabled"],
                )
                self.assertEqual(
                    (schedule.scheduler, schedule.dynamic_shift, schedule.messages),
                    (values["native.scheduler"], values["dynamic_shift.enabled"], ()),
                )

                spatial = (values["freefuse.enabled"], values["regional.enabled"], values["artist.enabled"])
                self.assertEqual(resolve_exclusive_features((FREEFUSE, REGIONAL, ARTIST_MIXER), spatial).enabled, spatial)

                refiner = resolve_freefuse_refiner_conflicts(
                    freefuse=values["freefuse.enabled"], refiner=values["native.refiner"], checkpoint=None
                )
                self.assertEqual((refiner.freefuse, refiner.refiner, refiner.messages), (values["freefuse.enabled"], values["native.refiner"], ()))

    def test_cfg_one_presets_enable_nothing_that_needs_the_unconditional_pass(self):
        for label in PRESETS:
            values = _resolved(label)
            if values["native.cfg"] > 1.0:
                continue
            with self.subTest(label):
                self.assertEqual(values["guidance.mode"], GUIDANCE_MODES[0])
                self.assertFalse(values["guidance.momentum_enabled"] or values["guidance.dcw_enabled"])

    def test_each_preset_matches_its_own_checkpoint_advice(self):
        for label in PRESETS:
            values = _resolved(label)
            checkpoint = "anima-turbo-v1.1" if "Turbo" in label else "anima-base-v1.0"
            with self.subTest(label):
                self.assertIsNone(turbo_settings_advice(checkpoint, values["native.cfg"], values["native.steps"]))

    def test_overrides_keep_each_control_value_type(self):
        for label, overrides in PRESETS.items():
            for name, value in overrides.items():
                with self.subTest(label=label, control=name):
                    self.assertIs(type(value), type(SAFE_BASE_AESTHETIC_PRESET[name]))

    def test_windows_and_boxes_are_well_formed(self):
        for label in PRESETS:
            values = _resolved(label)
            with self.subTest(label):
                self.assertGreater(values["native.steps"], 0)
                self.assertGreater(values["native.cfg"], 0.0)
                # Fractions of the schedule
                for start, end in (
                    ("guidance.skim_start", "guidance.skim_end"),
                    ("guidance.nag_start", "guidance.nag_end"),
                    ("guidance.range_start", "guidance.range_end"),
                    ("regional.start", "regional.end"),
                ):
                    self.assertTrue(0.0 <= values[start] <= values[end] <= 1.0, start)
                for region in (1, 2, 3):
                    x, y = values[f"regional.region{region}_x"], values[f"regional.region{region}_y"]
                    width, height = values[f"regional.region{region}_width"], values[f"regional.region{region}_height"]
                    self.assertTrue(0.0 <= x and 0.0 <= y and x + width <= 1.0 and y + height <= 1.0, region)
                self.assertLessEqual(values["hires_guard.base_mp"], values["hires_guard.hires_mp"])
                self.assertGreaterEqual(values["hires_guard.max_upscale"], 1.0)
                # The DAVE and Cross-Attention Boost panels' slider ranges
                self.assertTrue(0.0 <= values["dave.strength"] <= 1.0)
                self.assertTrue(0.01 <= values["dave.tau"] <= 1.0)
                self.assertTrue(1.0 <= values["xattn_boost.strength"] <= 3.0)
                self.assertTrue(0.5 <= values["xattn_boost.band"] <= 1.0)
                self.assertTrue(0.0 <= values["xattn_boost.renorm"] <= 1.0)

    def test_safe_preset_starts_every_optional_stack_disabled(self):
        switches = [name for name in SAFE_BASE_AESTHETIC_PRESET if name.endswith(("enabled", "native.hires", "native.refiner"))]
        for switch in ("hires_guard.enabled", "dave.enabled", "xattn_boost.enabled"):
            self.assertIn(switch, switches)
        self.assertEqual([name for name in switches if SAFE_BASE_AESTHETIC_PRESET[name] is not False], [])

    def test_block_selections_cover_every_anima_depth(self):
        # Read the way the panels read them at runtime: in the 2B layout
        depths = {size for pair in ANIMA_BLOCK_MAPPINGS for size in pair}
        self.assertEqual(depths, {28, 40, 52})
        for name, value in SAFE_BASE_AESTHETIC_PRESET.items():
            if name.endswith("blocks") and name not in REFERENCE_LAYOUT_BANDS:
                self.assertEqual(parse_block_list(value, 28, feature=name), set(range(28)), name)
                for depth in depths:
                    self.assertEqual(model_block_set(value, depth, feature=name), set(range(depth)), (name, depth))

    def test_full_block_selections_use_the_2b_layout_panel_default(self):
        # The same text the Regional and FreeFuse panels start with
        for name in ("regional.blocks", "freefuse.bias_blocks"):
            self.assertEqual(SAFE_BASE_AESTHETIC_PRESET[name], "0-27", name)

    def test_reference_layout_bands_are_proper_2b_bands(self):
        # Written in the 28-block (2B) layout and mapped onto larger models
        for name in REFERENCE_LAYOUT_BANDS:
            selected = parse_block_list(SAFE_BASE_AESTHETIC_PRESET[name], 28, feature=name)
            self.assertTrue(selected and selected < set(range(28)), name)

    def test_every_preset_feature_is_registered_by_a_panel(self):
        # A feature whose panel never registers its controls is silently left
        # as the user set it by "Apply preset". Per-control registration is
        # checked behaviourally by the extension suites (e.g. DAVE's
        # test_script.py); this catches a whole panel that registers nothing.
        registering = [
            source
            for source in (path.read_text(encoding="utf-8") for path in BUILTIN.glob("*/scripts/*.py"))
            # Direct calls, or an AnimaPatchPanel's PRESET_FIELDS
            if "register_preset_control(" in source or "PRESET_FIELDS = (" in source
        ]
        for feature in sorted({name.split(".", 1)[0] for name in SAFE_BASE_AESTHETIC_PRESET}):
            with self.subTest(feature):
                self.assertTrue(any(f'"{feature}.' in source for source in registering))


class PresetRegistryTests(unittest.TestCase):
    def test_registry_keeps_component_order_and_resolves_values(self):
        first = object()
        second = object()
        reset_preset_controls("test")
        register_preset_control("test", "native.steps", first)
        register_preset_control("test", "artist.optimization", second, "平衡")

        names, components, values = preset_controls("test")

        self.assertEqual(names, ("native.steps", "artist.optimization"))
        self.assertEqual(components, (first, second))
        self.assertEqual(values, (SAFE_BASE_AESTHETIC_PRESET["native.steps"], "平衡"))

    def test_registry_rejects_unowned_fields(self):
        reset_preset_controls("test")
        with self.assertRaises(KeyError):
            register_preset_control("test", "prompt.positive", object())

    def test_every_unlockable_conflict_control_has_a_preset_value(self):
        self.assertLessEqual(PRESET_CONFLICT_CONTROLS, SAFE_BASE_AESTHETIC_PRESET.keys())

    def test_every_preset_overrides_known_fields_and_is_described(self):
        for label, overrides in PRESETS.items():
            self.assertLessEqual(overrides.keys(), SAFE_BASE_AESTHETIC_PRESET.keys(), label)
            self.assertIn(label, PRESET_DESCRIPTIONS)

    def test_fields_a_preset_does_not_override_keep_the_registered_value(self):
        # e.g. a localized choice registered by the extension
        self.assertEqual(preset_value(TURBO_LABEL, "artist.optimization", "平衡"), "平衡")
        self.assertEqual(preset_value(SAFE_BASE_AESTHETIC_LABEL, "native.cfg", 4.5), 4.5)
        self.assertNotEqual(preset_value(TURBO_LABEL, "native.cfg", 4.5), 4.5)

    def test_turbo_nag_is_turbo_plus_nag(self):
        turbo, nag = _resolved(TURBO_LABEL), _resolved(TURBO_NAG_LABEL)
        changed = {name for name in turbo if turbo[name] != nag[name]}
        self.assertEqual(changed, {"guidance.enabled", "guidance.nag_enabled"})
        self.assertTrue(nag["guidance.enabled"] and nag["guidance.nag_enabled"])


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
