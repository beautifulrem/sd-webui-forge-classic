from __future__ import annotations

import unittest

from modules.anima_feature_conflicts import (
    record_conflict_resolution,
    resolve_exclusive_features,
    resolve_freefuse_refiner_conflicts,
    resolve_guidance_conflicts,
    resolve_schedule_conflicts,
)


class ConflictMetadataTests(unittest.TestCase):
    def test_processing_metadata_accumulates_unique_resolutions(self):
        process = type("Process", (), {"extra_generation_params": {}})()

        record_conflict_resolution(process, "Refiner cleared")
        record_conflict_resolution(process, "Momentum disabled")
        record_conflict_resolution(process, "Refiner cleared")

        self.assertEqual(
            process.extra_generation_params["Anima conflict resolution"],
            "Refiner cleared; Momentum disabled",
        )


class ExclusiveFeatureConflictTests(unittest.TestCase):
    def test_last_selected_feature_wins_and_locks_its_peers(self):
        state = resolve_exclusive_features(
            ("freefuse", "regional", "artist"),
            (True, True, False),
            selected="regional",
        )

        self.assertEqual(state.enabled, (False, True, False))
        self.assertEqual(state.interactive, (False, True, False))
        self.assertEqual(state.winner, "regional")

    def test_turning_off_winner_unlocks_every_feature(self):
        state = resolve_exclusive_features(
            ("freefuse", "regional", "artist"),
            (False, False, False),
            selected="freefuse",
        )

        self.assertEqual(state.enabled, (False, False, False))
        self.assertEqual(state.interactive, (True, True, True))
        self.assertIsNone(state.winner)

    def test_headless_processing_uses_declared_priority(self):
        state = resolve_exclusive_features(
            ("freefuse", "regional", "artist"),
            (True, True, True),
        )

        self.assertEqual(state.enabled, (True, False, False))
        self.assertEqual(state.winner, "freefuse")


class GuidanceConflictTests(unittest.TestCase):
    def test_selecting_fdg_turns_off_and_locks_momentum_and_dcw(self):
        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler",
            mode="FDG (experimental)",
            momentum=True,
            dcw=True,
            selected="mode",
        )

        self.assertEqual(state.mode, "FDG (experimental)")
        self.assertFalse(state.momentum)
        self.assertFalse(state.dcw)
        self.assertFalse(state.momentum_interactive)
        self.assertFalse(state.dcw_interactive)

    def test_selecting_momentum_reverts_conflicting_mode_to_standard(self):
        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler",
            mode="SMC-CFG",
            momentum=True,
            dcw=False,
            selected="momentum",
        )

        self.assertEqual(state.mode, "Standard / preserve existing")
        self.assertTrue(state.momentum)
        self.assertEqual(state.mode_choices, ("Standard / preserve existing",))

    def test_cns_sampler_removes_fdg_and_disables_momentum(self):
        state = resolve_guidance_conflicts(
            sampler="Anima ER SDE CNS",
            mode="FDG (experimental)",
            momentum=True,
            dcw=False,
            selected="sampler",
        )

        self.assertEqual(state.mode, "Standard / preserve existing")
        self.assertNotIn("FDG (experimental)", state.mode_choices)
        self.assertFalse(state.momentum)
        self.assertFalse(state.momentum_interactive)

    def test_cli_runtime_keeps_cfg_mode_and_drops_incompatible_switches(self):
        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler",
            mode="SMC-CFG",
            momentum=True,
            dcw=True,
        )

        self.assertEqual(state.mode, "SMC-CFG")
        self.assertFalse(state.momentum)
        self.assertTrue(state.dcw)


class ScheduleConflictTests(unittest.TestCase):
    def test_flow_sampler_locks_anima_scheduler_and_dynamic_shift(self):
        state = resolve_schedule_conflicts(
            sampler="Anima Flow PC3",
            scheduler="Beta57",
            dynamic_shift=True,
            selected="sampler",
        )

        self.assertEqual(state.scheduler, "Anima FlowMatch")
        self.assertFalse(state.dynamic_shift)
        self.assertFalse(state.scheduler_interactive)
        self.assertFalse(state.dynamic_shift_interactive)

    def test_selecting_dynamic_shift_replaces_beta57_and_locks_scheduler(self):
        state = resolve_schedule_conflicts(
            sampler="Euler",
            scheduler="Beta57",
            dynamic_shift=True,
            selected="dynamic_shift",
        )

        self.assertEqual(state.scheduler, "Automatic")
        self.assertTrue(state.dynamic_shift)
        self.assertFalse(state.scheduler_interactive)

    def test_dynamic_shift_automatic_update_does_not_turn_itself_off(self):
        state = resolve_schedule_conflicts(
            sampler="Euler",
            scheduler="Automatic",
            dynamic_shift=True,
            selected="scheduler",
        )

        self.assertTrue(state.dynamic_shift)
        self.assertFalse(state.scheduler_interactive)


class FreeFuseRefinerConflictTests(unittest.TestCase):
    def test_selecting_freefuse_clears_and_locks_refiner(self):
        state = resolve_freefuse_refiner_conflicts(
            freefuse=True,
            refiner=True,
            checkpoint="refiner.safetensors",
            selected="freefuse",
        )

        self.assertTrue(state.freefuse)
        self.assertFalse(state.refiner)
        self.assertEqual(state.checkpoint, "None")
        self.assertFalse(state.refiner_interactive)

    def test_selecting_configured_refiner_turns_off_and_locks_freefuse(self):
        state = resolve_freefuse_refiner_conflicts(
            freefuse=True,
            refiner=True,
            checkpoint="refiner.safetensors",
            selected="refiner",
        )

        self.assertFalse(state.freefuse)
        self.assertTrue(state.refiner)
        self.assertFalse(state.freefuse_interactive)

    def test_disabling_refiner_unlocks_freefuse(self):
        state = resolve_freefuse_refiner_conflicts(
            freefuse=False,
            refiner=False,
            checkpoint="refiner.safetensors",
            selected="refiner",
        )

        self.assertTrue(state.freefuse_interactive)
        self.assertTrue(state.refiner_interactive)

    def test_headless_processing_gives_explicit_freefuse_priority(self):
        state = resolve_freefuse_refiner_conflicts(
            freefuse=True,
            refiner=True,
            checkpoint="refiner.safetensors",
        )

        self.assertTrue(state.freefuse)
        self.assertFalse(state.refiner)


if __name__ == "__main__":
    unittest.main()
