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

    def test_cfg_zero_star_is_a_combine_mode_that_excludes_momentum_only(self):
        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler",
            mode="CFG-Zero*",
            momentum=True,
            dcw=True,
        )

        self.assertEqual(state.mode, "CFG-Zero*")
        self.assertFalse(state.momentum)
        self.assertTrue(state.dcw)
        self.assertTrue(state.dcw_interactive)
        self.assertEqual(state.messages, ("Momentum disabled: incompatible with CFG-Zero*",))

        # Unlike FDG it stays available with DCW and the CNS sampler.
        state = resolve_guidance_conflicts(sampler="Anima ER SDE CNS", mode="CFG-Zero*", momentum=False, dcw=True)
        self.assertEqual(state.mode, "CFG-Zero*")
        self.assertIn("CFG-Zero*", state.mode_choices)

        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler",
            mode="CFG-Zero*",
            momentum=True,
            dcw=False,
            selected="momentum",
        )
        self.assertEqual(state.mode, "Standard / preserve existing")
        self.assertTrue(state.momentum)

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


def test_sampler_names_match_the_registered_samplers():
    # The resolver stays dependency-free, so its sampler sets are constants;
    # keep them in step with the sampler table (a literal list; the module
    # itself only imports inside the initialised WebUI).
    import ast
    from pathlib import Path

    from modules.anima_feature_conflicts import CNS_SAMPLER, FLOW_SAMPLERS, MOMENTUM_SAMPLERS

    source = (Path(__file__).parents[1] / "modules" / "sd_samplers_kdiffusion.py").read_text(encoding="utf-8")
    table = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "samplers_k_diffusion"
    )
    labels, locked = set(), set()
    for entry in table.elts:
        label, options = entry.elts[0], ast.literal_eval(entry.elts[3])
        if isinstance(label, ast.Constant):
            labels.add(label.value)
            if options.get("lock_scheduler_override"):
                locked.add(label.value)

    assert FLOW_SAMPLERS == locked
    assert CNS_SAMPLER in labels
    assert MOMENTUM_SAMPLERS <= labels


def test_active_features_are_recorded_per_process():
    from types import SimpleNamespace

    from modules.anima_feature_conflicts import ARTIST_MIXER, FREEFUSE, REGIONAL, is_feature_active, set_feature_active

    first, second = SimpleNamespace(), SimpleNamespace()
    set_feature_active(first, FREEFUSE)
    set_feature_active(first, REGIONAL)
    assert is_feature_active(first, FREEFUSE) and not is_feature_active(second, FREEFUSE)
    assert not is_feature_active(first, ARTIST_MIXER)
    set_feature_active(first, FREEFUSE, False)
    assert not is_feature_active(first, FREEFUSE) and is_feature_active(first, REGIONAL)
    assert len({FREEFUSE, REGIONAL, ARTIST_MIXER}) == 3
