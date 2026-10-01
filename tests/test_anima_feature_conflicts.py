from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import modules
from modules.anima_feature_conflicts import (
    FLOW_SAMPLERS,
    FLOW_SCHEDULER,
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

    def test_apg_is_a_combine_mode_that_excludes_momentum_only(self):
        state = resolve_guidance_conflicts(sampler="Anima Flow Euler", mode="APG", momentum=True, dcw=True)
        self.assertEqual(state.mode, "APG")
        self.assertFalse(state.momentum)
        self.assertFalse(state.momentum_interactive)
        self.assertTrue(state.dcw)
        self.assertEqual(state.messages, ("Momentum disabled: incompatible with APG",))

        state = resolve_guidance_conflicts(sampler="Anima ER SDE CNS", mode="APG", momentum=False, dcw=True)
        self.assertEqual(state.mode, "APG")
        self.assertIn("APG", state.mode_choices)

        state = resolve_guidance_conflicts(
            sampler="Anima Flow Euler", mode="APG", momentum=True, dcw=False, selected="momentum"
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

    def test_dcw_is_disabled_for_multi_evaluation_samplers(self):
        for sampler in ("Heun", "DPM2", "DPM++ SDE", "DPM++ 2s a RF", "Restart", "Kohaku LoNyu Yog", "UniPC", "DDIM"):
            state = resolve_guidance_conflicts(sampler=sampler, mode="Standard / preserve existing", momentum=False, dcw=True)
            self.assertFalse(state.dcw, sampler)
            self.assertFalse(state.dcw_interactive, sampler)
            self.assertEqual(state.messages, (f"DCW disabled: unsupported sampler {sampler}",))

        for sampler in ("Euler", "DPM++ 2M", "ER SDE", "Anima Flow UniPC2", "Anima Flow PC3", "Anima SPEED"):
            state = resolve_guidance_conflicts(sampler=sampler, mode="Standard / preserve existing", momentum=False, dcw=True)
            self.assertTrue(state.dcw, sampler)
            self.assertTrue(state.dcw_interactive, sampler)
            self.assertEqual(state.messages, ())

    def test_dcw_is_disabled_with_an_inpaint_mask(self):
        state = resolve_guidance_conflicts(
            sampler="Euler", mode="Standard / preserve existing", momentum=False, dcw=True, inpaint_mask=True
        )
        self.assertFalse(state.dcw)
        self.assertEqual(state.messages, ("DCW disabled: incompatible with an inpaint mask",))
        # Without DCW the mask changes nothing.
        state = resolve_guidance_conflicts(
            sampler="Euler", mode="Standard / preserve existing", momentum=True, dcw=False, inpaint_mask=True
        )
        self.assertTrue(state.momentum)
        self.assertEqual(state.messages, ())


def _dropdown(*labels):
    """modules.sd_schedulers as far as the resolver reads it: the Schedule
    type dropdown's choices (sd_schedulers needs the WebUI's k_diffusion to import)."""
    return mock.patch.object(modules, "sd_schedulers", SimpleNamespace(schedulers=[SimpleNamespace(label=label) for label in labels]), create=True)


class ScheduleConflictTests(unittest.TestCase):
    def setUp(self):
        patcher = _dropdown("Automatic", "Beta57", FLOW_SCHEDULER)
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_a_hidden_flow_scheduler_shows_as_automatic(self):
        # "Hide schedulers" left it out of the dropdown at startup; get_sigmas still resolves "Automatic" to it
        with _dropdown("Automatic", "Beta57"):
            state = resolve_schedule_conflicts(sampler="Anima Flow PC3", scheduler="Beta57", dynamic_shift=False)

        self.assertEqual(state.scheduler, "Automatic")
        self.assertFalse(state.scheduler_interactive)

    def test_the_flow_samplers_and_their_scheduler_match_the_sampler_table(self):
        root = Path(__file__).resolve().parents[1]
        samplers = (root / "modules" / "sd_samplers_kdiffusion.py").read_text(encoding="utf-8")
        locked = dict(re.findall(r'\("([^"]+)", [^\n]*\{"scheduler": "(\w+)", "lock_scheduler_override": True\}\)', samplers))
        self.assertEqual(set(locked), FLOW_SAMPLERS)
        (name,) = set(locked.values())
        schedulers = (root / "modules" / "sd_schedulers.py").read_text(encoding="utf-8")
        self.assertIn(f'Scheduler("{name}", "{FLOW_SCHEDULER}",', schedulers)

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


def test_sampler_names_match_the_registered_samplers():
    # The resolver stays dependency-free, so its sampler sets are constants;
    # keep them in step with the sampler table (a literal list; the module
    # itself only imports inside the initialised WebUI).
    import ast
    from pathlib import Path

    from modules.anima_feature_conflicts import CNS_SAMPLER, DCW_SAMPLERS, FLOW_SAMPLERS, MOMENTUM_SAMPLERS, SPEED_SAMPLER

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
    assert SPEED_SAMPLER in FLOW_SAMPLERS
    assert CNS_SAMPLER in labels
    assert MOMENTUM_SAMPLERS <= labels
    assert DCW_SAMPLERS <= labels


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


SPATIAL = ("freefuse", "regional", "artist")


class _Checkbox:
    """Records the change handlers wired to it."""

    def __init__(self, name):
        self.name = name
        self.handlers = []

    def change(self, fn, inputs, outputs, **kwargs):
        self.handlers.append((fn, inputs, outputs))


def _tab_build(tab, *members):
    """Register ``members`` (checkboxes) in a fresh Gradio build of ``tab``."""
    import gradio as gr

    from modules.anima_feature_conflicts import register_exclusive_component

    with gr.Blocks():
        gr.Textbox(elem_id=f"{tab}_prompt")
        for member in members:
            register_exclusive_component(tab=tab, group="spatial", names=SPATIAL, name=member.name, component=member)


def _press(member, *values):
    """The updates of ``member``'s live handlers for checkbox ``values``."""
    results = [fn(*values) for fn, _, _ in member.handlers]
    return [[update.get("value") for update in result] for result in results if any(update.get("value") is not None for update in result)]


def test_spatial_group_is_wired_without_a_disabled_member():
    regional, freefuse = _Checkbox("regional"), _Checkbox("freefuse")
    _tab_build("txt2img", regional, freefuse)  # the Artist Mixer is disabled

    (fn, inputs, outputs), = regional.handlers
    assert inputs == outputs == [freefuse, regional]  # the group's order
    assert _press(regional, True, True) == [[False, True]]  # enabling Regional turns FreeFuse off
    assert _press(freefuse, True, True) == [[True, False]]


def test_a_new_tab_build_drops_the_previous_builds_components():
    old = [_Checkbox(name) for name in ("artist", "regional", "freefuse")]
    _tab_build("txt2img", *old)
    regional, freefuse = _Checkbox("regional"), _Checkbox("freefuse")
    _tab_build("txt2img", regional, freefuse)  # Reload UI with the Artist Mixer disabled

    (_, inputs, _), = regional.handlers
    assert inputs == [freefuse, regional] and not any(item in inputs for item in old)
    img2img = _Checkbox("artist")
    _tab_build("img2img", img2img)  # the other tab's group is its own
    assert img2img.handlers == [] and len(regional.handlers) == 1


def test_a_later_member_rewires_the_group_and_the_earlier_wiring_changes_nothing():
    artist, regional, freefuse = (_Checkbox(name) for name in ("artist", "regional", "freefuse"))
    _tab_build("txt2img", artist, regional, freefuse)

    assert artist.handlers[0][1] == [regional, artist]  # wired when Regional arrived
    assert artist.handlers[1][1] == [freefuse, regional, artist]  # ... and again with FreeFuse
    assert len(freefuse.handlers) == 1
    # Only the latest wiring answers: enabling the artist turns both others off
    assert _press(artist, True, False, True) == [[False, False, True]]
