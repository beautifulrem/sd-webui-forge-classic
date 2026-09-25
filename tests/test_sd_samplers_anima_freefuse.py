from types import SimpleNamespace

from modules.anima_guidance_state import (
    clear_stateful_guidance,
    register_stateful_guidance,
    reset_stateful_guidance,
)


class _State:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_probe_reset_resets_every_registered_state():
    process = SimpleNamespace()
    momentum, smc = _State(), _State()
    assert register_stateful_guidance(process, momentum) is momentum
    register_stateful_guidance(process, smc)

    reset_stateful_guidance(SimpleNamespace(p=process))

    assert (momentum.reset_calls, smc.reset_calls) == (1, 1)


def test_a_new_pass_starts_without_the_previous_states():
    process = SimpleNamespace()
    stale = register_stateful_guidance(process, _State())
    clear_stateful_guidance(process)

    reset_stateful_guidance(SimpleNamespace(p=process))
    reset_stateful_guidance(SimpleNamespace(p=None))  # no process: nothing to do

    assert stale.reset_calls == 0
