import logging
from types import SimpleNamespace

import pytest
import torch
from k_diffusion.sampling import sample_euler

import modules.sd_samplers_anima_freefuse as freefuse
from modules.anima_guidance_state import register_stateful_guidance

SIGMAS = torch.tensor([1.0, 0.7, 0.4, 0.0])  # 3 steps; collect_step=1 -> 2 probe steps


class _Runtime:
    collect_step = 1

    def __init__(self, events):
        self.events = events
        self.current_step = None

    def configure_steps(self, total):
        self.events.append(("configure_steps", total))

    def __getattr__(self, name):
        if name in {"begin_collect", "finish_collection", "begin_generate", "complete", "abort"}:
            return lambda: self.events.append(name)
        raise AttributeError(name)


class _Model:
    """Deterministic denoiser that, like CFGDenoiser, advances ``step`` per call."""

    def __init__(self, runtime=None, fail_at=None, error=RuntimeError("boom")):
        self.step = 0
        self.steps = []  # ``step`` as each call saw it
        self.inputs = []
        self.fail_at = fail_at
        self.error = error
        options = {"transformer_options": {"anima_freefuse_state": runtime}} if runtime else {}
        unet = SimpleNamespace(model_options=options)
        self.p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))

    def __call__(self, x, sigma, **_):
        if len(self.inputs) == self.fail_at:
            raise self.error
        self.inputs.append(x.clone())
        self.steps.append(self.step)
        self.step += 1
        return x * 0.5 + sigma.view(-1, 1) * 0.1


def _noise():
    return torch.linspace(-1.0, 1.0, 8).view(2, 4)


def test_real_pass_restarts_from_the_initial_noise():
    events = []
    model, x = _Model(_Runtime(events)), _noise()
    seen = []

    result = freefuse.sample_anima_freefuse_euler(
        model, x, SIGMAS, disable=True, callback=lambda d: seen.append((d["i"], d["x"].clone()))
    )

    assert len(model.inputs) == 2 + 3  # probe pass, then the full pass
    assert torch.equal(model.inputs[0], x) and torch.equal(model.inputs[2], x)
    assert not torch.equal(model.inputs[1], x)
    assert torch.equal(x, _noise())  # the caller's noise is untouched
    # Only the real pass reports progress, and it ends where plain Euler from x ends.
    assert [i for i, _ in seen] == [0, 1, 2] and torch.equal(seen[0][1], x)
    assert torch.allclose(result, sample_euler(_Model(), x, SIGMAS, disable=True))


def test_probe_pass_is_invisible_to_step_counters_and_runtime_lifecycle():
    events = []
    runtime = _Runtime(events)
    model = _Model(runtime)
    model.step = 5  # stale counter

    freefuse.sample_anima_freefuse_euler(model, _noise(), SIGMAS, disable=True)

    # Like the other Anima Flow samplers, each call sees its step index (prompt
    # schedules); the real pass starts over at 0 after the 2 probe calls.
    assert model.steps == [0, 1] + [0, 1, 2]
    assert model.step == 3
    assert runtime.current_step == 2
    assert events == [
        ("configure_steps", 3),
        "begin_collect",
        "finish_collection",
        "begin_generate",
        "complete",
    ]


def test_stateful_guidance_is_reset_between_passes():
    events = []
    model = _Model(_Runtime(events))
    register_stateful_guidance(model.p, SimpleNamespace(reset=lambda: events.append("guidance_reset")))

    freefuse.sample_anima_freefuse_euler(model, _noise(), SIGMAS, disable=True)

    assert events[1:] == ["begin_collect", "finish_collection", "guidance_reset", "begin_generate", "complete"]


def test_nonzero_churn_is_rejected_before_any_work():
    events = []
    model = _Model(_Runtime(events))

    with pytest.raises(ValueError, match="S-churn = 0"):
        freefuse.sample_anima_freefuse_euler(model, _noise(), SIGMAS, disable=True, s_churn=0.5)

    assert events == [] and model.inputs == []


@pytest.mark.parametrize("fail_at", [1, 3], ids=["probe_pass", "real_pass"])
def test_model_error_aborts_runtime_and_propagates(fail_at):
    events = []
    model = _Model(_Runtime(events), fail_at=fail_at)

    with pytest.raises(RuntimeError, match="boom"):
        freefuse.sample_anima_freefuse_euler(model, _noise(), SIGMAS, disable=True)

    assert events[-1] == "abort" and "complete" not in events
    assert ("begin_generate" in events) == (fail_at >= 2)


@pytest.mark.parametrize("fail_at", [1, 3], ids=["probe_pass", "real_pass"])
def test_interrupt_aborts_runtime_and_propagates(fail_at):
    class Interrupted(BaseException):  # like Forge's InterruptedException
        pass

    events = []
    model = _Model(_Runtime(events), fail_at=fail_at, error=Interrupted())

    with pytest.raises(Interrupted):
        freefuse.sample_anima_freefuse_euler(model, _noise(), SIGMAS, disable=True)

    assert events[-1] == "abort" and "complete" not in events


@pytest.mark.parametrize("s_churn", [0.0, 0.5])
def test_without_freefuse_runtime_falls_back_to_plain_euler(s_churn, caplog):
    x = _noise()

    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="AnimaFreeFuseSampler"):
        result = freefuse.sample_anima_freefuse_euler(_Model(), x, SIGMAS, disable=True, s_churn=s_churn)
    torch.manual_seed(0)
    expected = sample_euler(_Model(), x, SIGMAS, disable=True, s_churn=s_churn)

    assert torch.equal(result, expected)
    assert "without an active FreeFuse panel" in caplog.text


def test_in_place_input_edits_in_the_probe_pass_do_not_reach_the_real_pass():
    class EditingModel(_Model):
        def __call__(self, x, sigma, **extra):
            if "begin_generate" not in events:
                x.add_(1.0)  # e.g. a denoiser callback correcting the latent
            return super().__call__(x, sigma, **extra)

    events = []
    x = _noise()
    original = x.clone()

    freefuse.sample_anima_freefuse_euler(EditingModel(_Runtime(events)), x, SIGMAS, disable=True)

    assert torch.equal(x, original)
