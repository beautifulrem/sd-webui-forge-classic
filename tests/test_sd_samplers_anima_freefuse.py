from types import SimpleNamespace

from modules.anima_guidance_state import reset_stateful_guidance


class _State:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_probe_reset_includes_momentum_post_cfg_state():
    momentum = _State()

    def post_cfg(args):
        return args["denoised"]

    post_cfg._anima_momentum_state = momentum
    unet = SimpleNamespace(
        model_options={"sampler_post_cfg_function": [post_cfg]}
    )
    process = SimpleNamespace(
        sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))
    )
    model = SimpleNamespace(p=process)

    reset_stateful_guidance(model)

    assert momentum.reset_calls == 1
