"""Skip Layer Guidance (``modules/processing_scripts/slg.py``, upstream
be6fa5d4 + a83c1c82): a settings-only ``AlwaysVisible`` script. It loads in
Remi's runner without ``params()``, acts only on a Moss-Image model, leaves
every other model's sampling call untouched, and its infotext pastes back as
setting overrides.

No model: the sampling calls are spies and the model a stand-in."""

from types import SimpleNamespace

import pytest

from remi.testing import processing_script_classes, webui_state


@pytest.fixture(scope="module")
def slg():
    from backend.sampling import sampling_function
    from modules.processing_scripts import slg

    yield slg
    # importing the script patched the sampling function, as at startup; leave
    # the process as the other suites expect it
    sampling_function.calc_cond_uncond_batch = slg._calc_cond_uncond_batch


@pytest.fixture(scope="module")
def webui(slg):
    with webui_state(script_classes=processing_script_classes(("slg",))) as state:
        yield state


class _Spy:
    """Stands in for the unpatched ``calc_cond_uncond_batch``."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, cond, uncond, x, timestep, model_options):
        self.calls.append((model, cond, uncond, x, timestep, model_options))
        return [f"cond{len(self.calls)}", f"uncond{len(self.calls)}"]


@pytest.fixture
def spy(slg, monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(slg, "_calc_cond_uncond_batch", spy)
    return spy


class _Sigma:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return SimpleNamespace(item=lambda: self.value)


# region Settings and runner


def test_parse_layers(slg):
    assert slg.parse_layers("22") == [22]
    assert slg.parse_layers("22, 24,26") == [22, 24, 26]
    assert slg.parse_layers(" 22 ; x 7 ") == [22, 7]
    assert slg.parse_layers("") == slg.parse_layers("none") == []


def test_settings_are_remi_widgets_with_infotext_keys():
    from modules import shared_options
    from remi import ui

    templates = shared_options.options_templates
    assert templates["slg_enable"].default is True and templates["slg_enable"].component is None
    assert templates["slg_single_layers"].component is ui.Text and templates["slg_single_layers"].default == "22"
    assert templates["slg_start_percent"].component is ui.Number and templates["slg_start_percent"].default == 0.0
    assert templates["slg_end_percent"].component is ui.Number and templates["slg_end_percent"].default == 0.5
    infotext = {name: templates[name].infotext for name in ("slg_enable", "slg_single_layers", "slg_start_percent", "slg_end_percent")}
    assert infotext == {"slg_enable": "Skip Layer Guidance", "slg_single_layers": "SLG Single Layers", "slg_start_percent": "SLG Start Percent", "slg_end_percent": "SLG End Percent"}


def test_runs_in_the_runner_without_params(webui, slg):
    from modules import scripts

    for runner in (webui.txt2img, webui.img2img):
        script = next(script for script in runner.alwayson_scripts if isinstance(script, slg.SkipLayerGuidance))
        assert not scripts.has_params(script) and not scripts.legacy_ui_only(script)
        assert script not in [disabled for disabled, _reason in runner.disabled_scripts]
        assert runner.layout_for(script).names == [] and script.args_from == script.args_to


def test_infotext_pastes_back_as_setting_overrides(webui):
    from modules.infotext_utils import create_override_settings_dict, quote
    from remi.core.infotext import parse_infotext

    params = {"Steps": 50, "Sampler": "Euler", "Skip Layer Guidance": True, "SLG Single Layers": "22, 24, 26", "SLG Start Percent": 0.1, "SLG End Percent": 0.6}
    text = "a cat\nNegative prompt: \n" + ", ".join(f"{key}: {quote(value)}" for key, value in params.items())
    result = parse_infotext(text, "txt2img")

    overrides = result["core"]["override_settings"]
    # "Skip Layer Guidance: True" is the current value: no override
    assert sorted(overrides) == ["SLG End Percent: 0.6", "SLG Single Layers: 22, 24, 26", "SLG Start Percent: 0.1"]
    assert create_override_settings_dict(overrides) == {"slg_single_layers": "22, 24, 26", "slg_start_percent": 0.1, "slg_end_percent": 0.6}


# region The sampling patch


def test_import_patches_a_pass_through(slg, spy):
    """Without a ``sampler_calc_cond_batch_function`` (every model but a
    Moss-Image one with SLG on) the patched function is the original call:
    same arguments, same result, so other models' pixels do not change."""

    from backend.sampling import sampling_function

    assert sampling_function.calc_cond_uncond_batch is slg.calc_cond_uncond_batch
    model, cond, uncond, x, timestep = object(), [object()], [object()], object(), object()
    model_options = {"transformer_options": {}}

    result = slg.calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)
    assert result == ["cond1", "uncond1"]
    assert len(spy.calls) == 1 and all(a is b for a, b in zip(spy.calls[0], (model, cond, uncond, x, timestep, model_options), strict=True))


def test_patch_dispatches_to_the_sampler_function(slg, spy):
    seen = []

    def calc_batch_function(*args):
        seen.append(args)
        return ["c", "u"]

    model_options = {"sampler_calc_cond_batch_function": calc_batch_function}
    assert slg.calc_cond_uncond_batch("m", "c", "u", "x", "t", model_options) == ["c", "u"]
    assert seen == [("m", "c", "u", "x", "t", model_options)] and spy.calls == []


def test_revert_restores_the_original(slg):
    from backend.sampling import sampling_function

    try:
        slg.revert()
        assert sampling_function.calc_cond_uncond_batch is slg._calc_cond_uncond_batch
    finally:
        sampling_function.calc_cond_uncond_batch = slg.calc_cond_uncond_batch


def test_skips_the_layers_for_uncond_inside_the_window(slg, spy):
    calc = slg.SkipLayerGuidance.make_calc_cond_batch_function([], [22, 24], sigma_start=0.9, sigma_end=0.4)
    model_options = {"transformer_options": {"existing": 1}}

    out = calc("m", ["cond"], ["uncond"], "x", _Sigma(0.5), model_options)
    assert out == ["cond1", "uncond2"]
    (_, cond, uncond, _, _, options), (_, cond2, uncond2, _, _, slg_options) = spy.calls
    assert (cond, uncond, options) == (["cond"], None, model_options)  # cond: the full model
    assert (cond2, uncond2) == ([], ["uncond"])
    replace = slg_options["transformer_options"]["patches_replace"]["dit"]
    assert set(replace) == {("single_block", 22), ("single_block", 24)}
    args = {"img": "hidden"}
    assert replace[("single_block", 22)](args, {}) is args  # the block is skipped
    assert model_options == {"transformer_options": {"existing": 1}}  # the caller's options are not touched


@pytest.mark.parametrize("sigma, uncond", [(0.95, ["uncond"]), (0.3, ["uncond"]), (0.5, None)])
def test_one_plain_call_outside_the_window_or_without_uncond(slg, spy, sigma, uncond):
    calc = slg.SkipLayerGuidance.make_calc_cond_batch_function([], [22], sigma_start=0.9, sigma_end=0.4)
    model_options = {"transformer_options": {}}

    assert calc("m", ["cond"], uncond, "x", _Sigma(sigma), model_options) == ["cond1", "uncond1"]
    assert len(spy.calls) == 1 and spy.calls[0][2] is uncond and spy.calls[0][5] is model_options


# region process_before_every_sampling


class _Unet:
    def __init__(self):
        self.model = SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda percent: round(1.0 - percent, 6)))
        self.model_options = {"transformer_options": {}}
        self.clones = []

    def clone(self):
        clone = _Unet()
        self.clones.append(clone)
        return clone

    def set_model_sampler_calc_cond_batch_function(self, function):
        self.model_options["sampler_calc_cond_batch_function"] = function


def _p():
    unet = _Unet()
    return SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), extra_generation_params={}), unet


@pytest.mark.parametrize("moss, options", [(False, {}), (True, {"slg_enable": False}), (True, {"slg_single_layers": ""})])
def test_does_nothing_unless_moss_with_slg_on(webui, slg, monkeypatch, moss, options):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "moss", moss)
    monkeypatch.setattr(slg, "opts", webui.opts)  # bound at import, as at startup (after the options)
    for key, value in options.items():
        monkeypatch.setitem(webui.opts.data, key, value)
    p, unet = _p()

    slg.SkipLayerGuidance().process_before_every_sampling(p)
    assert p.sd_model.forge_objects.unet is unet and unet.clones == [] and p.extra_generation_params == {}


def test_patches_a_clone_of_the_moss_unet(webui, slg, monkeypatch):
    from backend.args import dynamic_args

    monkeypatch.setattr(dynamic_args, "moss", True)
    monkeypatch.setattr(slg, "opts", webui.opts)
    monkeypatch.setitem(webui.opts.data, "slg_single_layers", "22, 26")
    monkeypatch.setitem(webui.opts.data, "slg_start_percent", -1.0)  # clamped to 0..1
    monkeypatch.setitem(webui.opts.data, "slg_end_percent", 0.5)
    p, unet = _p()

    slg.SkipLayerGuidance().process_before_every_sampling(p)
    (clone,) = unet.clones
    assert p.sd_model.forge_objects.unet is clone and "sampler_calc_cond_batch_function" not in unet.model_options
    assert callable(clone.model_options["sampler_calc_cond_batch_function"])
    assert p.extra_generation_params == {"Skip Layer Guidance": True, "SLG Single Layers": "22, 26", "SLG Start Percent": 0.0, "SLG End Percent": 0.5}
