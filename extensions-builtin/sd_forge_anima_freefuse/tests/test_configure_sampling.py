import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.argv = sys.argv[:1]
EXTENSION_PATH = Path(__file__).parents[1]
# Shared with the DAVE / Cross-Attention Boost suites, loaded by path (see its docstring)
_spec = importlib.util.spec_from_file_location("extension_script_harness", EXTENSION_PATH.parents[1] / "tests" / "extension_script_harness.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def _load(monkeypatch):
    monkeypatch.syspath_prepend(str(EXTENSION_PATH))
    # modules.infotext_utils: set up by conftest.py for the whole session
    spec = importlib.util.spec_from_file_location("anima_freefuse_under_test", EXTENSION_PATH / "scripts" / "anima_freefuse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class _Engine:
    t5_tokenizer = _Tokenizer()

    def __init__(self, chunks=1):
        self.chunks = chunks

    def tokenize_line(self, prompt):
        # One chunk of T5 ids with the appended EOS, like AnimaTextProcessingEngine
        tokens = [ord(character) for character in prompt] + [1]
        return [SimpleNamespace(t5_tokens=tokens)] * self.chunks


class _Unet:
    def __init__(self, blocks, options=None):
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(blocks=[None] * blocks))
        self.model_options = dict(options or {})

    def clone(self):
        clone = _Unet(0, self.model_options)
        clone.model = self.model
        return clone

    def set_transformer_option(self, key, value):
        options = dict(self.model_options.get("transformer_options", {}))
        options[key] = value
        self.model_options["transformer_options"] = options


def _process(prompts=("alice and bob",), blocks=28, chunks=1):
    sd_model = SimpleNamespace(forge_objects=SimpleNamespace(unet=_Unet(blocks)), text_processing_engine_anima=_Engine(chunks))
    # firstpass_steps: the base pass's total steps, as setup_conds sets them
    return SimpleNamespace(sd_model=sd_model, prompts=list(prompts), negative_prompts=[""] * len(prompts), extra_generation_params={}, firstpass_steps=10)


SUBJECTS = (True, "alice_lora", "alice", True, "bob_lora", "bob", False, "", "")


def _values(subjects=SUBJECTS, collect_block=18, bias_blocks="0-51", background=""):
    # collect step, collect block, top-k, temperature, background, bg scale,
    # balance iterations, feather, routing strength, routing end, bias scale,
    # positive bias, bias blocks
    return (*subjects, 3, collect_block, 0.1, 300, background, 0.95, 15, 1, 1.0, 1.0, 6.0, 1.0, bias_blocks)


def _configure(module, p, values):
    """``_configure_sampling`` of an enabled panel with the script args after ``enable``."""
    module.AnimaFreeFuseScript._configure_sampling(p, module._arguments((True, *values)))


@pytest.fixture
def script(monkeypatch):
    module = _load(monkeypatch)
    applied = []
    monkeypatch.setattr(module, "apply_freefuse_patch", lambda unet, state: applied.append(state) or "patched")
    return module, applied


def test_valid_subjects_build_the_state(script):
    module, applied = script
    p = _process()
    _configure(module, p, _values())

    (state,) = applied
    assert p.sd_model.forge_objects.unet == "patched"
    assert [(a.name, a.selector, a.concept) for a in state.adapters] == [("subject_1", "alice_lora", "alice"), ("subject_2", "bob_lora", "bob")]
    assert state.token_positions == {"subject_1": [0, 1, 2, 3, 4], "subject_2": [10, 11, 12]}
    assert state.background_positions == [13]  # the EOS token
    params = p.extra_generation_params
    assert params["Anima FreeFuse subject 3 active"] is False and "Anima FreeFuse subject 3 LoRA" not in params
    assert params["Anima FreeFuse collect block"] == 18 and params["Anima FreeFuse bias blocks"] == "0-51"


SUMMARY_KEYS = {"Anima FreeFuse", "Anima FreeFuse collect", "Anima FreeFuse routing", "Anima FreeFuse bias"}


@pytest.mark.parametrize(
    "subjects, background, unwritten",
    [
        ((*SUBJECTS[:6], True, "carol_lora", "carol"), "forest", set()),
        # An inactive subject records only its switch, an empty background nothing
        (SUBJECTS, "", {"Anima FreeFuse subject 3 LoRA", "Anima FreeFuse subject 3 concept", "Anima FreeFuse background"}),
    ],
)
def test_written_infotext_restores_the_panel_controls(script, subjects, background, unwritten):
    module, applied = script
    panel = module.AnimaFreeFuseScript()
    controls = harness.build_ui(panel)
    fields = {key: component for component, key in panel.infotext_fields}
    assert list(fields) == panel.paste_field_names == list(module.INFOTEXT_KEYS)
    assert len(fields) == len(controls)
    p = _process(("alice and bob with carol in a forest",))
    module.set_feature_active(p, module.FREEFUSE)
    args = (True, *_values(subjects, background=background))

    panel.process_before_every_sampling(p, *args)

    assert len(applied) == 1
    written = p.extra_generation_params
    assert set(written) - set(fields) == SUMMARY_KEYS  # diagnostics only
    assert set(fields) - set(written) == unwritten
    for key in fields.keys() & written.keys():
        index = next(number for number, control in enumerate(controls) if control is fields[key])
        assert written[key] == args[index], key


@pytest.mark.parametrize(
    "subjects, message",
    [
        ((True, "alice_lora", "", True, "bob_lora", "bob", False, "", ""), "needs both a LoRA name and trigger phrase"),
        ((True, "", "alice", True, "bob_lora", "bob", False, "", ""), "needs both a LoRA name and trigger phrase"),
        ((True, "alice_lora", "alice", False, "bob_lora", "bob", False, "", ""), "two or three active subjects"),
        ((True, "alice_lora", "alice", True, "bob_lora", "carol", False, "", ""), "not found verbatim"),
    ],
)
def test_invalid_subjects_are_rejected(script, subjects, message):
    module, applied = script
    with pytest.raises(ValueError, match=message):
        _configure(module, _process(), _values(subjects))
    assert applied == []


def test_concepts_must_sit_at_the_same_tokens_across_the_batch(script):
    module, applied = script
    # Same positions in both prompts: accepted
    _configure(module, _process(("alice and bob", "alice and bob")), _values())
    assert len(applied) == 1
    with pytest.raises(ValueError, match="identical concept token positions"):
        _configure(module, _process(("alice and bob", "alice with bob")), _values())
    with pytest.raises(ValueError, match="single Anima conditioning chunk"):
        _configure(module, _process(chunks=2), _values())


class _RecordingEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.lines = []

    def tokenize_line(self, prompt):
        self.lines.append(prompt)
        return super().tokenize_line(prompt)


@pytest.mark.parametrize(
    "prompt, texts",
    [
        # Forge resolves prompt editing before the text encoder sees the prompt
        ("alice and bob [cat:dog:5]", ["alice and bob cat", "alice and bob dog"]),
        ("alice and bob [red|tan]", ["alice and bob red", "alice and bob tan"]),
        ("alice and bob", ["alice and bob"]),
    ],
)
def test_concept_positions_come_from_each_resolved_step_text(script, prompt, texts):
    module, applied = script
    p = _process((prompt,))
    p.sd_model.text_processing_engine_anima = engine = _RecordingEngine()
    _configure(module, p, _values())

    assert engine.lines == texts  # each distinct text once
    (state,) = applied
    assert state.token_positions == {"subject_1": [0, 1, 2, 3, 4], "subject_2": [10, 11, 12]}
    assert state.background_positions == [len(texts[0])]  # EOS after the resolved text


@pytest.mark.parametrize(
    "prompt, background",
    [
        ("[an |]alice and bob", ""),  # the edit moves the triggers
        ("alice and bob [cat:bird:5]", ""),  # ... or the EOS background slot
        ("alice and bob [cat:bird:5] in a forest", "forest"),  # ... or the background phrase
    ],
)
def test_prompt_editing_that_moves_token_positions_is_rejected(script, prompt, background):
    module, applied = script
    with pytest.raises(ValueError, match=r"prompt editing \(\[a:b:N\], \[a\|b\]\) to keep"):
        _configure(module, _process((prompt,)), _values(background=background))
    assert applied == []


def test_prompt_editing_after_the_background_phrase_is_accepted(script):
    module, applied = script
    _configure(module, _process(("alice and bob in a forest [cat:bird:5]",)), _values(background="forest"))
    (state,) = applied
    assert state.background_positions == [19, 20, 21, 22, 23, 24]


@pytest.mark.parametrize("prompts", [("alice and bob AND a forest",), ("alice and bob", "alice and bob AND rain:0.5")])
def test_and_prompts_are_rejected_before_sampling(script, prompts):
    module, applied = script
    # Forge would split these into sub-prompt conds: wrong concept positions and
    # one collected mask per sub-prompt row, failing only inside phase 2.
    with pytest.raises(ValueError, match="does not support AND prompts"):
        _configure(module, _process(prompts), _values())
    assert applied == []


def test_artist_mixer_in_the_same_pass_is_rejected(script):
    module, _ = script
    p = _process()
    module.set_feature_active(p, module.ARTIST_MIXER)
    with pytest.raises(ValueError, match="Artist Mixer"):
        _configure(module, p, _values())


@pytest.mark.parametrize(
    "blocks, collect, bias_text, bias",
    [
        (28, 18, "3-4", {3, 4}),
        # 2B block 1 -> model blocks 1 and 2 (its inserted copy); 18 -> 26 and 27
        (40, 26, "1,18", {1, 2, 26, 27}),
        (52, 1, "1", {1, 2, 3}),
    ],
)
def test_collect_and_bias_blocks_map_from_the_2b_layout(script, blocks, collect, bias_text, bias):
    module, applied = script
    collect_2b = 18 if blocks != 52 else 1
    p = _process(blocks=blocks)
    _configure(module, p, _values(collect_block=collect_2b, bias_blocks=bias_text))
    (state,) = applied
    assert state.collect_block == collect  # the original block, not a copy
    assert state.bias_blocks == bias
    assert p.extra_generation_params["Anima FreeFuse collect block"] == collect_2b  # pasted back as written


@pytest.mark.parametrize("blocks", [28, 40, 52])
def test_every_block_is_biased_by_the_full_range(script, blocks):
    module, applied = script
    for text in ("0-27", "0-51"):
        _configure(module, _process(blocks=blocks), _values(bias_blocks=text))
        assert applied.pop().bias_blocks == set(range(blocks))


def test_a_configuration_error_aborts_the_sampler(monkeypatch):
    module = _load(monkeypatch)
    from lib_anima_freefuse.runtime import RejectedFreeFuseState

    p = _process()
    unet = p.sd_model.forge_objects.unet
    module.set_feature_active(p, module.FREEFUSE)
    bad = _values((True, "alice_lora", "alice", False, "", "", False, "", ""))

    module.AnimaFreeFuseScript().process_before_every_sampling(p, True, *bad)

    patched = p.sd_model.forge_objects.unet
    assert patched is not unet and "transformer_options" not in unet.model_options
    rejected = patched.model_options["transformer_options"]["anima_freefuse_state"]
    assert isinstance(rejected, RejectedFreeFuseState)
    with pytest.raises(RuntimeError, match="Anima FreeFuse configuration failed: .*two or three active subjects"):
        rejected.configure_steps(20)
    assert p.extra_generation_params["Anima FreeFuse"] == "error: ValueError"


def test_disabled_or_inactive_feature_does_nothing(monkeypatch):
    module = _load(monkeypatch)
    for enable, active in ((False, True), (True, False)):
        p = _process()
        unet = p.sd_model.forge_objects.unet
        if active:
            module.set_feature_active(p, module.FREEFUSE)
        module.AnimaFreeFuseScript().process_before_every_sampling(p, enable, *_values())
        assert p.sd_model.forge_objects.unet is unet and p.extra_generation_params == {}



@pytest.mark.parametrize("values", [(*_values(), "extra"), _values()[:-1]], ids=["one-too-many", "one-short"])
def test_a_wrong_argument_count_is_refused_before_anything_changes(monkeypatch, values):
    from modules import anima_online_lora

    module = _load(monkeypatch)
    monkeypatch.setattr(anima_online_lora, "_leases", set())
    count = len(module.SCRIPT_ARG_NAMES)
    p = _process()
    p.sampler_name, p.enable_hr = "Euler", False
    unet = p.sd_model.forge_objects.unet
    with pytest.raises(TypeError, match=f"Anima FreeFuse takes {count} script arguments, got {len(values) + 1}"):
        module.AnimaFreeFuseScript().process(p, True, *values)
    assert p.sampler_name == "Euler" and not module.is_feature_active(p, module.FREEFUSE) and anima_online_lora._leases == set()
    module.set_feature_active(p, module.FREEFUSE)
    # Raised, not turned into the sampler's configuration error
    with pytest.raises(TypeError, match=f"got {len(values) + 1}"):
        module.AnimaFreeFuseScript().process_before_every_sampling(p, True, *values)
    assert p.sd_model.forge_objects.unet is unet and p.extra_generation_params == {}
