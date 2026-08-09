from types import SimpleNamespace

from lib_anima_guidance.prompts import effective_prompt_batch


def test_hires_pass_uses_hires_prompt_batch():
    process = SimpleNamespace(
        is_hr_pass=True,
        prompts=["base positive"],
        negative_prompts=["base negative"],
        hr_prompts=["hires one", "hires two"],
        hr_negative_prompts=["hires negative"],
    )

    positive, negative = effective_prompt_batch(process)

    assert positive == ["hires one", "hires two"]
    assert negative == ["hires negative", "hires negative"]


def test_hires_empty_batch_falls_back_to_base_prompts():
    process = SimpleNamespace(
        is_hr_pass=True,
        prompts=["base positive"],
        negative_prompts=["base negative"],
        hr_prompts=[],
        hr_negative_prompts=[],
    )

    assert effective_prompt_batch(process) == (
        ["base positive"],
        ["base negative"],
    )
