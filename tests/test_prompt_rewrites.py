from types import SimpleNamespace

from modules import prompt_rewrites


def test_chained_rewrites_restore_to_the_original():
    p = SimpleNamespace(prompt="[cat:dog:20]", hr_prompt="")
    # Prompt rescale, then Prompt Anchor, as their process() hooks run.
    prompt_rewrites.record(p, "prompt", p.prompt, "[cat:dog:6]")
    p.prompt = "[cat:dog:6]"
    prompt_rewrites.record(p, "prompt", p.prompt, "anchor, [cat:dog:6]")
    p.prompt = "anchor, [cat:dog:6]"

    prompt_rewrites.restore(p)  # before the next img2img batch image
    prompt_rewrites.restore(p)  # the other script's hook: no-op

    assert p.prompt == "[cat:dog:20]"


def test_fields_changed_by_others_are_left_alone():
    p = SimpleNamespace(prompt="a")
    prompt_rewrites.record(p, "prompt", "a", "anchor, a")
    p.prompt = "a, interrogated"  # e.g. Loopback's appended prompt

    prompt_rewrites.restore(p)

    assert p.prompt == "a, interrogated"
