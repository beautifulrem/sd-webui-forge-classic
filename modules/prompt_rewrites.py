"""Undoable rewrites of a processing object's prompt fields by scripts.

img2img batch, Loopback and SD upscale run the same processing object
through process_images() again, and setup_prompts() rebuilds the prompt
lists from p.prompt: a script that rewrites p.prompt in process() would
rewrite its own output on the next run. Scripts record their rewrites here
and every participant restores in before_process(), which runs before the
prompts are rebuilt; chained rewrites by several scripts restore to the
value before the first one.
"""

_KEY = "_script_prompt_rewrites"


def record(p, attr: str, before, after) -> None:
    rewrites = getattr(p, _KEY, None)
    if rewrites is None:
        rewrites = {}
        setattr(p, _KEY, rewrites)
    original = rewrites[attr][1] if attr in rewrites else before
    rewrites[attr] = (after, original)


def restore(p) -> None:
    """Put recorded fields back unless something else changed them since
    (e.g. Loopback appending an interrogated prompt)."""
    rewrites = getattr(p, _KEY, None)
    if not rewrites:
        return
    setattr(p, _KEY, None)
    for attr, (rewritten, original) in rewrites.items():
        if getattr(p, attr, None) == rewritten:
            setattr(p, attr, original)
