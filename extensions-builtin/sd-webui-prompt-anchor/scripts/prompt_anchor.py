"""
sd-webui-prompt-anchor
======================

Adds a secondary "Anchor prompt" editor next to the main txt2img / img2img
positive prompts. Whatever the user types into the anchor box is
PREPENDED to the regular positive prompt at generation time -- without
ever being written into the main prompt textarea.

The motivation is purely organizational. People build up a "stable
foundation" of artist tags / style tags / quality tags that they reuse
across many generations, and then vary just the *subject* portion. Until
now that meant either:

  (a) keeping the foundation pinned to the top of the main prompt box and
      carefully editing around it, or
  (b) using styles -- which are stored and appended at generation time
      but are a clunky UI for something you tweak constantly.

The anchor box gives you a second always-visible textarea you can park
your foundation in. The main box stays free for whatever varies
generation-to-generation, AND -- importantly -- any extension that
auto-fills the main txt2img prompt (booru-tag senders, image-captioners,
prompt-suggesters, etc.) will not touch the anchor. The two stay
cleanly separated.

Implementation overview
-----------------------

A `scripts.Script` subclass with `alwayson=True` that:
   - Declares the panel (`params()`: enable switch in the panel header,
     anchor prompt editor, separator field). `slot = "prompt_before"`
     places it directly above the main positive prompt (where the legacy
     prompt_anchor.js moved it); `section = "prompt"` keeps its arguments
     in the prompt category. The panel collapses to its header. The SPA keeps the values in the form draft
     (per browser, restored on reload).
   - Implements `process(p, *args)` to PREPEND the anchor text onto
     `p.prompt`, `p.all_prompts`, `p.hr_prompt`, and `p.all_hr_prompts`.
   - Writes the anchor text into infotext via `p.extra_generation_params`
     so it round-trips into PNG metadata and "send to txt2img" buttons.

Why `process()` and not the textarea
------------------------------------

Modifying `p.prompt` in `process()` runs BEFORE per-batch processing
(wildcards, dynamic prompts, etc.), so any extension that expands
wildcards will see -- and expand -- wildcards placed inside the anchor
box. And because the main prompt is never edited, no other extension
can race us, append to us, or overwrite us.

The opposite tradeoff is that the anchor text isn't visible in the
prompt textarea -- but that's the entire point of the feature. The
"Prompt used" sent to infotext IS the joined string, so the user can
always see what actually ran by hitting "Send to txt2img" on a generated
image (which will put the full joined prompt back).
"""

from __future__ import annotations

import re
import traceback

from modules import prompt_rewrites, scripts, script_callbacks, shared
from remi import ui


# ---------------------------------------------------------------------------
# Constants / settings keys
# ---------------------------------------------------------------------------

ENABLE_LABEL = "Anchor prompt (prepended to positive prompt at generation)"
PLACEHOLDER = (
    "Anchor text gets prepended to your positive prompt at "
    "generation time. Use this for styles, artists, quality "
    "tags -- anything you want to keep set-and-forget."
)
SEPARATOR_INFO = (
    "Inserted between the anchor and the main prompt. "
    "Default ', ' suits comma-separated tag prompts. "
    "Trailing commas/whitespace on the anchor and "
    "leading commas on the main prompt are stripped "
    "automatically to prevent doubled separators."
)

# Infotext key. Quoted so multi-line / comma-laden values survive the
# AUTOMATIC1111 "key: value, key: value" infotext parser.
INFOTEXT_KEY = "Anchor prompt"
# Written only when the separator is not the default ", " (same quoting).
SEPARATOR_INFOTEXT_KEY = "Anchor separator"


def _default_separator() -> str:
    """The string slotted between the anchor text and the main prompt
    when they're joined. Default ', ' matches the comma-separated tag
    style most commonly used; user can override per-tab.

    We deliberately do NOT auto-add a separator when the anchor text
    already ends with a comma (or the main prompt starts with one), to
    avoid producing duplicate commas in joined output. See `_join()`."""
    return ", "


def _configured_separator() -> str:
    """The separator a fresh form starts with: Settings -> Prompt Anchor."""
    value = shared.opts.data.get("prompt_anchor_default_separator")
    return value if isinstance(value, str) else _default_separator()


# ---------------------------------------------------------------------------
# Join helper
# ---------------------------------------------------------------------------

_TRAILING_PUNCT = re.compile(r"[,\s]+$")
_LEADING_PUNCT = re.compile(r"^[,\s]+")


def _join(anchor: str, main: str, separator: str) -> str:
    """Concatenate anchor + main with smart separator handling.

    Rules (in priority order):
      - Empty anchor -> return main unchanged.
      - Empty main -> return anchor unchanged (stripped of trailing junk).
      - Strip trailing commas/whitespace from anchor, leading commas
        from main, then join with `separator`. This guarantees we never
        emit ",, " or " , " regardless of how either side is written.

    The separator itself is used verbatim (after the cleanup above), so
    a user who wants newline-separated prompts can set it to "\\n" in
    settings.
    """
    a = (anchor or "")
    m = (main or "")
    if not a.strip():
        return m
    if not m.strip():
        return _TRAILING_PUNCT.sub("", a)
    a = _TRAILING_PUNCT.sub("", a)
    m = _LEADING_PUNCT.sub("", m)
    return f"{a}{separator}{m}"


# ---------------------------------------------------------------------------
# The Script
# ---------------------------------------------------------------------------


class PromptAnchorScript(scripts.Script):
    """One instance is constructed per tab (txt2img, img2img) by the
    webui's Script framework; each tab's form keeps its own anchor."""

    section = "prompt"
    # Above the main prompt, where the legacy prompt_anchor.js moved the box;
    # the section keeps the arguments in the prompt category
    slot = "prompt_before"

    def title(self):
        return "Prompt Anchor"

    def show(self, is_img2img):
        # AlwaysVisible -- we want this loaded on every generation, not
        # opt-in from the script dropdown.
        return scripts.AlwaysVisible

    # ---- UI ----------------------------------------------------------------

    def params(self, is_img2img):
        """The anchor panel, rendered in the core form's ``prompt_before``
        slot (this script's ``slot``), right above the main prompt. The enable
        switch sits in the panel header; collapsing the panel keeps it
        reachable. The SPA keeps the text in the form's draft, so it
        survives reloads per browser as the old localStorage shim did."""
        collapsed = bool(shared.opts.data.get("prompt_anchor_collapsed_by_default", False))
        return ui.Section(
            "Anchor prompt",
            open=not collapsed,
            enable=ui.Toggle("enabled", True, label=ENABLE_LABEL, paste=_paste_enabled),
            children=[
                ui.Text(
                    "anchor",
                    "",
                    lines=2,
                    max_lines=20,
                    prompt=True,
                    label="",
                    placeholder=PLACEHOLDER,
                    paste=_decode_infotext_anchor,
                ),
                ui.Group(
                    "Advanced",
                    ui.Text(
                        "separator",
                        _configured_separator,
                        lines=1,
                        label="Separator between anchor and main prompt",
                        info=SEPARATOR_INFO,
                        paste=_paste_separator,
                    ),
                    open=False,
                ),
            ],
        )

    # ---- Generation hooks --------------------------------------------------

    # The same p is run again by img2img batch, Loopback and SD upscale:
    # undo this script's rewrite before the next run rebuilds the prompts.
    # (Not after a run: Loopback / SD upscale report p.prompt afterwards.)
    def before_process(self, p, *args):
        prompt_rewrites.restore(p)

    def process(self, p, enabled, anchor, separator, *_):
        """Runs once per Generate click, BEFORE any per-batch processing
        (wildcards, dynamic prompts, etc.). We prepend the anchor here
        rather than in `before_process` so that:

          - other prompt-mutating scripts that read `p.all_prompts`
            during their own `process_batch` see the JOINED text
            (e.g. Dynamic Prompts will expand `__wildcard__` references
             placed inside the anchor), and
          - the joined string is what ends up baked into the infotext
            metadata stored with the saved image.

        Empty / disabled anchor -> we still attach the infotext field
        with an empty value so the parameter list is consistent across
        runs (makes diffing infotext between images simpler)."""
        try:
            if not enabled:
                return
            text = (anchor or "").strip()
            if not text:
                return

            sep = separator if separator is not None else _default_separator()

            # `p.prompt` -- single-string version, used for display and
            # for the no-batch path.
            original_prompt = getattr(p, "prompt", None)
            original_all_prompts = list(getattr(p, "all_prompts", None) or [])
            if hasattr(p, "prompt"):
                p.prompt = _join(text, p.prompt or "", sep)
                prompt_rewrites.record(p, "prompt", original_prompt, p.prompt)

            # `p.all_prompts` -- list, one entry per image in the batch.
            # Populated by process_images() BEFORE process() runs, so
            # we have to mutate the list in place / by-element.
            all_prompts = getattr(p, "all_prompts", None)
            if all_prompts:
                p.all_prompts = [_join(text, ap or "", sep) for ap in all_prompts]
                # setup_prompts() populated this before scripts.process().
                # Keep grid/video infotext (which uses main_prompt) aligned
                # with the per-image prompt list we just rewrote.
                p.main_prompt = p.all_prompts[0]

            # Hires-fix has its own prompt fields (`hr_prompt` and
            # `all_hr_prompts`). process_images() runs setup_prompts()
            # BEFORE scripts.process(), so both are already filled here
            # (copied from the original prompt when the user left the
            # hires prompt empty). Prepend unconditionally so the base
            # and hires passes use the same anchored conditioning.
            # A copy of the main prompt takes the anchored main prompt; a
            # custom hires prompt is anchored unless it already starts with
            # the anchor (e.g. pasted from an anchored image's infotext).
            def anchor_hires(hires, main, anchored_main):
                if hires == main:
                    return anchored_main
                if not hires or hires.startswith(text):
                    return hires
                return _join(text, hires, sep)

            hr_prompt = getattr(p, "hr_prompt", None)
            if isinstance(hr_prompt, str) and hr_prompt:
                p.hr_prompt = anchor_hires(hr_prompt, original_prompt, getattr(p, "prompt", None))
                prompt_rewrites.record(p, "hr_prompt", hr_prompt, p.hr_prompt)
            all_hr = getattr(p, "all_hr_prompts", None)
            if all_hr:
                anchored_all = getattr(p, "all_prompts", None) or []
                p.all_hr_prompts = [
                    anchor_hires(
                        hp,
                        original_all_prompts[index] if index < len(original_all_prompts) else None,
                        anchored_all[index] if index < len(anchored_all) else None,
                    )
                    for index, hp in enumerate(all_hr)
                ]

            # Record in infotext so the value survives into PNG metadata
            # and "send to txt2img" round-trips. Wrap in double quotes
            # so commas / newlines don't fragment the infotext parser.
            # Newlines inside the value are encoded as the literal two
            # characters '\n' (matching how A1111 encodes them); the
            # paste-fields code on the receiving end will decode them.
            extra = getattr(p, "extra_generation_params", None)
            if extra is not None:
                extra[INFOTEXT_KEY] = _encode_infotext_value(text)
                if sep != _default_separator():
                    extra[SEPARATOR_INFOTEXT_KEY] = _encode_infotext_value(sep)

        except Exception:
            # Never crash a generation just because the anchor failed.
            # Print to console and let the user generate without us.
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Infotext round-trip: the anchor param's paste function, so pasting
# parameters and "Send to" repopulate the anchor from PNG metadata.
# ---------------------------------------------------------------------------
#
# The value we wrote is wrapped in double quotes with escape sequences, and
# the standard parser hands us the raw text between the colon and the next
# comma, so the paste function does the unescape.

def _encode_infotext_value(text):
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _decode_infotext_value(raw):
    # The infotext parser already removed the quotes of a value written with
    # json quoting; whitespace inside a value (a separator) is significant.
    stripped = raw.strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        raw = stripped[1:-1]
    return raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _decode_infotext_anchor(infotext_dict):
    """Given the parsed infotext (a dict of key -> raw string value),
    return the cleaned anchor text. Reverses the encoding done in
    process(): strips wrapping quotes, decodes backslash escapes.

    Returns None (i.e. leaves the field unchanged) if no anchor info is
    present in this infotext -- avoids wiping a manually-typed anchor when
    the user pastes parameters from an image that wasn't generated with
    this extension.

    Signature note: paste functions receive a SINGLE argument (the
    params dict).
    """
    raw = infotext_dict.get(INFOTEXT_KEY)
    if raw is None:
        return None
    return _decode_infotext_value(raw)


def _paste_enabled(infotext_dict):
    """An anchored image turns the anchor on (else the pasted, de-anchored
    prompt would run without it); other infotexts leave the switch alone."""
    return True if INFOTEXT_KEY in infotext_dict else None


def _paste_separator(infotext_dict):
    """The separator the image was joined with: the recorded one, or the
    default ", " (only a non-default separator is written)."""
    if INFOTEXT_KEY not in infotext_dict:
        return None
    raw = infotext_dict.get(SEPARATOR_INFOTEXT_KEY)
    return _default_separator() if raw is None else _decode_infotext_value(raw)


def _strip_anchor(prompt, anchor, separator):
    """``prompt`` without the leading ``anchor + separator`` that
    ``process()`` joined onto it (the inverse of ``_join``); None when the
    prompt was not joined that way (e.g. hand-edited)."""
    if not isinstance(prompt, str) or not anchor.strip():
        return None
    head = _TRAILING_PUNCT.sub("", anchor)
    if not prompt.startswith(head):
        return None
    rest = prompt[len(head):]
    if not rest:
        return ""  # the main prompt was empty: the join is the anchor alone
    if rest.startswith(separator):
        return rest[len(separator):]
    return None


def _on_infotext_pasted(infotext, params):
    """Pasting an anchored image's parameters restores the main prompt as
    the user typed it: the infotext's ``Prompt`` is the joined text, and
    the pasted anchor is prepended again on the next run, so leaving it in
    would double the anchor (a different image and infotext)."""
    anchor = _decode_infotext_anchor(params)
    if not anchor:
        return
    main = _strip_anchor(params.get("Prompt"), anchor, _paste_separator(params))
    if main is not None:
        params["Prompt"] = main


# ---------------------------------------------------------------------------
# Settings page entry (optional - lets the user change the default
# separator globally; per-tab override still wins).
# ---------------------------------------------------------------------------

def _on_ui_settings():
    """Adds a section to Settings -> Prompt Anchor: the separator field's
    default (read on every page load) and the panel's initial open state
    (read when the layouts are built, hence Reload UI)."""
    try:
        section = ("prompt_anchor", "Prompt Anchor")
        shared.opts.add_option(
            "prompt_anchor_default_separator",
            shared.OptionInfo(
                _default_separator(),
                "Default separator between anchor and main prompt",
                ui.Text,
                section=section,
            ),
        )
        shared.opts.add_option(
            "prompt_anchor_collapsed_by_default",
            shared.OptionInfo(
                False,
                "Start the anchor box collapsed on page load",
                section=section,
            ).needs_reload_ui(),
        )
    except Exception:
        traceback.print_exc()


script_callbacks.on_ui_settings(_on_ui_settings)
script_callbacks.on_infotext_pasted(_on_infotext_pasted, name="prompt_anchor")
