"""
sd-webui-prompt-anchor
======================

Adds a secondary "Anchor prompt" textbox above the main txt2img / img2img
positive prompt boxes. Whatever the user types into the anchor box is
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

Two pieces co-operate:

1. A `scripts.Script` subclass with `alwayson=True` that:
   - Builds the UI (anchor textbox + enable checkbox + separator field).
     Gradio places this in the normal "always-on script accordion" area
     below the prompt; a small JS shim moves the DOM node up to sit
     directly above the prompt textarea on page load.
   - Implements `process(p, *args)` to PREPEND the anchor text onto
     `p.prompt`, `p.all_prompts`, `p.hr_prompt`, and `p.all_hr_prompts`.
   - Writes the anchor text into infotext via `p.extra_generation_params`
     so it round-trips into PNG metadata and "send to txt2img" buttons.

2. javascript/prompt_anchor.js handles:
   - The reparenting step above (move our wrapper to sit above the
     prompt textarea on the txt2img and img2img tabs).
   - localStorage persistence -- the anchor text and the enabled state
     are saved on every keystroke and restored on UI load.
   - Optional collapse/expand toggle so the box can be hidden when not
     in use without losing the saved content.

Why `process()` and not the textarea
------------------------------------

Modifying `p.prompt` in `process()` runs BEFORE per-batch processing
(wildcards, dynamic prompts, etc.), so any extension that expands
wildcards will see -- and expand -- wildcards placed inside the anchor
box. And because we never touch the textarea DOM, no other extension
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

import gradio as gr

from modules import scripts, script_callbacks, shared


# ---------------------------------------------------------------------------
# Constants / settings keys
# ---------------------------------------------------------------------------

# Distinctive elem_ids -- deliberately NOT matching "txt2img_prompt" /
# "img2img_prompt" so extensions targeting those selectors leave us alone.
ELEM_ID_TXT2IMG = "prompt_anchor_txt2img"
ELEM_ID_IMG2IMG = "prompt_anchor_img2img"
ELEM_ID_TXT2IMG_ENABLE = "prompt_anchor_enable_txt2img"
ELEM_ID_IMG2IMG_ENABLE = "prompt_anchor_enable_img2img"
ELEM_ID_TXT2IMG_GROUP = "prompt_anchor_group_txt2img"
ELEM_ID_IMG2IMG_GROUP = "prompt_anchor_group_img2img"

# Infotext key. Quoted so multi-line / comma-laden values survive the
# AUTOMATIC1111 "key: value, key: value" infotext parser.
INFOTEXT_KEY = "Anchor prompt"


def _default_separator() -> str:
    """The string slotted between the anchor text and the main prompt
    when they're joined. Default ', ' matches the comma-separated tag
    style most commonly used; user can override per-tab.

    We deliberately do NOT auto-add a separator when the anchor text
    already ends with a comma (or the main prompt starts with one), to
    avoid producing duplicate commas in joined output. See `_join()`."""
    return ", "


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
    webui's Script framework. We use `is_img2img` to pick distinct
    elem_ids so the two tabs don't share gradio components (they
    wouldn't anyway because they're in different Blocks, but the
    distinct elem_ids matter for our JS / localStorage)."""

    def title(self):
        return "Prompt Anchor"

    def show(self, is_img2img):
        # AlwaysVisible -- we want this loaded on every generation, not
        # opt-in from the script dropdown.
        return scripts.AlwaysVisible

    # ---- UI ----------------------------------------------------------------

    def ui(self, is_img2img):
        """Build the per-tab UI. Returns the list of components whose
        values will be passed (in order) to `process()` as *args.

        Layout note: we wrap everything in a gr.Group with a fixed
        elem_id so the JS shim can find and reparent the whole block in
        one move. Inside the group we use a thin gr.Row holding the
        checkbox + a label, with the textarea on its own line below --
        this matches the visual rhythm of the main prompt area (prompt
        textarea + small controls strip)."""
        group_id = ELEM_ID_IMG2IMG_GROUP if is_img2img else ELEM_ID_TXT2IMG_GROUP
        box_id = ELEM_ID_IMG2IMG if is_img2img else ELEM_ID_TXT2IMG
        toggle_id = ELEM_ID_IMG2IMG_ENABLE if is_img2img else ELEM_ID_TXT2IMG_ENABLE

        with gr.Group(elem_id=group_id, elem_classes=["prompt-anchor-group"]):
            with gr.Row(elem_classes=["prompt-anchor-header"]):
                enabled = gr.Checkbox(
                    label="Anchor prompt (prepended to positive prompt at generation)",
                    value=True,
                    elem_id=toggle_id,
                    elem_classes=["prompt-anchor-enabled"],
                    container=False,
                    scale=1,
                )
                # Small ghost button used by the JS to toggle the
                # collapsed/expanded visual state. Defined here so it
                # gets a stable elem_id; the click handler is purely
                # JS-side so we register no .click() listener.
                collapse_btn = gr.Button(
                    value="−",
                    elem_id=f"{group_id}_collapse",
                    elem_classes=["prompt-anchor-collapse"],
                    size="sm",
                    scale=0,
                    min_width=32,
                )

            # IMPORTANT: include "prompt" in elem_classes. The webui's
            # edit-attention.js (Ctrl+Up/Down weighting), the negative-
            # prompt detection, and most popular tag-autocomplete
            # extensions identify "real" prompt textareas by the
            # presence of the .prompt CSS class on the gradio wrapper.
            # Adding it here gets us all of that behaviour for free
            # without having to depend on internal selectors that might
            # change between webui versions. Our distinctive elem_id is
            # what protects us from extensions that target the MAIN
            # txt2img prompt by elem_id.
            anchor = gr.Textbox(
                label="",
                show_label=False,
                lines=2,
                max_lines=20,
                placeholder=(
                    "Anchor text gets prepended to your positive prompt at "
                    "generation time. Use this for styles, artists, quality "
                    "tags -- anything you want to keep set-and-forget."
                ),
                elem_id=box_id,
                elem_classes=["prompt", "prompt-anchor-textbox"],
                interactive=True,
                container=False,
            )

            # Forge Neo collects every always-on script's infotext fields
            # after ui() returns and folds them into the tab's canonical
            # paste-field list. Register here instead of calling
            # infotext_utils.add_paste_fields() from on_after_component:
            # add_paste_fields replaces the whole list and, at this point in
            # Neo's UI build, the host later replaces it again. The old hook
            # therefore either lost this field or risked clobbering native
            # paste targets depending on build order.
            self.infotext_fields = [(anchor, _decode_infotext_anchor)]

            # Separator lives in an accordion to keep the visible UI tight
            # for the 99% case where the default ", " is fine.
            with gr.Accordion(
                "Advanced", open=False,
                elem_classes=["prompt-anchor-advanced"],
            ):
                separator = gr.Textbox(
                    label="Separator between anchor and main prompt",
                    value=_default_separator(),
                    lines=1,
                    info=(
                        "Inserted between the anchor and the main prompt. "
                        "Default ', ' suits comma-separated tag prompts. "
                        "Trailing commas/whitespace on the anchor and "
                        "leading commas on the main prompt are stripped "
                        "automatically to prevent doubled separators."
                    ),
                    elem_classes=["prompt-anchor-separator"],
                )

        # IMPORTANT: collapse_btn is included here so Gradio keeps it as
        # part of the components list (otherwise an unused button can be
        # GC'd in some setups). Its value is discarded in process(); the
        # `*_` swallows any future additions too.
        return [enabled, anchor, separator, collapse_btn]

    # ---- Generation hooks --------------------------------------------------

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
            if hasattr(p, "prompt"):
                p.prompt = _join(text, p.prompt or "", sep)

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

            # Hires-fix has its own separate prompt fields (`hr_prompt`
            # and `all_hr_prompts`). When the user enables hires fix
            # without specifying a separate hires prompt, the webui
            # fills these with copies of the main prompt -- BUT it does
            # that copy in `setup_prompts()` which runs AFTER our
            # process(). So we need to handle two cases:
            #
            #   (a) `hr_prompt` already differs from `prompt` -- the
            #       user wrote a distinct hires prompt. Prepend our
            #       anchor to that one too, so the hires pass also gets
            #       the styles.
            #   (b) `hr_prompt` is empty or matches the original
            #       `prompt` -- it'll be auto-filled later from
            #       `p.prompt`, which we've already prepended into. Do
            #       NOTHING here; otherwise we'd double-prepend.
            #
            # We detect (a) by checking the hr_prompt against the
            # ORIGINAL prompt (the one before we joined). If they're
            # different and hr_prompt is non-empty, the user customised
            # the hires prompt and we should prepend separately.
            hr_prompt = getattr(p, "hr_prompt", None)
            # Stash a reference to the pre-joined prompt for the
            # comparison. (We've already overwritten p.prompt above.)
            # The simplest check: if hr_prompt is non-empty AND doesn't
            # start with our anchor text, it must be a user-authored
            # distinct hires prompt -- prepend.
            if hr_prompt and not hr_prompt.startswith(text):
                p.hr_prompt = _join(text, hr_prompt, sep)
                all_hr = getattr(p, "all_hr_prompts", None)
                if all_hr:
                    p.all_hr_prompts = [
                        _join(text, hp or "", sep) if hp and not hp.startswith(text) else hp
                        for hp in all_hr
                    ]

            # Record in infotext so the value survives into PNG metadata
            # and "send to txt2img" round-trips. Wrap in double quotes
            # so commas / newlines don't fragment the infotext parser.
            # Newlines inside the value are encoded as the literal two
            # characters '\n' (matching how A1111 encodes them); the
            # paste-fields code on the receiving end will decode them.
            extra = getattr(p, "extra_generation_params", None)
            if extra is not None:
                escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                extra[INFOTEXT_KEY] = f'"{escaped}"'

        except Exception:
            # Never crash a generation just because the anchor failed.
            # Print to console and let the user generate without us.
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Infotext round-trip: register paste fields so "Send to txt2img" buttons
# repopulate the anchor box from PNG metadata.
# ---------------------------------------------------------------------------
#
# When the user clicks "Send to txt2img" on a PNG-info image (or any of the
# other paste-into-tab buttons), the webui parses the infotext and walks a
# list of (component, infotext_key) pairs registered via
# `infotext_utils.add_paste_fields("txt2img", ...)`. We register our anchor
# textbox against the same key we write in process(), so the value comes
# back.
#
# We also need to handle DECODING -- the value we wrote is wrapped in
# double quotes with escape sequences, and the standard parser hands us
# the raw text between the colon and the next comma. So we register a
# *callable* paste field that does the unescape.

def _decode_infotext_anchor(infotext_dict):
    """Given the parsed infotext (a dict of key -> raw string value),
    return the cleaned anchor text. Reverses the encoding done in
    process(): strips wrapping quotes, decodes backslash escapes.

    Returns gr.update() with no `value=` (i.e. leaves the field
    unchanged) if no anchor info is present in this infotext -- avoids
    wiping a manually-typed anchor when the user pastes parameters from
    an image that wasn't generated with this extension.

    Signature note: webui's infotext_utils invokes the callable form
    of a paste field with a SINGLE argument (the params dict), not two.
    Adding a second arg breaks the paste flow with a TypeError that's
    silently swallowed -- so the field just never updates, looking
    like a bug in our code.
    """
    raw = infotext_dict.get(INFOTEXT_KEY)
    if raw is None:
        return gr.update()
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = raw[1:-1]
    decoded = raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    return decoded


# ---------------------------------------------------------------------------
# Settings page entry (optional - lets the user change the default
# separator globally; per-tab override still wins).
# ---------------------------------------------------------------------------

def _on_ui_settings():
    """Adds a section to Settings -> Prompt Anchor with a single field
    for the default separator. Stored under shared.opts and read by the
    JS shim to seed the per-tab separator field's default value on
    first load."""
    try:
        section = ("prompt_anchor", "Prompt Anchor")
        shared.opts.add_option(
            "prompt_anchor_default_separator",
            shared.OptionInfo(
                _default_separator(),
                "Default separator between anchor and main prompt",
                gr.Textbox,
                {"interactive": True},
                section=section,
            ),
        )
        shared.opts.add_option(
            "prompt_anchor_collapsed_by_default",
            shared.OptionInfo(
                False,
                "Start the anchor box collapsed on page load",
                gr.Checkbox,
                section=section,
            ),
        )
    except Exception:
        traceback.print_exc()


script_callbacks.on_ui_settings(_on_ui_settings)
