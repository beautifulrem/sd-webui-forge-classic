r"""
Prompt Blacklist — a Forge Neo / A1111-style extension.

Adds a collapsible "Prompt Blacklist" panel to both the txt2img and img2img
tabs. Paste a booru-style prompt into the positive prompt box, press
"Clean Prompt", and every tag that matches an entry in your blacklist is
removed. Nothing runs automatically — cleaning only happens on button press,
and only the positive prompt is ever touched.

Blacklist entries are comma- or newline-separated. Matching is:
  * case-insensitive
  * underscore/space agnostic  (blue_hair == blue hair)
  * weight/emphasis agnostic   ((blue hair:1.2), ((blue hair)) all match "blue hair")
  * escaped-paren agnostic     (ganyu \(genshin impact\) matches ganyu (genshin impact))
  * wildcard-capable           ("*_hair" removes blue_hair, long hair, etc.)

The blacklist is persisted under Forge's user-data directory.
"""

import os
import re
import fnmatch
import shutil

import gradio as gr

from modules import paths_internal, scripts, script_callbacks


# --------------------------------------------------------------------------- #
#  Persistence                                                                 #
# --------------------------------------------------------------------------- #

EXT_DIR = scripts.basedir()
STATE_DIR = os.path.join(
    paths_internal.data_path, "extension-data", "sd-forge-prompt-blacklist"
)
BLACKLIST_FILE = os.path.join(STATE_DIR, "blacklist.txt")
LEGACY_BLACKLIST_FILE = os.path.join(EXT_DIR, "blacklist.txt")


def _migrate_legacy_blacklist() -> None:
    if os.path.exists(BLACKLIST_FILE) or not os.path.isfile(LEGACY_BLACKLIST_FILE):
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        shutil.copy2(LEGACY_BLACKLIST_FILE, BLACKLIST_FILE)
    except OSError as e:
        print(f"[Prompt Blacklist] failed to migrate legacy blacklist.txt: {e}")


_migrate_legacy_blacklist()


def load_blacklist() -> str:
    path = BLACKLIST_FILE if os.path.isfile(BLACKLIST_FILE) else LEGACY_BLACKLIST_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def save_blacklist(text: str) -> str:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
            f.write((text or "").strip() + "\n")
        return "✅ Blacklist saved."
    except OSError as e:
        return f"⚠️ Could not save blacklist: {e}"


def reload_blacklist():
    return load_blacklist(), "🔄 Blacklist reloaded from file."


# --------------------------------------------------------------------------- #
#  Tag normalisation & cleaning                                                #
# --------------------------------------------------------------------------- #

# strips emphasis wrappers and trailing :weight from a token, e.g.
#   "((blue hair:1.2))" -> "blue hair"
_ESCAPED_OPEN = "\x00"
_ESCAPED_CLOSE = "\x01"

_WEIGHT_RE = re.compile(r":\s*[\d.]+\s*$")


def _strip_emphasis(token: str) -> str:
    """Remove surrounding ()/[]/<> emphasis and any trailing :weight."""
    # protect escaped parens (literal booru parens like `ganyu \(genshin impact\)`)
    s = token.replace(r"\(", _ESCAPED_OPEN).replace(r"\)", _ESCAPED_CLOSE).strip()

    changed = True
    while changed:
        changed = False
        s = s.strip()
        # trailing :1.2 style weight (possibly just before a closing paren)
        new = _WEIGHT_RE.sub("", s)
        if new != s:
            s, changed = new, True
        # matched wrapper pairs
        for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
            if len(s) >= 2 and s.startswith(open_c) and s.endswith(close_c):
                s, changed = s[1:-1], True

    return s.replace(_ESCAPED_OPEN, "(").replace(_ESCAPED_CLOSE, ")").strip()


def _normalize(tag: str) -> str:
    """Canonical form used for comparison."""
    s = _strip_emphasis(tag).lower()
    s = s.replace("\\", "")             # drop escape backslashes
    s = re.sub(r"[()\[\]{}]", " ", s)   # brackets are irrelevant for comparison
    s = s.replace("_", " ")             # underscore == space
    s = re.sub(r"\s+", " ", s)          # collapse whitespace
    return s.strip()


def _parse_blacklist(text: str):
    """Blacklist entries, normalised. Entries may contain * wildcards."""
    entries = []
    for chunk in re.split(r"[,\n]", text or ""):
        norm = _normalize(chunk)
        if norm:
            entries.append(norm)
    return entries


def _is_blacklisted(norm_tag: str, entries) -> bool:
    for entry in entries:
        if "*" in entry or "?" in entry:
            if fnmatch.fnmatchcase(norm_tag, entry):
                return True
        elif norm_tag == entry:
            return True
    return False


def clean_prompt(prompt: str, blacklist_text: str, remove_dupes: bool):
    """Remove blacklisted tags from the positive prompt. Returns (prompt, status)."""
    entries = _parse_blacklist(blacklist_text)
    if not entries and not remove_dupes:
        return prompt, "ℹ️ Blacklist is empty — nothing to remove."

    removed, seen = [], set()
    out_lines = []

    for line in (prompt or "").split("\n"):
        kept = []
        for raw in line.split(","):
            tag = raw.strip()
            if not tag:
                continue

            norm = _normalize(tag)

            # never touch structural keywords
            if norm in ("break", "and"):
                kept.append(tag)
                continue

            if _is_blacklisted(norm, entries):
                removed.append(tag)
                continue

            if remove_dupes:
                if norm in seen:
                    removed.append(f"{tag} (duplicate)")
                    continue
                seen.add(norm)

            kept.append(tag)
        out_lines.append(", ".join(kept))

    new_prompt = "\n".join(out_lines).strip()

    if removed:
        shown = ", ".join(f"`{t}`" for t in removed[:25])
        extra = f" … and {len(removed) - 25} more" if len(removed) > 25 else ""
        status = f"🧹 Removed **{len(removed)}** tag(s): {shown}{extra}"
    else:
        status = "✨ No blacklisted tags found — prompt unchanged."

    return new_prompt, status


# --------------------------------------------------------------------------- #
#  Grab the positive-prompt textboxes as the UI is built                       #
# --------------------------------------------------------------------------- #

PROMPT_BOXES = {}


def _on_after_component(component, **kwargs):
    elem_id = kwargs.get("elem_id")
    if elem_id in ("txt2img_prompt", "img2img_prompt"):
        PROMPT_BOXES[elem_id] = component


script_callbacks.on_after_component(_on_after_component)


# --------------------------------------------------------------------------- #
#  Script / UI                                                                 #
# --------------------------------------------------------------------------- #

class PromptBlacklistScript(scripts.Script):

    section = "prompt"
    create_group = False

    def title(self):
        return "Prompt Blacklist"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        prompt_box = PROMPT_BOXES.get(f"{tab}_prompt")

        with gr.Accordion("🧹 Prompt Blacklist", open=False, elem_id=f"pbl_accordion_{tab}"):
            gr.Markdown(
                "Tags listed below are stripped from the **positive prompt** when you "
                "press *Clean Prompt*. Separate entries with commas or new lines. "
                "`*` wildcards are supported (e.g. `*_username`). Matching ignores "
                "case, underscores vs. spaces, and `(emphasis:1.2)` weighting."
            )

            blacklist_tb = gr.Textbox(
                label="Blacklisted tags",
                value=load_blacklist(),
                lines=4,
                max_lines=12,
                placeholder="e.g. watermark, signature, artist name, *_username, jpeg artifacts",
                elem_id=f"pbl_blacklist_{tab}",
            )

            with gr.Row():
                clean_btn = gr.Button("🧹 Clean Prompt", variant="primary", elem_id=f"pbl_clean_{tab}")
                save_btn = gr.Button("💾 Save Blacklist", elem_id=f"pbl_save_{tab}")
                reload_btn = gr.Button("🔄 Reload", elem_id=f"pbl_reload_{tab}")

            remove_dupes = gr.Checkbox(
                label="Also remove duplicate tags",
                value=False,
                elem_id=f"pbl_dedupe_{tab}",
            )

            status_md = gr.Markdown("", elem_id=f"pbl_status_{tab}")

            if prompt_box is not None:
                clean_btn.click(
                    fn=clean_prompt,
                    inputs=[prompt_box, blacklist_tb, remove_dupes],
                    outputs=[prompt_box, status_md],
                    show_progress="hidden",
                )
            else:
                gr.Markdown("⚠️ Could not locate the positive prompt box — cleaning is disabled.")

            save_btn.click(
                fn=save_blacklist,
                inputs=[blacklist_tb],
                outputs=[status_md],
                show_progress="hidden",
            )

            reload_btn.click(
                fn=reload_blacklist,
                inputs=[],
                outputs=[blacklist_tb, status_md],
                show_progress="hidden",
            )

        # No components are passed to generation — this extension never runs
        # during image generation.
        return []

    def process(self, p, *args):
        return
