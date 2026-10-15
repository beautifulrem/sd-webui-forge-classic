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

from modules import paths_internal, scripts
from remi import ui


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


# Status messages: English catalog keys (localizations/zh_CN.json) whose
# {{placeholders}} the client fills after translating them.
MSG_SAVED = "Blacklist saved."
MSG_SAVE_FAILED = "Could not save blacklist: {{error}}"
MSG_SAVE_FAILED_TOAST = "Could not save the blacklist."
MSG_RELOADED = "Blacklist reloaded from file."
MSG_EMPTY = "Blacklist is empty, nothing to remove."
MSG_UNCHANGED = "No blacklisted tags found, prompt unchanged."
MSG_REMOVED = "Removed {{count}} tag(s): {{tags}}"
MSG_REMOVED_MORE = "Removed {{count}} tag(s): {{tags}} … and {{more}} more"
MSG_REMOVED_DUPES = "Removed {{count}} tag(s), {{duplicates}} of them duplicates: {{tags}}"
MSG_REMOVED_DUPES_MORE = "Removed {{count}} tag(s), {{duplicates}} of them duplicates: {{tags}} … and {{more}} more"
MESSAGES = (MSG_SAVED, MSG_SAVE_FAILED, MSG_SAVE_FAILED_TOAST, MSG_RELOADED, MSG_EMPTY, MSG_UNCHANGED, MSG_REMOVED, MSG_REMOVED_MORE, MSG_REMOVED_DUPES, MSG_REMOVED_DUPES_MORE)


def save_blacklist(text: str) -> tuple[bool, dict]:
    """Writes the blacklist file: ``(saved, status message)``."""
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(BLACKLIST_FILE, "w", encoding="utf-8") as f:
            f.write((text or "").strip() + "\n")
        return True, ui.message(MSG_SAVED)
    except OSError as e:
        return False, ui.message(MSG_SAVE_FAILED, error=str(e))


def reload_blacklist():
    return load_blacklist(), ui.message(MSG_RELOADED)


# --------------------------------------------------------------------------- #
#  Tag normalisation & cleaning                                                #
# --------------------------------------------------------------------------- #

# strips emphasis wrappers and trailing :weight from a token, e.g.
#   "((blue hair:1.2))" -> "blue hair"
_ESCAPED_OPEN = "\x00"
_ESCAPED_CLOSE = "\x01"

_WEIGHT_RE = re.compile(r":\s*-?[\d.]+\s*$")


def _strip_emphasis(token: str) -> str:
    """Remove surrounding ()/[]/{} emphasis and each group's trailing :weight.

    Only one weight belongs to each peeled group, so ``(16:9:1.1)`` keeps
    ``16:9`` and bare tags such as ``:3`` are never treated as weights.
    """
    # protect escaped parens (literal booru parens like `ganyu \(genshin impact\)`)
    s = token.replace(r"\(", _ESCAPED_OPEN).replace(r"\)", _ESCAPED_CLOSE).strip()

    weights = 0  # weights still allowed: one per peeled group level
    while True:
        s = s.strip()
        if weights:
            new = _WEIGHT_RE.sub("", s, count=1)
            if new != s:
                s, weights = new, weights - 1
                continue
        for open_c, close_c in (("(", ")"), ("[", "]"), ("{", "}")):
            if len(s) >= 2 and s.startswith(open_c) and s.endswith(close_c):
                s, weights = s[1:-1], weights + 1
                break
        else:
            break

    return s.replace(_ESCAPED_OPEN, "(").replace(_ESCAPED_CLOSE, ")").strip()


def _normalize(tag: str) -> str:
    """Canonical form used for comparison."""
    # A tag cut from the end of a weighted group ("blue eyes:1.2)") carries one
    # weight per unmatched closing bracket.
    _openers, closing = _bracket_residue(tag)
    group_ends = len(_CLOSING_SEGMENT.findall(closing))
    s = _strip_emphasis(tag).lower()
    s = s.replace("\\", "")             # drop escape backslashes
    s = re.sub(r"[()\[\]{}]", " ", s)   # brackets are irrelevant for comparison
    for _ in range(group_ends):
        s = _WEIGHT_RE.sub("", s.strip(), count=1)
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


_PAIRS = {")": "(", "]": "[", "}": "{"}
_TRAILING_WEIGHT = re.compile(r":\s*-?\d*\.?\d+\s*$")
# One closing bracket together with the weight of the group it closes.
_CLOSING_SEGMENT = re.compile(r"(:\s*-?[\d.]+\s*)?([)\]}])")


def _bracket_residue(tag: str):
    """Return (openers, closing) that removing ``tag`` must keep.

    Comma splitting cuts through emphasis groups such as
    ``(red hair, blue eyes:1.2)``; dropping ``(red hair`` must keep ``(``
    and dropping ``blue eyes:1.2)`` must keep ``:1.2)`` so the group stays
    balanced. Escaped brackets are literal text.
    """

    stack, first_unmatched_close, escaped = [], None, False
    for index, char in enumerate(tag):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char in "([{":
            stack.append(char)
        elif char in _PAIRS:
            if stack and stack[-1] == _PAIRS[char]:
                stack.pop()
            elif first_unmatched_close is None:
                first_unmatched_close = index
    closing = ""
    if first_unmatched_close is not None:
        weight = _TRAILING_WEIGHT.search(tag[:first_unmatched_close])
        closing = tag[weight.start() if weight else first_unmatched_close :]
    return "".join(stack), closing


def clean_prompt(prompt: str, blacklist_text: str, remove_dupes: bool):
    """Remove blacklisted tags from the positive prompt. Returns (prompt,
    status), the status a message ``{"text": catalog key, "params": ...}``."""
    entries = _parse_blacklist(blacklist_text)
    if not entries and not remove_dupes:
        return prompt, {"text": "Blacklist is empty, nothing to remove."}

    removed, seen, duplicates = [], set(), 0
    out_lines = []

    for line in (prompt or "").split("\n"):
        kept = []
        pending_openers = ""

        def drop(tag, label):
            nonlocal pending_openers
            removed.append(label)
            openers, closing = _bracket_residue(tag)
            # A group whose every tag was dropped disappears entirely, together
            # with its weight (which belongs to its first closing bracket).
            kept_closers = []
            for weight, bracket in _CLOSING_SEGMENT.findall(closing):
                if not kept_closers and pending_openers and _PAIRS[bracket] == pending_openers[-1]:
                    pending_openers = pending_openers[:-1]
                    continue
                kept_closers.append(weight + bracket)
            closing = "".join(kept_closers)
            pending_openers += openers
            if closing:
                if kept:
                    kept[-1] += closing
                else:
                    kept.append(closing)

        for raw in line.split(","):
            tag = raw.strip()
            if not tag:
                continue

            norm = _normalize(tag)

            # never touch structural keywords
            if norm in ("break", "and"):
                kept.append(pending_openers + tag)
                pending_openers = ""
                continue

            if _is_blacklisted(norm, entries):
                drop(tag, tag)
                continue

            if remove_dupes:
                if norm in seen:
                    drop(tag, tag)
                    duplicates += 1
                    continue
                seen.add(norm)

            kept.append(pending_openers + tag)
            pending_openers = ""
        if pending_openers:
            kept.append(pending_openers)
        out_lines.append(", ".join(kept))

    new_prompt = "\n".join(out_lines).strip()

    if removed:
        params = {"count": len(removed), "tags": ", ".join(removed[:25])}
        if duplicates:
            params["duplicates"] = duplicates
        if len(removed) > 25:
            params["more"] = len(removed) - 25
        text = "Removed {{count}} tag(s)" + (", {{duplicates}} of them duplicates" if duplicates else "") + ": {{tags}}" + (" … and {{more}} more" if len(removed) > 25 else "")
        status = {"text": text, "params": params}
    else:
        status = {"text": "No blacklisted tags found, prompt unchanged."}

    return new_prompt, status


# --------------------------------------------------------------------------- #
#  Script / UI                                                                 #
# --------------------------------------------------------------------------- #

INTRO = (
    "Tags listed below are stripped from the **positive prompt** when you "
    "press *Clean Prompt*. Separate entries with commas or new lines. "
    "`*` wildcards are supported (e.g. `*_username`). Matching ignores "
    "case, underscores vs. spaces, and `(emphasis:1.2)` weighting."
)


class PromptBlacklistScript(scripts.Script):

    section = "prompt"
    create_group = False

    def title(self):
        return "Prompt Blacklist"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        # The blacklist and the duplicate switch only feed the actions (this
        # extension never runs during image generation): never submitted with
        # a job nor kept in drafts, so neither a reused job nor a stale draft
        # can put an old blacklist back in the box (and a later Save write it).
        return ui.Section(
            "Prompt Blacklist",
            open=False,
            children=[
                ui.Markdown(INTRO),
                ui.Text(
                    "blacklist",
                    load_blacklist,
                    lines=4,
                    max_lines=12,
                    label="Blacklisted tags",
                    placeholder="e.g. watermark, signature, artist name, *_username, jpeg artifacts",
                    persist=False,
                    submit=False,
                ),
                ui.Row(
                    ui.Button("Clean Prompt", "clean", "primary"),
                    ui.Button("Save Blacklist", "save"),
                    ui.Button("Reload", "reload"),
                ),
                ui.Toggle("remove_dupes", False, label="Also remove duplicate tags", persist=False, submit=False),
                ui.Output("status", "text"),
            ],
        )

    @ui.action("clean")
    def clean(self, ctx, values):
        """Removes the blacklisted tags from the form's positive prompt."""

        prompt = str(ctx.core.get("prompt") or "")
        cleaned, status = clean_prompt(prompt, str(values.get("blacklist") or ""), bool(values.get("remove_dupes")))
        patch = {"__outputs__": {"status": status}}
        if cleaned != prompt:
            patch["core.prompt"] = {"value": cleaned}
        return patch

    @ui.action("save")
    def save(self, ctx, values):
        ok, status = save_blacklist(str(values.get("blacklist") or ""))
        return {"__outputs__": {"status": status}, **({} if ok else ui.toast(MSG_SAVE_FAILED_TOAST, "error"))}

    @ui.action("reload")
    def reload(self, ctx, values):
        text, status = reload_blacklist()
        return {"blacklist": {"value": text}, "__outputs__": {"status": status}}

    def process(self, p, *args):
        return
