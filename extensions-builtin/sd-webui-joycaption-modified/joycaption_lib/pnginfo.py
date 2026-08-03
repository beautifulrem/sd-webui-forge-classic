"""
PNG-info helpers for the JoyCaption bridge features.

Reads WebUI-style generation parameters out of an image (PNG `parameters`
text chunk or JPEG/WEBP EXIF UserComment) and extracts just the *positive
prompt* portion -- which is what we want to prepend to the JoyCaption
output before sending it back to txt2img.

The format we parse is the standard AUTOMATIC1111 / Forge layout:

    <positive prompt lines...>
    Negative prompt: <negative prompt lines...>
    Steps: 20, Sampler: Euler a, CFG scale: 7, ...

We don't try to compete with the WebUI's own parser -- if anything fancier
is needed (LoRA hashes, hires-fix params, etc.) the user can still use the
PNG info tab. We just need the positive prompt as plain text.
"""

from __future__ import annotations

from typing import Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is always present in forge
    Image = None  # type: ignore


def _extract_parameters_text(image) -> str:
    """Pull the raw 'parameters' string out of a PIL image, if present.

    Looks at:
      - PNG text chunks (image.info['parameters'])
      - JPEG/WEBP EXIF UserComment (image.info['exif'] - UNICODE-prefixed)

    Returns empty string if nothing found or the image can't be inspected.
    """
    if image is None or Image is None:
        return ""

    # Most webui-generated images: 'parameters' lives in image.info
    info = getattr(image, "info", None) or {}

    text = info.get("parameters") or ""
    if text:
        return text

    # Some images store it as 'Description' or 'comment'
    for key in ("Description", "description", "comment", "Comment"):
        v = info.get(key)
        if v:
            return v if isinstance(v, str) else v.decode("utf-8", errors="ignore")

    # JPEG EXIF UserComment fallback
    exif = info.get("exif")
    if exif and isinstance(exif, (bytes, bytearray)):
        try:
            # UserComment tag is 0x9286; PIL exposes it via getexif().get(37510)
            user_comment = image.getexif().get(37510)  # type: ignore[attr-defined]
            if user_comment:
                if isinstance(user_comment, bytes):
                    # Strip the 8-byte character code prefix (e.g. b'UNICODE\x00')
                    if user_comment.startswith(b"UNICODE\x00"):
                        return user_comment[8:].decode("utf-16-be", errors="ignore")
                    if user_comment.startswith(b"ASCII\x00\x00\x00"):
                        return user_comment[8:].decode("ascii", errors="ignore")
                    return user_comment.decode("utf-8", errors="ignore")
                return str(user_comment)
        except Exception:
            pass

    return ""


def extract_positive_prompt_from_text(raw: str) -> str:
    """Extract the positive prompt portion of an AUTOMATIC1111-format
    parameters string.

    Same parsing rules as extract_positive_prompt() but on a string the
    caller has already pulled out of somewhere (e.g. the PNG-info tab's
    visible plaintext textbox, or a clipboard paste).

    Returns an empty string for input that doesn't look like webui
    metadata at all.
    """
    if not raw:
        return ""

    # AUTOMATIC1111 format: positive prompt is everything BEFORE the first
    # line that starts with 'Negative prompt:'. If there's no Negative
    # line, the positive prompt is everything except the trailing params
    # line (the one that contains 'Steps: ...'). We use the same
    # heuristic as parse_generation_parameters: the last line is treated
    # as the params line if it has at least 3 'Key: Value' segments.
    lines = raw.strip().split("\n")
    positive_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Negative prompt:"):
            break
        positive_lines.append(line)

    # If we never hit a Negative line, drop the last line if it looks
    # like the params summary (e.g. "Steps: 20, Sampler: ..."). Cheap
    # detection: contains 'Steps:' and at least one comma.
    if positive_lines and len(positive_lines) > 1:
        last = positive_lines[-1].strip()
        if "Steps:" in last and "," in last:
            positive_lines = positive_lines[:-1]

    return "\n".join(positive_lines).strip()


def extract_positive_prompt(image) -> str:
    """Extract just the positive prompt from a WebUI-tagged image.

    Returns an empty string if the image has no recognisable metadata.
    """
    return extract_positive_prompt_from_text(_extract_parameters_text(image))


def extract_positive_prompt_from_path(path) -> str:
    """Convenience: open a file path and extract the positive prompt."""
    if Image is None or not path:
        return ""
    try:
        with Image.open(path) as im:
            im.load()
            return extract_positive_prompt(im)
    except Exception:
        return ""
