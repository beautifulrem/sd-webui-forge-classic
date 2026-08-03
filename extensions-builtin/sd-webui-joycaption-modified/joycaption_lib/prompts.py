"""
Caption type prompt map, extra-options list, and default system prompt
for JoyCaption Beta One — tuned for captioning diffusion-model training
datasets (booru tags + natural language, SFW + NSFW).
"""


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert image captioner. Your captions are used to train an image "
    "generation model that learns from a mixture of booru-style tags and natural-language "
    "descriptions, covering both SFW and NSFW imagery.\n"
    "\n"
    "Describe only what is visibly present, using concrete, specific terms — precise "
    "anatomical and visual vocabulary, booru-style where it fits, prose where prose reads "
    "better. Front-load the subject and what they are doing.\n"
    "\n"
    "Hard rules:\n"
    "- Never mention what is absent. No \"there is no watermark\", no \"the background lacks "
    "detail\", no negative descriptions of any kind.\n"
    "- No meta-commentary about the image itself. Do not write \"this image\", \"this photo\", "
    "\"the scene depicts\", \"we see\", \"captured in\", \"frozen in\", or anything similar.\n"
    "- No mood, atmosphere, or interpretive language. Do not write \"intimate\", \"playful\", "
    "\"mysterious\", \"evokes\", \"feels\", \"conveys\".\n"
    "- No quality or aesthetic judgments. Do not call anything \"tasteful\", \"beautiful\", "
    "\"stunning\", \"provocative\", or praise the technique.\n"
    "- For NSFW content, use direct anatomical and explicit terms without euphemism or hedging.\n"
    "- Write in present-tense, declarative sentences.\n"
    "- If readable text is visible, quote it verbatim. Otherwise, do not mention text at all."
)


# ---------------------------------------------------------------------------
# Caption types
# ---------------------------------------------------------------------------

NONE_CAPTION_TYPE = "(none — system prompt only)"


CAPTION_TYPE_MAP = {
    # --- No caption-type prompt; rely entirely on system prompt + extras ---
    NONE_CAPTION_TYPE: ["", "", ""],

    # --- Kept from upstream ------------------------------------------------
    "Descriptive (Formal)": [
        "Write a long detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.",
    ],

    # --- New: tuned for explicit / training-focused captioning ------------
    "NSFW Descriptive (Explicit)": [
        "Write an explicit, detailed description of this image. Use direct anatomical and sexual vocabulary without euphemism. Cover what is happening, who is involved, body positions, points of contact, state of arousal, and any relevant fluids or restraints. Front-load the main act.",
        "Write an explicit, detailed description of this image in {word_count} words or less. Use direct anatomical and sexual vocabulary without euphemism. Cover what is happening, who is involved, body positions, points of contact, state of arousal, and any relevant fluids or restraints. Front-load the main act.",
        "Write a {length} explicit, detailed description of this image. Use direct anatomical and sexual vocabulary without euphemism. Cover what is happening, who is involved, body positions, points of contact, state of arousal, and any relevant fluids or restraints. Front-load the main act.",
    ],
    "Booru Tags + Natural Language": [
        "Output booru-style general tags for this image (comma-separated, lowercase_underscores, no prefixes, no artist/copyright/character tags), then a blank line, then a natural-language description of the same image.",
        "Output booru-style general tags for this image (comma-separated, lowercase_underscores, no prefixes, no artist/copyright/character tags), then a blank line, then a natural-language description in {word_count} words or less.",
        "Output booru-style general tags for this image (comma-separated, lowercase_underscores, no prefixes, no artist/copyright/character tags), then a blank line, then a {length} natural-language description.",
    ],
    "Action-Focused": [
        "Describe what is happening in this image. Focus on actions, body positions, points of physical contact, and any motion implied. State what each visible figure is doing in concrete terms. Skip descriptions of static elements unless they are involved in the action.",
        "Describe what is happening in this image in {word_count} words or less. Focus on actions, body positions, points of physical contact, and any motion implied. State what each visible figure is doing in concrete terms. Skip descriptions of static elements unless they are involved in the action.",
        "Write a {length} description of what is happening in this image. Focus on actions, body positions, points of physical contact, and any motion implied. State what each visible figure is doing in concrete terms. Skip descriptions of static elements unless they are involved in the action.",
    ],
    "Character-Focused": [
        "Describe the character(s) in this image in detail. Cover species (if applicable), body type, hair, eyes, distinguishing features, clothing, accessories, and current pose and expression. Treat the character as the subject; mention the setting only briefly if at all.",
        "Describe the character(s) in this image in {word_count} words or less. Cover species (if applicable), body type, hair, eyes, distinguishing features, clothing, accessories, and current pose and expression. Treat the character as the subject; mention the setting only briefly if at all.",
        "Write a {length} description of the character(s) in this image. Cover species (if applicable), body type, hair, eyes, distinguishing features, clothing, accessories, and current pose and expression. Treat the character as the subject; mention the setting only briefly if at all.",
    ],
    "Anthro / Furry Descriptive": [
        "Describe this anthropomorphic / furry image in detail. Identify each character's species (or hybrid), fur / scale / feather color and pattern, body type (anthro, feral, or taur), species-specific features (muzzle, ears, tail, paw pads, claws, sheath, etc.), clothing or lack thereof, and current action or pose. Use precise furry-fandom terminology where it is standard.",
        "Describe this anthropomorphic / furry image in {word_count} words or less. Identify each character's species (or hybrid), fur / scale / feather color and pattern, body type (anthro, feral, or taur), species-specific features (muzzle, ears, tail, paw pads, claws, sheath, etc.), clothing or lack thereof, and current action or pose. Use precise furry-fandom terminology where it is standard.",
        "Write a {length} description of this anthropomorphic / furry image. Identify each character's species (or hybrid), fur / scale / feather color and pattern, body type (anthro, feral, or taur), species-specific features (muzzle, ears, tail, paw pads, claws, sheath, etc.), clothing or lack thereof, and current action or pose. Use precise furry-fandom terminology where it is standard.",
    ],

    # --- Pure tag-list modes (kept from upstream) -------------------------
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-Like Tag List": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Booru Tags (general only, no metadata)": [
        "Generate only comma-separated, lowercase_underscores booru-style general tags for this image. Cover subject counts (1girl, 1boy, 2girls, etc.), species if anthro, body features, clothing, accessories, pose, action, expression, setting, and any explicit content. Do NOT include artist, character, copyright, or meta tags. Do NOT include any prose or extra text.",
        "Generate only comma-separated, lowercase_underscores booru-style general tags for this image, {word_count} tags or less. Cover subject counts (1girl, 1boy, 2girls, etc.), species if anthro, body features, clothing, accessories, pose, action, expression, setting, and any explicit content. Do NOT include artist, character, copyright, or meta tags. Do NOT include any prose or extra text.",
        "Generate a {length} list of only comma-separated, lowercase_underscores booru-style general tags for this image. Cover subject counts (1girl, 1boy, 2girls, etc.), species if anthro, body features, clothing, accessories, pose, action, expression, setting, and any explicit content. Do NOT include artist, character, copyright, or meta tags. Do NOT include any prose or extra text.",
    ],
}


# ---------------------------------------------------------------------------
# Extra options — tripled (27 → 81) and ordered thematically so scrolling
# the list top-to-bottom feels coherent:
#   body & anatomy → NSFW specifics → anthro/furry → clothing → pose &
#   camera → setting → style/medium → voice/tone → prose rules → things
#   to omit.
# ---------------------------------------------------------------------------

EXTRA_OPTIONS = [
    # ─── Body & anatomy (neutral) ───
    "Include information about the ages of any people/characters when applicable.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    "Describe body type with specific terms (slim, athletic, curvy, chubby, muscular, etc.).",
    "Describe skin tone, complexion, and any visible texture (smooth, freckled, tanned, pale).",
    "Note any visible body hair where present.",
    "Mention apparent height or scale relative to other elements or characters.",
    "Describe any visible tattoos, piercings, scars, or other markings.",

    # ─── Anatomy & NSFW specifics ───
    "Note the degree of nudity (fully clothed, partial, topless, bottomless, fully nude).",
    "Explicitly describe any visible genitalia in anatomical detail.",
    "Describe breast size, shape, and any nipple visibility precisely.",
    "Describe the level of arousal visible (flushed skin, erect nipples, erection, etc.).",
    "If sexual activity is occurring, describe the specific act and position in detail.",
    "Note if the scene depicts oral, vaginal, anal, or other specific sexual contact.",
    "Describe penetration depth, point of contact, and what is being penetrated by what.",
    "Describe any visible bodily fluids (sweat, saliva, semen, etc.) explicitly.",
    "If applicable, describe any sex toys, restraints, or accessories in detail.",

    # ─── Anthro / furry ───
    "Identify the species (or hybrid species) of any anthropomorphic character.",
    "Describe fur color, pattern, length, and any markings.",
    "Describe the muzzle, ears, tail, claws, and any species-specific features.",
    "Distinguish between anthropomorphic (humanoid with animal features), feral, or taur body types.",
    "Note any visible paw pads, scales, feathers, or non-fur surface details.",
    "Describe the sexual dimorphism visible (e.g. sheath, knot, slit, etc. where applicable).",

    # ─── Clothing / styling ───
    "Describe each garment in detail, including color, material, fit, and condition.",
    "Note any specific clothing style, era, or subculture (gothic, casual, formal, fetish, etc.).",
    "Mention specific accessories (collars, gloves, stockings, jewelry, eyewear, etc.).",
    "If clothing is being removed, displaced, or sheer/see-through, describe how.",
    "If hair is styled, describe the style precisely (twin tails, braid, undercut, etc.) along with color.",
    "Note makeup, nail polish, or other cosmetic details.",

    # ─── Pose, action, expression, camera ───
    "Describe the exact pose using specific body-position language.",
    "Describe the position and gesture of each visible hand.",
    "Describe the position and gesture of each visible foot.",
    "Describe head tilt, gaze direction, and eye contact with the viewer.",
    "Note any movement implied (mid-jump, mid-stroke, falling, walking, etc.).",
    "Describe facial expression in concrete terms (e.g. \"mouth open, eyes half-closed\" rather than \"happy\").",
    "If multiple subjects are present, describe how they are positioned relative to each other and any physical contact between them.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "Explicitly specify the vantage height (eye-level, low-angle worm's-eye, bird's-eye, drone, rooftop, etc.).",
    "Include information about camera angle.",

    # ─── Setting & environment ───
    "Describe the location specifically (bedroom, classroom, bathroom, outdoors, etc.) and any defining objects.",
    "Describe time of day or implied time (morning light, night, golden hour, indoor lighting).",
    "Note the apparent weather or environmental conditions if outdoors.",
    "If the setting contains text, signs, or labels, transcribe them exactly.",
    "Describe any background characters, ambient activity, or context.",
    "Include information about lighting.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",

    # ─── Style & medium ───
    "Identify the art style (photorealistic, anime, semi-realistic, painterly, 3D render, etc.).",
    "Note any specific medium (digital painting, watercolor, oil, pencil sketch, screen-print, etc.).",
    "Describe linework quality (clean lineart, sketchy, no lines/painted, etc.).",
    "Describe color treatment (vivid, muted, monochrome, sepia, high-contrast, pastel).",
    "Note any obvious artistic influences or genre conventions visible.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Include information about whether there are JPEG artifacts or not.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",

    # ─── Voice / tone / subject reference ───
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    'Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.',
    "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
    "Do NOT include anything sexual; keep it PG.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "If it is a work of art, do not include the artist's name or the title of the work.",
    "ONLY describe the most important elements of the image.",

    # ─── Prose-style controls ───
    "Use present tense throughout.",
    "Begin the caption with the main subject and action, not the setting.",
    "Use sentence fragments where appropriate for booru-tag-like compactness.",
    "Front-load the most important visual details in the first sentence.",
    "Keep the description to a single flowing paragraph without line breaks.",
    "Avoid restating the same visual fact in multiple ways.",

    # ─── Things to omit (reinforces system prompt) ───
    "Do NOT speculate about what the subject is thinking, feeling, or intending.",
    'Do NOT describe the scene as "tasteful," "artistic," "provocative," or with any value judgment.',
    'Do NOT use phrases like "captured in a moment," "frozen in time," or other photography clichés.',
    "Do NOT describe what is NOT in the image (absences, negative space mentions).",
    'Do NOT use the words "image," "photo," "picture," "scene," or "depicts" to refer to the image itself.',
    "Do not mention the mood/feeling/etc of the image.",
    "Do NOT mention the image's resolution.",
    "Do NOT mention any text that is in the image.",
    "Do NOT use any ambiguous language.",
    "Include information about whether there is a watermark or not.",
    "If there is a watermark, you must mention it.",
    'Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.',
]


CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium", "long", "very long"] + [str(i) for i in range(20, 261, 10)]


# ---------------------------------------------------------------------------
# User-defined caption types — persisted to JSON next to the model weights
# ---------------------------------------------------------------------------

import json
from pathlib import Path


def _user_types_path() -> Path:
    try:
        from modules import paths_internal
        base = Path(paths_internal.models_path)
    except Exception:
        base = Path(__file__).resolve().parents[3] / "models"
    return base / "joycaption" / "user_caption_types.json"


def load_user_caption_types() -> dict:
    p = _user_types_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Normalise: each entry must be a 3-list of strings
        out = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) == 3 and all(isinstance(s, str) for s in v):
                out[k] = list(v)
        return out
    except Exception as e:
        print(f"[joycaption] Failed to load user caption types: {e}")
        return {}


def save_user_caption_types(d: dict) -> None:
    p = _user_types_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def all_caption_types() -> dict:
    """Built-in caption types merged with user-saved ones (user wins on name clash)."""
    return {**CAPTION_TYPE_MAP, **load_user_caption_types()}


def build_prompt(caption_type: str, caption_length: str, extra_options: list, name_input: str) -> str:
    """Build the user-turn prompt the same way the official Space does."""
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2

    types = all_caption_types()
    if caption_type not in types:
        # Unknown / deleted — fall back to no-prompt mode
        prompt = ""
    else:
        prompt = types[caption_type][map_idx]

    if extra_options:
        extras = " ".join(extra_options)
        prompt = (prompt + " " + extras).strip() if prompt else extras
    return prompt.format(
        name=(name_input or "{NAME}"),
        length=caption_length,
        word_count=caption_length,
    )
