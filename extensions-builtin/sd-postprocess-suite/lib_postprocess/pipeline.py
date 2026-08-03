"""
Post Processing Suite - pipeline definition.

A declarative spec drives BOTH the Gradio UI and the runtime. Each stage owns:
  - a stable key, a label, a default pipeline order
  - a list of parameter specs (id, widget type, default, widget kwargs)
  - an `effect` id used by the dispatcher

Per-stage the UI emits [enabled, order, *params]. The pipeline collects enabled
stages, sorts them by their (user-editable) order, and applies each in turn.
This is what makes the whole pipeline freely reorderable.
"""

from __future__ import annotations

import json
import numpy as np

from . import effects as E


def hex2rgb(h):
    if isinstance(h, (list, tuple)) and len(h) == 3:
        return tuple(float(x) for x in h)
    if not isinstance(h, str):
        return (0.0, 0.0, 0.0)
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---- parameter shorthand -------------------------------------------------
def sld(pid, default, mn, mx, step, label):
    return {"id": pid, "type": "slider", "default": default,
            "kw": {"minimum": mn, "maximum": mx, "step": step, "label": label}}


def num(pid, default, label, step=1):
    return {"id": pid, "type": "number", "default": default,
            "kw": {"label": label, "precision": 0, "step": step}}


def chk(pid, default, label):
    return {"id": pid, "type": "checkbox", "default": default, "kw": {"label": label}}


def col(pid, default, label):
    return {"id": pid, "type": "color", "default": default, "kw": {"label": label}}


def dd(pid, default, choices, label):
    return {"id": pid, "type": "dropdown", "default": default,
            "kw": {"choices": choices, "label": label}}


def fil(pid, label):
    return {"id": pid, "type": "file", "default": None,
            "kw": {"label": label, "file_types": [".cube"]}}


# ---- the stage spec ------------------------------------------------------
# order values are spaced so users can slot stages between them.
STAGES = [
    {"key": "descreen", "label": "Descreen / De-halftone (FFT)", "order": 5, "effect": "descreen",
     "params": [
         dd("mode", "luma", ["luma", "rgb"], "Mode (luma = color-safe)"),
         sld("strength", 0.9, 0.0, 1.0, 0.01, "Strength"),
         sld("protect", 0.04, 0.0, 0.3, 0.005, "Protect low-freq radius"),
         sld("threshold", 2.5, 1.0, 8.0, 0.1, "Peak threshold (std devs)"),
         num("notch_radius", 3, "Notch radius (px)"),
     ]},
    {"key": "lens_distortion", "label": "Lens Distortion", "order": 10, "effect": "lens_distortion",
     "params": [sld("k", 0.0, -0.6, 0.6, 0.01, "Distortion (− pincushion / + barrel)")]},
    {"key": "tone", "label": "Tone (exposure / contrast / levels)", "order": 20, "effect": "tone",
     "params": [
         sld("exposure", 0.0, -3.0, 3.0, 0.05, "Exposure (EV)"),
         sld("brightness", 0.0, -0.5, 0.5, 0.01, "Brightness"),
         sld("contrast", 1.0, 0.0, 2.5, 0.01, "Contrast"),
         sld("gamma", 1.0, 0.2, 3.0, 0.01, "Gamma"),
         sld("black", 0.0, 0.0, 0.5, 0.005, "Black point"),
         sld("white", 1.0, 0.5, 1.0, 0.005, "White point"),
         sld("highlights", 0.0, -0.5, 0.5, 0.01, "Highlights"),
         sld("shadows", 0.0, -0.5, 0.5, 0.01, "Shadows"),
     ]},
    {"key": "color", "label": "Color (saturation / hue / temp)", "order": 30, "effect": "color",
     "params": [
         sld("saturation", 1.0, 0.0, 3.0, 0.01, "Saturation"),
         sld("vibrance", 0.0, -1.0, 1.0, 0.01, "Vibrance"),
         sld("hue", 0.0, -180.0, 180.0, 1.0, "Hue shift (deg)"),
         sld("temperature", 0.0, -0.3, 0.3, 0.005, "Temperature (cool/warm)"),
         sld("tint", 0.0, -0.3, 0.3, 0.005, "Tint (green/magenta)"),
     ]},
    {"key": "color_balance", "label": "Color Balance (3-way)", "order": 35, "effect": "color_balance",
     "params": [
         sld("sh_r", 0.0, -0.3, 0.3, 0.005, "Shadows R"),
         sld("sh_g", 0.0, -0.3, 0.3, 0.005, "Shadows G"),
         sld("sh_b", 0.0, -0.3, 0.3, 0.005, "Shadows B"),
         sld("mid_r", 0.0, -0.3, 0.3, 0.005, "Midtones R"),
         sld("mid_g", 0.0, -0.3, 0.3, 0.005, "Midtones G"),
         sld("mid_b", 0.0, -0.3, 0.3, 0.005, "Midtones B"),
         sld("hi_r", 0.0, -0.3, 0.3, 0.005, "Highlights R"),
         sld("hi_g", 0.0, -0.3, 0.3, 0.005, "Highlights G"),
         sld("hi_b", 0.0, -0.3, 0.3, 0.005, "Highlights B"),
     ]},
    {"key": "split_tone", "label": "Split Toning", "order": 40, "effect": "split_tone",
     "params": [
         col("shadow_color", "#334d99", "Shadow color"),
         col("highlight_color", "#e6b34d", "Highlight color"),
         sld("balance", 0.0, -1.0, 1.0, 0.01, "Balance"),
         sld("strength", 0.0, 0.0, 1.0, 0.01, "Strength"),
     ]},
    {"key": "lut", "label": "Color LUT (.cube)", "order": 45, "effect": "lut",
     "params": [fil("lut_file", "3D LUT (.cube)"),
                sld("strength", 1.0, 0.0, 1.0, 0.01, "Strength")]},
    {"key": "blur", "label": "Blur", "order": 50, "effect": "blur",
     "params": [dd("kind", "gaussian", ["gaussian", "box"], "Kind"),
                sld("radius", 0.0, 0.0, 25.0, 0.1, "Radius")]},
    {"key": "sharpen", "label": "Sharpen (unsharp mask)", "order": 55, "effect": "sharpen",
     "params": [sld("amount", 0.0, 0.0, 5.0, 0.05, "Amount"),
                sld("radius", 2.0, 0.1, 10.0, 0.1, "Radius"),
                sld("threshold", 0.0, 0.0, 0.5, 0.005, "Threshold")]},
    {"key": "bloom", "label": "Bloom / Glow", "order": 60, "effect": "bloom",
     "params": [sld("threshold", 0.7, 0.0, 1.0, 0.01, "Threshold"),
                sld("radius", 8.0, 0.5, 40.0, 0.5, "Radius"),
                sld("intensity", 0.0, 0.0, 2.0, 0.01, "Intensity")]},
    {"key": "halation", "label": "Halation (film glow)", "order": 62, "effect": "halation",
     "params": [sld("threshold", 0.6, 0.0, 1.0, 0.01, "Threshold"),
                sld("radius", 6.0, 0.5, 40.0, 0.5, "Radius"),
                sld("intensity", 0.0, 0.0, 2.0, 0.01, "Intensity"),
                col("tint", "#ff331a", "Halation tint")]},
    {"key": "chromatic", "label": "Chromatic Aberration", "order": 65, "effect": "chromatic",
     "params": [sld("amount", 0.0, 0.0, 20.0, 0.1, "Amount (px)"),
                chk("radial", True, "Radial (off = horizontal split)")]},
    {"key": "vignette", "label": "Vignette", "order": 70, "effect": "vignette",
     "params": [sld("amount", 0.0, 0.0, 1.0, 0.01, "Amount"),
                sld("radius", 0.8, 0.0, 1.5, 0.01, "Radius"),
                sld("feather", 0.5, 0.0, 1.5, 0.01, "Feather"),
                sld("roundness", 1.0, 0.0, 1.0, 0.01, "Roundness")]},
    {"key": "film_grain", "label": "Film Grain", "order": 80, "effect": "film_grain",
     "params": [sld("intensity", 0.0, 0.0, 0.4, 0.005, "Intensity"),
                sld("size", 1.0, 1.0, 6.0, 0.1, "Grain size"),
                chk("colored", False, "Colored grain"),
                sld("shadow_weight", 0.5, 0.0, 1.0, 0.01, "Shadow weighting"),
                num("seed", 0, "Seed (0 = random)")]},
    {"key": "noise", "label": "Noise", "order": 82, "effect": "noise",
     "params": [sld("intensity", 0.0, 0.0, 0.3, 0.005, "Intensity"),
                dd("kind", "gaussian", ["gaussian", "uniform"], "Kind"),
                num("seed", 0, "Seed (0 = random)")]},
    {"key": "scanlines", "label": "Scanlines (CRT)", "order": 84, "effect": "scanlines",
     "params": [sld("intensity", 0.0, 0.0, 1.0, 0.01, "Intensity"),
                num("thickness", 1, "Line thickness"),
                num("gap", 1, "Gap")]},
    {"key": "sepia", "label": "Sepia", "order": 90, "effect": "sepia",
     "params": [sld("strength", 0.0, 0.0, 1.0, 0.01, "Strength")]},
    {"key": "duotone", "label": "Duotone", "order": 91, "effect": "duotone",
     "params": [col("dark", "#0d0d33", "Dark tone"),
                col("light", "#f2d999", "Light tone"),
                sld("strength", 0.0, 0.0, 1.0, 0.01, "Strength")]},
    {"key": "posterize", "label": "Posterize", "order": 92, "effect": "posterize",
     "params": [num("levels", 6, "Levels")]},
    {"key": "pixelate", "label": "Pixelate", "order": 93, "effect": "pixelate",
     "params": [num("block", 8, "Block size (px)")]},
    {"key": "edge", "label": "Edge Detect", "order": 94, "effect": "edge",
     "params": [sld("strength", 1.0, 0.1, 5.0, 0.1, "Strength"),
                chk("invert", True, "Invert (sketch lines)")]},
    {"key": "sketch", "label": "Pencil Sketch", "order": 95, "effect": "sketch",
     "params": [sld("strength", 0.0, 0.0, 1.0, 0.01, "Strength"),
                sld("radius", 6.0, 1.0, 30.0, 0.5, "Radius")]},
    {"key": "comic", "label": "Comic / Cel", "order": 96, "effect": "comic",
     "params": [num("levels", 5, "Color levels"),
                sld("edge_strength", 1.0, 0.0, 1.0, 0.01, "Edge strength"),
                sld("edge_threshold", 0.25, 0.01, 1.0, 0.01, "Edge threshold")]},
    {"key": "light_leak", "label": "Light Leak", "order": 97, "effect": "light_leak",
     "params": [col("color", "#ff6619", "Leak color"),
                sld("angle", 45.0, 0.0, 360.0, 1.0, "Angle"),
                sld("position", 0.8, 0.0, 1.0, 0.01, "Position"),
                sld("width", 0.5, 0.05, 1.5, 0.01, "Width"),
                sld("intensity", 0.0, 0.0, 1.0, 0.01, "Intensity"),
                dd("blend", "screen", ["screen", "add"], "Blend")]},
    {"key": "lens_flare", "label": "Lens Flare", "order": 98, "effect": "lens_flare",
     "params": [sld("x", 0.7, 0.0, 1.0, 0.01, "X position"),
                sld("y", 0.3, 0.0, 1.0, 0.01, "Y position"),
                sld("intensity", 0.0, 0.0, 1.0, 0.01, "Intensity"),
                col("color", "#ffe6b3", "Flare color"),
                num("ghosts", 4, "Ghost count")]},
    {"key": "jpeg_crush", "label": "JPEG Crush", "order": 99, "effect": "jpeg_crush",
     "params": [sld("quality", 60, 1, 100, 1, "Quality")]},
]

# --- category grouping (UI layout only; does NOT affect runtime order) ------
CATEGORY_ORDER = [
    "Cleanup & Optics", "Color & Tone", "Detail",
    "Film & Texture", "Stylize", "Overlays", "Output",
]
STAGE_CATEGORY = {
    "descreen": "Cleanup & Optics", "lens_distortion": "Cleanup & Optics",
    "chromatic": "Cleanup & Optics", "vignette": "Cleanup & Optics",
    "bloom": "Cleanup & Optics", "halation": "Cleanup & Optics",
    "tone": "Color & Tone", "color": "Color & Tone",
    "color_balance": "Color & Tone", "split_tone": "Color & Tone", "lut": "Color & Tone",
    "blur": "Detail", "sharpen": "Detail",
    "film_grain": "Film & Texture", "noise": "Film & Texture", "scanlines": "Film & Texture",
    "sepia": "Stylize", "duotone": "Stylize", "posterize": "Stylize",
    "pixelate": "Stylize", "edge": "Stylize", "sketch": "Stylize", "comic": "Stylize",
    "light_leak": "Overlays", "lens_flare": "Overlays",
    "jpeg_crush": "Output",
}


def ordered_stages():
    """STAGES grouped by category (stable within category). The single source
    of truth for arg ordering - ui() and unflatten() both follow this."""
    seen = set()
    out = []
    for cat in CATEGORY_ORDER:
        for st in STAGES:
            if STAGE_CATEGORY.get(st["key"]) == cat:
                out.append(st)
                seen.add(st["key"])
    for st in STAGES:  # any uncategorized stage falls through to the end
        if st["key"] not in seen:
            out.append(st)
    return out


def grouped_stages():
    """[(category, [stage, ...]), ...] in CATEGORY_ORDER."""
    groups = []
    for cat in CATEGORY_ORDER:
        members = [st for st in STAGES if STAGE_CATEGORY.get(st["key"]) == cat]
        if members:
            groups.append((cat, members))
    extras = [st for st in STAGES if st["key"] not in STAGE_CATEGORY]
    if extras:
        groups.append(("Other", extras))
    return groups


# number of flat ui() args owned by the pipeline (master + per-stage controls).
# Anything beyond this in the arg list is an extra control (e.g. apply-mode).
PIPELINE_ARG_COUNT = 1 + sum(2 + len(st["params"]) for st in STAGES)

# cache for parsed LUTs keyed by path
_LUT_CACHE: dict = {}


def _resolve_file(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("name") or v.get("path")
    if isinstance(v, str):
        return v
    return getattr(v, "name", None)


def apply_stage(effect: str, arr: np.ndarray, v: dict) -> np.ndarray:
    if effect == "descreen":
        return E.descreen(arr, strength=v["strength"], protect=v["protect"],
                          threshold=v["threshold"], notch_radius=int(v["notch_radius"]),
                          mode=v["mode"])
    if effect == "lens_distortion":
        return E.lens_distortion(arr, k=v["k"])
    if effect == "tone":
        return E.tone(arr, exposure=v["exposure"], brightness=v["brightness"],
                     contrast=v["contrast"], gamma=v["gamma"], black=v["black"],
                     white=v["white"], highlights=v["highlights"], shadows=v["shadows"])
    if effect == "color":
        return E.color(arr, saturation=v["saturation"], vibrance=v["vibrance"],
                      hue=v["hue"], temperature=v["temperature"], tint=v["tint"])
    if effect == "color_balance":
        return E.color_balance(arr, sh=(v["sh_r"], v["sh_g"], v["sh_b"]),
                              mid=(v["mid_r"], v["mid_g"], v["mid_b"]),
                              hi=(v["hi_r"], v["hi_g"], v["hi_b"]))
    if effect == "split_tone":
        return E.split_tone(arr, shadow_color=hex2rgb(v["shadow_color"]),
                           highlight_color=hex2rgb(v["highlight_color"]),
                           balance=v["balance"], strength=v["strength"])
    if effect == "lut":
        path = _resolve_file(v["lut_file"])
        if not path:
            return arr
        if path not in _LUT_CACHE:
            try:
                _LUT_CACHE[path] = E.parse_cube(path)
            except Exception as e:
                print(f"[PostProcess Suite] LUT load failed: {e}")
                _LUT_CACHE[path] = None
        cached = _LUT_CACHE[path]
        if cached is None:
            return arr
        lut, dmin, dmax = cached
        return E.apply_lut(arr, lut, dmin, dmax, strength=v["strength"])
    if effect == "blur":
        return E.blur(arr, radius=v["radius"], kind=v["kind"])
    if effect == "sharpen":
        return E.sharpen(arr, amount=v["amount"], radius=v["radius"], threshold=v["threshold"])
    if effect == "bloom":
        return E.bloom(arr, threshold=v["threshold"], radius=v["radius"], intensity=v["intensity"])
    if effect == "halation":
        return E.halation(arr, threshold=v["threshold"], radius=v["radius"],
                         intensity=v["intensity"], tint=hex2rgb(v["tint"]))
    if effect == "chromatic":
        return E.chromatic_aberration(arr, amount=v["amount"], radial=v["radial"])
    if effect == "vignette":
        return E.vignette(arr, amount=v["amount"], radius=v["radius"],
                         feather=v["feather"], roundness=v["roundness"])
    if effect == "film_grain":
        return E.film_grain(arr, intensity=v["intensity"], size=v["size"],
                           colored=v["colored"], shadow_weight=v["shadow_weight"],
                           seed=int(v["seed"]))
    if effect == "noise":
        return E.add_noise(arr, intensity=v["intensity"], kind=v["kind"], seed=int(v["seed"]))
    if effect == "scanlines":
        return E.scanlines(arr, intensity=v["intensity"], thickness=int(v["thickness"]),
                          gap=int(v["gap"]))
    if effect == "sepia":
        return E.sepia(arr, strength=v["strength"])
    if effect == "duotone":
        return E.duotone(arr, dark=hex2rgb(v["dark"]), light=hex2rgb(v["light"]),
                        strength=v["strength"])
    if effect == "posterize":
        return E.posterize(arr, levels=int(v["levels"]))
    if effect == "pixelate":
        return E.pixelate(arr, block=int(v["block"]))
    if effect == "edge":
        return E.edge_detect(arr, strength=v["strength"], invert=v["invert"])
    if effect == "sketch":
        return E.sketch(arr, strength=v["strength"], radius=v["radius"])
    if effect == "comic":
        return E.comic(arr, levels=int(v["levels"]), edge_strength=v["edge_strength"],
                      edge_threshold=v["edge_threshold"])
    if effect == "light_leak":
        return E.light_leak(arr, color_rgb=hex2rgb(v["color"]), angle=v["angle"],
                           position=v["position"], width=v["width"],
                           intensity=v["intensity"], blend=v["blend"])
    if effect == "lens_flare":
        return E.lens_flare(arr, x=v["x"], y=v["y"], intensity=v["intensity"],
                           color_rgb=hex2rgb(v["color"]), ghosts=int(v["ghosts"]))
    if effect == "jpeg_crush":
        return E.jpeg_crush(arr, quality=int(v["quality"]))
    return arr


def unflatten(args: list) -> tuple:
    """
    Map the flat ui() arg list back to (master_enabled, [stage_state, ...]).
    Layout: [master_enabled] then per stage: [enabled, order, *param_values]
    in the exact order of STAGES / its params.
    """
    it = iter(args)
    master_enabled = bool(next(it))
    states = []
    for st in ordered_stages():
        enabled = bool(next(it))
        order = next(it)
        vals = {}
        for p in st["params"]:
            vals[p["id"]] = next(it)
        states.append({"key": st["key"], "effect": st["effect"],
                       "enabled": enabled, "order": order, "vals": vals})
    return master_enabled, states


def run_pipeline(image, args: list):
    """image: PIL.Image -> PIL.Image (or unchanged if disabled / no active stages)."""
    master_enabled, states = unflatten(args)
    if not master_enabled:
        return image
    active = [(float(s["order"]), i, s) for i, s in enumerate(states) if s["enabled"]]
    if not active:
        return image
    active.sort(key=lambda t: (t[0], t[1]))

    arr, alpha = E.pil_to_arr(image)
    for _, _, s in active:
        try:
            arr = apply_stage(s["effect"], arr, s["vals"])
        except Exception as e:
            print(f"[PostProcess Suite] stage '{s['key']}' failed: {e}")
    arr = np.clip(arr, 0.0, 1.0)
    return E.arr_to_pil(arr, alpha)


def summarize(args: list) -> str:
    """Compact JSON of active stages for PNG infotext (record, not round-trip)."""
    master_enabled, states = unflatten(args)
    if not master_enabled:
        return ""
    out = {}
    for _, s in enumerate(states):
        if not s["enabled"]:
            continue
        vals = {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in s["vals"].items() if not isinstance(v, dict)}
        out[s["key"]] = {"order": s["order"], **vals}
    if not out:
        return ""
    return json.dumps(out, separators=(",", ":"))


# ----------------------------------------------------------------------------
# Presets: structured (de)serialization, robust to spec changes.
# ----------------------------------------------------------------------------
def _json_safe(v):
    if isinstance(v, dict):
        return _resolve_file(v)
    return v


def serialize_preset(args: list) -> dict:
    """Flat ui() args -> structured, JSON-safe preset dict."""
    master_enabled, states = unflatten(args)
    out = {"v": 1, "master": bool(master_enabled), "stages": {}}
    for s in states:
        out["stages"][s["key"]] = {
            "enabled": bool(s["enabled"]),
            "order": s["order"],
            "params": {k: _json_safe(val) for k, val in s["vals"].items()},
        }
    return out


def deserialize_to_values(preset: dict) -> list:
    """Structured preset -> flat value list in canonical (ordered_stages) order,
    length == PIPELINE_ARG_COUNT. Missing keys fall back to stage defaults."""
    vals = [bool(preset.get("master", False))]
    stages = preset.get("stages", {})
    for st in ordered_stages():
        sd = stages.get(st["key"], {})
        vals.append(bool(sd.get("enabled", False)))
        vals.append(sd.get("order", st["order"]))
        pvals = sd.get("params", {})
        for pspec in st["params"]:
            vals.append(pvals.get(pspec["id"], pspec["default"]))
    return vals
