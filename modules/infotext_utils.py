from __future__ import annotations

import json
import os
import re
from ast import literal_eval
from typing import Any


from backend.text_processing.emphasis import uses_emphasis
from modules import processing, shared

re_param_code = r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")


def quote(text: str) -> str:
    if "," not in str(text) and "\n" not in str(text) and ":" not in str(text):
        return text

    try:
        return json.dumps(text, ensure_ascii=False)
    except Exception:
        return text


def unquote(text: str) -> str:
    if not text or not (text.startswith('"') and text.endswith('"')):
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def restore_old_hires_fix_params(res: dict):
    if shared.opts.use_old_hires_fix_width_height:
        hires_width = int(res.get("Hires resize-1", 0))
        hires_height = int(res.get("Hires resize-2", 0))

        if hires_width and hires_height:
            res["Size-1"] = hires_width
            res["Size-2"] = hires_height
            return

    try:
        firstpass_width = int(res.get("First pass size-1", None))
        firstpass_height = int(res.get("First pass size-2", None))
    except TypeError:
        return

    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    if firstpass_width == 0 or firstpass_height == 0:
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    res["Size-1"] = firstpass_width
    res["Size-2"] = firstpass_height
    res["Hires resize-1"] = width
    res["Hires resize-2"] = height


def _extract_styles(res: dict, prompt: str, negative_prompt: str) -> tuple[str, str]:
    if shared.opts.infotext_styles == "Ignore":
        return prompt, negative_prompt

    found_styles, prompt_no_styles, negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

    same_hr_styles = True
    if "Hires prompt" in res or "Hires negative prompt" in res:
        hr_prompt, hr_negative_prompt = res.get("Hires prompt", prompt), res.get("Hires negative prompt", negative_prompt)
        hr_found_styles, hr_prompt_no_styles, hr_negative_prompt_no_styles = shared.prompt_styles.extract_styles_from_prompt(hr_prompt, hr_negative_prompt)
        if same_hr_styles := (found_styles == hr_found_styles):
            res["Hires prompt"] = "" if hr_prompt_no_styles == prompt_no_styles else hr_prompt_no_styles
            res["Hires negative prompt"] = "" if hr_negative_prompt_no_styles == negative_prompt_no_styles else hr_negative_prompt_no_styles

    if same_hr_styles:
        prompt, negative_prompt = prompt_no_styles, negative_prompt_no_styles
        if (shared.opts.infotext_styles == "Apply if any" and found_styles) or shared.opts.infotext_styles == "Apply":
            res["Styles array"] = found_styles

    return prompt, negative_prompt


def _populate_defaults(res: dict):
    if "Sampler" not in res:
        res["Sampler"] = "Euler"

    if "Schedule type" not in res:
        res["Schedule type"] = "Simple"

    if "RNG" not in res:
        res["RNG"] = "CPU"

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    restore_old_hires_fix_params(res)

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires schedule type" not in res:
        res["Hires schedule type"] = "Use same scheduler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    if "MaHiRo" not in res:
        res["MaHiRo"] = False

    if "Rescale CFG" not in res:
        res["Rescale CFG"] = 0.0


def parse_generation_parameters(x: str, skip_fields: list[str] | None = None):
    """
    parses infotext (the string under the Gallery in UI)
    returns a dict with field values
    """
    if skip_fields is None:
        skip_fields = shared.opts.infotext_skip_pasting

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    _prompts: list[str] = []
    _negative_prompts: list[str] = []
    _neg: bool = False

    for line in lines:
        if shared.opts.strip_whitespaces:
            line = line.strip()
            if line.startswith("Negative prompt:"):
                line = line.replace("Negative prompt:", "").strip()
                _neg = True
        else:
            if line.lstrip().startswith("Negative prompt: "):
                line = line.replace("Negative prompt: ", "")
                _neg = True
        (_negative_prompts if _neg else _prompts).append(line)

    prompt: str = "\n".join(_prompts)
    negative_prompt: str = "\n".join(_negative_prompts)

    lastline = lastline.replace("Sampler: Undefined,", "Sampler: Euler, Schedule type: Simple,")
    lastline = lastline.replace(", width:", ", Size-1:").replace(", height:", ", Size-2:")

    res: dict[str, Any] = {}

    for k, v in re_param.findall(lastline):
        if k == "Noise Schedule":
            continue
        try:
            v = unquote(v)
            if (m := re_imagesize.match(v)) is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f'Error parsing "{k}: {v}"')

    res["Prompt"], res["Negative prompt"] = prompt, negative_prompt = _extract_styles(res, prompt, negative_prompt)

    _populate_defaults(res)

    prompt_uses_emphasis: bool = uses_emphasis(prompt) or uses_emphasis(negative_prompt)
    if prompt_uses_emphasis and "Emphasis" not in res:
        res["Emphasis"] = "Original"

    if "Shift" in res:
        res["Distilled CFG Scale"] = res.pop("Shift")

    if "Hires Shift" in res:
        res["Hires Distilled CFG Scale"] = res.pop("Hires Shift")

    if "sd_model_name" in res:
        res["Model"] = res.pop("sd_model_name")

    if res.get("Model", None) == os.path.splitext(shared.opts.sd_model_checkpoint)[0]:
        res.pop("Model", None)

    for key in [*skip_fields, "Clip skip", "CLIP_stop_at_last_layers"]:
        res.pop(key, None)

    # VAE / TE
    modules, hr_modules = [], []

    if (vae := res.pop("VAE", None)) is not None:
        modules.append(vae)  # Classic

    from modules_forge import main_entry

    _keys = list(res.keys())
    known_modules = {os.path.splitext(m)[0]: m for m in main_entry.module_list.keys()}

    for key in _keys:
        if key.startswith("Module "):
            if (m := known_modules.get(res.pop(key), None)) is not None:
                modules.append(m)
        elif key.startswith("Hires Module "):
            if (m := known_modules.get(res.pop(key), None)) is not None:
                hr_modules.append(m)

    if modules != []:
        current_modules = shared.opts.forge_additional_modules
        basename_modules = []
        for m in current_modules:
            basename_modules.append(os.path.basename(m))

        if sorted(modules) != sorted(basename_modules):
            res["VAE/TE"] = modules

    # processing.py/StableDiffusionProcessingTxt2Img/init()
    if "Hires Module 1" in res:
        if res["Hires Module 1"] == "Use same choices":
            hr_modules = ["Use same choices"]
        elif res["Hires Module 1"] == "Built-in":
            hr_modules = []

        res["Hires VAE/TE"] = hr_modules
    else:
        res["Hires VAE/TE"] = ["Use same choices"]

    return res


INFOTEXT_TO_SETTING = [("VAE/TE", "forge_additional_modules")]


def create_override_settings_dict(text_pairs: list[str]) -> dict[str, Any]:
    """
    creates processing's override_settings parameters from "Infotext key: value" pairs
    >>> ["VAE/TE: []"]
    {"forge_additional_modules": []}
    """

    if not text_pairs:
        return {}

    params = {}

    for pair in text_pairs:
        k, v = pair.split(":", 1)
        params[k.strip()] = v.strip()

    res: dict[str, Any] = {}
    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]

    for param_name, setting_name in mapping + INFOTEXT_TO_SETTING:
        if (value := params.get(param_name, None)) is None:
            continue

        if setting_name == "forge_additional_modules":
            res[setting_name] = literal_eval(value)
            continue

        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res


def get_override_settings(params: dict[str, Any], *, skip_fields: list[str] = None) -> list[tuple[str, str, Any]]:
    """
    Returns a list of settings overrides from the infotext parameters dictionary

    >>> {"Clip skip": "2"}
    [("Clip skip", "CLIP_stop_at_last_layers", 2)]
    """

    res: list[tuple[str, str, Any]] = []
    mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]

    for param_name, setting_name in mapping + INFOTEXT_TO_SETTING:
        if param_name in (skip_fields or []):
            continue

        if (v := params.get(param_name, None)) is None:
            continue

        if setting_name == "sd_model_checkpoint" and shared.opts.disable_weights_auto_swap:
            continue
        if setting_name == "forge_additional_modules" and shared.opts.disable_modules_auto_swap:
            continue

        v = shared.opts.cast_value(setting_name, v)
        current_value = getattr(shared.opts, setting_name, None)

        if v != current_value:
            res.append((param_name, setting_name, v))

    return res
