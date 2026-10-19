"""ADetailer's parameters as a ``remi.ui`` layout.

The panel holds the main toggle, "Skip img2img" and one tab per model
(setting ``ad_max_models``) with that model's detector, prompts, detection,
mask, inpaint and ControlNet params. Each tab's params are named after the
``ALL_ARGS`` attributes with the tab's suffix (``ad_model``, ``ad_model_2nd``,
...) and written to / read from the same infotext keys as before.

Positionally (``param_order``) the script receives ``ad_enable``,
``ad_skip_img2img``, every tab's params in ``ALL_ARGS`` order, then one hidden
UI-only flag per tab (:data:`WORLD_FLAG`); :func:`pack_args` turns them back
into the one dict per model that ``AfterDetailerScript`` has always processed
(and that A1111 API clients still send through ``alwayson_scripts``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modules.launch_utils import is_installed
from remi import ui

from . import __version__
from .args import ALL_ARGS, MASK_MERGE_INVERT
from .controlnet import CNET_AVAILABLE, get_cn_models, get_cn_modules

MAX_MODELS = 8
"""Upper bound of the ``ad_max_models`` setting."""

REFRESH_ACTION = "refresh_models"
MODEL_CHANGED_ACTION = "ad_model_changed"
WORLD_FLAG = "ad_model_is_world"
"""Per-tab hidden flag: the tab's detector is a YOLO-World model (shows
"Detector Classes"); set by :data:`MODEL_CHANGED_ACTION`, never processed."""
SAME_SAMPLER = "Use same sampler"
SAME_SCHEDULER = "Use same scheduler"

ListSource = Callable[[], Sequence[str]]


@dataclass
class WebuiInfo:
    """Where the choice lists come from; read whenever a schema or a job is
    resolved, so the lists follow the server state."""

    ad_model_list: ListSource
    sampler_names: ListSource
    scheduler_names: ListSource
    checkpoints_list: ListSource
    vae_list: ListSource

    @property
    def checkpoint_default(self) -> str | None:
        """The checkpoint dropdown's default: the first one (none when empty)."""

        return next(iter(self.checkpoints_list()), None)


# region Utils


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def param_name(attr: str, n: int) -> str:
    """The param of ``attr`` in tab ``n`` (the legacy elem_id suffix)."""

    return attr + suffix(n, "_")


def infotext_key(attr: str, n: int) -> str:
    return ALL_ARGS_NAMES[attr] + suffix(n)


ALL_ARGS_NAMES: dict[str, str] = dict(ALL_ARGS)
MAIN_PARAMS = ("ad_enable", "ad_skip_img2img")


def is_world_model(model: Any) -> bool:
    """YOLO-World detectors take class names (the legacy ``"-world"`` test)."""

    return isinstance(model, str) and "-world" in model


def world_flags(values: Mapping[str, Any], num_models: int) -> dict[str, dict]:
    """Patch entries setting the :data:`WORLD_FLAG` of each tab whose detector
    is in ``values``."""

    models = {n: values[param_name("ad_model", n)] for n in range(num_models) if param_name("ad_model", n) in values}
    return {param_name(WORLD_FLAG, n): {"value": is_world_model(model)} for n, model in models.items()}


def param_order(num_models: int) -> tuple[str, ...]:
    """The positional order :func:`pack_args` reads."""

    return (
        *MAIN_PARAMS,
        *(param_name(attr, n) for n in range(num_models) for attr in ALL_ARGS.attrs),
        *(param_name(WORLD_FLAG, n) for n in range(num_models)),
    )


def pack_args(args: Sequence[Any]) -> tuple:
    """``(enable, skip_img2img, *model dicts)`` from the script's arguments.

    The form sends every tab's params as scalars (:func:`param_order`); these
    become one dict per tab and the trailing UI-only flags are dropped (scalars
    without the flags are accepted too). Arguments that already hold dicts (the
    A1111 API's ``alwayson_scripts``, scripts calling ADetailer) are kept as
    they are: the form params left at their defaults around them are not read."""

    args = tuple(args)
    if any(isinstance(arg, dict) for arg in args):
        return args
    width = len(ALL_ARGS)
    head, flat = args[: len(MAIN_PARAMS)], args[len(MAIN_PARAMS) :]
    # ALL_ARGS is longer than MAX_MODELS, so the two lengths never coincide
    count, rest = divmod(len(flat), width + 1)
    if not count or rest:
        count, rest = divmod(len(flat), width)
        if not count or rest:
            return args
    units = (dict(zip(ALL_ARGS.attrs, flat[start : start + width])) for start in range(0, count * width, width))
    return (*head, *units)


def _paste_enable(params: Mapping[str, Any]) -> Any:
    # Pasting a generation that used ADetailer switches it on (its infotext
    # has the models' keys, never "ADetailer enable")
    value = params.get("ADetailer enable")
    if value is not None:
        return value
    return True if any(infotext_key("ad_model", n) in params for n in range(MAX_MODELS)) else None


def _paste_tab_enable(n: int) -> Callable[[Mapping[str, Any]], Any]:
    # An ADetailer infotext lists every tab it ran: the others are switched
    # off so the pasted generation runs exactly those passes
    def paste(params: Mapping[str, Any]) -> Any:
        value = params.get(infotext_key("ad_tab_enable", n))
        if value is not None:
            return value
        if infotext_key("ad_model", n) in params:
            return True
        return False if any(infotext_key("ad_model", m) in params for m in range(MAX_MODELS)) else None

    return paste


# endregion

# region Layout


def adui(num_models: int, is_img2img: bool, webui_info: WebuiInfo) -> ui.Node:
    """The whole panel."""

    return ui.Section(
        "ADetailer",
        enable=ui.Toggle("ad_enable", False, label="ADetailer", infotext="ADetailer enable", paste=_paste_enable),
        children=[
            ui.Row(
                ui.Toggle("ad_skip_img2img", False, label="Skip img2img", infotext="ADetailer skip img2img", hidden=not is_img2img, width="fill"),
                ui.Markdown(f"ver. {__version__}"),
            ),
            ui.Tabs(*(ui.Tab(ordinal(n + 1), _ui_group(n, webui_info)) for n in range(num_models))),
        ],
    )


class _Tab:
    """Names, infotext keys and conditions of one model tab."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, attr: str) -> dict[str, Any]:
        """The common keyword arguments of ``attr``'s input (after its name)."""

        return {"infotext": infotext_key(attr, self.n)}

    def name(self, attr: str) -> str:
        return param_name(attr, self.n)

    def on(self, attr: str) -> dict:
        return ui.param(self.name(attr))


def _ui_group(n: int, webui_info: WebuiInfo) -> list[ui.Node]:
    t = _Tab(n)
    name = t.name
    clip = is_installed("clip")

    return [
        ui.Toggle(name("ad_tab_enable"), n == 0, label=f"Enable {ordinal(n + 1)} Tab", paste=_paste_tab_enable(n), **t("ad_tab_enable")),
        ui.Row(
            ui.Choice(name("ad_model"), lambda: ["None", *webui_info.ad_model_list()], "None", label="ADetailer Detector", width="fill", on_change=MODEL_CHANGED_ACTION, **t("ad_model")),
            ui.Button("Refresh models", REFRESH_ACTION, "tool"),
        ),
        # Follows the selected detector (also one found by Refresh later)
        ui.State(name(WORLD_FLAG), False, label="", submit=False),
        ui.Text(
            name("ad_model_classes"),
            "",
            label="Detector Classes",
            placeholder='Comma-separated class names to detect (e.g. "person,cat" ; default: COCO 80 classes)' if clip else "Detector Classes require Clip to be installed (https://github.com/ultralytics/CLIP.git)",
            lines=1,
            max_lines=1,
            visible_if=ui.param(name(WORLD_FLAG)),
            enabled_if=None if clip else False,
            **t("ad_model_classes"),
        ),
        ui.Text(
            name("ad_prompt"),
            "",
            label="ADetailer Prompt",
            lines=3,
            max_lines=6,
            placeholder="If blank, the original prompt is used",
            prompt=True,
            **t("ad_prompt"),
        ),
        ui.Text(
            name("ad_negative_prompt"),
            "",
            label="ADetailer Negative Prompt",
            lines=3,
            max_lines=6,
            placeholder="If blank, the original negative prompt is used",
            prompt=True,
            **t("ad_negative_prompt"),
        ),
        ui.Group("Detection", _detection(t), open=False),
        ui.Group("Mask Preprocessing", _mask_preprocessing(t), open=False),
        ui.Group("Inpaint Parameters", _inpainting(t, webui_info), open=False),
        ui.Group("ControlNet", _controlnet(t), open=False, visible_if=None if CNET_AVAILABLE else False),
    ]


def _detection(t: _Tab) -> list[ui.Node]:
    name = t.name
    return [
        ui.Slider(name("ad_confidence"), 0.0, 1.0, 0.3, 0.05, label="Confidence Threshold", **t("ad_confidence")),
        ui.Row(
            ui.Slider(name("ad_mask_min_ratio"), 0.0, 1.0, 0.0, 0.05, label="Min Area Ratio", **t("ad_mask_min_ratio")),
            ui.Slider(name("ad_mask_max_ratio"), 0.0, 1.0, 1.0, 0.05, label="Max Area Ratio", **t("ad_mask_max_ratio")),
        ),
        ui.Row(
            ui.Slider(name("ad_mask_k"), 0, 10, 0, 1, label="Keep only the top n masks", info="0 for unlimited", **t("ad_mask_k")),
            ui.Choice(name("ad_mask_filter_method"), ["Area", "Confidence"], "Area", style="radio", label="Filter Method", **t("ad_mask_filter_method")),
        ),
    ]


def _mask_preprocessing(t: _Tab) -> list[ui.Node]:
    name = t.name
    return [
        ui.Row(
            ui.Slider(name("ad_x_offset"), -256, 256, 0, 4, label="Offset: Right (+) / Left(-)", **t("ad_x_offset")),
            ui.Slider(name("ad_y_offset"), -256, 256, 0, 4, label="Offset: Up (+) / Down (-)", **t("ad_y_offset")),
            ui.Slider(name("ad_dilate_erode"), -256, 256, 4, 4, label="Mask: Dilation (+) / Erosion (-)", **t("ad_dilate_erode")),
        ),
        ui.Choice(name("ad_mask_merge_invert"), MASK_MERGE_INVERT, "None", style="radio", label="Masks Merge Mode", **t("ad_mask_merge_invert")),
    ]


def _inpainting(t: _Tab, webui_info: WebuiInfo) -> list[ui.Node]:
    name = t.name
    return [
        ui.Slider(name("ad_mask_blur"), 0, 64, 4, 4, label="Inpaint Mask Blur", **t("ad_mask_blur")),
        ui.Row(
            ui.Column(
                ui.Toggle(name("ad_inpaint_only_masked"), True, label="Inpaint Only Masked", **t("ad_inpaint_only_masked")),
                ui.Slider(name("ad_inpaint_only_masked_padding"), 0, 256, 32, 4, label="Only masked padding, pixels", visible_if=t.on("ad_inpaint_only_masked"), **t("ad_inpaint_only_masked_padding")),
            ),
            ui.Column(
                ui.Toggle(name("ad_use_noise_multiplier"), False, label="Use separate Noise Multiplier", **t("ad_use_noise_multiplier")),
                ui.Slider(name("ad_noise_multiplier"), 0.5, 1.5, 1.0, 0.05, label="Adetailer Noise Multiplier", visible_if=t.on("ad_use_noise_multiplier"), **t("ad_noise_multiplier")),
            ),
        ),
        ui.Column(
            ui.Toggle(name("ad_use_sampler"), False, label="Use different Sampler/Scheduler", **t("ad_use_sampler")),
            ui.Row(
                ui.Choice(name("ad_sampler"), lambda: [SAME_SAMPLER, *webui_info.sampler_names()], SAME_SAMPLER, label="ADetailer Sampler", **t("ad_sampler")),
                ui.Choice(name("ad_scheduler"), lambda: [SAME_SCHEDULER, *webui_info.scheduler_names()], SAME_SCHEDULER, label="ADetailer Scheduler", **t("ad_scheduler")),
                visible_if=t.on("ad_use_sampler"),
            ),
        ),
        ui.Column(
            ui.Toggle(name("ad_use_inpaint_width_height"), False, label="Use separate Width/Height", **t("ad_use_inpaint_width_height")),
            ui.Row(
                ui.Slider(name("ad_inpaint_width"), 64, 2048, 512, 64, label="Inpaint Width", **t("ad_inpaint_width")),
                ui.Slider(name("ad_inpaint_height"), 64, 2048, 512, 64, label="Inpaint Height", **t("ad_inpaint_height")),
                visible_if=t.on("ad_use_inpaint_width_height"),
            ),
        ),
        ui.Row(
            ui.Column(
                ui.Toggle(name("ad_use_steps"), False, label="Use separate Steps", **t("ad_use_steps")),
                ui.Slider(name("ad_steps"), 1, 150, 20, 1, label="ADetailer Steps", visible_if=t.on("ad_use_steps"), **t("ad_steps")),
            ),
            ui.Column(
                ui.Toggle(name("ad_use_cfg_scale"), False, label="Use separate CFG Scale", **t("ad_use_cfg_scale")),
                ui.Slider(name("ad_cfg_scale"), 1.0, 24.0, 4.0, 0.5, label="ADetailer CFG Scale", visible_if=t.on("ad_use_cfg_scale"), **t("ad_cfg_scale")),
            ),
        ),
        ui.Row(
            ui.Column(
                ui.Toggle(name("ad_use_checkpoint"), False, label="Use separate Checkpoint", **t("ad_use_checkpoint")),
                ui.Choice(name("ad_checkpoint"), lambda: list(webui_info.checkpoints_list()), lambda: webui_info.checkpoint_default, same=ui.without_model_hash, label="ADetailer Checkpoint", visible_if=t.on("ad_use_checkpoint"), **t("ad_checkpoint")),
            ),
            ui.Column(
                ui.Toggle(name("ad_use_vae"), False, label="Use separate VAE", **t("ad_use_vae")),
                ui.Choice(name("ad_vae"), lambda: list(webui_info.vae_list()), "None", label="ADetailer VAE", visible_if=t.on("ad_use_vae"), **t("ad_vae")),
            ),
        ),
        ui.Row(
            ui.Toggle(name("ad_restore_face"), False, label="Restore Faces after ADetailer", **t("ad_restore_face")),
            ui.Slider(name("ad_denoising_strength"), 0.0, 1.0, 0.5, 0.05, label="Inpaint Denoising Strength", width="fill", **t("ad_denoising_strength")),
        ),
    ]


def _controlnet(t: _Tab) -> list[ui.Node]:
    name = t.name
    return [
        ui.Row(
            ui.Choice(name("ad_controlnet_module"), get_cn_modules, "None", label="ControlNet Module", **t("ad_controlnet_module")),
            ui.Choice(name("ad_controlnet_model"), get_cn_models, "None", label="ControlNet Model", **t("ad_controlnet_model")),
        ),
        ui.Row(
            ui.Slider(name("ad_controlnet_weight"), 0.0, 1.0, 1.0, 0.05, label="ControlNet Weight", **t("ad_controlnet_weight")),
            ui.RangeSlider(name("ad_controlnet_guidance_start_end"), 0.0, 1.0, (0.0, 1.0), 0.05, label="ControlNet Guidance Start/End", **t("ad_controlnet_guidance_start_end")),
        ),
    ]


# endregion
