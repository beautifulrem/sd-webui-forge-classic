"""
Post Processing Suite for sd-webui-forge-classic (neo).

An AlwaysVisible script that appears in both txt2img and img2img. It runs a
fully reorderable stack of image effects in `postprocess_image_after_composite`
- the last hook before the image is saved - so every effect is baked into the
saved file, the gallery, and the API result.

txt2img also gets an "Apply PP -> Send to img2img" button (the ``pp-send``
widget) that runs the current settings on the selected result and drops the
processed image into the img2img form.

The panel is declared once from ``lib_postprocess.pipeline.STAGES``: one
section per stage, inside one collapsed section per category (the SPA only
mounts what is open), with every stage's params named ``<stage>_<param>``.
"""

import os
import shutil
import sys

import modules.scripts as scripts
from modules import paths_internal
from remi import ui

# make the sibling lib package importable
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

from lib_postprocess import pipeline as P  # noqa: E402

INFOTEXT_KEY = "PP Suite"

_PRESET_DIR = os.path.join(
    paths_internal.data_path, "extension-data", "sd-postprocess-suite", "presets"
)
_LEGACY_PRESET_DIR = os.path.join(_EXT_DIR, "presets")


def _migrate_legacy_presets() -> None:
    if os.path.isdir(_PRESET_DIR) or not os.path.isdir(_LEGACY_PRESET_DIR):
        return
    try:
        os.makedirs(os.path.dirname(_PRESET_DIR), exist_ok=True)
        shutil.copytree(_LEGACY_PRESET_DIR, _PRESET_DIR)
    except OSError as e:
        print(f"[PP Suite] legacy preset migration failed: {e}")


_migrate_legacy_presets()


def _sanitize(name: str) -> str:
    name = (name or "").strip()
    keep = [c for c in name if c.isalnum() or c in (" ", "-", "_", "(", ")")]
    return "".join(keep).strip()[:64]


def _preset_path(name: str) -> str:
    return os.path.join(_PRESET_DIR, _sanitize(name) + ".json")


def list_presets() -> list:
    try:
        names = set()
        for directory in (_PRESET_DIR, _LEGACY_PRESET_DIR):
            if os.path.isdir(directory):
                names.update(
                    f[:-5] for f in os.listdir(directory) if f.lower().endswith(".json")
                )
        return sorted(names, key=str.lower)
    except Exception:
        return []


def save_preset_file(name: str, args) -> bool:
    safe = _sanitize(name)
    if not safe:
        return False
    import json
    os.makedirs(_PRESET_DIR, exist_ok=True)
    try:
        data = P.serialize_preset(list(args))
        data["name"] = safe
        with open(_preset_path(safe), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[PP Suite] save preset failed: {e}")
        return False


def load_preset_file(name: str):
    import json
    try:
        path = _preset_path(name)
        if not os.path.isfile(path):
            path = os.path.join(_LEGACY_PRESET_DIR, _sanitize(name) + ".json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[PP Suite] load preset failed: {e}")
        return None


def delete_preset_file(name: str) -> None:
    try:
        os.remove(_preset_path(name))
    except Exception as e:
        print(f"[PP Suite] delete preset failed: {e}")


APPLY_MODES = ["Final output", "img2img input (pre-diffusion)", "Both"]


def stage_param_name(stage: dict, spec: dict) -> str:
    return f"{stage['key']}_{spec['id']}"


def pipeline_names() -> list[str]:
    """The pipeline's params in argument order: the master toggle, then per
    stage its toggle, order and params (``P.unflatten``'s layout)."""

    names = ["enable"]
    for stage in P.ordered_stages():
        names += [f"{stage['key']}_enabled", f"{stage['key']}_order"]
        names += [stage_param_name(stage, spec) for spec in stage["params"]]
    return names


def _input(name: str, spec: dict):
    kw = spec.get("kw", {})
    label = kw.get("label")
    kind = spec["type"]
    if kind == "slider":
        return ui.Slider(name, kw["minimum"], kw["maximum"], spec["default"], kw["step"], label=label)
    if kind == "number":
        return ui.Number(name, spec["default"], step=kw.get("step"), precision=kw.get("precision"), label=label)
    if kind == "checkbox":
        return ui.Toggle(name, spec["default"], label=label)
    if kind == "color":
        return ui.Color(name, spec["default"], label=label)
    if kind == "dropdown":
        return ui.Choice(name, kw["choices"], spec["default"], label=label)
    if kind == "file":
        return ui.File(name, accept=",".join(kw.get("file_types", ())) or None, label=label)
    raise ValueError(f"unknown component type {kind}")


def stage_section(stage: dict):
    """One stage: its toggle in the header, the order and the first two params
    in the first row, then two params per row."""

    created = [_input(stage_param_name(stage, spec), spec) for spec in stage["params"]]
    order = ui.Number(f"{stage['key']}_order", float(stage["order"]), minimum=0, step=1, precision=2, label="Order")
    rows = [ui.Row(order, *created[:2])]
    rows += [ui.Row(*created[i : i + 2]) for i in range(2, len(created), 2)]
    return ui.Section(stage["label"], enable=ui.Toggle(f"{stage['key']}_enabled", False, label=stage["label"]), children=rows)


# ---------------------------------------------------------------------------
# Uploads (LUT files, the image sent to img2img) live in the server's file store
# ---------------------------------------------------------------------------
def _file_store(ctx) -> ui.ActionFiles:
    """The action's files (``ctx.files``, filled by the server)."""

    if ctx.files is None:
        raise ui.ValidationError([ui.ParamError("this action needs the server's file store", owner=ctx.owner or None)])
    return ctx.files


def _upload_path(ctx, file_id) -> str | None:
    return _file_store(ctx).path(file_id)


def _lut_to_preset(ctx, file_id, preset: str) -> str | None:
    """A preset keeps its own copy of the LUT (uploads are pruned after a while)."""

    path = _upload_path(ctx, file_id)
    if path is None:
        return None
    directory = os.path.join(_PRESET_DIR, "luts")
    os.makedirs(directory, exist_ok=True)
    record = _file_store(ctx).get(file_id)
    base = _sanitize(os.path.splitext(record.name if record else "")[0]) or "lut"
    target = os.path.join(directory, f"{preset}-{base}.cube")
    shutil.copyfile(path, target)
    return target


def _preset_lut(name: str) -> str | None:
    """The LUT copy a saved preset owns (in ``presets/luts``), if any."""

    path = _preset_path(name)
    if not _sanitize(name) or not os.path.isfile(path):
        return None
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            lut = P.deserialize_to_values(json.load(f))[pipeline_names().index("lut_lut_file")]
    except Exception:
        return None
    luts = os.path.realpath(os.path.join(_PRESET_DIR, "luts"))
    if not isinstance(lut, str) or os.path.dirname(os.path.realpath(lut)) != luts:
        return None  # not a copy this suite made (a legacy path, an upload)
    return lut


def _remove_lut_copy(path: str | None) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as e:
            print(f"[PP Suite] could not remove the preset's LUT copy: {e}")


def _lut_from_preset(ctx, path) -> str | None:
    """The upload id of a preset's LUT file (None when it is gone)."""

    if not isinstance(path, str) or not path.lower().endswith(".cube") or not os.path.isfile(path):
        return None
    if os.path.getsize(path) > P._LUT_MAX_BYTES:
        return None
    with open(path, "rb") as file:
        return _file_store(ctx).store_bytes(file.read(), os.path.basename(path))


# ---------------------------------------------------------------------------
class PostProcessSuite(scripts.Script):
    section = "accordions"

    def title(self):
        return "Post Processing Suite"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        # Arguments: the pipeline, then (img2img) where it applies; the send
        # button and the preset picker are UI only and come last.
        self.param_order = (*pipeline_names(), *(("apply_mode",) if is_img2img else ("send",)), "preset")
        if is_img2img:
            where = ui.Choice(
                "apply_mode",
                APPLY_MODES,
                "Final output",
                style="radio",
                label="Apply post-processing to",
                info="'input' cleans/grades the (upscaled) init image right before it is encoded and diffused.",
            )
        else:
            where = ui.Custom("send", "pp-send", None, props={"action": "apply_send", "target": "img2img"}, label="")
        presets = ui.Group(
            "Presets",
            ui.Row(
                ui.Choice(
                    "preset",
                    list_presets,
                    "",
                    allow_custom=True,
                    label="Preset",
                    info="Pick a preset to load or delete, or type a new name to save the current settings.",
                    width="fill",
                ),
                ui.Button("Load", "preset_load", "secondary"),
                ui.Button("Save", "preset_save", "primary"),
                ui.Button("Delete", "preset_delete", "danger"),
                ui.Button("Refresh", "preset_refresh", "tool"),
            ),
            open=False,
        )
        categories = [ui.Section(category, children=[stage_section(stage) for stage in members]) for category, members in P.grouped_stages()]
        return ui.Section(
            "Post Processing Suite",
            enable=ui.Toggle("enable", False, label="Post Processing Suite"),
            children=[
                where,
                ui.Markdown("Enable a stage by ticking its toggle. Set each stage's **Order** (lower runs first) to reorder the pipeline."),
                presets,
                *categories,
            ],
        )

    # ---- actions -----------------------------------------------------------
    def _layout(self):
        if self.layout is None:
            self.layout = ui.ScriptLayout(self.params(self.is_img2img), owner="post_processing_suite", order=self.param_order)
        return self.layout

    def _pipeline_args(self, ctx, values) -> list:
        """The pipeline's processing arguments for the action's ``values``
        (validated; the LUT upload as its path, as a job passes it)."""

        coerced = self._layout().coerce(values, unknown="ignore")
        args = [coerced[name] for name in pipeline_names()]
        lut = pipeline_names().index("lut_lut_file")
        args[lut] = _upload_path(ctx, args[lut])
        return args

    @ui.action
    def preset_refresh(self, ctx, values):
        return {"preset": ui.update(choices=list_presets())}

    @ui.action
    def preset_save(self, ctx, values):
        name = _sanitize(values.get("preset") or "")
        if not name:
            return ui.toast("Type a name for the preset first", "warning")
        args = self._pipeline_args(ctx, values)
        lut = pipeline_names().index("lut_lut_file")
        previous = _preset_lut(name)
        args[lut] = _lut_to_preset(ctx, values.get("lut_lut_file"), name)
        if not save_preset_file(name, args):
            return ui.toast("Could not save the preset", "error")
        if previous is not None and previous != args[lut]:
            _remove_lut_copy(previous)  # the overwritten preset's LUT
        return {"preset": ui.update(choices=list_presets(), value=name), **ui.toast("Preset saved", "success")}

    @ui.action
    def preset_load(self, ctx, values):
        name = values.get("preset") or ""
        preset = load_preset_file(name) if name else None
        if not preset:
            return ui.toast("Pick a saved preset first", "warning")
        layout = self._layout()
        context = ui.Context()
        patch = {}
        for param, value in zip(pipeline_names(), P.deserialize_to_values(preset)):
            if param == "lut_lut_file":
                value = _lut_from_preset(ctx, value)
            try:
                patch[param] = ui.update(value=layout.by_name[param].coerce(value, context))
            except ui.ParamError:
                continue  # a value this version cannot take: the control keeps its own
        return {**patch, **ui.toast("Preset loaded", "success")}

    @ui.action
    def preset_delete(self, ctx, values):
        name = values.get("preset") or ""
        if not name or name not in list_presets():
            return ui.toast("Pick a saved preset first", "warning")
        lut = _preset_lut(name)
        delete_preset_file(name)
        if not os.path.isfile(_preset_path(name)):
            _remove_lut_copy(lut)
        return {"preset": ui.update(choices=list_presets(), value=""), **ui.toast("Preset deleted", "success")}

    @ui.action
    def apply_send(self, ctx, values):
        """Runs the pipeline on the result ``file_id`` (the action's own
        argument); answers the processed image's upload id as the output ``sent``."""

        from modules import images

        path = _upload_path(ctx, ctx.extra.get("file_id"))
        if path is None:
            return ui.toast("Select a generated image first", "warning")
        args = self._pipeline_args(ctx, values)
        image = images.read(path)
        image.load()
        try:
            result = P.run_pipeline(image, args)
        except Exception as e:
            print(f"[PP Suite] send-to-img2img failed: {e}")
            result = image
        return {"__outputs__": {"sent": _file_store(ctx).store_image(result, "post-processed.png")}}

    # ---- processing --------------------------------------------------------
    def _apply_mode(self, args):
        if self.is_img2img and len(args) > P.PIPELINE_ARG_COUNT:
            return args[P.PIPELINE_ARG_COUNT]
        return "Final output"

    def postprocess_image_after_composite(self, p, pp, *args):
        try:
            master_enabled, _ = P.unflatten(list(args))
        except Exception:
            return
        if not master_enabled:
            return
        if self._apply_mode(args) not in ("Final output", "Both"):
            return
        try:
            pp.image = P.run_pipeline(pp.image, list(args))
            summary = P.summarize(list(args))
            if summary:
                p.extra_generation_params[INFOTEXT_KEY] = summary
        except Exception as e:
            print(f"[PP Suite] postprocess failed: {e}")

    def before_process_init_images(self, p, pp, *args):
        """img2img only: clean/grade the (upscaled) init image before encoding."""
        if self._apply_mode(args) not in ("img2img input (pre-diffusion)", "Both"):
            return
        try:
            master_enabled, _ = P.unflatten(list(args))
        except Exception:
            return
        if not master_enabled:
            return
        # only the standard resize modes are safe to pre-resize (no-op the native
        # resize). skip inpaint full-res (crop) and latent-resize (mode 3).
        if isinstance(pp, dict) and pp.get("crop_region") is not None:
            return
        if getattr(p, "resize_mode", 0) not in (0, 1, 2):
            return
        # Inpainting derives its mask scale from the init image size, so keep
        # the original size there and only apply the effects.
        keep_size = isinstance(pp, dict) and pp.get("image_mask") is not None
        init_images = getattr(p, "init_images", None)
        if not init_images:
            return
        try:
            from modules import images, shared
            new_images = []
            for img in init_images:
                flat = images.flatten(img, shared.opts.img2img_background_color)
                if not keep_size:
                    flat = images.resize_image(p.resize_mode, flat, p.width, p.height)
                processed = P.run_pipeline(flat, list(args))
                new_images.append(processed)
            p.init_images = new_images
            summary = P.summarize(list(args))
            if summary:
                p.extra_generation_params[INFOTEXT_KEY + " (input)"] = summary
        except Exception as e:
            print(f"[PP Suite] img2img input post-process failed: {e}")
