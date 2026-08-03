"""
Post Processing Suite for sd-webui-forge-classic (neo).

An AlwaysVisible script that appears in both txt2img and img2img. It runs a
fully reorderable stack of image effects in `postprocess_image_after_composite`
- the last hook before the image is saved - so every effect is baked into the
saved file, the gallery, and the API result.

txt2img also gets an "Apply PP -> Send to img2img" button that runs the current
settings on the selected gallery image and drops the processed result straight
into the img2img canvas.
"""

import os
import shutil
import sys

import gradio as gr

import modules.scripts as scripts
from modules import paths_internal, script_callbacks
from modules.ui_components import InputAccordion

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


def _make_component(spec):
    t = spec["type"]
    kw = dict(spec.get("kw", {}))
    if t == "slider":
        return gr.Slider(value=spec["default"], **kw)
    if t == "number":
        return gr.Number(value=spec["default"], **kw)
    if t == "checkbox":
        return gr.Checkbox(value=spec["default"], **kw)
    if t == "color":
        return gr.ColorPicker(value=spec["default"], **kw)
    if t == "dropdown":
        return gr.Dropdown(value=spec["default"], **kw)
    if t == "file":
        return gr.File(value=spec["default"], **kw)
    raise ValueError(f"unknown component type {t}")


# ---------------------------------------------------------------------------
# Send-to-img2img wiring (deferred; needs cross-tab component handles)
# ---------------------------------------------------------------------------
_SEND = {"button": None, "comps": None, "gallery": None, "wired": False}


def _try_wire_send():
    if _SEND["wired"]:
        return
    if not (_SEND["button"] and _SEND["comps"] and _SEND["gallery"]):
        return
    try:
        from modules import infotext_utils
        dest = infotext_utils.paste_fields.get("img2img", {}).get("init_img")
    except Exception:
        dest = None
    if dest is None:
        return

    from modules.infotext_utils import image_from_url_text

    def _apply_and_send(image_data, *pp_args):
        img = image_from_url_text(image_data)
        if img is None:
            return gr.update()
        try:
            return P.run_pipeline(img, list(pp_args))
        except Exception as e:
            print(f"[PP Suite] send-to-img2img failed: {e}")
            return img

    _SEND["button"].click(
        fn=_apply_and_send,
        inputs=[_SEND["gallery"], *_SEND["comps"]],
        outputs=[dest],
        js="pp_extract_and_pass",
        show_progress=False,
    ).then(fn=None, js="switch_to_img2img")
    _SEND["wired"] = True


def _on_after_component(component, **kwargs):
    if kwargs.get("elem_id") == "txt2img_gallery":
        _SEND["gallery"] = component
    _try_wire_send()


script_callbacks.on_after_component(_on_after_component)


# ---------------------------------------------------------------------------
class PostProcessSuite(scripts.Script):
    def title(self):
        return "Post Processing Suite"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        comps = []

        with InputAccordion(False, label="Post Processing Suite") as master_enabled:
            comps.append(master_enabled)

            if is_img2img:
                apply_mode = gr.Radio(
                    ["Final output", "img2img input (pre-diffusion)", "Both"],
                    value="Final output",
                    label="Apply post-processing to",
                    info="'input' cleans/grades the (upscaled) init image right before "
                         "it is encoded and diffused.",
                )
            else:
                with gr.Row():
                    send_btn = gr.Button(
                        "\U0001F5BC\uFE0F Apply PP \u2192 Send to img2img",
                        variant="primary",
                        elem_id="pp_suite_send_to_img2img",
                    )
                _SEND["button"] = send_btn

            gr.Markdown(
                "Enable a stage by ticking its toggle. Set each stage's **Order** "
                "(lower runs first) to reorder the pipeline."
            )

            # ---- Presets -------------------------------------------------
            with gr.Accordion("\U0001F4BE Presets", open=False):
                with gr.Row():
                    preset_dd = gr.Dropdown(choices=list_presets(), label="Preset",
                                            scale=4, allow_custom_value=False)
                    preset_load = gr.Button("Load", scale=1)
                    preset_refresh = gr.Button("\U0001F504", scale=0, min_width=44)
                    preset_delete = gr.Button("Delete", scale=1, variant="stop")
                with gr.Row():
                    preset_name = gr.Textbox(label="Save as", placeholder="preset name",
                                             scale=4)
                    preset_save = gr.Button("Save", scale=1, variant="primary")

            for cat, members in P.grouped_stages():
                with gr.Accordion(cat, open=False):
                    for st in members:
                        with InputAccordion(False, label=st["label"]) as stage_enabled:
                            plist = st["params"]
                            created = []
                            with gr.Row():
                                order = gr.Number(value=st["order"], precision=2,
                                                  label="Order", minimum=0,
                                                  elem_classes=["pp-order"])
                                if len(plist) >= 1:
                                    created.append(_make_component(plist[0]))
                                if len(plist) >= 2:
                                    created.append(_make_component(plist[1]))
                            for i in range(2, len(plist), 2):
                                with gr.Row():
                                    for pspec in plist[i:i + 2]:
                                        created.append(_make_component(pspec))
                            comps.append(stage_enabled)
                            comps.append(order)
                            comps.extend(created)

            if is_img2img:
                comps.append(apply_mode)

        # ---- preset handlers (pipeline_comps are the load targets) -------
        pipeline_comps = comps[:P.PIPELINE_ARG_COUNT]

        def _do_save(name, *args):
            ok = save_preset_file(name, args)
            choices = list_presets()
            value = _sanitize(name) if ok else None
            return gr.update(choices=choices, value=value)

        def _do_load(name):
            preset = load_preset_file(name) if name else None
            if not preset:
                return [gr.update() for _ in pipeline_comps]
            values = P.deserialize_to_values(preset)
            return [gr.update(value=v) for v in values]

        def _do_delete(name):
            if name:
                delete_preset_file(name)
            return gr.update(choices=list_presets(), value=None)

        def _do_refresh():
            return gr.update(choices=list_presets())

        preset_save.click(fn=_do_save, inputs=[preset_name, *pipeline_comps],
                          outputs=[preset_dd], show_progress=False)
        preset_load.click(fn=_do_load, inputs=[preset_dd],
                          outputs=pipeline_comps, show_progress=False)
        preset_delete.click(fn=_do_delete, inputs=[preset_dd],
                            outputs=[preset_dd], show_progress=False)
        preset_refresh.click(fn=_do_refresh, inputs=[], outputs=[preset_dd],
                            show_progress=False)

        if not is_img2img:
            _SEND["comps"] = comps
            _try_wire_send()

        return comps

    @staticmethod
    def _apply_mode(args):
        if len(args) > P.PIPELINE_ARG_COUNT:
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
        init_images = getattr(p, "init_images", None)
        if not init_images:
            return
        try:
            from modules import images, shared
            new_images = []
            for img in init_images:
                flat = images.flatten(img, shared.opts.img2img_background_color)
                upscaled = images.resize_image(p.resize_mode, flat, p.width, p.height)
                processed = P.run_pipeline(upscaled, list(args))
                new_images.append(processed)
            p.init_images = new_images
            summary = P.summarize(list(args))
            if summary:
                p.extra_generation_params[INFOTEXT_KEY + " (input)"] = summary
        except Exception as e:
            print(f"[PP Suite] img2img input post-process failed: {e}")
