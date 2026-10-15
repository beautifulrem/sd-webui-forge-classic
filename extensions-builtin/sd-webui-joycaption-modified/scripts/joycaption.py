"""
sd-webui-joycaption — a "JoyCaption" page (``remi.pages``) for captioning
images locally with the JoyCaption Beta One Llava model.

The page keeps the former Gradio tab's controls, in the same order: model
precision and loading, caption settings, extra options, the caption type
editor, then "Single image" and "Batch (folder)". Model loads, captions and
folder runs are page tasks (behind generation, with progress and Cancel).
PNG Info offers "Send to JoyCaption" (the page's ``send_to``), and the Prompt
Workshop hands its downloaded post image over the same way. The heavy
lifting (model loading, prompt construction) lives in ``joycaption_lib``.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

# Forge puts the extension's root on sys.path before loading scripts/; be
# defensive in case something changes.
_EXT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from joycaption_lib import model as jcm  # noqa: E402
from joycaption_lib import pnginfo as jcpi  # noqa: E402
from joycaption_lib import prompts as jcp  # noqa: E402

from remi import pages, ui  # noqa: E402

PAGE_ID = "joycaption"

# ---------------------------------------------------------------------------
# Capability checks
# ---------------------------------------------------------------------------


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


_QUANT_TO_KEY = {
    "bf16 (best quality, ~17 GB VRAM)": "bf16",
    "int8 (~10 GB VRAM)": "int8",
    "nf4 (~6 GB VRAM)": "nf4",
}
_BF16 = "bf16 (best quality, ~17 GB VRAM)"


def _available_quantizations() -> list[str]:
    """bf16 always available; 4/8-bit only if bitsandbytes is installed."""
    if _has_bitsandbytes():
        return list(_QUANT_TO_KEY)
    return [_BF16]


# ---------------------------------------------------------------------------
# Pre-caption downscale
# ---------------------------------------------------------------------------
# JoyCaption's vision tower resizes everything to a small fixed resolution
# anyway, so a 3000x4000 download only burns time in preprocessing. Images
# are brought down to about 1 megapixel (aspect kept, never upscaled) before
# captioning; the files themselves are untouched.

_CAPTION_TARGET_PIXELS = 1024 * 1024  # ~1 MP


def _downscale_for_caption(image):
    """A copy of ``image`` with at most _CAPTION_TARGET_PIXELS pixels (the
    image itself when already small enough, or if anything goes wrong)."""
    try:
        if image is None:
            return image
        w, h = image.size
        if w <= 0 or h <= 0 or w * h <= _CAPTION_TARGET_PIXELS:
            return image
        from PIL import Image as _PILImage

        scale = math.sqrt(_CAPTION_TARGET_PIXELS / (w * h))
        return image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), _PILImage.LANCZOS)
    except Exception:
        traceback.print_exc()
        return image


# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------


def _status() -> dict:
    """The model status line, as a translatable message."""
    eng = jcm.get_engine()
    if eng.is_loaded():
        return ui.message("✅ Model loaded ({{quant}}): ready", quant=eng.loaded_quant)
    if jcm.is_model_downloaded():
        return ui.message("⚪ Weights present, model not loaded")
    return ui.message("⚪ Weights not downloaded yet (~17 GB are fetched on first use into {{path}})", path=str(jcm.model_local_path()))


def _build_prompt(values: dict) -> str:
    if values["use_custom"] and values["custom_prompt"].strip():
        return values["custom_prompt"].strip()
    return jcp.build_prompt(values["caption_type"], values["caption_length"], values["extra_options"] or [], values["name_input"] or "")


def _system_prompt(values: dict) -> str:
    return (values["system_prompt"] or jcp.DEFAULT_SYSTEM_PROMPT) if values["use_system_prompt"] else ""


def _join_caption(base: str, caption: str) -> str:
    """``caption`` after ``base`` (the image's prompt): ", " between them, or
    a space after a trailing comma."""
    cap = (caption or "").strip()
    base = (base or "").rstrip()
    if not cap:
        return base
    if not base:
        return cap
    return f"{base} {cap}" if base.endswith(",") else f"{base}, {cap}"


def _type_names() -> list[str]:
    return list(jcp.all_caption_types())


def _default_type(names: list[str]) -> str | None:
    return "Descriptive (Formal)" if "Descriptive (Formal)" in names else (names[0] if names else None)


def _editable_types() -> list:
    return [("Select a caption type", "")] + [name for name in _type_names() if name != jcp.NONE_CAPTION_TYPE]


# ---------------------------------------------------------------------------
# Model work (page tasks hold the generation lock: never fights a generation for VRAM)
# ---------------------------------------------------------------------------


def _ensure_loaded(task: pages.Task, quant_label: str) -> None:
    eng = jcm.get_engine()
    key = _QUANT_TO_KEY.get(quant_label, "bf16")
    if eng.is_loaded() and eng.loaded_quant == key:
        return
    if jcm.is_model_downloaded():
        task.progress(None, "Loading the model…")
    else:
        task.progress(None, "Downloading the model weights (~17 GB, first use only)…")
    eng.load(key)


def _caption(eng, image, prompt: str, values: dict) -> str:
    return eng.generate(
        image=_downscale_for_caption(image),
        prompt=prompt,
        system_prompt=_system_prompt(values),
        max_new_tokens=int(values["max_new_tokens"]),
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=int(values["top_k"]),
    )


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".jxl", ".avif", ".heif"}


def _scan_images(folder: str, recursive: bool) -> list[Path]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return []
    found = p.rglob("*") if recursive else p.glob("*")
    return sorted(f for f in found if f.suffix.lower() in _IMG_EXTS and f.is_file())


def _image_prompt(ctx, file_id) -> str:
    """The positive prompt recorded in an uploaded / generated image (WebUI
    PNG info, EXIF or stealth info), or ""."""
    store = pages.file_store(ctx)
    record = store.get(file_id) if store is not None and isinstance(file_id, str) and file_id else None
    if record is None or not record.is_image:
        return ""
    try:
        from PIL import Image

        from modules import images

        with Image.open(record.path) as image:
            infotext, _ = images.read_info_from_image(image)
            return jcpi.extract_positive_prompt_from_text(infotext or "") or jcpi.extract_positive_prompt(image)
    except Exception:
        traceback.print_exc()
        return ""


def _folders_hidden() -> bool:
    try:
        from modules import shared

        return bool(getattr(shared.cmd_opts, "hide_ui_dir_config", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


class JoyCaptionPage(pages.Page):
    def params(self):
        quants = _available_quantizations()
        types = _type_names()
        return ui.Column(
            ui.Markdown(
                "Captions images using "
                "[fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava). "
                "Weights download automatically to `models/joycaption/` on first use (~17 GB)."
            ),
            ui.Output("status", kind="text"),
            ui.Row(
                ui.Choice("quant", quants, quants[0], label="Precision"),
                ui.Toggle("keep_loaded", True, label="Keep model loaded after captioning"),
                ui.Custom("load", "task-runner", None, props={"action": "load", "label": "Load model"}, label="", persist=False, submit=False),
                ui.Button("Unload (free VRAM)", "unload"),
            ),
            None
            if _has_bitsandbytes()
            else ui.Markdown(
                "*Note: `bitsandbytes` not installed — only bf16 available. "
                "Launch Forge with `--bnb` (or `pip install bitsandbytes`) to enable 4-bit / 8-bit.*"
            ),
            ui.Section(
                "Caption settings",
                open=True,
                children=[
                    ui.Row(
                        ui.Choice("caption_type", lambda: _type_names(), _default_type(types), label="Caption type"),
                        ui.Choice("caption_length", jcp.CAPTION_LENGTH_CHOICES, "long", label="Caption length"),
                    ),
                    ui.Text("name_input", "", label="{name} replacement (used only if the {name} extra option is selected)"),
                    ui.Section(
                        "Custom prompt (overrides everything above)",
                        open=False,
                        children=[
                            ui.Toggle("use_custom", False, label="Use custom prompt"),
                            ui.Text("custom_prompt", "", lines=3, label="Custom prompt"),
                        ],
                    ),
                    ui.Section(
                        "Advanced generation settings",
                        open=False,
                        children=[
                            ui.Toggle("use_system_prompt", True, label="Use system prompt (uncheck to send only the caption-type prompt)"),
                            ui.Text("system_prompt", jcp.DEFAULT_SYSTEM_PROMPT, lines=8, label="System prompt"),
                            ui.Button("Reset system prompt to default", "reset_system_prompt", "tool"),
                            ui.Row(
                                ui.Slider("max_new_tokens", 64, 2048, 512, 16, label="Max new tokens"),
                                ui.Slider("temperature", 0.0, 2.0, 0.6, 0.05, label="Temperature"),
                            ),
                            ui.Row(
                                ui.Slider("top_p", 0.0, 1.0, 0.9, 0.05, label="Top-p"),
                                ui.Slider("top_k", 0, 200, 0, 1, label="Top-k (0 = disabled)"),
                            ),
                        ],
                    ),
                ],
            ),
            ui.Section(
                f"Extra options ({len(jcp.EXTRA_OPTIONS)} available)",
                open=False,
                children=[ui.MultiChoice("extra_options", jcp.EXTRA_OPTIONS, (), style="checkboxes", label="")],
            ),
            ui.Section(
                "Edit / add caption types",
                open=False,
                children=[
                    ui.Markdown(
                        "Load an existing type into the editor to modify it (saves a user override), "
                        "or type a new name to create one. User-saved types persist in "
                        "`models/joycaption/user_caption_types.json` and override built-ins of the same name. "
                        "Use `{length}` and `{word_count}` placeholders for length-aware variants; `{name}` for the name field."
                    ),
                    ui.Row(
                        ui.Choice("edit_select", lambda: _editable_types(), "", label="Load existing into editor", width="fill"),
                        ui.Button("Load", "load_type"),
                    ),
                    ui.Text("edit_name", "", label="Name"),
                    ui.Text("edit_any", "", lines=3, label='Prompt — "any" length'),
                    ui.Text("edit_word", "", lines=3, label="Prompt — when a {word_count} is given (use the placeholder)"),
                    ui.Text("edit_length", "", lines=3, label="Prompt — when a named {length} bucket is given (use the placeholder)"),
                    ui.Row(
                        ui.Button("Save", "save_type", "primary"),
                        ui.Button("Delete user override", "delete_type", "danger"),
                    ),
                    ui.Output("edit_status", kind="text"),
                ],
            ),
            ui.Tabs(
                ui.Tab(
                    "Single image",
                    children=[
                        ui.Row(
                            ui.Column(
                                ui.Image("image", label="Image", on_change="image_changed"),
                                ui.Custom("caption", "task-runner", None, props={"action": "caption", "label": "Caption image"}, label="", persist=False, submit=False),
                            ),
                            ui.Column(
                                ui.Text("caption_out", "", lines=10, label="Caption"),
                                ui.Text("prompt_out", "", lines=3, label="Prompt used"),
                            ),
                        ),
                        ui.Text(
                            "combined",
                            "",
                            lines=4,
                            prompt=True,
                            label="txt2img prompt + caption (editable; sent to txt2img below)",
                            placeholder="Auto-fills from the image's PNG-info prompt (if any) when you load an image, then appends the generated caption.",
                        ),
                        ui.Row(
                            ui.Custom(
                                "send_append",
                                "send-button",
                                None,
                                props={"to": "prompt", "source": "combined", "target": "txt2img", "mode": "append", "label": "→ Append to txt2img Prompt (does not replace)", "variant": "primary"},
                                label="",
                                persist=False,
                                submit=False,
                            ),
                            ui.Custom(
                                "send_replace",
                                "send-button",
                                None,
                                props={"to": "prompt", "source": "combined", "target": "txt2img", "mode": "replace", "label": "↻ Replace txt2img Prompt (overwrites existing)"},
                                label="",
                                persist=False,
                                submit=False,
                            ),
                        ),
                    ],
                ),
                ui.Tab(
                    "Batch (folder)",
                    children=[
                        ui.Markdown("Captions every image in a folder. Captions are saved as `<imagename>.<ext>` text files next to each image (ready for LoRA training)."),
                        ui.Row(
                            ui.Text("folder", "", label="Folder path (absolute)", placeholder=r"e.g. C:\training\anima-lora\1_subject", width="fill"),
                            ui.Toggle("recursive", False, label="Recursive"),
                            ui.Toggle("overwrite", False, label="Overwrite existing"),
                        ),
                        ui.Row(
                            ui.Text("prefix", "", label="Prefix (optional)"),
                            ui.Text("suffix", "", label="Suffix (optional)"),
                            ui.Choice("save_ext", [".txt", ".caption", ".tags"], ".txt", label="Save as"),
                        ),
                        ui.Custom("batch", "task-runner", None, props={"action": "batch", "label": "Run batch"}, label="", persist=False, submit=False),
                        ui.Output("log", kind="text", label="Log"),
                    ],
                ),
            ),
        )

    # ---- Model status / loading -------------------------------------------

    @ui.action("status")
    def status(self, ctx, values):
        return {"__outputs__": {"status": _status()}}

    @ui.action("load")
    def load(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)

        def work(task):
            _ensure_loaded(task, resolved["quant"])
            return {"__outputs__": {"status": _status()}, **ui.toast("JoyCaption model loaded", "success")}

        return self.start_task(ctx, "load", work, gpu=True)

    @ui.action("unload")
    def unload(self, ctx, values):
        # Not while the page's tasks use the model (of any user): a batch
        # would fail half-way, and the engine lock would hold this request.
        if self.busy("load", "caption", "batch"):
            return ui.toast("Wait for the running JoyCaption task to finish (or cancel it) before unloading", "warning")
        try:
            jcm.get_engine().unload()
        except Exception as e:
            return {"__outputs__": {"status": ui.message("❌ Failed to unload: {{error}}", error=str(e))}}
        return {"__outputs__": {"status": _status()}}

    @ui.action("reset_system_prompt")
    def reset_system_prompt(self, ctx, values):
        return {"system_prompt": {"value": jcp.DEFAULT_SYSTEM_PROMPT}}

    # ---- Caption type editor ----------------------------------------------

    @ui.action("load_type")
    def load_type(self, ctx, values):
        name = values.get("edit_select") or ""
        types = jcp.all_caption_types()
        prompts = types.get(name, ["", "", ""]) if name else ["", "", ""]
        return {
            "edit_name": {"value": name},
            "edit_any": {"value": prompts[0]},
            "edit_word": {"value": prompts[1]},
            "edit_length": {"value": prompts[2]},
        }

    def _types_patch(self, caption_type, edit_select, status) -> dict:
        names = _type_names()
        return {
            "caption_type": {"choices": names, "value": caption_type},
            "edit_select": {"choices": _editable_types(), "value": edit_select},
            "__outputs__": {"edit_status": status},
        }

    @ui.action("save_type")
    def save_type(self, ctx, values):
        name = str(values.get("edit_name") or "").strip()
        if not name:
            return {"__outputs__": {"edit_status": ui.message("❗ Name is required")}}
        if name == jcp.NONE_CAPTION_TYPE:
            return {"__outputs__": {"edit_status": ui.message("❗ '{{name}}' is reserved", name=name)}}
        user = jcp.load_user_caption_types()
        user[name] = [str(values.get(key) or "") for key in ("edit_any", "edit_word", "edit_length")]
        jcp.save_user_caption_types(user)
        return self._types_patch(name, name, ui.message("✅ Saved '{{name}}'", name=name))

    @ui.action("delete_type")
    def delete_type(self, ctx, values):
        name = str(values.get("edit_name") or "").strip()
        user = jcp.load_user_caption_types()
        if not name:
            status = ui.message("❗ Nothing selected")
        elif name in user:
            del user[name]
            jcp.save_user_caption_types(user)
            if name in jcp.CAPTION_TYPE_MAP:
                status = ui.message("✅ Deleted user override '{{name}}' (built-in restored)", name=name)
            else:
                status = ui.message("✅ Deleted user override '{{name}}'", name=name)
        elif name in jcp.CAPTION_TYPE_MAP:
            status = ui.message("❗ '{{name}}' is built-in. Delete only removes user overrides.", name=name)
        else:
            status = ui.message("❗ '{{name}}' not found", name=name)
        return self._types_patch(_default_type(_type_names()), "", status)

    # ---- Single image ---------------------------------------------------------

    @ui.action("image_changed")
    def image_changed(self, ctx, values):
        """A new image (upload, PNG Info, Prompt Workshop) fills the combined
        box with its PNG-info prompt; removing it empties the box."""
        return {"combined": {"value": _image_prompt(ctx, values.get("image"))}}

    @ui.action("caption")
    def caption(self, ctx, values):
        resolved = self.values(ctx, values)
        image = resolved["image"]
        if image is None:
            return ui.toast("❗ No image provided.", "warning")
        image_prompt = _image_prompt(ctx, values.get("image"))

        def work(task):
            _ensure_loaded(task, resolved["quant"])
            eng = jcm.get_engine()
            prompt = _build_prompt(resolved)
            task.check()
            task.progress(0.5, "Generating…")
            caption = _caption(eng, image, prompt, resolved)
            if not resolved["keep_loaded"]:
                eng.unload()
            # The combined box is rebuilt from scratch on each caption: the
            # image's own prompt, then the new caption.
            return {
                "caption_out": {"value": caption},
                "prompt_out": {"value": prompt},
                "combined": {"value": _join_caption(image_prompt, caption)},
                "__outputs__": {"status": _status()},
            }

        return self.start_task(ctx, "caption", work, gpu=True)

    # ---- Batch ------------------------------------------------------------------

    @ui.action("batch")
    def batch(self, ctx, values):
        resolved = self.values(ctx, values, uploads=False)
        if _folders_hidden():
            return ui.toast("Folder captioning is disabled on this server (--hide-ui-dir-config).", "error")
        folder = resolved["folder"].strip()
        if not folder:
            return ui.toast("❗ Please provide a folder path.", "warning")
        images = _scan_images(folder, resolved["recursive"])
        if not images:
            return ui.toast(ui.message("❗ No images found in: {{folder}}", folder=folder), "warning")

        def work(task):
            from PIL import Image

            _ensure_loaded(task, resolved["quant"])
            eng = jcm.get_engine()
            prompt = _build_prompt(resolved)
            log_lines = [f"Found {len(images)} image(s). Prompt:\n{prompt}\n"]
            task.publish(log="\n".join(log_lines))
            done = skipped = errors = 0
            ext = resolved["save_ext"] if resolved["save_ext"].startswith(".") else f".{resolved['save_ext']}"
            try:
                for i, img_path in enumerate(images):
                    task.check()
                    task.progress(i / len(images), "{{index}}/{{total}}  {{name}}", index=i + 1, total=len(images), name=img_path.name)
                    out_path = img_path.with_suffix(ext)
                    if out_path.exists() and not resolved["overwrite"]:
                        log_lines.append(f"⏭  {img_path.name} — skipped (caption exists)")
                        skipped += 1
                    else:
                        try:
                            with Image.open(img_path) as im:
                                caption = _caption(eng, im, prompt, resolved)
                            out_path.write_text(f"{resolved['prefix']}{caption}{resolved['suffix']}", encoding="utf-8")
                            log_lines.append(f"✅ {img_path.name}\n     → {out_path.name}")
                            done += 1
                        except Exception as e:
                            log_lines.append(f"❌ {img_path.name} — {e}")
                            errors += 1
                    task.publish(log="\n".join(log_lines[-200:]))
            except pages.TaskCancelled:
                log_lines.append("")
                log_lines.append(f"Cancelled. Captioned {done}, skipped {skipped}, errors {errors}.")
                return {"__outputs__": {"log": "\n".join(log_lines[-400:]), "status": _status()}, **ui.toast("Cancelled", "info")}
            if not resolved["keep_loaded"]:
                eng.unload()
            log_lines.append("")
            log_lines.append(f"Done. Captioned {done}, skipped {skipped}, errors {errors}.")
            summary = ui.message("Done. Captioned {{done}}, skipped {{skipped}}, errors {{errors}}.", done=done, skipped=skipped, errors=errors)
            return {"__outputs__": {"log": "\n".join(log_lines[-400:]), "status": _status()}, **ui.toast(summary, "success" if not errors else "warning")}

        return self.start_task(ctx, "batch", work, gpu=True)


pages.register(JoyCaptionPage(PAGE_ID, "JoyCaption", "camera", on_load="status", send_to=[pages.SendTarget("image", "JoyCaption")]))
