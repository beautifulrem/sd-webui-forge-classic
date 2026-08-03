"""
sd-webui-joycaption — adds a "JoyCaption" tab to Forge / A1111-style WebUIs
for captioning images locally with the JoyCaption Beta One Llava model.

This script registers the tab via `script_callbacks.on_ui_tabs`. All the
heavy lifting (model loading, prompt construction) lives in the sibling
modules `joycaption_prompts` and `joycaption_model`.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import traceback
from pathlib import Path

import gradio as gr

# Forge already adds the extension's root dir to sys.path before loading
# scripts/, so `joycaption_lib` (sitting at the extension root) is
# importable from here. Be defensive in case something changes.
_EXT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from joycaption_lib import prompts as jcp  # noqa: E402
from joycaption_lib import model as jcm    # noqa: E402
from joycaption_lib import pnginfo as jcpi  # noqa: E402

from modules import script_callbacks  # noqa: E402

# Forge Neo renamed `generation_parameters_copypaste` to `infotext_utils`.
# The new name is canonical in current Forge; the old name is still aliased
# in upstream A1111. Try the new name first and fall back so the extension
# keeps working across forks.
try:
    from modules import infotext_utils as _infotext  # type: ignore
except Exception:  # pragma: no cover - older webuis
    try:
        from modules import generation_parameters_copypaste as _infotext  # type: ignore
    except Exception:
        _infotext = None  # type: ignore


# ---------------------------------------------------------------------------
# Bridge: shared temp folder for images dropped in by the Prompt Workshop
# ---------------------------------------------------------------------------
# When the Workshop's "Send Image to JoyCaption" button fires, it downloads
# the booru image into this folder and the JoyCaption tab picks it up. The
# folder is wiped on extension load (i.e. every webui startup) so it never
# accumulates -- it's just a handoff scratchpad, not a cache.

import shutil  # noqa: E402
import tempfile  # noqa: E402

_BRIDGE_TMP_DIR = Path(tempfile.gettempdir()) / "joycaption_bridge"
try:
    if _BRIDGE_TMP_DIR.exists():
        shutil.rmtree(_BRIDGE_TMP_DIR, ignore_errors=True)
    _BRIDGE_TMP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # Non-fatal: bridge will just fail later with a clear error.
    traceback.print_exc()


def bridge_tmp_dir() -> Path:
    """Shared temp folder used by the Workshop -> JoyCaption image handoff.
    Created once at module load and wiped on every webui startup."""
    return _BRIDGE_TMP_DIR


# ---------------------------------------------------------------------------
# Capability checks
# ---------------------------------------------------------------------------

def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _available_quantizations() -> list[str]:
    """bf16 always available; 4/8-bit only if bitsandbytes is installed."""
    quants = ["bf16 (best quality, ~17 GB VRAM)"]
    if _has_bitsandbytes():
        quants += [
            "int8 (~10 GB VRAM)",
            "nf4 (~6 GB VRAM)",
        ]
    return quants


_QUANT_TO_KEY = {
    "bf16 (best quality, ~17 GB VRAM)": "bf16",
    "int8 (~10 GB VRAM)": "int8",
    "nf4 (~6 GB VRAM)": "nf4",
}


# ---------------------------------------------------------------------------
# Pre-caption downscale
# ---------------------------------------------------------------------------
# JoyCaption's vision tower internally resizes everything to a fixed (small)
# resolution anyway, so feeding it the original 3000x4000 booru download
# just burns time on the upload + processor preprocessing without changing
# the caption. We downscale to roughly 1 megapixel (1024 * 1024 total
# pixels) BEFORE handing the image to the engine. Aspect ratio is
# preserved; only the total pixel count is targeted. Images already at or
# below the target are left alone -- we never upscale.
#
# This applies only to the captioning path. The single-image preview, the
# bridge-folder copy on disk, and anything the Workshop downloads stay at
# original resolution.

_CAPTION_TARGET_PIXELS = 1024 * 1024  # ~1 MP


def _downscale_for_caption(image):
    """Return a copy of `image` whose total pixel count is <=
    _CAPTION_TARGET_PIXELS, preserving aspect ratio. Returns `image`
    unchanged if it is already at or below the target, or if anything
    goes wrong (we'd rather caption a too-large image than crash)."""
    try:
        if image is None:
            return image
        w, h = image.size
        if w <= 0 or h <= 0:
            return image
        cur = w * h
        if cur <= _CAPTION_TARGET_PIXELS:
            return image
        import math
        from PIL import Image as _PILImage
        scale = math.sqrt(_CAPTION_TARGET_PIXELS / cur)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # LANCZOS is the highest-quality downscale filter PIL offers.
        return image.resize((new_w, new_h), _PILImage.LANCZOS)
    except Exception:
        traceback.print_exc()
        return image


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def _gpu_lock():
    """Forge's queue_lock serializes all GPU work (txt2img, img2img, etc.).
    Acquiring it here means JoyCaption never fights an in-flight generation
    for VRAM — if the user clicks Load/Caption mid-generation, we queue."""
    try:
        from modules.call_queue import queue_lock
        return queue_lock
    except Exception:
        return None


def _load_model(quant_label: str, progress=gr.Progress(track_tqdm=True)):
    lock = _gpu_lock()
    held = False
    if lock is not None:
        if not lock.acquire(blocking=False):
            yield "🟡 Queued — waiting for current GPU task to finish…"
            lock.acquire()  # blocks until txt2img/img2img/etc. releases
        held = True
    try:
        progress(0.0, desc="Loading…")
        eng = jcm.get_engine()
        eng.load(_QUANT_TO_KEY[quant_label])
        progress(1.0, desc="Loaded")
        yield _status_text()
    except Exception as e:
        traceback.print_exc()
        yield f"❌ Failed to load: {e}"
    finally:
        if held:
            lock.release()


def _unload_model():
    try:
        jcm.get_engine().unload()
        return _status_text()
    except Exception as e:
        return f"❌ Failed to unload: {e}"


def _status_text() -> str:
    eng = jcm.get_engine()
    if eng.is_loaded():
        return f"✅ Model loaded ({eng.loaded_quant}) — ready"
    if jcm.is_model_downloaded():
        return "⚪ Weights present, model not loaded"
    return f"⚪ Weights not downloaded yet (~17 GB will be fetched on first use → {jcm.model_local_path()})"


def _build_prompt(caption_type, caption_length, extra_options, name_input, use_custom, custom_prompt) -> str:
    if use_custom and custom_prompt.strip():
        return custom_prompt.strip()
    return jcp.build_prompt(caption_type, caption_length, extra_options or [], name_input or "")


def _caption_single(
    image,
    caption_type, caption_length, extra_options, name_input,
    use_custom, custom_prompt, system_prompt, use_system_prompt,
    max_new_tokens, temperature, top_p, top_k,
    quant_label,
    keep_loaded,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        yield "", "", "❗ No image provided."
        return

    # Downscale to ~1 MP before captioning. Saves significant time on
    # large booru downloads / hi-res screenshots without changing what
    # JoyCaption sees (its vision tower resizes internally anyway).
    # See _downscale_for_caption() for details.
    image = _downscale_for_caption(image)

    lock = _gpu_lock()
    held = False
    if lock is not None:
        if not lock.acquire(blocking=False):
            yield "", "", "🟡 Queued — waiting for current GPU task to finish…"
            lock.acquire()
        held = True

    try:
        # Ensure loaded
        eng = jcm.get_engine()
        if not eng.is_loaded() or eng.loaded_quant != _QUANT_TO_KEY[quant_label]:
            progress(0.0, desc="Loading model…")
            eng.load(_QUANT_TO_KEY[quant_label])

        prompt = _build_prompt(caption_type, caption_length, extra_options, name_input, use_custom, custom_prompt)
        progress(0.5, desc="Generating…")
        caption = eng.generate(
            image=image,
            prompt=prompt,
            system_prompt=(system_prompt or jcp.DEFAULT_SYSTEM_PROMPT) if use_system_prompt else "",
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
        progress(1.0, desc="Done")

        if not keep_loaded:
            eng.unload()

        yield caption, prompt, _status_text()
    except Exception as e:
        traceback.print_exc()
        yield "", "", f"❌ {e}"
    finally:
        if held:
            lock.release()


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".jxl", ".avif", ".heif"}


def _scan_images(folder: str, recursive: bool) -> list[Path]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return []
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted([f for f in it if f.suffix.lower() in _IMG_EXTS and f.is_file()])


def _caption_batch(
    folder, recursive, overwrite, prefix, suffix, save_ext,
    caption_type, caption_length, extra_options, name_input,
    use_custom, custom_prompt, system_prompt, use_system_prompt,
    max_new_tokens, temperature, top_p, top_k,
    quant_label,
    keep_loaded,
    progress=gr.Progress(track_tqdm=True),
):
    from PIL import Image

    if not folder:
        yield "", "❗ Please provide a folder path."
        return

    images = _scan_images(folder, recursive)
    if not images:
        yield "", f"❗ No images found in: {folder}"
        return

    lock = _gpu_lock()
    held = False
    if lock is not None:
        if not lock.acquire(blocking=False):
            yield "⏳ Queued — waiting for current GPU task to finish…", "🟡 Queued"
            lock.acquire()
        held = True

    try:
        # Load model once
        eng = jcm.get_engine()
        if not eng.is_loaded() or eng.loaded_quant != _QUANT_TO_KEY[quant_label]:
            progress(0.0, desc="Loading model…")
            eng.load(_QUANT_TO_KEY[quant_label])

        prompt = _build_prompt(caption_type, caption_length, extra_options, name_input, use_custom, custom_prompt)
        log_lines: list[str] = [f"Found {len(images)} image(s). Prompt:\n{prompt}\n"]
        yield "\n".join(log_lines), _status_text()

        done = 0
        skipped = 0
        errors = 0
        for i, img_path in enumerate(images):
            progress(i / len(images), desc=f"{i+1}/{len(images)}  {img_path.name}")
            out_path = img_path.with_suffix(save_ext if save_ext.startswith(".") else f".{save_ext}")

            if out_path.exists() and not overwrite:
                log_lines.append(f"⏭  {img_path.name} — skipped (caption exists)")
                skipped += 1
                yield "\n".join(log_lines[-200:]), _status_text()
                continue

            try:
                with Image.open(img_path) as im:
                    # Match the single-image path: downscale to ~1 MP
                    # before captioning so a folder full of 4K images
                    # doesn't take forever. Files on disk are untouched.
                    im_for_caption = _downscale_for_caption(im)
                    caption = eng.generate(
                        image=im_for_caption,
                        prompt=prompt,
                        system_prompt=(system_prompt or jcp.DEFAULT_SYSTEM_PROMPT) if use_system_prompt else "",
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                    )
                text = f"{prefix}{caption}{suffix}"
                out_path.write_text(text, encoding="utf-8")
                log_lines.append(f"✅ {img_path.name}\n     → {out_path.name}")
                done += 1
            except Exception as e:
                log_lines.append(f"❌ {img_path.name} — {e}")
                errors += 1

            yield "\n".join(log_lines[-200:]), _status_text()

        if not keep_loaded:
            eng.unload()

        log_lines.append("")
        log_lines.append(f"Done. Captioned {done}, skipped {skipped}, errors {errors}.")
        yield "\n".join(log_lines[-400:]), _status_text()
    finally:
        if held:
            lock.release()


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------

def _build_ui() -> gr.Blocks:
    quant_choices = _available_quantizations()
    bnb_warn = "" if _has_bitsandbytes() else (
        "*Note: `bitsandbytes` not installed — only bf16 available. "
        "Launch Forge with `--bnb` (or `pip install bitsandbytes`) to enable 4-bit / 8-bit.*"
    )

    css = """
    #joycaption-extras-scroll {
        max-height: 300px !important;
        overflow-y: auto !important;
        overflow-x: hidden;
        padding: 6px 10px 6px 8px;
        border: 1px solid var(--border-color-primary, #444);
        border-radius: 6px;
        background: var(--background-fill-secondary, transparent);
    }
    /* Tighten the checkbox layout so 81 fit comfortably */
    #joycaption-extras-scroll label {
        padding: 2px 4px !important;
        margin: 0 !important;
        font-size: 0.85em;
        line-height: 1.25;
    }
    #joycaption-extras-scroll .wrap {
        gap: 0 !important;
    }
    #joycaption-extras-scroll::-webkit-scrollbar { width: 10px; }
    #joycaption-extras-scroll::-webkit-scrollbar-thumb {
        background: var(--border-color-accent, #666);
        border-radius: 5px;
    }
    """

    with gr.Blocks(analytics_enabled=False, css=css) as ui:
        gr.Markdown("## JoyCaption — local image captioning")
        gr.Markdown(
            "Captions images using "
            "[fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava). "
            "Weights download automatically to `models/joycaption/` on first use (~17 GB)."
        )

        # ---- Model status / loading ----
        with gr.Row():
            status = gr.Markdown(_status_text())
        with gr.Row():
            quant = gr.Dropdown(label="Precision", choices=quant_choices, value=quant_choices[0])
            keep_loaded = gr.Checkbox(label="Keep model loaded after captioning", value=True)
            load_btn = gr.Button("Load model", variant="primary")
            unload_btn = gr.Button("Unload (free VRAM)")
        if bnb_warn:
            gr.Markdown(bnb_warn)

        load_btn.click(_load_model, inputs=[quant], outputs=[status])
        unload_btn.click(_unload_model, outputs=[status])

        # ---- Shared caption settings ----
        with gr.Accordion("Caption settings", open=True):
            with gr.Row():
                _initial_types = list(jcp.all_caption_types().keys())
                caption_type = gr.Dropdown(
                    label="Caption type",
                    choices=_initial_types,
                    value="Descriptive (Formal)" if "Descriptive (Formal)" in _initial_types else _initial_types[0],
                )
                caption_length = gr.Dropdown(
                    label="Caption length",
                    choices=jcp.CAPTION_LENGTH_CHOICES,
                    value="long",
                )
            name_input = gr.Textbox(
                label="{name} replacement (used only if the {name} extra option is selected)",
                value="",
            )

            with gr.Accordion("Custom prompt (overrides everything above)", open=False):
                use_custom = gr.Checkbox(label="Use custom prompt", value=False)
                custom_prompt = gr.Textbox(label="Custom prompt", lines=3, value="")

            with gr.Accordion("Advanced generation settings", open=False):
                use_system_prompt = gr.Checkbox(
                    label="Use system prompt (uncheck to send only the caption-type prompt)",
                    value=True,
                )
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value=jcp.DEFAULT_SYSTEM_PROMPT,
                    lines=8,
                    elem_id="joycaption-system-prompt-v2",
                )
                sysprompt_reset = gr.Button("Reset system prompt to default", size="sm")
                sysprompt_reset.click(
                    fn=lambda: jcp.DEFAULT_SYSTEM_PROMPT,
                    outputs=[system_prompt],
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(64, 2048, value=512, step=16, label="Max new tokens")
                    temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Temperature")
                with gr.Row():
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                    top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0 = disabled)")

        # ---- Extra options (separate accordion, closed by default) ----
        with gr.Accordion(f"Extra options ({len(jcp.EXTRA_OPTIONS)} available)", open=False):
            with gr.Column(elem_id="joycaption-extras-scroll"):
                extra_options = gr.CheckboxGroup(
                    show_label=False,
                    choices=jcp.EXTRA_OPTIONS,
                    value=[],
                    container=False,
                )

        # ---- Caption type editor (closed by default) ----
        with gr.Accordion("Edit / add caption types", open=False):
            gr.Markdown(
                "Load an existing type into the editor to modify it (saves a user override), "
                "or type a new name to create one. User-saved types persist in "
                "`models/joycaption/user_caption_types.json` and override built-ins of the same name. "
                "Use `{length}` and `{word_count}` placeholders for length-aware variants; `{name}` for the name field."
            )
            with gr.Row():
                edit_select = gr.Dropdown(
                    label="Load existing into editor",
                    choices=[k for k in _initial_types if k != jcp.NONE_CAPTION_TYPE],
                    value=None,
                )
                edit_load_btn = gr.Button("Load")
            edit_name = gr.Textbox(label="Name", value="")
            edit_any = gr.Textbox(label='Prompt — "any" length', lines=3, value="")
            edit_word = gr.Textbox(label='Prompt — when a {word_count} is given (use the placeholder)', lines=3, value="")
            edit_length = gr.Textbox(label='Prompt — when a named {length} bucket is given (use the placeholder)', lines=3, value="")
            with gr.Row():
                edit_save_btn = gr.Button("Save", variant="primary")
                edit_delete_btn = gr.Button("Delete user override")
            edit_status = gr.Markdown("")

        # --- Editor handlers ---
        def _do_load_type(name):
            if not name:
                return "", "", "", ""
            types = jcp.all_caption_types()
            if name not in types:
                return name, "", "", ""
            a, w, lng = types[name]
            return name, a, w, lng

        def _do_save_type(name, p_any, p_word, p_length):
            name = (name or "").strip()
            if not name:
                return gr.update(), gr.update(), "❗ Name is required"
            if name == jcp.NONE_CAPTION_TYPE:
                return gr.update(), gr.update(), f"❗ '{name}' is reserved"
            user = jcp.load_user_caption_types()
            user[name] = [p_any or "", p_word or "", p_length or ""]
            jcp.save_user_caption_types(user)
            all_keys = list(jcp.all_caption_types().keys())
            edit_keys = [k for k in all_keys if k != jcp.NONE_CAPTION_TYPE]
            return (
                gr.update(choices=all_keys, value=name),
                gr.update(choices=edit_keys, value=name),
                f"✅ Saved '{name}'",
            )

        def _do_delete_type(name):
            name = (name or "").strip()
            user = jcp.load_user_caption_types()
            if not name:
                msg = "❗ Nothing selected"
            elif name in user:
                del user[name]
                jcp.save_user_caption_types(user)
                msg = f"✅ Deleted user override '{name}'" + (
                    " (built-in restored)" if name in jcp.CAPTION_TYPE_MAP else ""
                )
            elif name in jcp.CAPTION_TYPE_MAP:
                msg = f"❗ '{name}' is built-in. Delete only removes user overrides."
            else:
                msg = f"❗ '{name}' not found"
            all_keys = list(jcp.all_caption_types().keys())
            edit_keys = [k for k in all_keys if k != jcp.NONE_CAPTION_TYPE]
            new_val = "Descriptive (Formal)" if "Descriptive (Formal)" in all_keys else (all_keys[0] if all_keys else None)
            return gr.update(choices=all_keys, value=new_val), gr.update(choices=edit_keys, value=None), msg

        edit_load_btn.click(
            _do_load_type,
            inputs=[edit_select],
            outputs=[edit_name, edit_any, edit_word, edit_length],
        )
        edit_save_btn.click(
            _do_save_type,
            inputs=[edit_name, edit_any, edit_word, edit_length],
            outputs=[caption_type, edit_select, edit_status],
        )
        edit_delete_btn.click(
            _do_delete_type,
            inputs=[edit_name],
            outputs=[caption_type, edit_select, edit_status],
        )

        # ---- Tabs: Single vs Batch ----
        with gr.Tabs():
            with gr.TabItem("Single image"):
                with gr.Row():
                    with gr.Column():
                        image_in = gr.Image(
                            label="Image",
                            type="pil",
                            height=480,
                            elem_id="joycaption_single_image",
                        )
                        run_btn = gr.Button("Caption image", variant="primary")
                    with gr.Column():
                        caption_out = gr.Textbox(label="Caption", lines=10, show_copy_button=True)
                        prompt_out = gr.Textbox(label="Prompt used", lines=3, show_copy_button=True)

                # ---- Bridge: txt2img prompt + caption combo box ----------
                # Auto-populated when an image is loaded: if the image has
                # webui PNG-info metadata, the positive prompt goes here.
                # After captioning, the generated caption is *appended* to
                # whatever's already in this box. The user can edit it
                # freely before sending it back to txt2img.
                txt2img_combined = gr.Textbox(
                    label="txt2img prompt + caption (editable; sent to txt2img below)",
                    lines=4,
                    interactive=True,
                    show_copy_button=True,
                    elem_id="joycaption_txt2img_combined",
                    placeholder=(
                        "Auto-fills from the image's PNG-info prompt (if any) when you "
                        "load an image, then appends the generated caption."
                    ),
                )
                with gr.Row():
                    send_to_t2i_btn = gr.Button(
                        "→ Append to txt2img Prompt (does not replace)",
                        elem_id="joycaption_send_to_t2i",
                        variant="primary",
                    )
                    # Sibling action: replace the txt2img prompt entirely
                    # rather than appending. Wired purely in JS (see
                    # javascript/joycaption.js) to match the append
                    # button's flow. Especially useful in combination
                    # with the sd-webui-prompt-anchor extension: with
                    # your style/quality foundation parked in the
                    # anchor box, the main txt2img prompt becomes pure
                    # subject-of-this-image content -- so REPLACING it
                    # with a fresh JoyCaption result (which describes
                    # exactly the subject) is exactly what you want,
                    # rather than appending the caption to whatever
                    # subject the previous image had. Without the
                    # anchor extension this button is still useful for
                    # quick "swap the prompt" workflows.
                    #
                    # Styled as "secondary" (vs the primary Append
                    # button) so the destructive option doesn't visually
                    # dominate the safer one.
                    replace_t2i_btn = gr.Button(
                        "↻ Replace txt2img Prompt (overwrites existing)",
                        elem_id="joycaption_replace_t2i",
                        variant="secondary",
                    )

                # When a new image is dropped/uploaded, refresh the combined
                # box with just the image's positive prompt (or empty it if
                # the image has no webui metadata). We use `.upload()` and
                # `.clear()` rather than `.change()` because `.change()`
                # fires every time Gradio re-renders the component (e.g.
                # right after captioning, which would wipe our appended
                # caption).
                def _on_image_loaded(img):
                    return jcpi.extract_positive_prompt(img)

                image_in.upload(
                    fn=_on_image_loaded,
                    inputs=[image_in],
                    outputs=[txt2img_combined],
                )
                image_in.clear(
                    fn=lambda: "",
                    inputs=[],
                    outputs=[txt2img_combined],
                )

                # Caption flow: run the model, then append the caption to
                # the combined box. We chain via `.then(...)` so the second
                # step has access to whatever the user (or _on_image_loaded)
                # put in the combined box before clicking.
                def _append_caption_to_combined(caption: str, existing: str) -> str:
                    cap = (caption or "").strip()
                    if not cap:
                        return existing
                    base = (existing or "").rstrip()
                    if not base:
                        return cap
                    # If base ends with a comma already, just add a space;
                    # otherwise insert ", " separator. Mirrors the same
                    # joining behaviour the Workshop's JS uses.
                    if base.endswith(","):
                        return base + " " + cap
                    return base + ", " + cap

                # On each caption run we REBUILD the combined box from
                # scratch: positive prompt extracted from the current
                # image (if any) + the freshly generated caption. This
                # way captioning a second image overwrites whatever the
                # first image left behind, instead of accumulating.
                # User edits to the combined box between caption runs
                # are intentionally discarded -- if they want to preserve
                # them, they can hit "Append to txt2img Prompt" first.
                def _rebuild_combined(image, caption):
                    base = jcpi.extract_positive_prompt(image) if image is not None else ""
                    cap = (caption or "").strip()
                    if not cap:
                        return base
                    if not base:
                        return cap
                    if base.rstrip().endswith(","):
                        return base.rstrip() + " " + cap
                    return base.rstrip() + ", " + cap

                run_btn.click(
                    _caption_single,
                    inputs=[
                        image_in,
                        caption_type, caption_length, extra_options, name_input,
                        use_custom, custom_prompt, system_prompt, use_system_prompt,
                        max_new_tokens, temperature, top_p, top_k,
                        quant, keep_loaded,
                    ],
                    outputs=[caption_out, prompt_out, status],
                ).then(
                    fn=_rebuild_combined,
                    inputs=[image_in, caption_out],
                    outputs=[txt2img_combined],
                )

                # The "Send to txt2img" button is wired up in
                # javascript/joycaption.js -- it needs to reach into a
                # different tab's textarea, which is awkward via Gradio.

            with gr.TabItem("Batch (folder)"):
                gr.Markdown(
                    "Captions every image in a folder. Captions are saved as "
                    "`<imagename>.<ext>` text files next to each image (ready for LoRA training)."
                )
                with gr.Row():
                    folder = gr.Textbox(
                        label="Folder path (absolute)",
                        placeholder=r"e.g. C:\training\anima-lora\1_subject",
                        scale=4,
                    )
                    recursive = gr.Checkbox(label="Recursive", value=False)
                    overwrite = gr.Checkbox(label="Overwrite existing", value=False)
                with gr.Row():
                    prefix = gr.Textbox(label="Prefix (optional)", value="", scale=2)
                    suffix = gr.Textbox(label="Suffix (optional)", value="", scale=2)
                    save_ext = gr.Dropdown(
                        label="Save as",
                        choices=[".txt", ".caption", ".tags"],
                        value=".txt",
                    )
                batch_btn = gr.Button("Run batch", variant="primary")
                log_out = gr.Textbox(label="Log", lines=20, max_lines=50)

                batch_btn.click(
                    _caption_batch,
                    inputs=[
                        folder, recursive, overwrite, prefix, suffix, save_ext,
                        caption_type, caption_length, extra_options, name_input,
                        use_custom, custom_prompt, system_prompt, use_system_prompt,
                        max_new_tokens, temperature, top_p, top_k,
                        quant, keep_loaded,
                    ],
                    outputs=[log_out, status],
                )

        # ---- Bridge: register JoyCaption as a paste destination ----------
        # This is what makes the PNG-info "Send to JoyCaption" button work
        # for the *image* part of the hand-off. We deliberately pass an
        # empty fields list because the standard paste-fields machinery
        # ends up stringifying gr.update() dicts into our combined-prompt
        # textbox in this particular Gradio version (4.40 inside forge
        # Neo) -- the value that reaches the textbox is literally
        # "[{'value': '...', '__type__': 'update'}]" rather than the
        # prompt string. Instead, the text half of the hand-off is wired
        # by hand below: we attach a second .click() to the PNG-info
        # button that pulls the positive prompt out of the text component
        # ourselves and writes a plain string into txt2img_combined.
        if _infotext is not None:
            try:
                _infotext.add_paste_fields("joycaption", image_in, [])
            except Exception:
                traceback.print_exc()

        # Stash handles so the on_after_component hook (registered at
        # module load, below) can wire up the PNG-info button against
        # this tab's actual gradio components.
        global _JC_IMAGE_IN, _JC_COMBINED
        _JC_IMAGE_IN = image_in
        _JC_COMBINED = txt2img_combined

        # If the PNG-info "Send to JoyCaption" button was created before
        # this UI was built (which it always is -- built-in tabs come
        # first), wire its text-extraction click here. The button's
        # standard image-copy click was registered at button-creation
        # time; we chain a SECOND click handler that does the text part.
        # Multiple .click() listeners on one button are allowed -- they
        # fire as independent event chains.
        #
        # _PNGINFO_CLICK_WIRED guards against attaching duplicate
        # listeners if _build_ui ends up running more than once (which
        # can happen on some UI-reload paths). Without this guard a
        # second invocation would chain a second copy of the text
        # extractor onto the (still-the-same) button.
        global _PNGINFO_CLICK_WIRED
        if (_PNGINFO_SEND_BTN is not None and _PNGINFO_TEXT is not None
                and not _PNGINFO_CLICK_WIRED):
            try:
                def _extract_prompt_for_jc(infotext_str):
                    # Plain string in, plain string out. No gr.update
                    # wrapping anywhere in the chain.
                    return jcpi.extract_positive_prompt_from_text(infotext_str or "")

                _PNGINFO_SEND_BTN.click(
                    fn=_extract_prompt_for_jc,
                    inputs=[_PNGINFO_TEXT],
                    outputs=[txt2img_combined],
                    show_progress=False,
                )
                # Also switch to the JoyCaption tab so the user sees the
                # image + prompt land. The standard ParamBinding flow
                # tries to call switch_to_<tabname> in JS, but that
                # helper is only auto-generated for the four built-in
                # tabs; for our custom tab we need to do the switch
                # explicitly via a small JS shim defined in
                # javascript/joycaption.js.
                _PNGINFO_SEND_BTN.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    _js="joycaption_switch_to_tab",
                )
                _PNGINFO_CLICK_WIRED = True
            except Exception:
                traceback.print_exc()

    return ui


# ---------------------------------------------------------------------------
# Tab registration
# ---------------------------------------------------------------------------

# Handles to the JoyCaption tab's image and combined-prompt components.
# Populated by _build_ui() the first time the UI is constructed; consumed
# by the on_after_component hook below when the PNG-info tab is built.
_JC_IMAGE_IN = None
_JC_COMBINED = None

# PNG-info components we need to bind against. We watch for them via
# on_after_component because they live in a different tab built by the
# WebUI itself -- we have no direct gradio reference. Populated as the
# UI is constructed.
_PNGINFO_IMAGE = None
_PNGINFO_TEXT = None
_PNGINFO_SEND_BTN = None  # the "Send to JoyCaption" button we inject
_PNGINFO_BTN_REGISTERED = False
_PNGINFO_CLICK_WIRED = False  # second-click (text + tab-switch) wired?


def on_ui_tabs():
    try:
        ui = _build_ui()
        return [(ui, "JoyCaption", "joycaption")]
    except Exception:
        traceback.print_exc()
        return []


def _on_after_component(component, **kwargs):
    """Watches the rest of the UI being built and injects a 'Send to
    JoyCaption' button into the PNG-info tab once we've seen both its
    image input (elem_id='pnginfo_image') and its parsed-text display
    (elem_id='pnginfo_generation_info', 'html2', or 'html_info_pnginfo'
    depending on webui version).

    Crucial timing detail: built-in tabs (txt2img, ..., PNG info) are
    constructed BEFORE on_ui_tabs callbacks fire, so by the time our
    JoyCaption tab is being built the PNG-info Blocks context is closed
    and we can no longer add buttons to it. This function therefore
    creates the button *during* PNG-info construction, while that
    context is still open. The ParamBinding we register doesn't need
    JoyCaption to exist yet -- it just appends to a list that
    connect_paste_params_buttons() walks at the very end of UI setup,
    by which time both tabs are registered and the wiring can resolve.
    """
    global _PNGINFO_IMAGE, _PNGINFO_TEXT, _PNGINFO_SEND_BTN, _PNGINFO_BTN_REGISTERED

    if _infotext is None or _PNGINFO_BTN_REGISTERED:
        return

    elem_id = kwargs.get("elem_id") or getattr(component, "elem_id", None)
    if not elem_id:
        return

    if elem_id == "pnginfo_image":
        _PNGINFO_IMAGE = component
    # Different webui versions use different elem_ids for the metadata
    # display. We accept the first one that shows up.
    elif elem_id in ("pnginfo_generation_info", "html2", "html_info_pnginfo"):
        if _PNGINFO_TEXT is None:
            _PNGINFO_TEXT = component

    if _PNGINFO_IMAGE is None or _PNGINFO_TEXT is None:
        return

    # Skip our own button if we see it bouncing back through the
    # callback. (Creating gr.Button below re-fires after_component_callback
    # for the new button, which re-enters this function. The guard further
    # down -- set BEFORE the button is constructed -- catches that case;
    # this elem_id check is belt-and-braces in case any later edits move
    # the guard.)
    if elem_id == "pnginfo_send_to_joycaption":
        return

    # Both PNG-info components seen. The on_after_component callback is
    # firing inside the PNG-info Blocks context (i.e. right after the
    # component we just received was constructed). Adding a gr.Button
    # here drops it into the PNG-info tab next to the existing Send-to-X
    # buttons. The ParamBinding registers tabname="joycaption" -- when
    # JoyCaption later calls add_paste_fields("joycaption", ...), the
    # framework resolves the link.
    #
    # CRITICAL: set the guard BEFORE creating the button. Constructing a
    # gradio component inside an after_component_callback re-fires that
    # same callback for the new component, which re-enters this function.
    # If we don't flip the guard first, every re-entry creates another
    # button (and another, and another -- it doesn't truly recurse
    # forever because forge eventually hits a different RecursionError
    # in its own error reporting path, but we still pile up dozens of
    # ghost buttons on the PNG-info tab before that happens).
    _PNGINFO_BTN_REGISTERED = True
    try:
        btn = gr.Button("Send to JoyCaption", elem_id="pnginfo_send_to_joycaption")
        _infotext.register_paste_params_button(_infotext.ParamBinding(
            paste_button=btn,
            tabname="joycaption",
            source_image_component=_PNGINFO_IMAGE,
            # source_text_component intentionally omitted -- we wire the
            # text-extraction half ourselves in _build_ui (see the
            # add_paste_fields call there), bypassing the standard
            # paste_fields machinery which stringifies gr.update() dicts
            # into our textbox in this Gradio version.
        ))
        _PNGINFO_SEND_BTN = btn
        print("[joycaption] registered 'Send to JoyCaption' button on PNG-info tab")
    except Exception:
        # If button creation failed, allow another attempt later -- a
        # subsequent component might still trigger a successful retry.
        _PNGINFO_BTN_REGISTERED = False
        traceback.print_exc()


# Register hooks. The on_after_component hook is what wires the PNG-info
# button; we also need on_ui_tabs to build the JoyCaption tab itself.
script_callbacks.on_ui_tabs(on_ui_tabs)
try:
    script_callbacks.on_after_component(_on_after_component)
except Exception:
    # Older WebUIs might not expose this; the PNG-info button just
    # won't be created. The JoyCaption tab still works.
    traceback.print_exc()
