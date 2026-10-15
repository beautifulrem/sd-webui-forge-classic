"""Bootstrap, schemas, actions, presets and conflict rules, infotext, i18n,
models, styles, token counts, options and extension pages."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response

from remi.server import i18n
from remi.server import models as m
from remi.server.deps import ServicesDep, UserDep, not_found, opts, publish_options_changed
from remi.server.runtime import BUILTIN_TABS

router = APIRouter()

BOOTSTRAP_OPTIONS = (
    "forge_preset",
    "sd_model_checkpoint",
    "forge_additional_modules",
    "forge_unet_storage_dtype",
    "localization",
    "quicksettings_list",
    "ui_reorder_list",
    "ui_tab_order",
    "hidden_tabs",
    "samples_format",
    "return_grid",
    "do_not_show_images",
    "gallery_height",
    "live_previews_enable",
    "live_previews_image_format",
    "live_preview_fps",
    "show_progress_every_n_steps",
    "show_progressbar",
    "show_progress_in_title",
    "js_modal_lightbox",
    "js_modal_lightbox_initially_zoomed",
    "js_live_preview_in_modal_lightbox",
    "notification_audio",
    "notification_volume",
    "keyedit_precision_attention",
    "keyedit_precision_extra",
    "keyedit_delimiters",
    "keyedit_delimiters_whitespace",
    "keyedit_move",
    "disable_token_counters",
    "include_styles_into_token_counters",
    "extra_networks_default_multiplier",
    "extra_networks_add_text_separator",
    "sd_checkpoint_dropdown_use_short",
    "sd_modules_dropdown_use_short",
    "lora_preset_filter",
    "extra_networks_card_order_field",  # the network browser's default sort
    "extra_networks_card_order",
    "ctrl_enter_interrupt",
    "prevent_screen_sleep_during_generation",
    "send_seed",
    "send_size",
    "live_preview_refresh_period",
    "confirm_leave",
    # The image editor (webui/src/canvas)
    "forge_canvas_consistent_brush",
    "forge_canvas_plain",
    "forge_canvas_plain_color",
    "forge_canvas_height",
    "forge_canvas_toolbar_always",
)

FEATURES = ("jobs.v1", "ws.v1", "uploads.v1", "outputs.v1", "networks.v1", "options.v1", "styles.v1", "tokens.v1", "presets.v1")


# region Pages registry (remi.pages, when present)


def _pages_module() -> Any:
    try:
        from remi import pages
    except ImportError:
        return None
    return pages


def _registered_pages() -> list[Any]:
    pages = _pages_module()
    if pages is None or not hasattr(pages, "registered"):
        return []
    return list(pages.registered())


def _page(page_id: str) -> Any:
    pages = _pages_module()
    page = pages.get(page_id) if pages is not None and hasattr(pages, "get") else None
    if page is None:
        raise not_found(f"no page {page_id!r}")
    return page


# endregion


def builtin_tabs() -> list[str]:
    """Route ids of the built-in tabs in the user's order, hidden ones out."""

    options = opts()
    order = {title: index for index, title in enumerate(getattr(options, "ui_tab_order", []) or [])}
    hidden = set(getattr(options, "hidden_tabs", []) or [])
    tabs = sorted(BUILTIN_TABS, key=lambda item: order.get(item[1], len(order) + BUILTIN_TABS.index(item)))
    return [route for route, title in tabs if title not in hidden]


def models_summary() -> dict[str, Any]:
    from remi.networks import module_roots

    options = opts()
    modules = [path for path in getattr(options, "forge_additional_modules", []) or [] if isinstance(path, str)]
    roots = module_roots()

    def under(path: str, kind: str) -> bool:
        real = os.path.abspath(path)
        return any(real.startswith(os.path.abspath(root).rstrip(os.sep) + os.sep) for root in roots[kind] if root)

    vae = [os.path.basename(path) for path in modules if under(path, "vae")]
    return {
        "checkpoint": getattr(options, "sd_model_checkpoint", None) or None,
        "vae": vae[0] if vae else None,
        "text_encoders": [os.path.basename(path) for path in modules if not under(path, "vae")],
        "modules": [os.path.basename(path) for path in modules],
        "preset": getattr(options, "forge_preset", None),
        "unet_dtype": getattr(options, "forge_unet_storage_dtype", None),
    }


@router.get("/app", response_model=m.Bootstrap, summary="Bootstrap: version, features, tabs, locales, options subset, models")
def bootstrap(request: Request, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    from modules import shared
    from modules_forge import forge_version
    from remi.server.routes.system import user_assets
    from remi.server.runtime import not_ported

    options = opts()
    auth = request.app.state.auth
    features = list(FEATURES)
    if auth.required:
        features.append("auth")
    if request.app.state.sdapi:
        features.append("sdapi")
    if _pages_module() is not None:
        features.append("pages")
    if not getattr(shared.cmd_opts, "hide_ui_dir_config", False):
        features.append("folders.open")  # "open folder" actions (networks, gallery)
    pages = [{"id": page.id, "title": page.title, "icon": getattr(page, "icon", None), "widget": getattr(page, "widget", None)} for page in _registered_pages()]
    return {
        "version": f"{forge_version.version} {forge_version.release}",
        "features": features,
        "tabs": builtin_tabs(),
        "pages": pages,
        "locales": i18n.locales(),
        "locale": i18n.default_locale(),
        "options": {key: options.data.get(key, info.default) for key in BOOTSTRAP_OPTIONS if (info := options.data_labels.get(key)) is not None},
        "models": models_summary(),
        "not_ported": not_ported(),
        "auth": {"required": auth.required, "user": user},
        "queue": services.queue.status(),
        "user_assets": sorted(user_assets()),
    }


@router.get("/schema/{tab}", response_model=m.TabSchema, response_model_exclude_unset=True, summary="A generation tab's core form and scripts (schema-format.md §4)")
def schema(tab: m.FormTab) -> dict[str, Any]:
    from remi.core.tabs import tab_schema

    return tab_schema(tab)


@router.get("/pages", response_model=list[m.PageInfo], summary="Extension pages")
def pages() -> list[dict[str, Any]]:
    return [{"id": page.id, "title": page.title, "icon": getattr(page, "icon", None), "widget": getattr(page, "widget", None)} for page in _registered_pages()]


@router.get("/pages/{page_id}", response_model=m.PageSchema, response_model_exclude_unset=True, summary="One extension page's schema (schema-format.md §8)")
def page_schema(page_id: str) -> dict[str, Any]:
    pages_module = _pages_module()
    _page(page_id)
    return pages_module.page_schema(page_id)


# region Actions, rules, presets


def _action_target(owner: str, tab: str | None) -> Any:
    from remi.core.actions import core_actions

    if owner == "core":
        return core_actions
    if owner.startswith("page:"):
        return _page(owner[5:])
    from remi.core.tabs import runner_for

    tab = tab or "txt2img"
    try:
        runner = runner_for(tab)
    except KeyError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if runner is None:
        from modules import scripts

        from remi.core.extras import assign_args

        for operation, layout in assign_args(scripts.scripts_postproc):
            if layout.owner == owner:
                return operation
    elif owner in runner.by_id:
        return runner.by_id[owner]
    raise not_found(f"no owner {owner!r} on {tab}")


@router.post(
    "/actions/{owner}/{action}",
    response_model=m.ActionPatch,
    responses={422: {"model": m.ValidationErrorBody}, 404: {"model": m.Detail}},
    summary="Run an owner's action; returns a patch (schema-format.md §5)",
)
def run_action(owner: str, action: str, body: m.ActionRequest, request: Request, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    from remi import ui

    extra = dict(body.model_extra or {})
    if owner == "settings":
        return _settings_action(action, services)
    target = _action_target(owner, body.tab)
    spec = ui.actions_of(target).get(action)
    if spec is None:
        raise not_found(f"{owner} has no action {action!r}")
    core = body.values if owner == "core" else (body.core or {})
    ctx = ui.ActionContext(tab=body.tab, owner=owner, core=core, scripts=body.scripts or {}, user=user, request=request, extra=extra)

    def call() -> dict[str, Any]:
        return ui.call_action(target, action, ctx, body.values)

    if spec.gpu:
        with services.queue.lock:
            patch = services.queue.dispatch(call)
    else:
        patch = call()
    settings = patch.get("__settings__")
    if isinstance(settings, dict):
        publish_options_changed(services, list(settings))
    return patch


def _settings_action(action: str, services: Any) -> dict[str, Any]:
    from remi.ui.settings import refresh_setting

    if not action.startswith("refresh:"):
        raise not_found(f"settings has no action {action!r}")
    key = action.split(":", 1)[1]
    options = opts()
    if key not in options.data_labels or options.data_labels[key].refresh is None:
        raise not_found(f"setting {key!r} has no refresh action")
    return refresh_setting(options, key)


@router.post("/rules/derive", response_model=m.ActionPatch, responses={422: {"model": m.ValidationErrorBody}}, summary="Anima conflict locks for the current form values")
def derive_rules(body: m.RulesRequest) -> dict[str, Any]:
    from remi.core.conflicts import conflict_patch

    return conflict_patch(body.tab, {**body.scripts, "core": body.core}, body.selected)


@router.get("/presets", response_model=m.PresetList, summary="Anima presets and UI presets")
def preset_list() -> dict[str, Any]:
    from modules.anima_presets import PRESETS, SAFE_BASE_AESTHETIC_LABEL
    from modules_forge.presets import PresetArch

    return {"anima": [SAFE_BASE_AESTHETIC_LABEL, *[label for label in PRESETS if label != SAFE_BASE_AESTHETIC_LABEL]], "forge": PresetArch.choices(), "current": getattr(opts(), "forge_preset", None)}


@router.get("/presets/{name:path}", response_model=m.ActionPatch, summary="The form patch of an Anima preset")
def preset_patch(name: str, tab: m.FormTab = "txt2img") -> dict[str, Any]:
    from modules.anima_presets import PRESETS, SAFE_BASE_AESTHETIC_LABEL
    from remi.core.presets import anima_preset_patch

    if name != SAFE_BASE_AESTHETIC_LABEL and name not in PRESETS:
        raise not_found(f"no preset {name!r}")
    if tab == "extras":
        raise HTTPException(status_code=422, detail="presets apply to txt2img and img2img")
    return anima_preset_patch(tab, name)


# endregion

# region Infotext


@router.post("/infotext/parse", response_model=m.InfotextParsed, summary="Infotext to named values (schema-format.md §7)")
def parse_infotext(body: m.InfotextRequest) -> dict[str, Any]:
    from remi.core.infotext import parse_infotext as parse

    return parse(body.text, body.tab)


@router.post("/infotext/image", response_model=m.ImageInfo, responses={404: {"model": m.Detail}}, summary="PNG Info of an uploaded or generated image")
def image_info(body: m.ImageInfoRequest, services: ServicesDep) -> dict[str, Any]:
    from PIL import Image

    from modules import images
    from remi.core.infotext import parse_infotext as parse

    record = services.files.get(body.file_id)
    if record is None or not record.is_image:
        raise not_found("no such image")
    with Image.open(record.path) as image:
        infotext, items = images.read_info_from_image(image)
    parsed = parse(infotext, body.tab) if infotext else None
    return {"infotext": infotext, "items": {str(k): str(v) for k, v in items.items() if not isinstance(v, bytes)}, "parsed": parsed}


# endregion

# region i18n


@router.get("/i18n/{locale}", response_model=dict[str, str], responses={404: {"model": m.Detail}}, summary="A merged locale catalog (English text → translation)")
def locale_catalog(locale: str, response: Response, if_none_match: str | None = Header(None)) -> Any:
    name = i18n.normalize(locale)
    if name is None:
        raise not_found(f"no locale {locale!r}")
    data, etag = i18n.catalog(name)
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return data


# endregion

# region Models and styles


@router.get("/models", response_model=m.Models, summary="Checkpoints, modules, samplers, schedulers, upscalers, styles, presets")
def list_models() -> dict[str, Any]:
    from modules import sd_models, sd_samplers, sd_schedulers, shared
    from modules_forge.presets import PresetArch
    from remi.networks import module_files
    from remi.server.runtime import unet_dtypes

    def module_refs(kind: str) -> list[dict[str, Any]]:
        return [{"name": name, "title": os.path.relpath(path, root).replace("\\", "/"), "size": _size(path)} for name, path, root in module_files(kind)]

    vae, text_encoders = module_refs("vae"), module_refs("text_encoders")
    return {
        "checkpoints": [{"name": info.title, "title": info.title, "hash": info.shorthash, "size": _size(info.filename)} for info in list(sd_models.checkpoints_list.values())],
        "vae": vae,
        "text_encoders": text_encoders,
        "modules": sorted(vae + text_encoders, key=lambda ref: ref["name"].lower()),
        "samplers": [{"name": sampler.name, "aliases": list(sampler.aliases or [])} for sampler in sd_samplers.visible_samplers()],
        "schedulers": [{"name": scheduler.name, "label": scheduler.label} for scheduler in sd_schedulers.schedulers],
        "upscalers": [{"name": upscaler.name} for upscaler in shared.sd_upscalers or []],
        "styles": _styles(),
        "unet_dtypes": unet_dtypes(),
        "presets": PresetArch.choices(),
    }


def _size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


@router.post("/models/refresh", response_model=m.Ok, summary="Re-scan checkpoints, VAE and text encoders")
def refresh_models(services: ServicesDep) -> dict[str, Any]:
    from remi.server.runtime import refresh_models as refresh

    refresh()
    kinds = ("checkpoints", "vae", "text_encoders")
    services.networks.invalidate(*kinds)  # the registries were just re-scanned
    services.bus.publish("options.changed", keys=["models"])
    for kind in kinds:
        services.bus.publish("networks.changed", kind=kind)
    return {"ok": True}


@router.post("/models/unload", response_model=m.Ok, summary="Unload the model from memory (behind generation)")
def unload_model(services: ServicesDep) -> dict[str, Any]:
    from modules import sd_models

    with services.queue.lock:
        services.queue.dispatch(sd_models.unload_model_weights)
    return {"ok": True}


def _styles() -> list[dict[str, Any]]:
    from modules import shared

    return [{"name": style.name, "prompt": style.prompt or "", "negative_prompt": style.negative_prompt or ""} for style in shared.prompt_styles.styles.values()]


@router.get("/styles", response_model=list[m.PromptStyle], summary="Prompt styles")
def list_styles() -> list[dict[str, Any]]:
    return _styles()


@router.put("/styles/{name}", response_model=m.PromptStyle, summary="Create or replace a style")
def save_style(name: str, body: m.StyleBody, services: ServicesDep) -> dict[str, Any]:
    from modules import shared, styles

    name = name.strip()
    if not name or name == "None":
        raise HTTPException(status_code=422, detail="a style needs a name")
    database = shared.prompt_styles
    existing = database.styles.get(name)
    database.styles[name] = styles.PromptStyle(name, body.prompt, body.negative_prompt, existing.path if existing else str(database.default_path))
    database.save_styles()
    services.bus.publish("options.changed", keys=["styles"])
    return {"name": name, "prompt": body.prompt, "negative_prompt": body.negative_prompt}


@router.delete("/styles/{name}", status_code=204, responses={404: {"model": m.Detail}}, summary="Delete a style")
def delete_style(name: str, services: ServicesDep) -> Response:
    from modules import shared

    if shared.prompt_styles.styles.pop(name, None) is None:
        raise not_found(f"no style {name!r}")
    shared.prompt_styles.save_styles()
    services.bus.publish("options.changed", keys=["styles"])
    return Response(status_code=204)


@router.post("/styles/apply", response_model=m.StylesApplied, summary="Write styles into the prompts")
def apply_styles(body: m.StylesApplyRequest) -> dict[str, Any]:
    from modules import shared

    database = shared.prompt_styles
    return {"prompt": database.apply_styles_to_prompt(body.prompt, body.styles), "negative_prompt": database.apply_negative_styles_to_prompt(body.negative_prompt, body.styles)}


@router.post("/prompt/tokens", response_model=m.TokenCount, summary="Token count of a prompt (null while no model is loaded)")
def count_tokens(body: m.TokenCountRequest) -> dict[str, Any]:
    return token_count(body.text, body.steps, body.styles, positive=not body.negative)


def token_count(text: str, steps: int, styles: list[str], *, positive: bool = True) -> dict[str, Any]:
    from functools import reduce

    from modules import extra_networks, prompt_parser, script_callbacks, sd_models, shared
    from modules.processing_scripts.comments import strip_comments

    params = script_callbacks.BeforeTokenCounterParams(text, steps, styles, is_positive=positive)
    script_callbacks.before_token_counter_callback(params)
    text, steps, styles, positive = params.prompt, params.steps, params.styles, params.is_positive
    if shared.opts.include_styles_into_token_counters:
        apply = shared.prompt_styles.apply_styles_to_prompt if positive else shared.prompt_styles.apply_negative_styles_to_prompt
        text = apply(text, styles)
    text = strip_comments(text).strip()
    try:
        text, _ = extra_networks.parse_prompt(text)
        flat = prompt_parser.get_multicond_prompt_list([text])[1] if positive else [text]
        schedules = prompt_parser.get_learned_conditioning_prompt_schedules(flat, steps)
    except Exception:
        schedules = [[[steps, text]]]
    lengths = getattr(getattr(sd_models.model_data, "sd_model", None), "get_prompt_lengths_on_ui", None)
    if lengths is None:
        return {"tokens": None, "max_length": None}
    from backend.text_processing.parsing import strip_emphasis

    prompts = [strip_emphasis(prompt) for _, prompt in reduce(lambda a, b: a + b, schedules)]
    tokens, max_length = max((lengths(prompt) for prompt in prompts), key=lambda item: item[0])
    return {"tokens": int(tokens), "max_length": int(max_length)}


# endregion

# region Options


def _settings_schema() -> dict[str, Any]:
    from modules import shared
    from remi.server.options import frozen_reason
    from remi.ui.settings import settings_schema

    options = opts()
    return settings_schema(options, locked=lambda key: frozen_reason(options, key, shared.cmd_opts))


@router.get("/options", response_model=m.SettingsSchema, summary="Settings layout and values (schema-format.md §6)")
def get_options() -> dict[str, Any]:
    return _settings_schema()


@router.patch("/options", response_model=m.SettingsSchema, responses={422: {"model": m.ValidationErrorBody}}, summary="Change settings (validated; saved; broadcast)")
def patch_options(values: dict[str, Any], services: ServicesDep) -> dict[str, Any]:
    from modules import shared
    from remi.server.options import apply_values, coerce_values
    from remi.ui import ValidationError

    options = opts()
    coerced = coerce_values(options, values, shared.cmd_opts)
    changed, failed = apply_values(options, coerced, None if shared.cmd_opts.freeze_settings else shared.config_filename)
    publish_options_changed(services, changed)
    if failed:
        raise ValidationError(failed)
    return _settings_schema()


# endregion
