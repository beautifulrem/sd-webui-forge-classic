"""Settings of the Remi server, added to the WebUI options."""

from __future__ import annotations

from typing import Any


def server_options() -> dict[str, Any]:
    from modules.options import OptionInfo, options_section
    from remi import ui

    return {
        **options_section(
            ("preview", "Live Previews", "ui"),
            {
                "live_preview_fps": OptionInfo(6, "Live preview frames per second", ui.Slider, {"minimum": 1, "maximum": 30, "step": 1}).info("WebSocket previews; only the newest frame is sent"),
            },
        ),
        **options_section(
            ("system", "System", "system"),
            {
                "remi_upload_max_mb": OptionInfo(64, "Largest upload", ui.Slider, {"minimum": 1, "maximum": 2048, "step": 1}).info("in MB"),
                "remi_upload_retention_days": OptionInfo(7, "Keep uploads and unsaved images for", ui.Slider, {"minimum": 1, "maximum": 365, "step": 1}).info("days; uploads still used by queued jobs are kept; images generated with saving turned off are deleted after this"),
            },
        ),
    }


def register_options(opts: Any = None) -> None:
    """Add the server's options (once; values already loaded are kept)."""

    if opts is None:
        from modules import shared

        opts = shared.opts
    for key, info in server_options().items():
        if key not in opts.data_labels:
            opts.add_option(key, info)


def option(name: str, default: Any) -> Any:
    from modules import shared

    value = getattr(shared.opts, name, default)
    return default if value is None else value
