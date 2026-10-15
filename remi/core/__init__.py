"""Remi's core generation model on top of ``remi.ui``: the txt2img/img2img/
extras forms, the tab schema, infotext parsing, presets and conflict locks.

Each submodule is importable without Gradio; the WebUI modules it reads
(``modules.shared``, ``modules.scripts``, ...) are imported when used.
"""

from remi.core.job import JobContext

__all__ = ["JobContext"]
