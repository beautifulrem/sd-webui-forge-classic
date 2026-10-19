"""Registers the "Checkpoint Merger" page (``remi/core/merger.py``) when the
scripts load, so it follows the script life cycle like extension pages: a
"Reload UI" drops it and registers it again."""

from remi.core import merger

merger.register()
