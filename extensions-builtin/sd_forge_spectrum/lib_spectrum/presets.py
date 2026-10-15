import os.path
import threading
from json import dump, load
from typing import Final

from lib_spectrum import logger

PRESET_FILE: Final[os.PathLike] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets.json")
# Infotext key of the enable switch: written only when Spectrum runs, so
# pasting (UI, "Send to", API infotext) turns Spectrum back on. It is not a
# CONTROLS entry: presets store the settings, never the switch.
ENABLE_KEY: Final[str] = "spec_enabled"
# The script arguments after ``enable``, in API order:
# (infotext key, Anima preset control name, type, default)
CONTROLS: Final[tuple] = (
    ("spec_w", "spectrum.weight", float, 0.25),
    ("spec_m", "spectrum.degree", int, 4),
    ("spec_lam", "spectrum.regularization", float, 0.1),
    ("spec_window_size", "spectrum.window", int, 2),
    ("spec_flex_window", "spectrum.window_growth", float, 0.0),
    ("spec_warmup_steps", "spectrum.warmup", int, 6),
    ("spec_stop_caching_step", "spectrum.stop", float, 0.9),
    ("spec_tail_actual_steps", "spectrum.tail_actual", int, 3),
    ("spec_history_size", "spectrum.history", int, 10),
    ("spec_schedule", "spectrum.schedule", str, "Window"),
    ("spec_refresh_ratio", "spectrum.refresh_ratio", float, 0.0),
    ("spec_sea_beta", "spectrum.sea_beta", float, 2.0),
    ("spec_compat_policy", "spectrum.policy", str, "Conservative"),
    ("spec_verbose", "spectrum.verbose", bool, False),
)
INFOTEXT_KEYS: Final[tuple[str, ...]] = tuple(control[0] for control in CONTROLS)
PARAMS: Final[list[type]] = [control[2] for control in CONTROLS]
DEFAULTS: Final[list] = [control[3] for control in CONTROLS]
DEFAULT: Final[dict] = dict(zip(INFOTEXT_KEYS, DEFAULTS))
# Save / Delete are server actions run in a thread pool: one writer at a time.
_WRITE_LOCK = threading.Lock()


class PresetManager:
    presets: dict[str, list[float]] = None

    @classmethod
    def load_presets(cls):
        if cls.presets is not None:
            return

        if not os.path.isfile(PRESET_FILE):
            with open(PRESET_FILE, "w+", encoding="utf-8") as json_file:
                dump({}, json_file)

            logger.debug("Creating new empty Presets...")
            cls.presets = {}
            return

        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as json_file:
                cls.presets = load(json_file)
        except Exception:
            logger.error("Failed to load Presets...")
            cls.presets = {}
        else:
            logger.debug("Loaded Presets...")

    @classmethod
    def list_preset(cls) -> list[str]:
        return list(cls.presets.keys())

    @classmethod
    def get_preset(cls, preset_name: str) -> list | None:
        """The settings saved as ``preset_name`` in CONTROLS order (None when
        there is no such preset)."""

        if (preset := cls.presets.get(preset_name, None)) is None:
            logger.error(f'Preset "{preset_name}" was not found...')
            return None

        # Presets from the previous seven-control Spectrum UI remain valid.
        # A hand-edited entry that does not convert raises ValueError.
        try:
            values = [*preset, *DEFAULTS[len(preset) :]]
            return [obj(val) for obj, val in zip(PARAMS, values)]
        except (TypeError, ValueError) as error:
            raise ValueError(f'Preset "{preset_name}" is invalid: {error}') from None

    @classmethod
    def save_preset(cls, preset_name: str, *args) -> list[str] | None:
        """Save the settings ``args`` (CONTROLS order); returns the preset
        names, or None for an empty name."""

        if preset_name is None or not preset_name.strip():
            logger.error("Invalid Preset Name...")
            return None

        with _WRITE_LOCK:
            cls.presets.update({preset_name: [*args]})
            cls._write()
            names = cls.list_preset()

        logger.info(f'Preset "{preset_name}" Saved!')
        return names

    @classmethod
    def delete_preset(cls, preset_name: str) -> list[str] | None:
        """Delete ``preset_name``; returns the remaining preset names, or None
        when there is no such preset."""

        with _WRITE_LOCK:
            if preset_name not in cls.presets:
                logger.error(f'Preset "{preset_name}" was not found...')
                return None
            del cls.presets[preset_name]
            cls._write()
            names = cls.list_preset()

        logger.info(f'Preset "{preset_name}" Deleted!')
        return names

    @classmethod
    def _write(cls) -> None:
        with open(PRESET_FILE, "w", encoding="utf-8") as json_file:
            dump(cls.presets, json_file)
