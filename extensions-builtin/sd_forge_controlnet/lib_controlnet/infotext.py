from lib_controlnet import external_code
from lib_controlnet.logging import logger

from modules.processing import StableDiffusionProcessing


def field_to_displaytext(fieldname: str) -> str:
    return " ".join([word.capitalize() for word in fieldname.split("_")])


def displaytext_to_field(text: str) -> str:
    return "_".join([word.lower() for word in text.split(" ")])


def parse_value(value: str) -> bool | int | float | str:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def serialize_unit(unit: external_code.ControlNetUnit) -> str:
    log_value = {field_to_displaytext(field): getattr(unit, field) for field in external_code.ControlNetUnit.infotext_fields() if getattr(unit, field) != -1}

    if not all("," not in str(v) and ":" not in str(v) for v in log_value.values()):
        logger.error(f"Unexpected tokens encountered:\n{log_value}")
        return ""

    return ", ".join(f"{field}: {value}" for field, value in log_value.items())


def parse_unit_fields(text: str) -> dict:
    """The infotext fields (``ControlNetUnit.infotext_fields()``) a unit's
    entry sets, e.g. ``"Module: canny, Weight: 1.0"`` -> ``{"module":
    "canny", "weight": 1.0}``; ``ValueError`` for a malformed entry."""

    known = set(external_code.ControlNetUnit.infotext_fields())
    fields = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, separator, value = item.strip().partition(": ")
        if not separator:
            raise ValueError(f"cannot read {item.strip()!r}")
        field = displaytext_to_field(key)
        if field in known:
            fields[field] = parse_value(value)
    return fields


class Infotext:
    @staticmethod
    def unit_prefix(unit_index: int) -> str:
        return f"ControlNet {unit_index}"

    @staticmethod
    def write_infotext(units: list[external_code.ControlNetUnit], p: StableDiffusionProcessing):
        """Write infotext to p"""
        p.extra_generation_params.update({Infotext.unit_prefix(i): serialize_unit(unit) for i, unit in enumerate(units) if unit.enabled})
