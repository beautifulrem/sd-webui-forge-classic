import ast
from pathlib import Path

import pytest

# main_entry only imports inside the initialised WebUI; the migration is pure
_SOURCE = Path(__file__).parents[1] / "modules_forge" / "main_entry.py"
_tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_names = {"_migrate_scaled_fp8_setting", "anima_quant_options"}
_nodes = [n for n in _tree.body if getattr(n, "name", None) in _names or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) in _names)]
_ns = {}
exec(compile(ast.Module(_nodes, []), str(_SOURCE), "exec"), _ns)  # noqa: S102 - our own source file
migrate, anima_quant_options = _ns["_migrate_scaled_fp8_setting"], _ns["anima_quant_options"]


class _Options:
    def __init__(self, data, dtype):
        self.data = dict(data)
        self.forge_unet_storage_dtype = dtype
        self.saved = 0

    def set(self, key, value):
        setattr(self, key, value)

    def save(self, filename):
        self.saved += 1


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("float8-e4m3fn", "float8-e4m3fn (Anima scaled)"),
        ("float8-e4m3fn (fp16 LoRA)", "float8-e4m3fn (Anima scaled, fp16 LoRA)"),
        ("Automatic", "Automatic"),
    ],
)
def test_legacy_scaled_fp8_setting_becomes_a_low_bits_choice(dtype, expected):
    options = _Options({"anima_scaled_fp8": True}, dtype)

    migrate(options, "config.json")

    assert options.forge_unet_storage_dtype == expected
    assert "anima_scaled_fp8" not in options.data and options.saved == 1
    assert expected == "Automatic" or expected in anima_quant_options


def test_nothing_happens_without_the_legacy_setting():
    options = _Options({}, "float8-e4m3fn")

    migrate(options, "config.json")

    assert options.forge_unet_storage_dtype == "float8-e4m3fn" and options.saved == 0
