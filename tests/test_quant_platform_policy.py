"""Backend capability policy tests that do not require a GPU or PyTorch."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, relative, "exec"), namespace)
    return namespace[name]


class QuantPlatformPolicyTests(unittest.TestCase):
    def test_int8_policy_and_fallback_formats(self):
        for backend in ("mps", "xpu", "directml", "cuda", "cpu"):
            with self.subTest(backend=backend):
                namespace = {
                    "is_device_mps": lambda device: device == "mps",
                    "mps_mode": lambda: backend == "mps",
                    "is_intel_xpu": lambda: backend == "xpu",
                    "is_directml_enabled": lambda: backend == "directml",
                }
                supports_int8 = load_function("backend/memory_management.py", "supports_int8_compute", namespace)
                expected = backend in ("cuda", "cpu")
                self.assertEqual(supports_int8(backend), expected)
                management = SimpleNamespace(
                    supports_int8_compute=supports_int8,
                    supports_nvfp4_compute=lambda device: False,
                    supports_mxfp8_compute=lambda device: False,
                    supports_fp8_compute=lambda device: False,
                )
                disabled = load_function("backend/operations.py", "_get_disabled_quants", {"memory_management": management})(backend)
                self.assertEqual(bool(disabled & {"int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8"}), not expected)
                self.assertTrue({"float8_e4m3fn", "float8_e5m2", "nvfp4", "mxfp8"} <= disabled)

    def test_explicit_mps_device_disables_int8_outside_mps_mode(self):
        namespace = {
            "is_device_mps": lambda device: device == "mps",
            "mps_mode": lambda: False,
            "is_intel_xpu": lambda: False,
            "is_directml_enabled": lambda: False,
        }
        supports_int8 = load_function("backend/memory_management.py", "supports_int8_compute", namespace)
        self.assertFalse(supports_int8("mps"))
        self.assertTrue(supports_int8("cuda"))


if __name__ == "__main__":
    unittest.main()
