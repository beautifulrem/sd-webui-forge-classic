"""Loads ``adetailer_testlib`` before collection (it prepares the WebUI imports),
registered under its unique name for the test modules."""

import importlib.util
import sys
from pathlib import Path

if "adetailer_testlib" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("adetailer_testlib", Path(__file__).with_name("adetailer_testlib.py"))
    sys.modules["adetailer_testlib"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["adetailer_testlib"])
