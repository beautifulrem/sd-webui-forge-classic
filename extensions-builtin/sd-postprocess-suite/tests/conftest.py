"""Loads ``pp_suite_testlib`` before collection (it prepares the WebUI imports),
registered under its unique name for the test modules."""

import importlib.util
import sys
from pathlib import Path

if "pp_suite_testlib" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("pp_suite_testlib", Path(__file__).with_name("pp_suite_testlib.py"))
    sys.modules["pp_suite_testlib"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["pp_suite_testlib"])
