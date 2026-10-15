"""Loads ``detail_daemon_testlib`` before collection (it prepares the WebUI imports),
registered under its unique name for the test modules."""

import importlib.util
import sys
from pathlib import Path

if "detail_daemon_testlib" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("detail_daemon_testlib", Path(__file__).with_name("detail_daemon_testlib.py"))
    sys.modules["detail_daemon_testlib"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["detail_daemon_testlib"])
