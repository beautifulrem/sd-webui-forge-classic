"""``modules.redact``: the fallbacks the security suite does not reach
(tests/test_remi_security.py covers the regular paths)."""

from __future__ import annotations

import sys

from modules import redact


def test_known_flags_fall_back_to_the_static_lists(monkeypatch):
    monkeypatch.setitem(sys.modules, "modules.cmd_args", None)  # the launch parser cannot be imported
    assert set(redact._known_flags()) == {*redact.SECRET_FLAGS, "--api", "--auth-file", "--ngrok-region"}
    # abbreviations still resolve against the static lists
    assert redact.argv(["--api-a", "a:b", "--api", "--ngrok-r", "eu"]) == ["--api-a", redact.HIDDEN, "--api", "--ngrok-r", "eu"]


def test_known_flags_come_from_the_launch_parser():
    flags = redact._known_flags()
    assert "--auth" in flags and "--listen" in flags and all(flag.startswith("--") for flag in flags)


def test_an_unparsable_command_line_is_hidden_when_it_may_hold_a_secret():
    assert redact.command_line("--listen --auth 'a:b") == redact.HIDDEN  # unbalanced quote
    assert redact.command_line("--ckpt 'https://u:tok@host/m") == f"--ckpt 'https://{redact.HIDDEN}@host/m"
    assert redact.command_line("--listen 'x") == "--listen 'x"
