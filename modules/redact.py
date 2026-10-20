"""Keeping secrets out of logs and the sysinfo dump: login flags, tokens,
API keys, and credentials inside URLs."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from typing import Any

HIDDEN = "<hidden>"

SECRET_FLAGS = ("--auth", "--api-auth", "--ngrok", "--ngrok-options", "--gradio-auth")
"""command-line flags whose value is a password or token. ``--gradio-auth``
(the old name of ``--auth``) is no option any more, but an old launch command
can still carry it: the launcher prints the arguments (``launch_utils``)
before the parser refuses the unknown option, and ``COMMANDLINE_ARGS`` goes
into the sysinfo dump, so its password stays hidden there."""

_OTHER_FLAGS = ("--api", "--auth-file", "--ngrok-region")
"""flags sharing a prefix with a secret one (argparse accepts unique prefixes)"""


def _known_flags() -> tuple[str, ...]:
    """Every long option of the launch parser (the static lists if it cannot be imported)."""

    try:
        from modules.cmd_args import parser
    except Exception:
        return (*SECRET_FLAGS, *_OTHER_FLAGS)
    return tuple(option for action in parser._actions for option in action.option_strings if option.startswith("--"))

SECRET_KEY_RE = re.compile(r"(?:^|[_.-])(api_?key|key|token|secret|password|passwd|credentials?|auth)(?:$|[_.-])", re.I)
_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@")


def _is_secret_flag(token: str) -> bool:
    if token in SECRET_FLAGS:
        return True
    if not token.startswith("--") or len(token) < 5:
        return False
    known = _known_flags()
    if token in known:  # an exact non-secret flag, e.g. --api (a prefix of --api-auth)
        return False
    # an abbreviation argparse resolves to a secret flag (and to nothing else)
    candidates = [flag for flag in {*known, *SECRET_FLAGS, *_OTHER_FLAGS} if flag.startswith(token)]
    return bool(candidates) and all(flag in SECRET_FLAGS for flag in candidates)


def url(text: str) -> str:
    """``https://user:token@host/x`` → ``https://<hidden>@host/x`` (anywhere in ``text``)."""

    return _URL_CREDENTIALS_RE.sub(lambda match: f"{match['scheme']}{HIDDEN}@", text)


def argv(args: Iterable[str]) -> list[str]:
    """Command-line arguments with the values of :data:`SECRET_FLAGS`
    hidden (``--auth x``, ``--auth=x`` and abbreviations)."""

    result: list[str] = []
    hide_next = False
    for arg in args:
        if hide_next:
            result.append(HIDDEN)
            hide_next = False
            continue
        flag, sep, _value = arg.partition("=")
        if _is_secret_flag(flag):
            if sep:
                result.append(f"{flag}={HIDDEN}")
            else:
                result.append(arg)
                hide_next = True
            continue
        result.append(url(arg))
    return result


def command_line(text: str) -> str:
    """A shell command line (``COMMANDLINE_ARGS``) with secrets hidden."""

    try:
        return shlex.join(argv(shlex.split(text)))
    except ValueError:  # unbalanced quotes: hide the whole thing if it may hold a secret
        return HIDDEN if any(flag in text for flag in SECRET_FLAGS) else url(text)


def mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Settings with secret-looking keys hidden (non-empty strings only) and
    URL credentials removed."""

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and value and SECRET_KEY_RE.search(str(key)):
            result[key] = HIDDEN
        elif isinstance(value, str):
            result[key] = url(value)
        else:
            result[key] = value
    return result
