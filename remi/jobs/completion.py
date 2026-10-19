"""What happens when the queue is done: the setting ``queue_completion_action``
(the Agent Scheduler's, with its choices, so a saved value carries over).

"Stop webui" stops the server the way the console's Ctrl+C does; "Sleep",
"Hibernate", "Restart" and "Shut down" run the operating system's command
(the ones the Agent Scheduler ran) and, for the last two, stop the server
too. The queue runs the action once and sets the setting back to "Do
nothing" (:meth:`remi.jobs.queue.JobQueue._queue_done`): every generation
is a queue job in Remi, so a saved "Shut down" would otherwise power the
computer off after the next single Generate.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any

log = logging.getLogger(__name__)

NOTHING = "Do nothing"
STOP = "Stop webui"
ACTIONS = (NOTHING, STOP, "Sleep", "Hibernate", "Restart", "Shut down")
DELAY = 5.0
"""seconds between the last job and the action (its events reach the browsers)"""


def power_command(action: str, platform: str = sys.platform) -> list[str] | None:
    """The command for a power action on ``platform`` (None: none, or Windows
    sleep, which is a system call: :func:`_windows_sleep`)."""

    if platform == "win32":
        return {"Shut down": ["shutdown", "/s", "/hybrid", "/t", "0"], "Restart": ["shutdown", "/r", "/t", "0"], "Hibernate": ["shutdown", "/h"]}.get(action)
    if platform == "darwin":
        return {
            "Shut down": ["osascript", "-e", 'tell application "System Events" to shut down'],
            "Restart": ["osascript", "-e", 'tell application "System Events" to restart'],
            "Sleep": ["pmset", "sleepnow"],
            "Hibernate": ["pmset", "sleepnow"],
        }.get(action)
    return {
        "Shut down": ["systemctl", "poweroff"],
        "Restart": ["systemctl", "reboot"],
        "Sleep": ["sh", "-c", "systemctl hybrid-sleep || systemctl suspend"],
        "Hibernate": ["systemctl", "hibernate"],
    }.get(action)


def _windows_sleep() -> None:
    import ctypes

    if not ctypes.windll.PowrProf.SetSuspendState(False, False, False):  # type: ignore[attr-defined]
        log.error("Could not put the computer to sleep (error %s)", ctypes.GetLastError())  # type: ignore[attr-defined]


def _stop_server() -> None:
    from modules import shared

    shared.state.server_command = "stop"


def perform(action: str, *, platform: str = sys.platform, run: Callable[[Sequence[str]], Any] = subprocess.Popen, stop_server: Callable[[], None] = _stop_server) -> bool:
    """Runs ``action`` now; False for "Do nothing" or an unknown action."""

    if action not in ACTIONS or action == NOTHING:
        return False
    log.warning("The queue is done: %s", action)
    if action == STOP:
        stop_server()
        return True
    command = power_command(action, platform)
    try:
        if command is not None:
            run(command)
        elif action == "Sleep" and platform == "win32":
            _windows_sleep()
        else:
            log.error("%s is not supported on %s", action, platform)
            return False
    except Exception:
        log.exception("running the queue completion action %s", action)
        return False
    if action in ("Shut down", "Restart"):
        stop_server()
    return True


def perform_later(action: str, delay: float = DELAY) -> None:
    """:func:`perform` on its own thread after ``delay`` seconds."""

    timer = threading.Timer(delay, perform, args=(action,))
    timer.daemon = True
    timer.name = "remi-queue-done"
    timer.start()
