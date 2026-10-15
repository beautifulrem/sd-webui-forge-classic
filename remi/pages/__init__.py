"""``remi.pages``: extension pages (the former ``on_ui_tabs`` tabs).

An extension registers a page when its script is imported::

    from remi import pages, ui

    class Captioner(pages.Page):
        def params(self):
            return ui.Column(ui.Image("image"), ui.Button("Caption", "caption"), ui.Output("caption"))

        @ui.action("caption")
        def caption(self, ctx, values):
            return self.start_task(ctx, "caption", lambda task: {...}, gpu=True)

    pages.register(Captioner("captioner", "Captioner", icon="camera"))

or passes the pieces: ``Page(id, title, icon, params=node_or_callable,
actions=object_with_ui_actions, widget=None, send_to=[...])``. A page is
schema-rendered (``params``) or a dedicated frontend view (``widget``, in
``webui/src/pages/custom/<widget>``). Its owner id is ``page:<id>``: the SPA
keeps its values in that form and calls ``POST /actions/page:<id>/<action>``.

The server reads :func:`registered`, :func:`get` and :func:`page_schema`
(schema-format.md §8). Registrations follow the script life cycle: a page is
dropped when its scripts are unloaded ("Reload UI"), so a disabled extension's
page disappears and an enabled one registers again.

Long work (model downloads, captioning a folder) runs as a page *task*
(:meth:`Page.start_task`): the action answers at once, the work runs on a
thread (behind generation with ``gpu=True``) and reports progress; the
``task-runner`` widget polls the built-in ``__task__`` action and applies the
task's result patch when it ends.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import types
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from remi import ui
from remi.ui.actions import ActionSpec, actions_of, check_patch, message
from remi.ui.errors import SchemaError

log = logging.getLogger(__name__)

PAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SEND_KINDS = ("image",)

__all__ = [
    "Page",
    "SendTarget",
    "Task",
    "TaskCancelled",
    "clear",
    "file_store",
    "get",
    "page_schema",
    "register",
    "registered",
    "services",
    "tab_titles",
    "unregister",
]


@dataclass(frozen=True)
class SendTarget:
    """Something other views can send to this page: ``kind="image"`` puts an
    image (upload or output file id) into the page's ``param``. The SPA
    offers it as "Send to <label>" wherever it sends images (PNG Info)."""

    param: str
    label: str
    kind: str = "image"

    def schema(self) -> dict[str, str]:
        return {"kind": self.kind, "param": self.param, "label": self.label}


# region Tasks


class TaskCancelled(Exception):
    """Raised by :meth:`Task.check` once the user cancelled the task."""


TASK_STATES = ("queued", "running", "completed", "failed", "cancelled")
_FINISHED = frozenset({"completed", "failed", "cancelled"})
_TASK_TTL = 3600.0  # finished tasks are kept this long for late polls


class Task:
    """A page's long-running work. The work function receives the task and
    reports with :meth:`progress`; it returns the patch (schema-format.md
    §5) applied to the page when it ends."""

    def __init__(self, page_id: str, name: str, user: str | None):
        self.id = uuid.uuid4().hex[:16]
        self.page_id, self.name, self.user = page_id, name, user
        self.state = "queued"
        self.fraction: float | None = None
        self.text: str | dict | None = None
        self.error: str | None = None
        self.result: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.started = time.time()
        self.finished: float | None = None
        self.delivered = False  # the result patch reached a view
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def progress(self, fraction: float | None = None, text: str | None = None, **params: Any) -> None:
        """Report progress: ``fraction`` in 0…1 (``None``: unknown) and an
        English status text (a catalog key; ``{{name}}`` placeholders are
        filled from ``params`` after translation)."""

        with self._lock:
            if fraction is not None:
                self.fraction = min(1.0, max(0.0, float(fraction)))
            if text is not None:
                self.text = message(text, **params) if params else text

    def publish(self, **outputs: Any) -> None:
        """Output values shown while the task runs (e.g. a log so far);
        each poll answers the latest ones."""

        with self._lock:
            self.outputs.update(outputs)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def check(self) -> None:
        """Stop here when the user cancelled (raises :class:`TaskCancelled`)."""

        if self._cancel.is_set():
            raise TaskCancelled()

    @property
    def done(self) -> bool:
        return self.state in _FINISHED

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "state": self.state,
                "progress": self.fraction,
                "text": self.text,
                "error": self.error,
                "done": self.done,
                "cancelling": self._cancel.is_set() and not self.done,
            }

    def _finish(self, state: str, *, result: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            self.result = dict(result or {})
            self.error = error
            if state == "completed":
                self.fraction = 1.0
            self.finished = time.time()


_tasks: dict[tuple[str, str, str | None], Task] = {}
_tasks_lock = threading.Lock()


def _prune_tasks(now: float) -> None:
    for key, task in list(_tasks.items()):
        if task.finished is not None and now - task.finished > _TASK_TTL:
            del _tasks[key]


def _generation_lock(ctx: Any) -> Any:
    found = services(ctx)
    if found is not None:
        return found.queue.lock
    from modules.call_queue import queue_lock

    return queue_lock


# endregion


class Page:
    """One extension page. Subclasses may override :meth:`params` and
    declare ``@ui.action`` methods; or pass ``params=`` (a layout node, or a
    callable building one per request, e.g. for choices read from a file)
    and ``actions=`` (an object, class or module whose ``@ui.action``
    callables become the page's actions; they receive ``(ctx, values)``).
    ``on_load`` names an action the SPA runs each time the page is shown
    (e.g. to refresh a status line); ``send_to`` lists what other views can
    send here."""

    def __init__(
        self,
        id: str,
        title: str,
        icon: str | None = None,
        *,
        params: ui.Node | Callable[[], ui.Node | None] | None = None,
        actions: Any = None,
        widget: str | None = None,
        send_to: Iterable[SendTarget] = (),
        on_load: str | None = None,
    ):
        if not isinstance(id, str) or not PAGE_ID_RE.fullmatch(id):
            raise SchemaError(f"bad page id {id!r} (lowercase letters, digits, - and _)")
        if not isinstance(title, str) or not title.strip():
            raise SchemaError(f"page {id!r}: a title is required")
        if widget is not None and params is not None:
            raise SchemaError(f"page {id!r}: a page has params or a widget, not both")
        self.id, self.title, self.icon, self.widget = id, title, icon, widget
        self._params_source = params
        self.on_load = on_load
        self.send_to = tuple(send_to)
        for target in self.send_to:
            if not isinstance(target, SendTarget) or target.kind not in SEND_KINDS:
                raise SchemaError(f"page {id!r}: send_to takes SendTarget(kind in {SEND_KINDS})")
        if actions is not None:
            self._adopt_actions(actions)
        if on_load is not None and on_load not in actions_of(type(self)):
            raise SchemaError(f"page {id!r}: on_load names no action of the page ({on_load!r})")

    @property
    def owner(self) -> str:
        return f"page:{self.id}"

    def __repr__(self) -> str:
        return f"Page({self.id!r})"

    # region Layout and schema

    def params(self) -> ui.Node | None:
        """The page's layout tree (``None`` for a ``widget`` page)."""

        source = self._params_source
        return source() if callable(source) else source

    def layout(self) -> ui.ScriptLayout:
        """The validated layout (unique names, known references, send targets)."""

        layout = ui.ScriptLayout(None if self.widget else self.params(), owner=self.owner)
        for target in self.send_to:
            node = layout.by_name.get(target.param)
            if not isinstance(node, ui.Image):
                raise SchemaError(f"page {self.id!r}: send target {target.param!r} is not an Image input")
        return layout

    def schema(self) -> dict[str, Any]:
        """schema-format.md §8."""

        ctx = ui.Context()
        layout = self.layout()
        data: dict[str, Any] = {
            "version": ui.SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "icon": self.icon,
            "owner": self.owner,
            "widget": self.widget,
            "layout": layout.schema(ctx),
            "defaults": ui.jsonable(layout.defaults(ctx)),
        }
        if self.send_to:
            data["send_to"] = [target.schema() for target in self.send_to]
        if self.on_load:
            data["on_load"] = self.on_load
        return data

    # endregion

    # region Values

    def values(self, ctx: ui.ActionContext, values: Mapping[str, Any] | None, *, uploads: bool = True) -> dict[str, Any]:
        """The page's values as its code uses them: validated (unknown names
        ignored, missing ones defaulted) and, with ``uploads``, images
        resolved to PIL images and files to paths. Raises
        :class:`remi.ui.ValidationError`."""

        store = file_store(ctx) if uploads else None
        resolve = ui.Context(resolve_upload=store.resolve_upload if store is not None else None)
        layout = self.layout()
        coerced = layout.coerce(values, resolve, unknown="ignore")
        if not uploads:
            return coerced
        result = {}
        errors = []
        for node in (*layout.inputs, *layout.tools):
            try:
                result[node.name] = node.to_arg(coerced[node.name], resolve) if isinstance(node, (ui.Image, ui.ImageList, ui.File)) else coerced[node.name]
            except ui.ParamError as error:
                error.param, error.owner = node.name, self.owner
                errors.append(error)
        if errors:
            raise ui.ValidationError(errors)
        return result

    # endregion

    # region Tasks

    def start_task(self, ctx: ui.ActionContext, name: str, work: Callable[[Task], Mapping[str, Any] | None], *, gpu: bool = False) -> dict[str, Any]:
        """Run ``work(task)`` on a thread (``gpu``: behind generation, holding
        the generation lock while it runs) and answer the task's status
        (``__outputs__`` ``task:<name>``). One task per page, name and user
        runs at a time: starting it again while it runs answers its status
        with a warning."""

        key = (self.id, name, ctx.user)
        with _tasks_lock:
            _prune_tasks(time.time())
            running = _tasks.get(key)
            if running is not None and not running.done:
                return {**_task_patch(running), **ui.toast("This task is already running", "warning")}
            task = _tasks[key] = Task(self.id, name, ctx.user)
        lock = _generation_lock(ctx) if gpu else None

        def run() -> None:
            held = False
            try:
                if lock is not None:
                    if not lock.acquire(blocking=False):
                        task.progress(text="Waiting for the current generation to finish…")
                        while not lock.acquire(timeout=0.5):
                            task.check()
                    held = True
                task.check()
                task.state = "running"
                result = check_patch(work(task) or {})
            except TaskCancelled:
                task._finish("cancelled", result=ui.toast("Cancelled", "info"))
            except ui.ValidationError as error:
                task._finish("failed", error="; ".join(str(item.message) for item in error.errors))
            except Exception as error:
                log.exception("page %s: task %s failed", self.id, name)
                task._finish("failed", error=str(error) or type(error).__name__)
            else:
                task._finish("completed", result=result)
            finally:
                if held:
                    lock.release()

        threading.Thread(target=run, name=f"remi-page-{self.id}-{name}", daemon=True).start()
        return _task_patch(task)

    def task(self, ctx: ui.ActionContext, name: str) -> Task | None:
        with _tasks_lock:
            return _tasks.get((self.id, name, ctx.user))

    def busy(self, *names: str) -> bool:
        """Whether a task of this page (of any user; only those ``names``
        when given) is queued or running: e.g. before freeing a model the
        page's tasks use."""

        with _tasks_lock:
            return any(task.page_id == self.id and not task.done and (not names or task.name in names) for task in _tasks.values())

    @ui.action("__task__")
    def _task_status(self, ctx: ui.ActionContext, values: dict[str, Any]) -> dict[str, Any]:
        """``{task: name, id?, resume?}`` → the task's status; once it has
        ended, also its result patch: for the poll that asked for that task
        ``id``, or for a ``resume`` (a view that mounts again, e.g. after a
        reload) while no view has received the result yet."""

        name = ctx.extra.get("task")
        if not isinstance(name, str):
            raise ui.ValidationError([ui.ParamError("task: expected a task name", owner=self.owner)])
        task = self.task(ctx, name)
        if task is None:
            return {"__outputs__": {f"task:{name}": None}}
        patch = _task_patch(task)
        if not task.done:
            return patch
        resume = bool(ctx.extra.get("resume"))
        with task._lock:
            deliver = (not resume and ctx.extra.get("id") == task.id) or (resume and not task.delivered)
            task.delivered = task.delivered or deliver
        if deliver:
            if task.state == "failed":
                patch.update(ui.toast(message("Failed: {{error}}", error=task.error or ""), "error"))
            patch = {**task.result, **patch, "__outputs__": {**task.result.get("__outputs__", {}), **patch["__outputs__"]}}
        return patch

    @ui.action("__cancel__")
    def _task_cancel(self, ctx: ui.ActionContext, values: dict[str, Any]) -> dict[str, Any]:
        """``{task: name}``: ask the running task to stop."""

        name = ctx.extra.get("task")
        task = self.task(ctx, name) if isinstance(name, str) else None
        if task is None or task.done:
            return {}
        task.cancel()
        return _task_patch(task)

    # endregion

    def _adopt_actions(self, source: Any) -> None:
        """Make ``source``'s actions (an object's methods, or a module's
        functions) this page's: actions are looked up on the owner's class,
        so the page gets a subclass of its own."""

        specs = actions_of(source)
        clash = sorted(set(specs) & set(actions_of(type(self))))
        if clash:
            raise SchemaError(f"page {self.id!r}: actions {clash} clash with the page's own")
        members = {}
        for index, (name, spec) in enumerate(specs.items()):
            bound = spec.func if isinstance(source, types.ModuleType) else spec.func.__get__(source, type(source))
            members[f"_remi_action_{index}"] = _forward(name, spec, bound)
        self.__class__ = type(f"{type(self).__name__}[{self.id}]", (type(self),), members)


def _forward(name: str, spec: ActionSpec, bound: Callable[..., Any]) -> Callable[..., Any]:
    def call(self: Page, ctx: ui.ActionContext, values: dict[str, Any]) -> Any:
        return bound(ctx, values)

    call.__name__ = name
    return ui.action(name, gpu=spec.gpu)(call)


def _task_patch(task: Task) -> dict[str, Any]:
    with task._lock:
        outputs = dict(task.outputs) if not task.done else {}
    return {"__outputs__": {**outputs, f"task:{task.name}": task.status()}}


# region Registry

_pages: dict[str, Page] = {}
_registry_lock = threading.Lock()


def register(page: Page) -> Page:
    """Add ``page`` (replacing one with the same id). It is dropped again
    when the scripts are unloaded (a "Reload UI" loads them anew)."""

    if not isinstance(page, Page):
        raise SchemaError(f"register() takes a Page, got {type(page).__name__}")
    with _registry_lock:
        previous = _pages.get(page.id)
        if previous is not None and previous is not page:
            log.info("page %r replaced", page.id)
        _pages[page.id] = page
    try:
        from modules import script_callbacks

        script_callbacks.on_script_unloaded(lambda: unregister(page.id, page), name=f"remi.pages:{page.id}")
    except Exception:  # outside the WebUI (tools, tests without modules)
        log.debug("page %r: no script life cycle to follow", page.id, exc_info=True)
    return page


def unregister(page_id: str, page: Page | None = None) -> None:
    """Remove a page (only ``page`` itself, when given)."""

    with _registry_lock:
        current = _pages.get(page_id)
        if current is not None and (page is None or current is page):
            del _pages[page_id]


def clear() -> None:
    with _registry_lock:
        _pages.clear()


def registered(*, include_hidden: bool = False) -> list[Page]:
    """The pages the UI lists: in the user's "UI Tab Order", those in "Hide
    UI Tabs" left out (both settings store page titles, like the built-in
    tabs'); unordered pages follow in registration order. ``include_hidden``:
    every page, in registration order."""

    with _registry_lock:
        found = list(_pages.values())
    if include_hidden:
        return found
    order, hidden = _tab_settings()
    rank = {title: index for index, title in enumerate(order)}
    found = [page for page in found if page.title not in hidden]
    return sorted(found, key=lambda page: rank.get(page.title, len(rank)))  # stable: registration order otherwise


def tab_titles() -> list[str]:
    """Titles of every registered page, for the tab settings' choices."""

    return [page.title for page in registered(include_hidden=True)]


def _tab_settings() -> tuple[list[str], set[str]]:
    try:
        from modules import shared

        options = shared.opts
        order = [title for title in getattr(options, "ui_tab_order", None) or () if isinstance(title, str)]
        hidden = {title for title in getattr(options, "hidden_tabs", None) or () if isinstance(title, str)}
        return order, hidden
    except Exception:  # outside the WebUI
        return [], set()


def get(page_id: str) -> Page | None:
    with _registry_lock:
        return _pages.get(page_id)


def page_schema(page_id: str) -> dict[str, Any]:
    page = get(page_id)
    if page is None:
        raise KeyError(page_id)
    return page.schema()


# endregion

# region Server services


def services(ctx: ui.ActionContext | None) -> Any:
    """The server's services for an action (``None`` outside a server)."""

    request = getattr(ctx, "request", None)
    found = getattr(getattr(getattr(request, "app", None), "state", None), "services", None)
    if found is not None:
        return found
    try:
        from remi.server.services import default_services

        return default_services()
    except Exception:
        log.debug("no server services", exc_info=True)
        return None


def file_store(ctx: ui.ActionContext | None) -> Any:
    """The server's file index (uploads, outputs), or ``None``."""

    found = services(ctx)
    return getattr(found, "files", None)


# endregion
