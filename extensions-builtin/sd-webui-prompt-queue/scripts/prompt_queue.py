"""
Prompt Queue — queue up prompts for txt2img / img2img and run them back-to-back.

Backend design:
  * A small thread-safe in-memory store, persisted under Forge's user-data
    directory (survives webui restarts without modifying built-in sources).
  * A handful of lightweight FastAPI endpoints under /prompt-queue/*.
    All rendering is done client-side (javascript/prompt_queue.js) against
    /prompt-queue/state, which is cheap to poll: the response carries a
    `version` counter, so the UI only re-renders when something changed.
  * The actual generation is driven from the browser (fill prompt fields,
    click Generate). The server decides *when* it is safe to dispatch the
    next item via an atomic /pop that checks the webui's real busy state
    (shared.state.job + modules.progress task registry).
"""

import json
import os
import shutil
import threading
import time
import uuid

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modules import paths_internal, script_callbacks, shared

try:
    from modules import progress as webui_progress
except Exception:
    webui_progress = None


MAX_PENDING = 100          # hard cap requested by the user
MAX_FINISHED_KEPT = 50     # finished/failed history kept for display
STALE_RUNNING_SECONDS = 120

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(
    paths_internal.data_path, "extension-data", "sd-webui-prompt-queue"
)
STATE_FILE = os.path.join(STATE_DIR, "queue.json")
LEGACY_STATE_FILE = os.path.join(EXT_DIR, "queue.json")


def _migrate_legacy_state():
    if os.path.exists(STATE_FILE) or not os.path.isfile(LEGACY_STATE_FILE):
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        shutil.copy2(LEGACY_STATE_FILE, STATE_FILE)
    except OSError as e:
        print(f"[Prompt Queue] failed to migrate legacy queue.json: {e}")


_migrate_legacy_state()

VALID_TABS = ("txt2img", "img2img")


class _Store:
    """Thread-safe queue state."""

    def __init__(self):
        self.lock = threading.RLock()
        self.items = []          # list of dicts, oldest first
        self.enabled = True      # auto-run toggle
        self.version = 0
        self._load()

    # ---------- persistence ----------

    def _load(self):
        path = STATE_FILE if os.path.isfile(STATE_FILE) else LEGACY_STATE_FILE
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.items = [i for i in data.get("items", []) if isinstance(i, dict)]
                # Anything that was mid-flight when the webui stopped goes back to pending.
                restored_pending = False
                for item in self.items:
                    if item.get("status") == "running":
                        item["status"] = "pending"
                    if item.get("status") == "pending":
                        restored_pending = True
                # Don't surprise the user with generations right after a restart:
                # if we restored unfinished work, start paused.
                if restored_pending:
                    self.enabled = False
        except Exception as e:
            print(f"[Prompt Queue] failed to load {STATE_FILE}: {e}")
            self.items = []

    def _save(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"items": self.items}, f, ensure_ascii=False)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            print(f"[Prompt Queue] failed to save {STATE_FILE}: {e}")

    def _bump(self):
        self.version += 1
        self._save()

    # ---------- helpers ----------

    def _pending(self):
        return [i for i in self.items if i["status"] == "pending"]

    def _prune_finished(self):
        finished = [i for i in self.items if i["status"] in ("done", "failed")]
        overflow = len(finished) - MAX_FINISHED_KEPT
        if overflow > 0:
            drop_ids = {i["id"] for i in finished[:overflow]}
            self.items = [i for i in self.items if i["id"] not in drop_ids]

    # ---------- mutations ----------

    def add(self, tab, prompt, negative):
        with self.lock:
            if len(self._pending()) >= MAX_PENDING:
                return None, f"Queue is full ({MAX_PENDING} prompts max)."
            item = {
                "id": uuid.uuid4().hex[:12],
                "tab": tab,
                "prompt": prompt or "",
                "negative": negative or "",
                "status": "pending",
                "created": time.time(),
                "started": None,
                "finished": None,
            }
            self.items.append(item)
            self._prune_finished()
            self._bump()
            return item, None

    def remove(self, item_id):
        with self.lock:
            before = len(self.items)
            self.items = [i for i in self.items if i["id"] != item_id]
            if len(self.items) != before:
                self._bump()
                return True
            return False

    def move(self, item_id, direction):
        """Reorder within pending items only."""
        with self.lock:
            pending = self._pending()
            idx = next((k for k, i in enumerate(pending) if i["id"] == item_id), None)
            if idx is None:
                return False
            swap = idx - 1 if direction == "up" else idx + 1
            if swap < 0 or swap >= len(pending):
                return False
            a, b = pending[idx], pending[swap]
            ia, ib = self.items.index(a), self.items.index(b)
            self.items[ia], self.items[ib] = self.items[ib], self.items[ia]
            self._bump()
            return True

    def clear(self, which):
        with self.lock:
            if which == "pending":
                self.items = [i for i in self.items if i["status"] != "pending"]
            elif which == "finished":
                self.items = [i for i in self.items if i["status"] not in ("done", "failed")]
            self._bump()

    def set_enabled(self, enabled):
        with self.lock:
            self.enabled = bool(enabled)
            self._bump()

    def pop_next(self):
        """Atomically hand the next pending item to the runner, or None."""
        with self.lock:
            if not self.enabled or webui_is_busy():
                return None
            now = time.time()
            for i in self.items:
                if i["status"] == "running":
                    # A runner already owns an item. If it looks abandoned
                    # (e.g. the browser tab was closed mid-dispatch), fail it
                    # and move on; otherwise refuse to double-dispatch.
                    if now - (i.get("started") or now) > STALE_RUNNING_SECONDS:
                        i["status"] = "failed"
                        i["finished"] = now
                    else:
                        return None
            for i in self.items:
                if i["status"] == "pending":
                    i["status"] = "running"
                    i["started"] = now
                    self._bump()
                    return dict(i)
            return None

    def finish(self, item_id, status):
        with self.lock:
            for i in self.items:
                if i["id"] == item_id and i["status"] == "running":
                    i["status"] = status if status in ("done", "failed") else "done"
                    i["finished"] = time.time()
                    self._prune_finished()
                    self._bump()
                    return True
            return False

    def snapshot(self):
        with self.lock:
            return {
                "version": self.version,
                "enabled": self.enabled,
                "busy": webui_is_busy(),
                "max": MAX_PENDING,
                "pending": len(self._pending()),
                "items": [dict(i) for i in self.items],
            }


def webui_is_busy():
    """True while the webui is generating or has its own tasks queued."""
    try:
        if getattr(shared.state, "job", ""):
            return True
    except Exception:
        pass
    if webui_progress is not None:
        try:
            if webui_progress.current_task:
                return True
            if webui_progress.pending_tasks:
                return True
        except Exception:
            pass
    return False


STORE = _Store()


# ---------------------------------------------------------------- API

class AddRequest(BaseModel):
    tab: str = Field(...)
    prompt: str = Field(default="")
    negative: str = Field(default="")


class IdRequest(BaseModel):
    id: str


class MoveRequest(BaseModel):
    id: str
    direction: str  # "up" | "down"


class RunRequest(BaseModel):
    enabled: bool


class ClearRequest(BaseModel):
    which: str  # "pending" | "finished"


class FinishRequest(BaseModel):
    id: str
    status: str = "done"


def register_api(_demo, app: FastAPI):
    prefix = "/prompt-queue"

    @app.get(prefix + "/state")
    def state():
        return STORE.snapshot()

    @app.post(prefix + "/add")
    def add(req: AddRequest):
        tab = req.tab if req.tab in VALID_TABS else "txt2img"
        if not (req.prompt or "").strip() and not (req.negative or "").strip():
            return {"ok": False, "error": "Prompt is empty."}
        item, err = STORE.add(tab, req.prompt, req.negative)
        if err:
            return {"ok": False, "error": err}
        return {"ok": True, "item": item, "pending": STORE.snapshot()["pending"]}

    @app.post(prefix + "/remove")
    def remove(req: IdRequest):
        return {"ok": STORE.remove(req.id)}

    @app.post(prefix + "/move")
    def move(req: MoveRequest):
        return {"ok": STORE.move(req.id, "up" if req.direction == "up" else "down")}

    @app.post(prefix + "/clear")
    def clear(req: ClearRequest):
        STORE.clear("pending" if req.which == "pending" else "finished")
        return {"ok": True}

    @app.post(prefix + "/run")
    def run(req: RunRequest):
        STORE.set_enabled(req.enabled)
        return {"ok": True, "enabled": STORE.enabled}

    @app.post(prefix + "/pop")
    def pop():
        return {"item": STORE.pop_next()}

    @app.post(prefix + "/finish")
    def finish(req: FinishRequest):
        return {"ok": STORE.finish(req.id, req.status)}


# ---------------------------------------------------------------- UI tab

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.HTML(
            '<div id="prompt-queue-root">'
            '<div class="pq-loading">Loading queue…</div>'
            "</div>"
        )
    return [(tab, "Queue", "prompt_queue")]


script_callbacks.on_app_started(register_api)
script_callbacks.on_ui_tabs(on_ui_tabs)
