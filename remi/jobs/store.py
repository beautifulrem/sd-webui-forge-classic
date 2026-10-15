"""Persistent job records (SQLite, WAL) for the generation queue.

Queued jobs run by priority (higher first), then by position (the order the
user arranged), then by submission. A job that was running when the server
stopped is marked ``interrupted`` on the next start (retry it to run it
again); queued jobs survive restarts.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remi.files import decode_cursor, encode_cursor, connect

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED, INTERRUPTED = "queued", "running", "completed", "failed", "cancelled", "interrupted"
STATUSES = (QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED, INTERRUPTED)
FINISHED = frozenset({COMPLETED, FAILED, CANCELLED, INTERRUPTED})
MAX_REORDER = 500
"""ids :meth:`JobStore.reorder` considers (SQLite's variable limit)"""


@dataclass
class Job:
    id: str
    seq: int
    tab: str
    core: dict[str, Any]
    scripts: dict[str, Any]
    layout_hash: str = ""
    schema_version: int = 1
    priority: int = 0
    position: float = 0.0
    status: str = QUEUED
    created: float = 0.0
    started: float | None = None
    finished: float | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    infotext: str | None = None
    error: str | None = None
    errors: list[dict[str, Any]] | None = None
    summary: str = ""
    forever: bool = False
    user: str | None = None
    retry_of: str | None = None

    @property
    def id_task(self) -> str:
        """The task id of the progress registry (``/internal/progress``)."""

        return f"task(remi-{self.id})"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    tab TEXT NOT NULL,
    core TEXT NOT NULL,
    scripts TEXT NOT NULL,
    layout_hash TEXT NOT NULL DEFAULT '',
    schema_version INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 0,
    position REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created REAL NOT NULL,
    started REAL,
    finished REAL,
    outputs TEXT NOT NULL DEFAULT '[]',
    infotext TEXT,
    error TEXT,
    errors TEXT,
    summary TEXT NOT NULL DEFAULT '',
    forever INTEGER NOT NULL DEFAULT 0,
    user TEXT,
    retry_of TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, priority DESC, position, seq);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_COLUMNS = "id, seq, tab, core, scripts, layout_hash, schema_version, priority, position, status, created, started, finished, outputs, infotext, error, errors, summary, forever, user, retry_of"
_QUEUE_ORDER = "priority DESC, position, seq"


def _job(row: tuple) -> Job:
    values = dict(zip(_COLUMNS.split(", "), row, strict=True))
    for key in ("core", "scripts", "outputs"):
        values[key] = json.loads(values[key])
    values["errors"] = json.loads(values["errors"]) if values["errors"] else None
    values["forever"] = bool(values["forever"])
    return Job(**values)


class JobStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = threading.RLock()
        self._db = connect(Path(path))
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _fetch(self, sql: str, args: tuple | list = ()) -> list[Job]:
        with self._lock:
            return [_job(row) for row in self._db.execute(sql, args).fetchall()]

    def add(self, tab: str, core: dict, scripts: dict, *, priority: int = 0, layout_hash: str = "", schema_version: int = 1, summary: str = "", forever: bool = False, user: str | None = None, retry_of: str | None = None) -> Job:
        job_id = secrets.token_hex(8)
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO jobs (id, tab, core, scripts, layout_hash, schema_version, priority, status, created, summary, forever, user, retry_of) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, tab, json.dumps(core), json.dumps(scripts), layout_hash, schema_version, int(priority), QUEUED, now, summary, int(forever), user, retry_of),
            )
            seq = cursor.lastrowid
            self._db.execute("UPDATE jobs SET position = ? WHERE seq = ?", (float(seq), seq))
        return self.get(job_id)

    def get(self, job_id: str) -> Job | None:
        jobs = self._fetch(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return jobs[0] if jobs else None

    def queued(self) -> list[Job]:
        """Queued jobs in run order."""

        return self._fetch(f"SELECT {_COLUMNS} FROM jobs WHERE status = ? ORDER BY {_QUEUE_ORDER}", (QUEUED,))

    def count(self, status: str = QUEUED) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)).fetchone()[0]

    def claim_next(self) -> Job | None:
        """Mark the first queued job running and return it."""

        with self._lock:
            row = self._db.execute(f"SELECT id FROM jobs WHERE status = ? ORDER BY {_QUEUE_ORDER} LIMIT 1", (QUEUED,)).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET status = ?, started = ? WHERE id = ?", (RUNNING, time.time(), row[0]))
            return self.get(row[0])

    def finish(self, job_id: str, status: str, *, outputs: list[dict] | None = None, infotext: str | None = None, error: str | None = None, errors: list[dict] | None = None) -> Job | None:
        if status not in FINISHED:
            raise ValueError(f"not a final status: {status}")
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, finished = ?, outputs = ?, infotext = ?, error = ?, errors = ? WHERE id = ?",
                (status, time.time(), json.dumps(outputs or []), infotext, error, json.dumps(errors) if errors else None, job_id),
            )
        return self.get(job_id)

    def cancel_queued(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._db.execute("UPDATE jobs SET status = ?, finished = ? WHERE id = ? AND status = ?", (CANCELLED, time.time(), job_id, QUEUED))
            return cursor.rowcount > 0

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM jobs WHERE id = ? AND status != ?", (job_id, RUNNING)).rowcount > 0

    def update(self, job_id: str, *, priority: int | None = None, forever: bool | None = None) -> Job | None:
        with self._lock:
            if priority is not None:
                self._db.execute("UPDATE jobs SET priority = ? WHERE id = ?", (int(priority), job_id))
            if forever is not None:
                self._db.execute("UPDATE jobs SET forever = ? WHERE id = ?", (int(forever), job_id))
        return self.get(job_id)

    def requeue_forever(self, job_id: str) -> Job | None:
        """Queue a copy of ``job_id`` if it is still marked generate-forever
        (atomic with :meth:`clear_forever`)."""

        with self._lock:
            job = self.get(job_id)
            if job is None or not job.forever:
                return None
            return self.add(job.tab, job.core, job.scripts, priority=job.priority, layout_hash=job.layout_hash, schema_version=job.schema_version, summary=job.summary, forever=True, user=job.user, retry_of=job.id)

    def clear_forever(self) -> int:
        with self._lock:
            return self._db.execute("UPDATE jobs SET forever = 0 WHERE forever = 1").rowcount

    def reorder(self, ids: list[str]) -> None:
        """Put the listed queued jobs in this order, at the place of the
        earliest of them; the other queued jobs keep their positions."""

        ids = list(dict.fromkeys(ids))[:MAX_REORDER]
        with self._lock:
            rows = self._db.execute(f"SELECT id, position FROM jobs WHERE status = ? AND id IN ({','.join('?' * len(ids))})", (QUEUED, *ids)).fetchall() if ids else []
            positions = sorted(position for _id, position in rows)
            known = {job_id for job_id, _position in rows}
            ordered = [job_id for job_id in dict.fromkeys(ids) if job_id in known]
            for job_id, position in zip(ordered, positions, strict=True):
                self._db.execute("UPDATE jobs SET position = ? WHERE id = ?", (position, job_id))

    def history(self, *, cursor: str | None = None, limit: int = 50, status: str | None = None) -> tuple[list[Job], str | None, int]:
        """Newest first: ``(jobs, next_cursor, total)``; ``status`` filters
        (a comma-separated list is allowed)."""

        where, args = ["1"], []
        if status:
            wanted = [item for item in status.split(",") if item in STATUSES]
            where.append(f"status IN ({','.join('?' * len(wanted))})" if wanted else "0")
            args += wanted
        filters = " AND ".join(where)
        page = filters
        page_args = list(args)
        position = decode_cursor(cursor, 1)
        if position is not None and position[0].isdigit():
            page += " AND seq < ?"
            page_args.append(int(position[0]))
        limit = min(max(int(limit), 1), 200)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM jobs WHERE {filters}", args).fetchone()[0]
        jobs = self._fetch(f"SELECT {_COLUMNS} FROM jobs WHERE {page} ORDER BY seq DESC LIMIT ?", [*page_args, limit + 1])
        next_cursor = encode_cursor(jobs[limit - 1].seq) if len(jobs) > limit else None
        return jobs[:limit], next_cursor, total

    def recover(self) -> int:
        """Mark jobs left running by a previous process interrupted."""

        with self._lock:
            return self._db.execute("UPDATE jobs SET status = ?, finished = ?, error = ? WHERE status = ?", (INTERRUPTED, time.time(), "the server stopped while the job ran", RUNNING)).rowcount

    def referenced_strings(self) -> set[str]:
        """Every string value in the queued jobs' parameters (upload ids the
        queue still needs)."""

        found: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, str):
                found.add(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for job in self._fetch(f"SELECT {_COLUMNS} FROM jobs WHERE status IN (?, ?)", (QUEUED, RUNNING)):
            walk(job.core)
            walk(job.scripts)
        return found

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
