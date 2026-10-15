"""Persistent job records (SQLite, WAL) for the generation queue.

Queued jobs run by priority (higher first), then by position (the order the
user arranged), then by submission. A job that was running when the server
stopped is marked ``interrupted`` on the next start (retry it to run it
again); queued jobs survive restarts.

Jobs can be starred (kept by "clear history"), searched (prompt, infotext,
parameters) and carry ``meta``: where an imported job came from, notes on
its mapping and, for jobs whose arguments could not all be mapped to the
current parameters, the unmapped arguments (``replayable`` is then false:
such a job is never run or retried as it is).
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
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
    starred: bool = False
    replayable: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    """``source`` (an importer), ``source_id``, ``name``, ``notes`` (list of
    strings) and ``opaque`` (arguments kept as they were, not replayable)"""

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
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created DESC, seq DESC);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS imports (
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    PRIMARY KEY (source, source_id)
);
"""
# ``imports``: records an importer already brought in (importing twice is a no-op)

_ADDED_COLUMNS = {
    "starred": "INTEGER NOT NULL DEFAULT 0",
    "replayable": "INTEGER NOT NULL DEFAULT 1",
    "meta": "TEXT",
}
"""columns added after the first release (``ALTER TABLE`` on open)"""

_COLUMNS = "id, seq, tab, core, scripts, layout_hash, schema_version, priority, position, status, created, started, finished, outputs, infotext, error, errors, summary, forever, user, retry_of, starred, replayable, meta"
_QUEUE_ORDER = "priority DESC, position, seq"
MAX_QUERY_TERMS = 8


def _job(row: tuple) -> Job:
    values = dict(zip(_COLUMNS.split(", "), row, strict=True))
    for key in ("core", "scripts", "outputs"):
        values[key] = json.loads(values[key])
    values["errors"] = json.loads(values["errors"]) if values["errors"] else None
    values["forever"] = bool(values["forever"])
    values["starred"] = bool(values["starred"])
    values["replayable"] = bool(values["replayable"])
    values["meta"] = json.loads(values["meta"]) if values["meta"] else {}
    return Job(**values)


def like_terms(query: str | None) -> list[str]:
    """``LIKE`` patterns (escaped with ``\\``) for the words of ``query``."""

    terms = [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in (query or "").split() if term]
    return [f"%{term}%" for term in terms[:MAX_QUERY_TERMS]]


class JobStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = threading.RLock()
        self._db = connect(Path(path))
        self._db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            present = {row[1] for row in self._db.execute("PRAGMA table_info(jobs)").fetchall()}
            for name, declaration in _ADDED_COLUMNS.items():
                if name not in present:
                    self._db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _fetch(self, sql: str, args: tuple | list = ()) -> list[Job]:
        with self._lock:
            return [_job(row) for row in self._db.execute(sql, args).fetchall()]

    def add(self, tab: str, core: dict, scripts: dict, *, priority: int = 0, layout_hash: str = "", schema_version: int = 1, summary: str = "", forever: bool = False, user: str | None = None, retry_of: str | None = None, after: str | None = None, meta: dict | None = None) -> Job:
        """Queue a job at the end, or right after the queued job ``after``
        (same priority, so it runs next to it)."""

        job_id = secrets.token_hex(8)
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO jobs (id, tab, core, scripts, layout_hash, schema_version, priority, status, created, summary, forever, user, retry_of, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, tab, json.dumps(core), json.dumps(scripts), layout_hash, schema_version, int(priority), QUEUED, now, summary, int(forever), user, retry_of, json.dumps(meta) if meta else None),
            )
            seq = cursor.lastrowid
            position = float(seq)
            anchor = self._db.execute("SELECT position, priority FROM jobs WHERE id = ? AND status = ?", (after, QUEUED)).fetchone() if after else None
            if anchor is not None:
                following = self._db.execute("SELECT MIN(position) FROM jobs WHERE status = ? AND priority = ? AND position > ? AND seq != ?", (QUEUED, anchor[1], anchor[0], seq)).fetchone()[0]
                position = (anchor[0] + following) / 2 if following is not None else anchor[0] + 1
                self._db.execute("UPDATE jobs SET priority = ? WHERE seq = ?", (anchor[1], seq))
            self._db.execute("UPDATE jobs SET position = ? WHERE seq = ?", (position, seq))
        return self.get(job_id)

    def add_record(self, job: Job) -> Job:
        """Insert a job as it is (an imported one: any status, its own times
        and outputs); a new id and sequence number are given. A job that is
        not replayable is never stored queued (it is stored cancelled)."""

        job_id = secrets.token_hex(8)
        if job.status in (QUEUED, RUNNING) and not job.replayable:
            job = replace(job, status=CANCELLED, finished=job.finished or time.time())
        with self._lock:
            cursor = self._db.execute(
                f"INSERT INTO jobs ({_COLUMNS.replace('id, seq, ', '', 1)}, id) VALUES ({', '.join('?' * (len(_COLUMNS.split(', ')) - 1))})",
                (
                    job.tab, json.dumps(job.core), json.dumps(job.scripts), job.layout_hash, job.schema_version, int(job.priority), 0.0, job.status,
                    job.created or time.time(), job.started, job.finished, json.dumps(job.outputs or []), job.infotext, job.error,
                    json.dumps(job.errors) if job.errors else None, job.summary, int(job.forever), job.user, job.retry_of,
                    int(job.starred), int(job.replayable), json.dumps(job.meta) if job.meta else None, job_id,
                ),
            )
            self._db.execute("UPDATE jobs SET position = ? WHERE seq = ?", (float(cursor.lastrowid), cursor.lastrowid))
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
        """Mark the first queued job running and return it. A queued job
        that is not replayable (none should be: :meth:`add_record` stores
        them cancelled) is cancelled on the way, so the queue never counts
        a job it will not run."""

        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, finished = ?, error = COALESCE(error, ?) WHERE status = ? AND replayable = 0",
                (CANCELLED, time.time(), "not replayable: its arguments do not map to the current parameters", QUEUED),
            )
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

    def move(self, job_id: str, *, after: str | None = None, before: str | None = None, priority: int | None = None) -> bool:
        """Move a queued job right after the queued job ``after`` (else right
        before ``before``) of its priority group, given the new ``priority``
        first. Neighbours that are not queued in that group are ignored (they
        started meanwhile); with neither, only the priority changes. False
        when ``job_id`` is not queued."""

        with self._lock:
            row = self._db.execute("SELECT priority FROM jobs WHERE id = ? AND status = ?", (job_id, QUEUED)).fetchone()
            if row is None:
                return False
            group = row[0] if priority is None else int(priority)
            if priority is not None:
                self._db.execute("UPDATE jobs SET priority = ? WHERE id = ?", (group, job_id))
            for _attempt in range(2):
                spot = self._spot(job_id, group, after, before)
                if spot is None:
                    return True
                if spot is not False:
                    self._db.execute("UPDATE jobs SET position = ? WHERE id = ?", (spot, job_id))
                    return True
                self._renumber()  # positions too close (or tied) to place it between
            return True

    def _spot(self, job_id: str, group: int, after: str | None, before: str | None) -> float | None | bool:
        """The position between the neighbours; None: no neighbour; False:
        no float fits between them."""

        def position_of(other: str | None) -> float | None:
            if not other or other == job_id:
                return None
            found = self._db.execute("SELECT position FROM jobs WHERE id = ? AND status = ? AND priority = ?", (other, QUEUED, group)).fetchone()
            return found[0] if found else None

        tied = self._db.execute("SELECT COUNT(*) - COUNT(DISTINCT position) FROM jobs WHERE status = ? AND priority = ? AND id != ?", (QUEUED, group, job_id)).fetchone()[0]
        if tied:
            return False
        low, high = position_of(after), None
        if low is not None:
            high = self._db.execute("SELECT MIN(position) FROM jobs WHERE status = ? AND priority = ? AND position > ? AND id != ?", (QUEUED, group, low, job_id)).fetchone()[0]
        else:
            high = position_of(before)
            if high is None:
                return None
            low = self._db.execute("SELECT MAX(position) FROM jobs WHERE status = ? AND priority = ? AND position < ? AND id != ?", (QUEUED, group, high, job_id)).fetchone()[0]
        if low is None:
            return high - 1.0
        if high is None:
            return low + 1.0
        middle = (low + high) / 2
        return middle if low < middle < high else False

    def _renumber(self) -> None:
        """Give the queued jobs the positions 1, 2, … in run order."""

        ids = [row[0] for row in self._db.execute(f"SELECT id FROM jobs WHERE status = ? ORDER BY {_QUEUE_ORDER}", (QUEUED,)).fetchall()]
        self._db.executemany("UPDATE jobs SET position = ? WHERE id = ?", [(float(index + 1), job_id) for index, job_id in enumerate(ids)])

    def history(self, *, cursor: str | None = None, limit: int = 50, status: str | None = None, query: str | None = None, starred: bool | None = None, tab: str | None = None) -> tuple[list[Job], str | None, int]:
        """Newest first (by submission time: imported jobs keep theirs, so
        they sort among the user's own by when they were made):
        ``(jobs, next_cursor, total)``; ``status`` filters
        (a comma-separated list is allowed), every word of ``query`` must
        appear in the prompt summary, the infotext or the parameters,
        ``starred`` and ``tab`` filter too."""

        where, args = ["1"], []
        if status:
            wanted = [item for item in status.split(",") if item in STATUSES]
            where.append(f"status IN ({','.join('?' * len(wanted))})" if wanted else "0")
            args += wanted
        for pattern in like_terms(query):
            where.append("(summary LIKE ? ESCAPE '\\' OR infotext LIKE ? ESCAPE '\\' OR core LIKE ? ESCAPE '\\' OR meta LIKE ? ESCAPE '\\')")
            args += [pattern] * 4
        if starred is not None:
            where.append("starred = ?")
            args.append(int(starred))
        if tab:
            where.append("tab = ?")
            args.append(tab)
        filters = " AND ".join(where)
        page = filters
        page_args = list(args)
        position = decode_cursor(cursor, 2)
        if position is not None:
            try:
                created, seq = float(position[0]), int(position[1])
            except ValueError:
                pass
            else:
                page += " AND (created < ? OR (created = ? AND seq < ?))"
                page_args += [created, created, seq]
        limit = min(max(int(limit), 1), 200)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM jobs WHERE {filters}", args).fetchone()[0]
        jobs = self._fetch(f"SELECT {_COLUMNS} FROM jobs WHERE {page} ORDER BY created DESC, seq DESC LIMIT ?", [*page_args, limit + 1])
        next_cursor = encode_cursor(repr(jobs[limit - 1].created), jobs[limit - 1].seq) if len(jobs) > limit else None
        return jobs[:limit], next_cursor, total

    def set_starred(self, ids: list[str], starred: bool) -> int:
        ids = list(dict.fromkeys(ids))[:MAX_REORDER]
        if not ids:
            return 0
        with self._lock:
            return self._db.execute(f"UPDATE jobs SET starred = ? WHERE id IN ({','.join('?' * len(ids))})", (int(starred), *ids)).rowcount

    def update_params(self, job_id: str, core: dict, scripts: dict, *, layout_hash: str, schema_version: int, summary: str) -> Job | None:
        """Replace a queued job's parameters; None when it is not queued (any more)."""

        with self._lock:
            changed = self._db.execute(
                "UPDATE jobs SET core = ?, scripts = ?, layout_hash = ?, schema_version = ?, summary = ? WHERE id = ? AND status = ?",
                (json.dumps(core), json.dumps(scripts), layout_hash, schema_version, summary, job_id, QUEUED),
            ).rowcount
        return self.get(job_id) if changed else None

    def clear_finished(self, *, keep_starred: bool = True) -> int:
        """Delete finished jobs (but the starred ones, with ``keep_starred``)."""

        with self._lock:
            sql = f"DELETE FROM jobs WHERE status IN ({','.join('?' * len(FINISHED))})"
            if keep_starred:
                sql += " AND starred = 0"
            return self._db.execute(sql, tuple(sorted(FINISHED))).rowcount

    def imported_job(self, source: str, source_id: str) -> str | None:
        """The job an importer made of ``source_id`` (while it exists)."""

        with self._lock:
            row = self._db.execute("SELECT imports.job_id FROM imports JOIN jobs ON jobs.id = imports.job_id WHERE source = ? AND source_id = ?", (source, source_id)).fetchone()
        return row[0] if row else None

    def record_import(self, source: str, source_id: str, job_id: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO imports (source, source_id, job_id) VALUES (?, ?, ?)", (source, source_id, job_id))

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
