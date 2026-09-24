from pathlib import Path
from sqlalchemy import create_engine, inspect, text, String, Text

from modules import shared

from .base import Base, metadata, db_file
from .app_state import AppStateKey, AppState, AppStateManager
from .task import TaskStatus, Task, TaskManager

version = "2"

state_manager = AppStateManager()
task_manager = TaskManager()


def init():
    engine = create_engine(f"sqlite:///{db_file}")

    metadata.create_all(engine)

    state_manager.set_value(AppStateKey.Version, version)
    # check if app state exists
    if state_manager.get_value(AppStateKey.QueueState) is None:
        # create app state
        state_manager.set_value(AppStateKey.QueueState, "running")

    inspector = inspect(engine)
    with engine.connect() as conn:
        task_columns = inspector.get_columns("task")
        # add result column
        if not any(col["name"] == "result" for col in task_columns):
            conn.execute(text("ALTER TABLE task ADD COLUMN result TEXT"))

        # add api_task_id column
        if not any(col["name"] == "api_task_id" for col in task_columns):
            conn.execute(text("ALTER TABLE task ADD COLUMN api_task_id VARCHAR(64)"))

        # add api_task_callback column
        if not any(col["name"] == "api_task_callback" for col in task_columns):
            conn.execute(text("ALTER TABLE task ADD COLUMN api_task_callback VARCHAR(255)"))

        # add name column
        if not any(col["name"] == "name" for col in task_columns):
            conn.execute(text("ALTER TABLE task ADD COLUMN name VARCHAR(255)"))

        # add bookmarked column
        if not any(col["name"] == "bookmarked" for col in task_columns):
            conn.execute(text("ALTER TABLE task ADD COLUMN bookmarked BOOLEAN DEFAULT FALSE"))

        params_column = next(col for col in task_columns if col["name"] == "params")
        if version > "1" and not isinstance(params_column["type"], Text):
            transaction = conn.begin()
            conn.execute(
                text(
                    """
                    CREATE TABLE task_temp (
                        id VARCHAR(64) NOT NULL,
                        type VARCHAR(20) NOT NULL,
                        params TEXT NOT NULL,
                        script_params BLOB NOT NULL,
                        priority INTEGER NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        created_at DATETIME DEFAULT (datetime('now')) NOT NULL,
                        updated_at DATETIME DEFAULT (datetime('now')) NOT NULL,
                        result TEXT,
                        PRIMARY KEY (id)
                    )"""
                )
            )
            conn.execute(text("INSERT INTO task_temp SELECT * FROM task"))
            conn.execute(text("DROP TABLE task"))
            conn.execute(text("ALTER TABLE task_temp RENAME TO task"))
            transaction.commit()

        conn.close()

    if getattr(shared.cmd_opts, "agent_scheduler_trust_unsigned_params", False):
        _trust_unsigned_script_params(engine)


# Result prefix of tasks that failed because their params were unsigned.
UNSIGNED_FAILURE = "Could not load task: task script params are not signed"


def _trust_unsigned_script_params(engine):
    """Sign every stored script param without a valid signature, on the
    user's explicit word that this database holds no crafted imports. Tasks
    that failed only because they were unsigned go back to the queue."""
    from .. import signing
    from ..helpers import log

    signing.ensure_key()
    if not signing.key_is_persistent:
        log.error("[AgentScheduler] Not trusting unsigned tasks: the signing key file cannot be used")
        return
    signed = requeued = 0
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, script_params, status, result FROM task")).fetchall()
        for task_id, script_params, status, result in rows:
            if script_params is None or signing.verified_payload(script_params) is not None:
                continue
            values = {"value": signing.sign(signing.strip_signature(script_params)), "id": task_id}
            if status == "failed" and (result or "").startswith(UNSIGNED_FAILURE):
                conn.execute(
                    text("UPDATE task SET script_params = :value, status = 'pending', result = NULL WHERE id = :id"),
                    values,
                )
                requeued += 1
            else:
                conn.execute(text("UPDATE task SET script_params = :value WHERE id = :id"), values)
            signed += 1
    log.warning(
        f"[AgentScheduler] Trusted and signed {signed} stored task(s), {requeued} requeued; "
        "remove --agent-scheduler-trust-unsigned-params for later starts"
    )


__all__ = [
    "init",
    "Base",
    "metadata",
    "db_file",
    "AppStateKey",
    "AppState",
    "TaskStatus",
    "Task",
    "task_manager",
    "state_manager",
]
