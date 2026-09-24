from pathlib import Path
from sqlalchemy import create_engine, inspect, text, String, Text

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

    _sign_existing_script_params(engine)


def _sign_existing_script_params(engine):
    """Sign script params stored before signing existed, so the queue and
    history keep working. Runs once: anything unsigned that shows up later is
    refused instead of being trusted.

    Failed tasks are left unsigned: the previous pickle filter refused crafted
    imports by failing them, and signing those would let a requeue run them.
    """
    import os

    from ..signing import is_signed, migration_marker, sign

    marker = migration_marker()
    if os.path.exists(marker):
        return
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, script_params FROM task WHERE status != :failed"),
            {"failed": "failed"},
        ).fetchall()
        for task_id, script_params in rows:
            if script_params is not None and not is_signed(script_params):
                conn.execute(
                    text("UPDATE task SET script_params = :value WHERE id = :id"),
                    {"value": sign(bytes(script_params)), "id": task_id},
                )
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("signed\n")


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
