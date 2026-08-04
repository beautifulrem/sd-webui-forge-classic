import os
import shutil

from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.orm import declarative_base

from modules import scripts
from modules import shared
from modules import paths_internal

if hasattr(shared.cmd_opts, "agent_scheduler_sqlite_file"):
    # Relative database paths belong to Forge's user-data directory. Built-in
    # source trees may be read-only and must remain clean across normal use.
    if not os.path.isabs(shared.cmd_opts.agent_scheduler_sqlite_file):
        db_file = os.path.join(paths_internal.data_path, shared.cmd_opts.agent_scheduler_sqlite_file)
    else:
        db_file = os.path.abspath(shared.cmd_opts.agent_scheduler_sqlite_file)
else:
    db_file = os.path.join(
        paths_internal.data_path,
        "extension-data",
        "agent-scheduler",
        "task_scheduler.sqlite3",
    )

legacy_db_file = os.path.join(scripts.basedir(), "task_scheduler.sqlite3")
try:
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    if not os.path.exists(db_file) and os.path.isfile(legacy_db_file):
        shutil.copy2(legacy_db_file, db_file)
except OSError as e:
    print(f"Agent Scheduler database migration failed: {e}")

print(f"Using sqlite file: {db_file}")


Base = declarative_base()
metadata: MetaData = Base.metadata

class BaseTableManager:
    def __init__(self, engine = None):
        # Get the db connection object, making the file and tables if needed.
        try:
            self.engine = engine if engine else create_engine(f"sqlite:///{db_file}")
        except Exception as e:
            print(f"Exception connecting to database: {e}")
            raise e

    def get_engine(self):
        return self.engine

    # Commit and close the database connection.
    def quit(self):
        self.engine.dispose()
