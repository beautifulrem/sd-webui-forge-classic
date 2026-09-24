# preload.py is used for cmd line arguments
def preload(parser):
    parser.add_argument(
        "--agent-scheduler-sqlite-file",
        help="SQLite database path. Absolute paths are used as-is; relative paths are resolved from Forge's data directory.",
        default="extension-data/agent-scheduler/task_scheduler.sqlite3",
    )
    parser.add_argument(
        "--agent-scheduler-trust-unsigned-params",
        action="store_true",
        help="Sign stored task params that have no valid signature (tasks saved by older versions) at startup. "
        "Use once, and only if nobody else could have imported tasks into this database.",
    )
