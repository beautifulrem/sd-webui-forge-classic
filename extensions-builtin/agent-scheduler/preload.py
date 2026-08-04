# preload.py is used for cmd line arguments
def preload(parser):
    parser.add_argument(
        "--agent-scheduler-sqlite-file",
        help="SQLite database path. Absolute paths are used as-is; relative paths are resolved from Forge's data directory.",
        default="extension-data/agent-scheduler/task_scheduler.sqlite3",
    )
