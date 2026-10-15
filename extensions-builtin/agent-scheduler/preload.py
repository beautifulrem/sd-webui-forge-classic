# The Agent Scheduler is superseded by the Remi Queue (Queue page). These
# flags stay so existing launch commands keep working: the Queue's importer
# ("Import from Agent Scheduler" in the Queue page menu) reads the database
# --agent-scheduler-sqlite-file names.
def preload(parser):
    parser.add_argument(
        "--agent-scheduler-sqlite-file",
        help="Agent Scheduler database to import into the Queue. Absolute paths are used as-is; relative paths are resolved from Forge's data directory.",
        default="extension-data/agent-scheduler/task_scheduler.sqlite3",
    )
    parser.add_argument(
        "--agent-scheduler-trust-unsigned-params",
        action="store_true",
        help="No longer used: the Queue's importer reads stored script arguments with a restricted loader, signed or not.",
    )
