# Agent Scheduler (superseded)

The Agent Scheduler's task queue, history, bookmarks and "Enqueue" button are
part of the Remi WebUI itself now: the **Queue** page (pending jobs with
drag-to-reorder and priorities, pause / resume, editing, history with search
and stars) and the **Add to queue** button next to Generate (加入队列 in the
Chinese UI).

Its HTTP API (`/agent-scheduler/v1/*`) is not kept: scripts use
`POST /api/v2/jobs` and `/api/v2/queue/*` (see `docs/remi-webui/architecture.md`).

This folder only keeps the command line flags of the extension, so existing
launch commands keep working.

## Bringing your tasks over

Open **Queue → ⋯ → Import from Agent Scheduler**. The importer reads the
database (`--agent-scheduler-sqlite-file`, by default
`<data>/extension-data/agent-scheduler/task_scheduler.sqlite3`) without
changing it, and can be run again safely (tasks imported before are skipped):

- pending tasks are queued (the queue is paused first, so you can review them),
- finished tasks go to the history with their images (when the files still
  exist inside the output folders and were not removed from the gallery),
  bookmarks become stars, task names are kept; old tasks sort by when they
  were made, below your recent jobs,
- script arguments, stored by position, are named through the current script
  layout; arguments past the last known parameter cannot be named. Such tasks
  are kept in the history as *not replayable* and listed in the import report.
  Open one in its form (Edit) to check the parameters and queue it from there.

Stored script arguments are read with a restricted loader (plain data, arrays
and images only), so an imported database cannot run code.
