# Agent Scheduler (superseded)

The Agent Scheduler's task queue, history, bookmarks and "Enqueue" button are
part of the Remi WebUI itself now: the **Queue** page (pending jobs with
drag-to-reorder and priorities, pause / resume, editing, history with search
and stars) and the **Add to queue** button next to Generate (加入队列 in the
Chinese UI).

Its HTTP API (`/agent-scheduler/v1/*`) is not kept (its clients get
`410 Gone`): scripts use `POST /api/v2/jobs` and `/api/v2/queue/*` (see
`docs/remi-webui/architecture.md`).

The **Add to queue** keyboard shortcut is Ctrl+Shift+Enter (⌘⇧Enter on
macOS), next to Ctrl+Enter for Generate, on the Generate and Extras pages.

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

## Settings of the old extension

The queue's settings keep their keys, so saved values carry over
(Settings → System → Queue, run by the server, `remi/jobs/queue.py`):

| Setting | In Remi |
|---|---|
| `queue_history_retention_days` | Kept, same choices; starred jobs are kept and the images stay in the gallery. The default is "Keep forever" (it was "30 days"): the history now holds every generation, not only queued tasks, so a default clean-up would delete what you never queued. A saved value is used as it is. |
| `queue_automatic_requeue_failed_task` | Kept: a failed job goes to the end of the queue again, up to 3 attempts in all (the old one retried forever, so a job that always fails kept the queue busy); cancelled and interrupted jobs are not retried. |
| `queue_completion_action` | Kept, same actions (do nothing, stop the WebUI, sleep, hibernate, restart, shut down). It runs once, after the last queued job, and then goes back to "Do nothing", so a restarted WebUI does not shut the computer down again when its first job ends. Settable only from the WebUI, not from the API (`restrict_api`). |

### Not carried over

These settings have no equivalent, for the reason given
(`tests/test_remi_server_jobs.py` checks that every old setting is either a
setting here or a row of this table):

| Setting | Why |
|---|---|
| `queue_paused` | It stored the pause state of the queue. The Queue page's pause / resume does that now, and the state is kept across restarts (the queue's database), so a separate "disable auto-processing" setting would be a second switch for the same thing. |
| `queue_button_hide_checkpoint` | It hid the checkpoint dropdown next to "Enqueue", which never had an effect in Forge Neo: the task got the override `Model hash: <checkpoint>`, and no setting reads that infotext key (the checkpoint's is `Model`), so queued tasks ran with whatever checkpoint was loaded. A queued job runs with the model bar's selection when it starts, as before. The importer keeps the old override on imported tasks, so the job's form shows which checkpoint a task was queued for. |
| `queue_button_placement` | "Add to queue" has one place in every form, next to Generate / Process, in the run controls that stay visible while you scroll; Ctrl+Shift+Enter reaches it from anywhere on the page. |
| `queue_ui_placement` | The queue is its own page, one click away in the navigation, which shows how many jobs wait; the header shows a running job's progress on every page. There is no Gradio tab layout to append it to. |
| `queue_grid_page_size` | The history loads 50 jobs a page as you scroll and only draws the visible rows, so a page size is no longer a speed setting. |
| `queue_keyboard_shortcut` | Replaced by a fixed shortcut, Ctrl+Shift+Enter (⌘⇧Enter on macOS), the Generate shortcut plus Shift. The old default, Ctrl+E (⌘E), is the browser's own shortcut for its search bar in Chrome and Firefox on Windows and Linux, and a remappable shortcut would need its own conflict checks against the prompt editor's keys. |
