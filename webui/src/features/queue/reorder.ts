/**
 * Moving a queued job. The queue runs by priority (higher first), then by the
 * user's order. A move is sent as "put it after / before this neighbour"
 * (`POST /queue/move`), so it costs the same in a queue of any length; a job
 * dropped among jobs of another priority takes theirs, so the list stays in
 * run order. Moves are planned by id against the latest queue: jobs that
 * started while the user was dragging do not shift the target.
 */

export type QueuedRef = { id: string; priority: number }

export type MovePlan = {
  /** The moved job. */
  id: string
  /** The list after the move (ids, run order). */
  order: string[]
  /** The moved job's new priority, when it changes. */
  priority: number | null
  /** The job it now follows in its priority group (null: first of the group). */
  after: string | null
  /** The job it now precedes in its priority group (null: last of the group). */
  before: string | null
}

export function moveIndex<T>(list: readonly T[], from: number, to: number): T[] {
  const next = [...list]
  if (from < 0 || from >= next.length) return next
  const [item] = next.splice(from, 1)
  next.splice(Math.max(0, Math.min(to, next.length)), 0, item as T)
  return next
}

/** The plan for moving the job at `from` to index `to`; null when nothing changes. */
export function planMove(jobs: readonly QueuedRef[], from: number, to: number): MovePlan | null {
  if (from === to || from < 0 || from >= jobs.length) return null
  const moved = moveIndex(jobs, from, to)
  const index = moved.findIndex((job) => job.id === jobs[from]!.id)
  const job = moved[index]!
  const above = moved[index - 1]
  const below = moved[index + 1]
  // Between two groups it keeps its priority if either neighbour shares it.
  let priority = job.priority
  if (above?.priority !== priority && below?.priority !== priority) priority = above?.priority ?? below?.priority ?? priority
  return {
    id: job.id,
    order: moved.map((item) => item.id),
    priority: priority === job.priority ? null : priority,
    after: above?.priority === priority ? above.id : null,
    before: below?.priority === priority ? below.id : null,
  }
}

/**
 * Where the job `id`, placed in `arranged` (the order the user sees, taken
 * when the drag began), lands in `latest` (the queue now): after the nearest
 * job above it that is still queued. Null when `id` left the queue.
 */
export function dropTarget(latest: readonly QueuedRef[], arranged: readonly QueuedRef[], id: string): { from: number; to: number } | null {
  const from = latest.findIndex((job) => job.id === id)
  if (from === -1) return null
  const rest = latest.filter((job) => job.id !== id)
  const queued = new Set(rest.map((job) => job.id))
  const at = arranged.findIndex((job) => job.id === id)
  for (let index = at - 1; index >= 0; index--) {
    const anchor = arranged[index]!.id
    if (queued.has(anchor)) return { from, to: rest.findIndex((job) => job.id === anchor) + 1 }
  }
  return { from, to: 0 }
}

/** Priority presets offered in the menus (any integer works on the server). */
export const PRIORITIES = [
  { value: 10, label: "High" },
  { value: 0, label: "Normal" },
  { value: -10, label: "Low" },
] as const

export function priorityLabel(priority: number): string | null {
  if (priority === 0) return null
  const preset = PRIORITIES.find((item) => item.value === priority)
  return preset ? preset.label : priority > 0 ? "High" : "Low"
}
