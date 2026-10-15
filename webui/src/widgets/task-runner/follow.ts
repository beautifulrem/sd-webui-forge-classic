/**
 * Following a page task (`remi.pages.Page.start_task`): poll the owner's
 * `__task__` action until the task ends; every answer is applied like an
 * action's patch (the status lands in the outputs as `task:<name>`, the
 * final answer carries the task's result). One follower per form, owner and
 * task; it outlives the widget, so a result still lands when the user has
 * left the page meanwhile.
 */
import { api } from "@/api/endpoints"
import type { ActionExtra } from "@/api/types"
import { type ActionContext, actionRequest, applyResponse, splitResponse } from "@/ui/actions"
import type { Message } from "@/ui/messages"

export type TaskStatus = {
  id: string
  name: string
  state: "queued" | "running" | "completed" | "failed" | "cancelled"
  progress: number | null
  text: string | Message | null
  error: string | null
  done: boolean
  cancelling?: boolean
}

export const outputKey = (task: string) => `task:${task}`

export function isTaskStatus(value: unknown): value is TaskStatus {
  if (typeof value !== "object" || value === null) return false
  const status = value as Record<string, unknown>
  return typeof status.id === "string" && typeof status.state === "string" && typeof status.done === "boolean"
}

const followers = new Map<string, Promise<void>>()
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export function isFollowing(ctx: ActionContext, owner: string, task: string): boolean {
  return followers.has(`${ctx.tab}/${owner}/${task}`)
}

/** Asks for the current status only (a view that mounts again); applies it. */
export async function resumeTask(ctx: ActionContext, owner: string, task: string): Promise<TaskStatus | null> {
  try {
    return await poll(ctx, owner, task, { resume: true })
  } catch {
    return null
  }
}

/**
 * Polls until task `id` ends (`interval` ms apart; network errors back off up
 * to 5 s and never toast). Resolves when it has ended or was replaced.
 */
export function followTask(ctx: ActionContext, owner: string, task: string, id: string, interval = 750): Promise<void> {
  const key = `${ctx.tab}/${owner}/${task}`
  const running = followers.get(key)
  if (running) return running
  const loop = (async () => {
    let delay = interval
    for (;;) {
      await sleep(delay)
      try {
        const status = await poll(ctx, owner, task, { id })
        delay = interval
        if (!status || status.done || status.id !== id) return
      } catch {
        delay = Math.min(delay * 2, 5000)
      }
    }
  })().finally(() => followers.delete(key))
  followers.set(key, loop)
  return loop
}

async function poll(ctx: ActionContext, owner: string, task: string, extra: Record<string, unknown>): Promise<TaskStatus | null> {
  // The task's own arguments (ActionExtra types the core actions' only)
  const body = actionRequest(ctx, owner, { task, ...extra } as unknown as ActionExtra)
  const split = splitResponse(owner, await api.action(owner, "__task__", body))
  applyResponse(ctx.store, owner, split)
  const status = split.outputs?.[outputKey(task)]
  return isTaskStatus(status) ? status : null
}
