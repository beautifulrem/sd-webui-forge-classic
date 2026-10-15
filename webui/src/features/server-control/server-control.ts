/**
 * "Reload UI" and "Restart server" with a reconnecting overlay.
 *
 * The server answers the command first, then (after the running job, for a
 * reload) stops serving and comes back as a new app. The client watches
 * `/internal/ping`, whose `instance` differs for every app: once a ping
 * answers with another instance than before the command, the page reloads so
 * the new build, schemas and extensions take effect. (A server without
 * instance ids counts as back after it has failed a ping and answers again.)
 * Pure state machine + an effect runner, so tests drive it with a fake clock
 * and ping.
 */
import { create } from "zustand"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { askBeforeLeaving, leaveWithoutAsking, unsavedTotal } from "./unsaved"

export type ServerCommand = "reload" | "restart"

export type ServerPhase =
  /** Nothing happening. */
  | "idle"
  /** The command request is in flight. */
  | "requesting"
  /** Accepted; the server still answers (a reload waits for the running job). */
  | "waiting-down"
  /** The server went away; waiting for it to answer again. */
  | "waiting-up"
  /** Back: the page is reloading. */
  | "done"
  /** The command failed or the server did not come back in time. */
  | "failed"

type ServerControlState = {
  /** A command waiting for the user's confirmation. */
  pending: ServerCommand | null
  command: ServerCommand | null
  phase: ServerPhase
  /** `Date.now()` when the command was sent. */
  startedAt: number
  /** English message (translated at render) or a server message. */
  error: string | null
  /** After a timeout the server may still be starting: the user can keep waiting. */
  canWait: boolean
}

const IDLE: ServerControlState = { pending: null, command: null, phase: "idle", startedAt: 0, error: null, canWait: false }

export const useServerControl = create<ServerControlState>()(() => IDLE)

/** How often the server is pinged, and how long each ping may take. */
export const PING_INTERVAL = 800
const PING_TIMEOUT = 2500
/** Give up waiting for the server to come back after this long. */
export const UP_TIMEOUT = 5 * 60_000

export type ServerControlDeps = {
  send: (command: ServerCommand) => Promise<unknown>
  /** Resolves with the server's instance id when it answers, rejects when it does not. */
  ping: () => Promise<string | null>
  reloadPage: () => void
  sleep: (ms: number) => Promise<void>
  now: () => number
}

const defaultDeps: ServerControlDeps = {
  send: (command) => (command === "reload" ? api.system.reload() : api.system.restart()),
  ping: async () => (await api.system.ping(AbortSignal.timeout(PING_TIMEOUT))).instance ?? null,
  reloadPage: () => {
    leaveWithoutAsking()
    location.reload()
  },
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  now: () => Date.now(),
}

let generation = 0
/** The instance that answered before the command (null: unknown). */
let before: string | null = null

type Probe = { up: false } | { up: true; instance: string | null }

async function probe(deps: ServerControlDeps): Promise<Probe> {
  try {
    return { up: true, instance: await deps.ping() }
  } catch {
    return { up: false }
  }
}

/** Whether an answer comes from the new server: another instance, or (no ids) any answer after it was down. */
function isNew(answer: Probe, wentDown: boolean): boolean {
  if (!answer.up) return false
  if (before !== null && answer.instance !== null) return answer.instance !== before
  return wentDown
}

/**
 * Sends `command` and follows the server until it is back, then reloads the
 * page. Resolves when the overlay has reached `done` or `failed`.
 */
export async function runServerCommand(command: ServerCommand, deps: ServerControlDeps = defaultDeps): Promise<void> {
  const run = ++generation
  const set = (patch: Partial<ServerControlState>) => {
    if (run === generation) useServerControl.setState(patch)
  }
  useServerControl.setState({ pending: null, command, phase: "requesting", startedAt: deps.now(), error: null, canWait: false })
  const first = await probe(deps)
  if (run !== generation) return
  before = first.up ? first.instance : null
  try {
    await deps.send(command)
  } catch (error) {
    // The connection dropping mid-request means the server is already going down.
    if (error instanceof ApiError) {
      set({ phase: "failed", error: error.status === 501 ? "This server cannot restart itself; restart it from its console." : error.message })
      return
    }
  }
  set({ phase: "waiting-down" })
  let downSince: number | null = null
  while (run === generation) {
    await deps.sleep(PING_INTERVAL)
    if (run !== generation) return
    const answer = await probe(deps)
    if (run !== generation) return
    if (isNew(answer, downSince !== null)) {
      set({ phase: "done" })
      deps.reloadPage()
      return
    }
    if (!answer.up) {
      if (downSince === null) {
        downSince = deps.now()
        set({ phase: "waiting-up" })
      } else if (deps.now() - downSince > UP_TIMEOUT) {
        set({ phase: "failed", error: "The server did not come back. Check its console, then reload this page.", canWait: true })
        return
      }
    } else if (downSince !== null) {
      // The old server answered again (a missed ping): it has not gone down yet.
      downSince = null
      set({ phase: "waiting-down" })
    }
  }
}

/** Keep waiting after a timeout (the server may still be starting). */
export function keepWaiting(deps: ServerControlDeps = defaultDeps) {
  const { command } = useServerControl.getState()
  const run = ++generation
  useServerControl.setState({ phase: "waiting-up", error: null, canWait: false })
  void (async () => {
    while (run === generation) {
      await deps.sleep(PING_INTERVAL)
      if (run !== generation) return
      if (isNew(await probe(deps), true) && run === generation) {
        useServerControl.setState({ phase: "done", command })
        deps.reloadPage()
        return
      }
    }
  })()
}

/** Closes the overlay (stops watching). */
export function dismissServerCommand() {
  generation++
  askBeforeLeaving()
  useServerControl.setState(IDLE)
}

/** Asks the user to confirm `command` (the shell's dialog), then runs it. */
export function requestServerCommand(command: ServerCommand) {
  if (useServerControl.getState().phase !== "idle") return
  useServerControl.setState({ pending: command })
}

export function cancelServerCommand() {
  useServerControl.setState({ pending: null })
}

/**
 * Runs `command` right away after an explicit "apply" by the user, asking
 * first only when unsaved edits elsewhere would be lost.
 */
export function runServerCommandAfterApply(command: ServerCommand) {
  if (unsavedTotal() > 0) requestServerCommand(command)
  else void runServerCommand(command)
}
