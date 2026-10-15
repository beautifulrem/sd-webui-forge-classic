import { afterEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/api/http"
import { dismissServerCommand, keepWaiting, requestServerCommand, runServerCommand, type ServerControlDeps, UP_TIMEOUT, useServerControl } from "./server-control"
import { askBeforeLeaving, installBeforeUnloadGuard, leaveWithoutAsking, unsavedTotal, useUnsaved } from "./unsaved"

/**
 * A server that answers pings per `script`, one entry per ping, then stays as
 * the last one: an instance id (up), `""` (up, no instance ids) or null (down).
 */
function fakeServer(script: (string | null)[], extra: Partial<ServerControlDeps> = {}) {
  let clock = 0
  let pings = 0
  const reloadPage = vi.fn()
  const deps: ServerControlDeps = {
    send: vi.fn(async () => ({ ok: true })),
    ping: async () => {
      const answer = script[Math.min(pings++, script.length - 1)]
      if (answer === null || answer === undefined) throw new Error("down")
      return answer || null
    },
    reloadPage,
    sleep: async (ms) => {
      clock += ms
    },
    now: () => clock,
    ...extra,
  }
  return { deps, reloadPage, pings: () => pings }
}

afterEach(() => dismissServerCommand())

describe("server control", () => {
  it("waits for the server to go down and come back as a new instance, then reloads the page", async () => {
    const phases: string[] = []
    const stop = useServerControl.subscribe((s) => phases.push(s.phase))
    const server = fakeServer(["old", "old", null, null, "new"])
    await runServerCommand("reload", server.deps)
    stop()
    expect(server.deps.send).toHaveBeenCalledWith("reload")
    expect(server.reloadPage).toHaveBeenCalledTimes(1)
    expect([...new Set(phases)]).toEqual(["requesting", "waiting-down", "waiting-up", "done"])
  })

  it("reloads as soon as another instance answers, even without a missed ping", async () => {
    const server = fakeServer(["old", "new"])
    await runServerCommand("restart", server.deps)
    expect(server.reloadPage).toHaveBeenCalledTimes(1)
    expect(server.pings()).toBe(2)
  })

  it("keeps waiting while the old server answers, also after a missed ping", async () => {
    const phases: string[] = []
    const stop = useServerControl.subscribe((s) => phases.push(s.phase))
    // busy with the running job: one ping times out, then the old app answers again
    const server = fakeServer(["old", "old", null, "old", "old", "new"])
    await runServerCommand("reload", server.deps)
    stop()
    expect(server.pings()).toBe(6)
    expect(server.reloadPage).toHaveBeenCalledTimes(1)
    expect(phases.filter((phase, index) => phase !== phases[index - 1])).toEqual(["requesting", "waiting-down", "waiting-up", "waiting-down", "done"])
  })

  it("without instance ids, waits for a missed ping before trusting an answer", async () => {
    const server = fakeServer(["", "", "", null, ""])
    await runServerCommand("reload", server.deps)
    expect(server.pings()).toBe(5)
    expect(server.reloadPage).toHaveBeenCalledTimes(1)
  })

  it("fails with a readable message when the server cannot restart", async () => {
    const server = fakeServer(["old"], { send: async () => Promise.reject(new ApiError(501, "no launcher")) })
    await runServerCommand("restart", server.deps)
    expect(useServerControl.getState()).toMatchObject({ phase: "failed", canWait: false, error: "This server cannot restart itself; restart it from its console." })
    expect(server.reloadPage).not.toHaveBeenCalled()
  })

  it("gives up after the timeout and can keep waiting", async () => {
    const down = fakeServer(["old", null])
    await runServerCommand("reload", down.deps)
    expect(useServerControl.getState()).toMatchObject({ phase: "failed", canWait: true })
    expect(down.deps.now()).toBeGreaterThan(UP_TIMEOUT)
    const back = fakeServer(["old", "new"])
    keepWaiting(back.deps)
    await vi.waitFor(() => expect(back.reloadPage).toHaveBeenCalled())
    expect(back.pings()).toBe(2)
  })

  it("a dismissed command stops following the server", async () => {
    const server = fakeServer(["old"])
    const run = runServerCommand("reload", server.deps)
    dismissServerCommand()
    await run
    expect(useServerControl.getState().phase).toBe("idle")
    expect(server.reloadPage).not.toHaveBeenCalled()
  })

  it("asks for confirmation first, and only while idle", () => {
    requestServerCommand("restart")
    expect(useServerControl.getState().pending).toBe("restart")
  })
})

describe("unsaved changes registry", () => {
  it("sums sources and asks before unloading until told otherwise", () => {
    installBeforeUnloadGuard()
    useUnsaved.getState().set("settings", 2)
    useUnsaved.getState().set("extensions", 1)
    expect(unsavedTotal()).toBe(3)
    const event = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(true)
    leaveWithoutAsking()
    const second = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(second)
    expect(second.defaultPrevented).toBe(false)
    askBeforeLeaving()
    const third = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(third)
    expect(third.defaultPrevented).toBe(true)
    useUnsaved.getState().set("settings", 0)
    useUnsaved.getState().set("extensions", 0)
    expect(useUnsaved.getState().sources).toEqual({})
  })
})
