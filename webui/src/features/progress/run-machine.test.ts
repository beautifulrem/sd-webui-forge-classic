import { describe, expect, it } from "vitest"
import { IDLE, isBusy, phaseOf, reduce, type RunEffect, type RunEvent, type RunOptions, type RunState } from "./run-machine"

const restart: RunOptions = { interruptOnly: false }

/** Feeds events in order; returns the final state and every effect produced. */
function run(events: RunEvent[], options: RunOptions = restart, from: RunState = IDLE) {
  let state = from
  const effects: RunEffect[] = []
  for (const event of events) {
    const [next, produced] = reduce(state, event, options)
    state = next
    effects.push(...produced)
  }
  return { state, effects, phase: phaseOf(state) }
}

const submits = (effects: RunEffect[]) => effects.filter((e) => e.type === "submit").length

describe("run machine", () => {
  it("upscales a result with one job of its own (the ✨ button)", () => {
    const source = { fileId: "out-1", key: "a:0" }
    const { state, effects, phase } = run([{ type: "upscale", source }])
    expect(phase).toBe("submitting")
    expect(effects).toEqual([{ type: "submit", upscale: source }])
    const done = run([{ type: "submitted", jobIds: ["u"] }, { type: "started", jobId: "u" }, { type: "finished", jobId: "u", outcome: "completed" }], restart, state)
    expect(done.phase).toBe("idle")
    expect(submits(done.effects)).toBe(0)
  })

  it("goes idle → submitting → queued → running → idle", () => {
    expect(phaseOf(IDLE)).toBe("idle")
    expect(run([{ type: "generate" }]).phase).toBe("submitting")
    expect(run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }]).phase).toBe("queued")
    expect(run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }]).phase).toBe("running")
    const done = run([
      { type: "generate" },
      { type: "submitted", jobIds: ["a"] },
      { type: "started", jobId: "a" },
      { type: "finished", jobId: "a", outcome: "completed" },
    ])
    expect(done.phase).toBe("idle")
    expect(submits(done.effects)).toBe(1)
  })

  it("queues another job when Generate is pressed while one runs", () => {
    const { state, effects } = run([
      { type: "generate" },
      { type: "submitted", jobIds: ["a"] },
      { type: "started", jobId: "a" },
      { type: "generate" },
      { type: "submitted", jobIds: ["b"] },
    ])
    expect(state.jobs).toEqual(["a", "b"])
    expect(submits(effects)).toBe(2)
  })

  it("ignores events of other clients' jobs", () => {
    const { state } = run([{ type: "started", jobId: "x" }, { type: "finished", jobId: "x", outcome: "completed" }])
    expect(state).toEqual(IDLE)
  })

  it("rejected submits end forever and repeats", () => {
    const { state, phase } = run([{ type: "forever", on: true }, { type: "rejected" }])
    expect(phase).toBe("idle")
    expect(state.forever).toBe(false)
  })

  describe("Ctrl/Cmd+Enter", () => {
    const running = run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }]).state

    it("generates when nothing of this form runs", () => {
      expect(run([{ type: "shortcut" }]).effects).toEqual([{ type: "submit" }])
    })

    it("a second press while the first submit is in flight queues no duplicate", () => {
      const { state, effects } = run([{ type: "shortcut" }, { type: "shortcut" }])
      expect(submits(effects)).toBe(1)
      expect(state.pending).toBe(1)
    })

    it("interrupts the running job and generates again once it has ended", () => {
      const interrupted = run([{ type: "shortcut" }], restart, running)
      expect(interrupted.effects).toEqual([{ type: "interrupt", jobId: "a" }])
      expect(interrupted.phase).toBe("interrupting")
      const after = run([{ type: "finished", jobId: "a", outcome: "interrupted" }], restart, interrupted.state)
      expect(after.effects).toEqual([{ type: "submit" }])
      expect(after.phase).toBe("submitting")
    })

    it("only interrupts with ctrl_enter_interrupt", () => {
      const options = { interruptOnly: true }
      const interrupted = run([{ type: "shortcut" }], options, running)
      const after = run([{ type: "finished", jobId: "a", outcome: "interrupted" }], options, interrupted.state)
      expect(after.effects).toEqual([])
      expect(after.phase).toBe("idle")
    })
  })

  describe("interrupt and skip", () => {
    it("interrupts the running job and cancels this form's queued ones", () => {
      const { state, effects } = run([
        { type: "generate" },
        { type: "submitted", jobIds: ["a"] },
        { type: "generate" },
        { type: "submitted", jobIds: ["b"] },
        { type: "started", jobId: "a" },
        { type: "interrupt" },
      ])
      expect(effects.slice(-2)).toEqual([
        { type: "interrupt", jobId: "a" },
        { type: "cancel", jobId: "b" },
      ])
      expect(state.jobs).toEqual(["a"])
      expect(phaseOf(state)).toBe("interrupting")
    })

    it("cancels a job whose submit was in flight when interrupted", () => {
      const { state, effects } = run([{ type: "generate" }, { type: "interrupt" }, { type: "submitted", jobIds: ["late"] }])
      expect(effects.at(-1)).toEqual({ type: "cancel", jobId: "late" })
      expect(isBusy(state)).toBe(false)
    })

    it("cancels every submit in flight, not just the first to answer", () => {
      // "Queue another" twice while a job runs, then Interrupt before either POST answered.
      const running = run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }]).state
      const { state, effects } = run(
        [
          { type: "generate" },
          { type: "generate" },
          { type: "interrupt" },
          { type: "submitted", jobIds: ["b"] },
          { type: "submitted", jobIds: ["c1", "c2"] },
          { type: "finished", jobId: "a", outcome: "interrupted" },
        ],
        restart,
        running,
      )
      expect(effects.filter((e) => e.type === "cancel")).toEqual([
        { type: "cancel", jobId: "b" },
        { type: "cancel", jobId: "c1" },
        { type: "cancel", jobId: "c2" },
      ])
      expect(state.jobs).toEqual([])
      expect(phaseOf(state)).toBe("idle")
    })

    it("a submit answered after the interrupt does not stay busy when another was rejected", () => {
      const { state } = run([{ type: "generate" }, { type: "generate" }, { type: "interrupt" }, { type: "rejected" }, { type: "submitted", jobIds: ["x"] }])
      expect(state).toMatchObject({ pending: 0, dropPending: 0, jobs: [] })
    })

    it("skip acts on the running job only", () => {
      expect(run([{ type: "skip" }]).effects).toEqual([])
      const { effects } = run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }, { type: "skip" }])
      expect(effects.at(-1)).toEqual({ type: "skip", jobId: "a" })
    })
  })

  describe("forever and repeat", () => {
    it("generate forever resubmits after every finished job until stopped", () => {
      const first = run([{ type: "forever", on: true }, { type: "submitted", jobIds: ["a"] }, { type: "finished", jobId: "a", outcome: "completed" }])
      expect(submits(first.effects)).toBe(2)
      expect(first.state.forever).toBe(true)
      const stopped = run([{ type: "submitted", jobIds: ["b"] }, { type: "forever", on: false }, { type: "finished", jobId: "b", outcome: "completed" }], restart, first.state)
      expect(submits(stopped.effects)).toBe(0)
      expect(stopped.phase).toBe("idle")
    })

    it("turning forever on while busy waits for the current job", () => {
      const busy = run([{ type: "generate" }, { type: "submitted", jobIds: ["a"] }])
      const on = run([{ type: "forever", on: true }], restart, busy.state)
      expect(on.effects).toEqual([])
      expect(run([{ type: "finished", jobId: "a", outcome: "completed" }], restart, on.state).effects).toEqual([{ type: "submit" }])
    })

    it("interrupt and failures end forever", () => {
      const forever = run([{ type: "forever", on: true }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }])
      expect(run([{ type: "interrupt" }, { type: "finished", jobId: "a", outcome: "interrupted" }], restart, forever.state).state.forever).toBe(false)
      const failed = run([{ type: "finished", jobId: "a", outcome: "failed" }], restart, forever.state)
      expect(failed.state.forever).toBe(false)
      expect(submits(failed.effects)).toBe(0)
    })

    it("repeat N submits N times, one after another", () => {
      let { state, effects } = run([{ type: "generate", runs: 3 }])
      expect(state.repeatLeft).toBe(2)
      for (const id of ["a", "b", "c"]) {
        const step = run([{ type: "submitted", jobIds: [id] }, { type: "finished", jobId: id, outcome: "completed" }], restart, state)
        state = step.state
        effects = [...effects, ...step.effects]
      }
      expect(submits(effects)).toBe(3)
      expect(phaseOf(state)).toBe("idle")
    })

    it("skip keeps the repeats going; stop-repeats ends them", () => {
      const started = run([{ type: "generate", runs: 3 }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }, { type: "skip" }])
      expect(started.state.repeatLeft).toBe(2)
      const stopped = run([{ type: "stop-repeats" }, { type: "finished", jobId: "a", outcome: "completed" }], restart, started.state)
      expect(submits(stopped.effects)).toBe(0)
    })
  })

  it("restores jobs found after a reload", () => {
    const { state, phase } = run([{ type: "restore", jobs: ["a", "b"], running: "a" }])
    expect(state.jobs).toEqual(["a", "b"])
    expect(phase).toBe("running")
  })
})
