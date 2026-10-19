/**
 * A step's progress drawn with its preview. The server sends a step's preview
 * frame and, a few dozen milliseconds later (when the sampler has started the
 * next step), that step's `job.progress`; drawn as they arrive, the page
 * renders and paints twice per step. While previews arrive for every step of
 * the running job, a progress event waits for the next preview frame and is
 * applied in the same animation frame (api/paint-batch), so the step counter,
 * percentage and ETA change together with the image, once per step. If the
 * next frame is late (more than one step interval, see {@link holdMs}), the
 * progress is drawn on its own.
 */
import { paintBatch } from "./paint-batch"

/** The longest a progress event waits for a preview, whatever the step interval. */
export const MAX_HOLD_MS = 10_000

type Progress = { seq?: number }
type Frame = { jobSeq: number; step: number }

export type ProgressGate<P extends Progress> = {
  /** A `job.progress` event: applied now, or held until the next preview frame. */
  progress: (event: P) => void
  /** A preview frame arrived (before it is decoded): releases the held progress into its paint. */
  frame: (frame: Frame) => void
  /** Applies a held progress now (the job ended or another one started). */
  flush: () => void
}

/** How long a progress event may wait after the last frame: one step interval more than expected. */
export function holdMs(interval: number): number {
  return Math.min(2 * interval, MAX_HOLD_MS)
}

export function createProgressGate<P extends Progress>(apply: (event: P) => void, now: () => number = () => performance.now()): ProgressGate<P> {
  let last: (Frame & { at: number }) | null = null
  /** ms between the last two frames when they were consecutive steps of one job, else 0 */
  let interval = 0
  let held: P | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  function release() {
    clearTimeout(timer)
    timer = undefined
    const event = held
    held = null
    if (event) paintBatch(() => apply(event))
  }

  return {
    progress(event) {
      const flowing = last !== null && interval > 0 && event.seq === last.jobSeq
      const wait = flowing && last ? last.at + holdMs(interval) - now() : 0
      if (wait <= 0) {
        release()
        paintBatch(() => apply(event))
        return
      }
      held = event
      clearTimeout(timer)
      timer = setTimeout(release, wait)
    },
    frame(frame) {
      const at = now()
      interval = last && last.jobSeq === frame.jobSeq && frame.step === last.step + 1 ? at - last.at : 0
      last = { ...frame, at }
      release()
    },
    flush() {
      release()
      last = null
      interval = 0
    },
  }
}
