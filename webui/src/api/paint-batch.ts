/**
 * One paint per generation step. A step reaches the page as a binary preview
 * frame and a `job.progress` event; applied to the stores as each message
 * arrived, the page rendered and drew once per message and once more when the
 * new preview image had decoded (about four frames per step, each a layout,
 * paint and composite while the GPU samples).
 *
 * Updates queued with {@link paintBatch} run together in the next animation
 * frame, so React commits once and the browser draws once. A preview that is
 * still decoding holds the batch ({@link holdPaint}, at most
 * {@link MAX_HOLD_MS}), so the step's progress (released by
 * `progress-gate.ts` when the frame arrives) and its decoded image appear in
 * the same frame. Code that must see the queued updates first (a job ending)
 * calls {@link flushPaintBatch}. While the page is hidden nothing is drawn and
 * updates apply at once.
 */

/** Longest a decoding preview may hold back the other updates. */
export const MAX_HOLD_MS = 100

let pending: (() => void)[] = []
let holds = 0
let frame: number | null = null
/** Ends a hold that lasted MAX_HOLD_MS. */
let holdTimer: ReturnType<typeof setTimeout> | undefined
let expired = false

function visible(): boolean {
  return typeof requestAnimationFrame === "function" && typeof document !== "undefined" && document.visibilityState === "visible"
}

function schedule() {
  if (pending.length === 0) return
  if (!visible()) {
    flushPaintBatch()
    return
  }
  if (holds > 0 && !expired) {
    // no animation frames are requested while holding (each would be drawn): the release schedules one
    holdTimer ??= setTimeout(() => {
      holdTimer = undefined
      expired = true
      schedule()
    }, MAX_HOLD_MS)
    return
  }
  frame ??= requestAnimationFrame(onFrame)
}

function onFrame() {
  frame = null
  // a hold that began after this frame was requested (a preview arriving right after its progress)
  if (holds > 0 && !expired) schedule()
  else flushPaintBatch()
}

/** Applies `update` (a store write that changes what is drawn) in the next animation frame. */
export function paintBatch(update: () => void): void {
  pending.push(update)
  schedule()
}

/** Holds the next batch until the returned release is called (e.g. a preview decoding). */
export function holdPaint(): () => void {
  holds++
  let released = false
  return () => {
    if (released) return
    released = true
    holds--
    schedule()
  }
}

/** Applies every queued update now, in order. */
export function flushPaintBatch(): void {
  if (frame !== null) cancelAnimationFrame(frame)
  frame = null
  clearTimeout(holdTimer)
  holdTimer = undefined
  expired = false
  const updates = pending
  pending = []
  for (const update of updates) {
    try {
      update()
    } catch (error) {
      console.error("[paint-batch] update failed", error)
    }
  }
}
