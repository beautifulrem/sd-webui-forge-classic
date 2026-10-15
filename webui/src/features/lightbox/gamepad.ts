/** Gamepad navigation in the lightbox (`js_modal_lightbox_gamepad`): left stick or d-pad left / right. */
import { useEffect, useRef } from "react"

type PadLike = { axes: readonly number[]; buttons: readonly { pressed: boolean }[] } | null

const DEAD_ZONE = 0.5
/** Standard mapping: d-pad left / right. */
const DPAD_LEFT = 14
const DPAD_RIGHT = 15

/** -1 (previous), 1 (next) or 0 from the first pad pushed left or right. */
export function padDirection(pads: readonly PadLike[]): -1 | 0 | 1 {
  for (const pad of pads) {
    if (!pad) continue
    const x = pad.axes[0] ?? 0
    if (x < -DEAD_ZONE || pad.buttons[DPAD_LEFT]?.pressed) return -1
    if (x > DEAD_ZONE || pad.buttons[DPAD_RIGHT]?.pressed) return 1
  }
  return 0
}

/** Polls the gamepads while `enabled`; a held direction repeats every `repeatMs`. */
export function useGamepadNavigation(enabled: boolean, repeatMs: number, onStep: (delta: -1 | 1) => void) {
  const step = useRef(onStep)
  useEffect(() => {
    step.current = onStep
  })
  useEffect(() => {
    if (!enabled || typeof navigator.getGamepads !== "function") return
    let frame = 0
    let last: number | null = null
    const poll = (now: number) => {
      const direction = padDirection(navigator.getGamepads())
      if (direction === 0) last = null
      else if (last === null || now - last >= repeatMs) {
        last = now
        step.current(direction)
      }
      frame = requestAnimationFrame(poll)
    }
    frame = requestAnimationFrame(poll)
    return () => cancelAnimationFrame(frame)
  }, [enabled, repeatMs])
}
