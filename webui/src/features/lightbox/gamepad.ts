/**
 * Lightbox navigation besides the keys: a gamepad (`js_modal_lightbox_gamepad`:
 * left stick or d-pad left / right) and the horizontal wheel (a tilt wheel or a
 * sideways two-finger swipe), as A1111's imageviewerGamepad.js did.
 */
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

/** A sideways wheel this strong (pixels, or lines / pages) turns the page. */
const WHEEL_THRESHOLD = 0.6

/**
 * -1 (previous), 1 (next) or 0 for a wheel event: only a mostly horizontal
 * one counts (a vertical wheel zooms, Ctrl + wheel is a trackpad pinch).
 */
export function wheelDirection(event: { deltaX: number; deltaY: number; ctrlKey?: boolean }): -1 | 0 | 1 {
  if (event.ctrlKey || Math.abs(event.deltaX) <= Math.abs(event.deltaY)) return 0
  if (event.deltaX <= -WHEEL_THRESHOLD) return -1
  if (event.deltaX >= WHEEL_THRESHOLD) return 1
  return 0
}

/**
 * A wheel handler step: turns the page at most once per `repeatMs` (a swipe
 * sends dozens of events). Returns whether the event was a sideways one (then
 * it must not zoom).
 */
export function useWheelNavigation(repeatMs: number, onStep: (delta: -1 | 1) => void) {
  const last = useRef(Number.NEGATIVE_INFINITY)
  return (event: { deltaX: number; deltaY: number; ctrlKey?: boolean; timeStamp: number }): boolean => {
    const direction = wheelDirection(event)
    if (direction === 0) return false
    if (event.timeStamp - last.current >= repeatMs) {
      last.current = event.timeStamp
      onStep(direction)
    }
    return true
  }
}
