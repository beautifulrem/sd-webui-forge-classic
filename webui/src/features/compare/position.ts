/** Divider positions of the comparison slider (0 = all "after", 1 = all "before"). */

/** 0..1, rounded to 0.1 % so keyboard steps land on round numbers. */
export function clampPosition(value: number): number {
  if (!Number.isFinite(value)) return 0.5
  return Math.round(Math.min(1, Math.max(0, value)) * 1000) / 1000
}

/** The position after a key press, or null when the key does nothing. */
export function keyPosition(position: number, key: string, big: boolean): number | null {
  const step = big ? 0.1 : 0.01
  switch (key) {
    case "ArrowLeft":
    case "ArrowDown":
      return clampPosition(position - step)
    case "ArrowRight":
    case "ArrowUp":
      return clampPosition(position + step)
    case "PageDown":
      return clampPosition(position - 0.1)
    case "PageUp":
      return clampPosition(position + 0.1)
    case "Home":
      return 0
    case "End":
      return 1
    default:
      return null
  }
}
