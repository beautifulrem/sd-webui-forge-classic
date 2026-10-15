/** Image size helpers: aspect presets at a pixel budget, and "W x H" text. */

export type Dimensions = { width: number; height: number }

export const ASPECTS: readonly (readonly [number, number])[] = [
  [1, 1],
  [4, 3],
  [3, 4],
  [3, 2],
  [2, 3],
  [16, 9],
  [9, 16],
  [21, 9],
  [9, 21],
]

type Bounds = { step?: number; minimum?: number; maximum?: number }

/**
 * The size with aspect `w:h` whose area is closest to `area`, on a `step`
 * grid (default 64) and within bounds.
 */
export function aspectSize([w, h]: readonly [number, number], area: number, { step = 64, minimum = 64, maximum = 4096 }: Bounds = {}): Dimensions {
  const snap = (value: number) => Math.min(maximum, Math.max(minimum, Math.round(value / step) * step))
  const width = Math.sqrt((area * w) / h)
  return { width: snap(width), height: snap(area / width) }
}

/** The ratio closest to `size` among `ASPECTS`, if within 3 %. */
export function matchAspect({ width, height }: Dimensions): readonly [number, number] | null {
  if (width <= 0 || height <= 0) return null
  const ratio = width / height
  let best: readonly [number, number] | null = null
  let error = Infinity
  for (const aspect of ASPECTS) {
    const e = Math.abs(Math.log(ratio / (aspect[0] / aspect[1])))
    if (e < error) {
      best = aspect
      error = e
    }
  }
  return error < 0.03 ? best : null
}

/** "1024x768", "1024 × 768", "1024*768", "1024, 768" → dimensions. */
export function parseSize(text: string): Dimensions | null {
  const match = /^\s*(\d{2,5})\s*(?:[x×X*,]|\s)\s*(\d{2,5})\s*$/.exec(text)
  if (!match) return null
  const width = Number(match[1])
  const height = Number(match[2])
  return width > 0 && height > 0 ? { width, height } : null
}
