/**
 * Brush strokes as dabs (pure): a stroke is a polyline of input points
 * (image coordinates, pen pressure); the walker places round dabs along it at
 * a spacing proportional to the dab size, carrying the remainder across
 * batches. The same input gives the same dabs however it is split into
 * batches, so a stroke painted live (pointer events coalesced per frame) and
 * replayed by redo render identically.
 */

export type StrokePoint = { x: number; y: number; pressure: number }

/** Dabs as a flat `[x, y, radius, …]` list. */
export type Dabs = number[]

export type DabSpec = {
  /** Brush diameter in image pixels at full pressure. */
  size: number
  /** Distance between dabs as a fraction of the dab diameter. */
  spacing: number
  /** Whether pen pressure scales the dab size. */
  pressure: boolean
}

/** Smallest dab radius (sub-pixel brushes still leave a trace). */
const MIN_RADIUS = 0.35

export class StrokeWalker {
  private last: StrokePoint | null = null
  /** Distance travelled since the last dab. */
  private carry = 0
  private readonly spec: DabSpec

  constructor(spec: DabSpec) {
    this.spec = spec
  }

  private radius(pressure: number): number {
    const p = this.spec.pressure ? Math.max(0.05, Math.min(1, pressure)) : 1
    return Math.max(MIN_RADIUS, (this.spec.size * p) / 2)
  }

  private step(radius: number): number {
    return Math.max(0.5, radius * 2 * this.spec.spacing)
  }

  /** Adds points; returns the new dabs. */
  add(points: readonly StrokePoint[]): Dabs {
    const dabs: Dabs = []
    for (const point of points) {
      const last = this.last
      if (!last) {
        dabs.push(point.x, point.y, this.radius(point.pressure))
        this.last = point
        this.carry = 0
        continue
      }
      const dx = point.x - last.x
      const dy = point.y - last.y
      const length = Math.hypot(dx, dy)
      if (length === 0) continue
      let travelled = 0
      // Walk the segment, one dab per `step` of the size at that point.
      for (;;) {
        const t = travelled / length
        const pressure = last.pressure + (point.pressure - last.pressure) * t
        const need = this.step(this.radius(pressure)) - this.carry
        if (travelled + need > length) {
          this.carry += length - travelled
          break
        }
        travelled += need
        this.carry = 0
        const u = travelled / length
        const p = last.pressure + (point.pressure - last.pressure) * u
        dabs.push(last.x + dx * u, last.y + dy * u, this.radius(p))
      }
      this.last = point
    }
    return dabs
  }
}

/** All dabs of a whole stroke (what redo replays). */
export function strokeDabs(points: readonly StrokePoint[], spec: DabSpec): Dabs {
  return new StrokeWalker(spec).add(points)
}

/** Bounds `[x0, y0, x1, y1]` of dabs, grown by `pad` px for antialiasing. */
export function dabBounds(dabs: ArrayLike<number>, pad = 1): [number, number, number, number] | null {
  if (dabs.length < 3) return null
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let i = 0; i + 2 < dabs.length; i += 3) {
    const x = dabs[i]!
    const y = dabs[i + 1]!
    const r = dabs[i + 2]! + pad
    if (x - r < x0) x0 = x - r
    if (y - r < y0) y0 = y - r
    if (x + r > x1) x1 = x + r
    if (y + r > y1) y1 = y + r
  }
  return [x0, y0, x1, y1]
}
