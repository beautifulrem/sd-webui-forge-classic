/**
 * Sampling-window math of the Anima LoRA panels, mirrored from
 * modules/anima_lora_support.py (`stage_window`, `window_curve_factor`) so the
 * stage-timeline and artist-mix-table widgets draw what a run applies.
 * Not a widget (no `index.tsx`); `schedule.test.ts` checks it against Python.
 */

export type StageKind = "composition" | "character" | "style"
export type Window = { start: number; end: number; peak: number }
export type CurveShape = "hold" | "front" | "back" | "triangle" | "smooth"

export const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), high)

/** The sampling-progress window of a template stage at sampler `shift`. */
export function stageWindow(kind: StageKind, shift: number): Window {
  const move = clamp((clamp(Number.isFinite(shift) ? shift : 3, 1, 24) - 3) / 21, -0.1, 1) * 0.12
  if (kind === "composition") {
    const end = clamp(0.32 + move, 0.22, 0.46)
    return { start: 0, end, peak: clamp(0.08 + move * 0.25, 0, end) }
  }
  if (kind === "style") {
    const start = clamp(0.62 + move, 0.48, 0.82)
    return { start, end: 1, peak: clamp(0.86 + move * 0.35, start, 1) }
  }
  const start = clamp(0.2 + move * 0.65, 0.12, 0.36)
  const end = clamp(0.74 + move * 0.45, 0.58, 0.88)
  return { start, end, peak: clamp((start + end) * 0.5, start, end) }
}

/**
 * Strength factor (0..1) at sampling `progress` of the window [start, end]
 * (either order); 0 outside. "front" falls to `floor`, "back" rises from it,
 * "triangle" peaks linearly at `peak`, anything else is the eased triangle.
 */
export function windowCurveFactor(progress: number, start: number, end: number, peak: number, shape: string, floor = 0): number {
  let lo = clamp(start, 0, 1)
  let hi = clamp(end, 0, 1)
  if (lo > hi) [lo, hi] = [hi, lo]
  if (progress < lo || progress > hi) return 0
  if (hi - lo <= 1e-6) return 1
  const local = (progress - lo) / (hi - lo)
  if (shape === "hold") return 1
  if (shape === "front") return 1 - (1 - floor) * local
  if (shape === "back") return floor + (1 - floor) * local
  const peakLocal = (clamp(peak, lo, hi) - lo) / (hi - lo)
  const raw = clamp(local <= peakLocal ? local / Math.max(peakLocal, 1e-6) : (1 - local) / Math.max(1 - peakLocal, 1e-6), 0, 1)
  return shape === "triangle" ? raw : raw * raw * (3 - 2 * raw)
}

/** A flow model's sigma at sampling `progress` (0 = first step) under time shift `shift`. */
export function flowSigma(progress: number, shift: number): number {
  const t = clamp(1 - progress, 0, 1)
  const s = Number.isFinite(shift) && shift > 0 ? shift : 3
  return (s * t) / (1 + (s - 1) * t)
}

/** An SVG path of `factor(x)` over x in [0, 1], `samples` points, in a `width` × `height` box (y up). */
export function curvePath(factor: (x: number) => number, width: number, height: number, samples = 96): string {
  const points: string[] = []
  for (let i = 0; i <= samples; i++) {
    const x = i / samples
    const y = clamp(factor(x), 0, 1)
    points.push(`${(x * width).toFixed(2)},${(height - y * height).toFixed(2)}`)
  }
  return `M${points.join("L")}`
}

/** Tidy number text: at most `digits` decimals, no trailing zeros. */
export function fmt(value: number, digits = 2): string {
  return String(Number(value.toFixed(digits)))
}
