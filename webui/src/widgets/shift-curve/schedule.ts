/**
 * Dynamic-shift sigma schedule, a port of
 * extensions-builtin/sd-forge-dynamic-shift/lib_dynshift/schedule.py
 * (`compute_sigmas`): the flow-matching time-SNR shift
 * `sigma(t, s) = s·t / (1 + (s − 1)·t)` with `s` interpolated per step from
 * the first-step shift to the last-step shift. Display only: the server
 * computes the schedule that runs.
 */
export const CURVES = ["Linear", "Cosine", "Exponential"] as const

const T_MIN = 1 / 1000

export function timeSnrShift(shift: number, t: number): number {
  return (shift * t) / (1 + (shift - 1) * t)
}

/** The shift at progress `u` ∈ [0, 1] (0 = first step). */
export function interpolateShift(s0: number, s1: number, u: number, curve: string): number {
  if (curve === "Exponential") return s0 * (s1 / s0) ** u
  const w = curve === "Cosine" ? (1 - Math.cos(Math.PI * u)) / 2 : u
  return s0 + (s1 - s0) * w
}

/** `n` + 1 strictly decreasing sigmas ending in 0 (as the Python helper). */
export function computeSigmas(n: number, shiftStart: number, shiftEnd: number, curve = "Linear"): number[] {
  const steps = Math.max(Math.trunc(n), 1)
  const s0 = Math.max(shiftStart, 0.01)
  const s1 = Math.max(shiftEnd, 0.01)
  const sigmas: number[] = []
  for (let i = 0; i < steps; i++) {
    const u = i / Math.max(steps - 1, 1)
    const t = 1 + (T_MIN - 1) * u
    sigmas.push(timeSnrShift(interpolateShift(s0, s1, u, curve), t))
  }
  // A rising ramp can locally raise sigma: keep the schedule strictly decreasing.
  for (let i = 1; i < steps; i++) {
    if (sigmas[i]! >= sigmas[i - 1]!) sigmas[i] = sigmas[i - 1]! * 0.9999
  }
  sigmas.push(0)
  return sigmas
}

/** SVG path through `values` (0…1, top = 1) in a `width` × `height` box with `pad`. */
export function sigmaPath(values: readonly number[], width: number, height: number, pad: number): string {
  return values
    .map((value, i) => {
      const x = pad + (width - 2 * pad) * (i / Math.max(values.length - 1, 1))
      const y = pad + (height - 2 * pad) * (1 - value)
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}
