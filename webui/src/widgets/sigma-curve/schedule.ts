/**
 * Detail Daemon's schedule (`Script.make_schedule` in
 * extensions-builtin/sd-webui-detail-daemon/scripts/detail_daemon.py),
 * computed the same way: numpy's linspace, Python's round-half-to-even, the
 * same operation order. `schedule.fixture.json` holds the extension's own
 * output for a set of inputs; tests on both sides keep them in step.
 */

export type ScheduleParams = {
  start: number
  end: number
  bias: number
  amount: number
  exponent: number
  start_offset: number
  end_offset: number
  fade: number
  smooth: boolean
}

export const SCHEDULE_DEFAULTS: ScheduleParams = {
  start: 0.2,
  end: 0.8,
  bias: 0.5,
  amount: 0.1,
  exponent: 1,
  start_offset: 0,
  end_offset: 0,
  fade: 0,
  smooth: true,
}

/** Python's `round()` of a float: halves go to the even neighbour. */
export function pyRound(x: number): number {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x)
}

/** `numpy.linspace(start, stop, num)` (endpoint included). */
export function linspace(start: number, stop: number, num: number): number[] {
  if (num <= 0) return []
  if (num === 1) return [start]
  const step = (stop - start) / (num - 1)
  const values = Array.from({ length: num }, (_, i) => i * step + start)
  values[num - 1] = stop
  return values
}

function shape(values: number[], smooth: boolean, exponent: number, amount: number, offset: number): number[] {
  let out = smooth ? values.map((v) => 0.5 * (1 - Math.cos(v * Math.PI))) : values
  out = out.map((v) => v ** exponent)
  if (out.some((v) => v !== 0)) out = out.map((v) => v * (amount - offset) + offset)
  return out
}

/** The multiplier of each of `steps` sampling steps. */
export function makeSchedule(steps: number, p: ScheduleParams): number[] {
  const start = Math.min(p.start, p.end)
  const mid = start + p.bias * (p.end - start)
  const multipliers = new Array<number>(steps).fill(0)
  const clamp = (i: number) => Math.min(Math.max(i, 0), steps)
  const [startIdx, midIdx, endIdx] = [start, mid, p.end].map((x) => pyRound(x * (steps - 1))) as [number, number, number]

  const startValues = shape(linspace(0, 1, midIdx - startIdx + 1), p.smooth, p.exponent, p.amount, p.start_offset)
  const endValues = shape(linspace(1, 0, endIdx - midIdx + 1), p.smooth, p.exponent, p.amount, p.end_offset)

  // multipliers[start:mid+1] = start_values; multipliers[mid:end+1] = end_values
  for (let i = clamp(startIdx); i < clamp(midIdx + 1); i++) multipliers[i] = startValues[i - startIdx]!
  for (let i = clamp(midIdx); i < clamp(endIdx + 1); i++) multipliers[i] = endValues[i - midIdx]!
  for (let i = 0; i < clamp(startIdx); i++) multipliers[i] = p.start_offset
  for (let i = clamp(endIdx + 1); i < steps; i++) multipliers[i] = p.end_offset
  return multipliers.map((v) => v * (1 - p.fade))
}

/** What the old matplotlib preview showed besides the curve: mean sign, peak and the bias line. */
export function summarize(values: readonly number[], p: ScheduleParams): { mean: number; peak: number; mid: number } {
  const mean = values.reduce((sum, v) => sum + v, 0) / Math.max(values.length, 1)
  const peak = Math.min(Math.max(Math.max(0, ...values.map(Math.abs)), -1), 1)
  const start = Math.min(p.start, p.end)
  return { mean, peak, mid: start + p.bias * (p.end - start) }
}
