/** Pure helpers of the `sigma-curve` widget: a daemon's params from the form values, the curve path. */
import { SCHEDULE_DEFAULTS, type ScheduleParams } from "./schedule"

export type DaemonState = ScheduleParams & { active: boolean; hires: boolean; mode: string }

const NUMBER_KEYS = ["start", "end", "bias", "amount", "exponent", "start_offset", "end_offset", "fade"] as const

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

/** A daemon's params from the owner's values (defaults for anything missing). */
export function daemonState(values: Record<string, unknown> | undefined, prefix: string): DaemonState {
  const read = (name: string) => values?.[`${prefix}${name}`]
  const state = { ...SCHEDULE_DEFAULTS, active: read("active") === true, hires: read("hires") === true, mode: String(read("mode") ?? "both") }
  for (const key of NUMBER_KEYS) state[key] = num(read(key), SCHEDULE_DEFAULTS[key])
  state.smooth = read("smooth") === undefined ? SCHEDULE_DEFAULTS.smooth : read("smooth") === true
  return state
}

/** SVG path of the schedule over a `width` × `height` box, y from -1 (bottom) to 1 (top). */
export function curvePath(values: readonly number[], width: number, height: number): string {
  const x = (i: number) => (values.length > 1 ? (i / (values.length - 1)) * width : 0)
  const y = (v: number) => ((1 - Math.min(Math.max(v, -1), 1)) / 2) * height
  return values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ")
}
