/**
 * What a stage-scheduler panel applies over the sampling timeline, mirrored
 * from lib_stage_scheduler.runtime (`PassConfig.finalize_timing`,
 * `_stage_curve_factor`): with auto timing the template's parts (or its
 * stage) on their Shift windows, else the panel's own window, which blocks
 * the LoRA when the template only blocks stages.
 */
import { clamp, stageWindow, type StageKind, type Window, windowCurveFactor } from "../_anima/schedule"

export type Timing = { auto: boolean; start: number; end: number; peak: number; strength: number; curve: string }
export type Part = { stage: string; mode: "allow" | "ban" | string; curve: string; strength: number }
export type Template = { label: string; builtin: boolean; auto: boolean; stage: string; curve: string | null; parts: Part[] }
export type Choice = { value: string; label: string }
export type Curve = Choice & { shape?: string }
export type Stage = Choice & { kind?: StageKind }

export const DEFAULT_TIMING: Timing = { auto: true, start: 0.62, end: 1, peak: 0.86, strength: 1, curve: "" }

/** A timing value, whatever the form holds (missing keys: the defaults). */
export function timingOf(value: unknown, fallbackCurve: string): Timing {
  const raw = (value && typeof value === "object" ? value : {}) as Partial<Record<keyof Timing, unknown>>
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d)
  return {
    auto: typeof raw.auto === "boolean" ? raw.auto : DEFAULT_TIMING.auto,
    start: num(raw.start, DEFAULT_TIMING.start),
    end: num(raw.end, DEFAULT_TIMING.end),
    peak: num(raw.peak, DEFAULT_TIMING.peak),
    strength: num(raw.strength, DEFAULT_TIMING.strength),
    curve: typeof raw.curve === "string" && raw.curve ? raw.curve : fallbackCurve,
  }
}

export type Resolved = {
  /** The window the panel shows (with auto timing: the template stage's at the Shift). */
  window: Window
  /** Template parts that run (auto timing only), each on its stage's window. */
  parts: (Part & { kind: StageKind; window: Window })[]
  /** Manual timing of a template that only blocks stages: the window blocks the LoRA. */
  banWindow: boolean
  strength: number
  shape: string
}

export function kindOf(stage: string, stages: readonly Stage[]): StageKind {
  return stages.find((item) => item.value === stage)?.kind ?? "character"
}

export function shapeOf(curve: string, curves: readonly Curve[]): string {
  return curves.find((item) => item.value === curve)?.shape ?? "smooth"
}

/** What the run does with ``timing`` on ``template`` at ``shift``. */
export function resolve(timing: Timing, template: Template | undefined, shift: number, stages: readonly Stage[], curves: readonly Curve[]): Resolved {
  const parts = template?.parts ?? []
  const strength = clamp(timing.strength, 0, 4)
  const shape = shapeOf(timing.curve, curves)
  if (timing.auto) {
    const kind = kindOf(template?.stage ?? "", stages)
    return {
      window: stageWindow(kind, shift),
      parts: parts.map((part) => ({ ...part, kind: kindOf(part.stage, stages), window: stageWindow(kindOf(part.stage, stages), shift) })),
      banWindow: false,
      strength,
      shape,
    }
  }
  let start = clamp(timing.start, 0, 1)
  let end = clamp(timing.end, 0, 1)
  if (end < start) [start, end] = [end, start]
  return {
    window: { start, end, peak: clamp(timing.peak, start, end) },
    parts: [],
    banWindow: parts.length > 0 && parts.every((part) => part.mode === "ban"),
    strength,
    shape,
  }
}

/** The LoRA strength multiplier at sampling ``progress`` (``_stage_curve_factor``). */
export function factorAt(resolved: Resolved, progress: number, curves: readonly Curve[]): number {
  const { strength } = resolved
  if (resolved.parts.length) {
    const allow: number[] = []
    let banned = false
    for (const part of resolved.parts) {
      const { start, end, peak } = part.window
      if (part.mode === "ban") {
        if (Math.min(start, end) <= progress && progress <= Math.max(start, end)) banned = true
        continue
      }
      allow.push(clamp(windowCurveFactor(progress, start, end, peak, shapeOf(part.curve, curves)) * clamp(part.strength, 0, 4), 0, 4))
    }
    const factor = allow.length ? Math.max(...allow) * strength : strength
    return banned ? 0 : clamp(factor, 0, 4)
  }
  const { start, end, peak } = resolved.window
  if (resolved.banWindow) return start <= progress && progress <= end ? 0 : strength
  return clamp(windowCurveFactor(progress, start, end, peak, resolved.shape) * strength, 0, 4)
}
