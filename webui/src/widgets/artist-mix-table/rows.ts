/**
 * Rows of the Artist Scheduled Mixer's tables, read the way the run reads
 * them (`_normalize_rows` of scripts/anima_artist_scheduled_mixer.py):
 * Yes/No and label values are accepted, numbers clamped, and a row on a
 * stage with Auto Shift takes that stage's window at the current Shift.
 */
import { clamp, stageWindow, type StageKind, type Window } from "../_anima/schedule"

export type Row = {
  enabled: boolean
  artist: string
  weight: number
  blocks: string
  start: number
  end: number
  peak: number
  curve: string
  stage: string
  auto_shift: boolean
}

export const CUSTOM = "Custom"
export const HOLD = "Hold"

const TRUE = new Set(["1", "true", "yes", "y", "on", "启用", "是", "开"])
const FALSE = new Set(["0", "false", "no", "n", "off", "禁用", "否", "关"])

export function toBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value
  if (value === null || value === undefined) return fallback
  const text = String(value).trim().toLowerCase()
  return TRUE.has(text) ? true : FALSE.has(text) ? false : fallback
}

function toNumber(value: unknown, fallback: number): number {
  const number = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN
  return Number.isFinite(number) ? number : fallback
}

/** One choice key from its key or label (case-insensitive); else `fallback`. */
function choiceKey(value: unknown, choices: readonly { value: string; label: string }[], fallback: string): string {
  const text = String(value ?? "").trim().toLowerCase()
  return choices.find((choice) => choice.value.toLowerCase() === text || choice.label.toLowerCase() === text)?.value ?? fallback
}

export type Choices = { curves: readonly { value: string; label: string }[]; stages: readonly { value: string; label: string; kind?: StageKind }[] }

/** A row as the run reads it (without Auto Shift's window: see `windowOf`). */
export function rowOf(raw: unknown, defaultBlocks: string, choices: Choices): Row {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>
  return {
    enabled: toBool(r.enabled, true),
    artist: String(r.artist ?? "").trim() === "" ? "" : String(r.artist),
    weight: clamp(toNumber(r.weight, 1), -4, 4),
    blocks: String(r.blocks ?? "").trim() || defaultBlocks,
    start: clamp(toNumber(r.start, 0), 0, 1),
    end: clamp(toNumber(r.end, 1), 0, 1),
    peak: clamp(toNumber(r.peak, 0.5), 0, 1),
    curve: choiceKey(r.curve, choices.curves, "Smooth"),
    stage: choiceKey(r.stage, choices.stages, CUSTOM),
    auto_shift: toBool(r.auto_shift, true),
  }
}

export function newRow(blocks: string): Row {
  return { enabled: true, artist: "", weight: 1, blocks, start: 0, end: 1, peak: 0.5, curve: HOLD, stage: CUSTOM, auto_shift: false }
}

export function stageKind(stage: string, choices: Choices): StageKind | undefined {
  return choices.stages.find((choice) => choice.value === stage)?.kind
}

/** Whether Auto Shift times the row (a stage other than Custom). */
export function isAuto(row: Row, choices: Choices): boolean {
  return row.auto_shift && row.stage !== CUSTOM && stageKind(row.stage, choices) !== undefined
}

const round4 = (value: number) => Number(value.toFixed(4))

/** The window a row samples in at `shift` (its stage's with Auto Shift). */
export function windowOf(row: Row, shift: number, choices: Choices): Window {
  if (isAuto(row, choices)) {
    const window = stageWindow(stageKind(row.stage, choices)!, shift)
    return { start: round4(window.start), end: round4(window.end), peak: round4(window.peak) }
  }
  return { start: row.start, end: row.end, peak: row.peak }
}

/** Choosing a stage: a stage other than Custom times the row from the Shift and turns Auto Shift on. */
export function withStage(row: Row, stage: string, shift: number, choices: Choices): Row {
  const next = { ...row, stage }
  if (stage === CUSTOM || stageKind(stage, choices) === undefined) return { ...next, auto_shift: false }
  return { ...next, auto_shift: true, ...windowOf({ ...next, auto_shift: true }, shift, choices) }
}

/** Toggling Auto Shift: on re-times the row, off keeps the window shown. */
export function withAuto(row: Row, auto: boolean, shift: number, choices: Choices): Row {
  return { ...row, ...windowOf(row, shift, choices), auto_shift: auto }
}

/** The artists a chain mixes (the names of the enabled rows; a row may hold a comma-separated chain). */
export function artistNames(rows: readonly Row[]): string[] {
  return rows.filter((row) => row.enabled).flatMap((row) => row.artist.replaceAll("，", ",").split(/[,\r\n]+/).map((name) => name.trim()).filter(Boolean))
}
