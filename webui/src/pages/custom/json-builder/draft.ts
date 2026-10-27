/**
 * The JSON Builder's editing model without UI (upstream `main.js` state and
 * helpers): the canvas size, the summary, the items with their boxes in
 * pixels, and what turns them into a draft, a prompt or back.
 */
import { compilePrompt } from "./compile"
import type { Box, CanvasType, Label, LayoutDraft } from "./decompile"

export type Item = { id: number; label: Label; caption: string; text: string; bbox: Box }

export type BuilderDraft = {
  width: number
  height: number
  /** The summary caption. */
  caption: string
  /** Meta tags, comma separated as typed. */
  meta: string
  /** Text appended after the compiled prompt (a LoRA tag, a trigger word). */
  extra: string
  canvas: CanvasType
  items: Item[]
}

/** Display names of the item labels (the prompt keeps the lower-case values). */
export const LABEL_NAMES = { object: "Object", character: "Character", text: "Text" } as const

export const ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"] as const
export type AspectRatio = (typeof ASPECT_RATIOS)[number]

/** Pixels per megapixel in the resolution tool (2^20, as upstream). */
export const MP_PIXELS = 1024 * 1024
/** The smallest box side a resize leaves, in pixels. */
export const MIN_BOX = 16

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
export const round16 = (v: number) => Math.max(16, Math.round(v / 16) * 16)

/** A new item's box: 30 % of the shorter side, 10 % in from the top-left corner. */
export function defaultBbox(width: number, height: number): Box {
  const m = Math.round(Math.min(width, height) * 0.1)
  const s = Math.round(Math.min(width, height) * 0.3)
  return [m, m, m + s, m + s]
}

/** A new item's label (a prompt value, not a display text). */
const NEW_LABEL: Label = "object"

export function newItem(id: number, width: number, height: number): Item {
  return { id, label: NEW_LABEL, caption: "", text: "", bbox: defaultBbox(width, height) }
}

/** Width and height (multiples of 16) for `mp` megapixels at aspect ratio `ar` ("16:9"). */
export function resolutionFor(mp: number, ar: string): { width: number; height: number } | null {
  const [aw, ah] = ar.split(":").map(Number)
  if (!(mp > 0) || !aw || !ah) return null
  const area = mp * MP_PIXELS
  const w = Math.sqrt((area * aw) / ah)
  return { width: round16(w), height: round16(area / w) }
}

/** The box shrunk to fit the canvas if larger, then moved inside it, keeping its size. */
export function clampBox(bbox: Box, width: number, height: number): Box {
  const bw = Math.min(bbox[2] - bbox[0], width)
  const bh = Math.min(bbox[3] - bbox[1], height)
  const x1 = clamp(bbox[0], 0, width - bw)
  const y1 = clamp(bbox[1], 0, height - bh)
  return [x1, y1, x1 + bw, y1 + bh]
}

/** `bbox` moved by (dx, dy) pixels, kept inside the canvas. */
export function moveBox(bbox: Box, dx: number, dy: number, width: number, height: number): Box {
  const w = bbox[2] - bbox[0]
  const h = bbox[3] - bbox[1]
  const x1 = clamp(Math.round(bbox[0] + dx), 0, width - w)
  const y1 = clamp(Math.round(bbox[1] + dy), 0, height - h)
  return [x1, y1, x1 + w, y1 + h]
}

/** `bbox` with its bottom-right corner moved by (dx, dy) pixels: at least {@link MIN_BOX}, inside the canvas. */
export function resizeBox(bbox: Box, dx: number, dy: number, width: number, height: number): Box {
  const x2 = clamp(Math.round(bbox[2] + dx), bbox[0] + MIN_BOX, width)
  const y2 = clamp(Math.round(bbox[3] + dy), bbox[1] + MIN_BOX, height)
  return [bbox[0], bbox[1], x2, y2]
}

export function metaTags(meta: string): string[] {
  return meta
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
}

/** The Layout draft the compiler takes (and "Save JSON" writes), keys in upstream's order. */
export function buildDraft(state: BuilderDraft) {
  const meta = metaTags(state.meta)
  const { canvas, caption } = state
  const summary: { meta?: string[]; canvas: CanvasType; caption: string } = meta.length ? { meta, canvas, caption } : { canvas, caption }
  return {
    width: state.width,
    height: state.height,
    bbox_coord_space: "abs" as const,
    bbox_format: "xyxy" as const,
    summary,
    items: state.items.map((item) => {
      const out: { label: Label; caption: string; bbox: number[]; text?: string } = { label: item.label, caption: item.caption, bbox: item.bbox.map((n) => Math.round(n)) }
      if (item.label === "text") out.text = item.text
      return out
    }),
  }
}

/** A field left empty that the prompt needs. */
export type Missing = "caption" | `item:${number}:caption` | `item:${number}:text`

/** The empty fields the prompt needs (upstream marks them red instead of compiling). */
export function missingFields(state: BuilderDraft): Set<Missing> {
  const missing = new Set<Missing>()
  if (state.caption.trim() === "") missing.add("caption")
  for (const item of state.items) {
    if (item.caption.trim() === "") missing.add(`item:${item.id}:caption`)
    if (item.label === "text" && item.text.trim() === "") missing.add(`item:${item.id}:text`)
  }
  return missing
}

/**
 * The prompt to send: the compiled draft, then the extra text, on one line
 * (newlines become spaces, as upstream writes it into the prompt box).
 * Throws a `DraftError` when the draft is not valid.
 */
export function finalPrompt(state: BuilderDraft): string {
  let prompt = compilePrompt(JSON.stringify(buildDraft(state)))
  if (state.extra.trim() !== "") prompt += `\n${state.extra}`
  return prompt.replaceAll("\n", " ")
}

/**
 * A compiled prompt's escaped parentheses as typed (`\(` → `(`): the
 * compiler escapes every parenthesis again, so the prompt does not change.
 */
export const unescapeParens = (text: string) => text.replace(/\\+([()])/g, "$1")

/**
 * The editor's state for a loaded draft; item ids start after `lastId`. The
 * extra text is cleared, parentheses are shown unescaped.
 */
export function fromDraft(draft: LayoutDraft, lastId: number): BuilderDraft & { lastId: number } {
  let id = lastId
  const items = draft.items.map((item) => ({
    id: ++id,
    label: item.label,
    caption: unescapeParens(item.caption),
    text: unescapeParens(item.text ?? ""),
    bbox: item.bbox.slice() as Box,
  }))
  return {
    width: draft.width,
    height: draft.height,
    caption: unescapeParens(draft.summary.caption),
    meta: draft.summary.meta.map(unescapeParens).join(", "),
    extra: "",
    canvas: draft.summary.canvas,
    items,
    lastId: id,
  }
}
