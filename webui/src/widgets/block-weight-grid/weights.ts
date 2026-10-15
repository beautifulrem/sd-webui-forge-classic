/**
 * The Anima LoRA weight texts the block-weight-grid widget edits, read as
 * modules/anima_lora_support.py reads them (`parse_layer_weights`,
 * `parse_module_weights`, `fold_weight_text`) and written back in a
 * canonical form ("all=1.0, 0-9=0.6, 19-27=0.75"; one "module=weight" per line).
 */

/** Full-width forms, the ideographic space and colon/stop, dash variants: as `fold_weight_text`. */
export function foldWeightText(text: string): string {
  let out = ""
  for (const ch of text) {
    const code = ch.codePointAt(0)!
    if (ch === "：") out += "="
    else if (code >= 0xff01 && code <= 0xff5e) out += String.fromCodePoint(code - 0xfee0)
    else if (code === 0x3000) out += " "
    else if (ch === "。") out += "."
    else if ([0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xfe63].includes(code)) out += "-"
    else out += ch
  }
  return out
}

const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/

/** A plain decimal number (as `plain_number`), else null. */
export function plainNumber(token: string): number | null {
  const text = token.trim()
  if (!NUMBER.test(text)) return null
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}

/** Python's `str(float)` for the weights the grid writes: 1 → "1.0", 0.75 → "0.75". */
export function formatWeight(value: number): string {
  const rounded = Number(value.toFixed(4))
  return Number.isInteger(rounded) ? rounded.toFixed(1) : String(rounded)
}

type Entry = [selector: string | null, value: string]

function entries(text: string, spacedNames = false): Entry[] {
  const source = foldWeightText(text).trim().replace(/\bblock_lrs?\s*=/gi, "")
  if (!source.includes("=")) {
    return source
      .split(/[\s,;]+/)
      .filter(Boolean)
      .map((token) => [null, token])
  }
  const out: Entry[] = []
  for (const raw of source.split(/[,;\n]+/)) {
    let part = raw
      .split("=")
      .map((piece) => piece.trim())
      .join("=")
    part = part.replace(/(?<![\w.])(\d+)\s*-\s*(\d+)(?==)/g, "$1-$2")
    for (const token of spacedNames ? [part] : part.split(/\s+/)) {
      if (!token) continue
      const eq = token.indexOf("=")
      out.push(eq >= 0 ? [token.slice(0, eq).trim(), token.slice(eq + 1).trim()] : [null, token])
    }
  }
  return out
}

/** Past the deepest Anima DiT (52 blocks) and the Qwen3 text encoder (28 layers). */
export const MAX_LAYER_INDEX = 255

export type LayerWeights = {
  /** The `all=` weight of every index not listed. */
  fallback: number
  values: Map<number, number>
  /** Tokens a run ignores (with a warning); editing the grid drops them. */
  ignored: string[]
}

const DEFAULT_SELECTORS = new Set(["*", "all", "default"])

function index(text: string): number | null {
  return /^\d+$/.test(text) ? Number(text) : null
}

/** Block (or layer) weights of a weight text, as `parse_layer_weights`. */
export function parseLayerWeights(text: string): LayerWeights {
  const values = new Map<number, number>()
  const ignored: string[] = []
  const bare: [string, number][] = []
  let fallback = 1
  for (const [selector, token] of entries(text)) {
    if (selector === null) {
      const value = plainNumber(token)
      if (value === null) ignored.push(token)
      else bare.push([token, value])
      continue
    }
    const left = selector.toLowerCase()
    const value = plainNumber(token)
    if (value === null) {
      ignored.push(`${selector}=${token}`)
      continue
    }
    const range = /^(\d+)\s*-\s*(\d+)$/.exec(left)
    if (DEFAULT_SELECTORS.has(left)) fallback = value
    else if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])].sort((x, y) => x - y) as [number, number]
      if (a > MAX_LAYER_INDEX) ignored.push(`${selector}=${token}`)
      else for (let i = a; i <= Math.min(b, MAX_LAYER_INDEX); i++) values.set(i, value)
    } else {
      const at = index(left)
      if (at !== null && at <= MAX_LAYER_INDEX) values.set(at, value)
      else ignored.push(`${selector}=${token}`)
    }
  }
  if (bare.length && values.size) ignored.push(...bare.map(([token]) => token))
  else bare.forEach(([, value], i) => values.set(i, value))
  return { fallback, values, ignored }
}

/**
 * The text of per-index `weights` with `fallback` for the rest: "all=F" and
 * one entry per run of equal weights that differ from it. `emptyWhenFlat`:
 * an all-1.0 grid is "" (e.g. the Qwen layers: empty = unchanged).
 */
export function serializeLayerWeights(fallback: number, weights: readonly number[], emptyWhenFlat = false): string {
  const runs: string[] = []
  let i = 0
  while (i < weights.length) {
    const value = weights[i]!
    let j = i
    while (j + 1 < weights.length && weights[j + 1] === value) j++
    if (value !== fallback) runs.push(`${i === j ? i : `${i}-${j}`}=${formatWeight(value)}`)
    i = j + 1
  }
  if (emptyWhenFlat && !runs.length && fallback === 1) return ""
  return [`all=${formatWeight(fallback)}`, ...runs].join(", ")
}

/** The grid of a parsed text over `count` indexes. */
export function gridOf(weights: LayerWeights, count: number): number[] {
  return Array.from({ length: count }, (_, i) => weights.values.get(i) ?? weights.fallback)
}

const MODULE_ALIASES: Record<string, string> = {
  self: "self_attn",
  selfattn: "self_attn",
  self_attention: "self_attn",
  cross: "cross_attn",
  crossattn: "cross_attn",
  cross_attention: "cross_attn",
  ada: "adaln",
  adaln_modulation: "adaln",
  modulation: "adaln",
  norms: "norm",
  layernorm: "norm",
  feedforward: "mlp",
  ffn: "mlp",
}

/** A module name as `module_key` normalises it. */
export function moduleKey(name: string): string {
  const key = name.trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_")
  return MODULE_ALIASES[key] ?? key
}

export type ModuleWeights = {
  /** Each grid module's weight (1.0 when unset). */
  values: Record<string, number>
  /** Grid modules the text writes (kept on their own line even at 1.0). */
  written: string[]
  /** Valid entries of modules outside the grid: a run applies them; kept as written. */
  kept: string[]
  /** Entries a run ignores (no valid weight); editing the grid drops them. */
  ignored: string[]
}

/** Module weights of a text over `modules`, as `parse_module_weights`. */
export function parseModuleWeights(text: string, modules: readonly string[]): ModuleWeights {
  const values = Object.fromEntries(modules.map((module) => [module, 1])) as Record<string, number>
  const written = new Set<string>()
  const kept: string[] = []
  const ignored: string[] = []
  for (const [name, token] of entries(text, true)) {
    if (name === null) {
      if (token) ignored.push(token)
      continue
    }
    const value = plainNumber(token)
    const key = moduleKey(name)
    if (value === null || !key) ignored.push(`${name}=${token}`)
    else if (!modules.includes(key)) kept.push(`${name}=${token}`)
    else {
      values[key] = value
      written.add(key)
    }
  }
  return { values, written: [...written], kept, ignored }
}

/**
 * One "module=weight" line per grid module the text wrote or that is not
 * 1.0 (an unset module is 1.0), then the kept entries as written.
 */
export function serializeModuleWeights(weights: Pick<ModuleWeights, "values" | "written" | "kept">, modules: readonly string[]): string {
  const lines = modules.filter((module) => weights.written.includes(module) || (weights.values[module] ?? 1) !== 1).map((module) => `${module}=${formatWeight(weights.values[module] ?? 1)}`)
  return [...lines, ...weights.kept].join("\n")
}

/**
 * The text of a grid edit that shows `layout` indexes: `weights` (and
 * `fallback`) for those, and the values the text has past the layout kept
 * (a 28-block view of a 40-block text must not drop blocks 28-39).
 */
export function serializeLayout(parsed: LayerWeights, weights: readonly number[], fallback: number, emptyWhenFlat = false): string {
  const deepest = Math.max(-1, ...parsed.values.keys())
  const tail = Array.from({ length: Math.max(0, deepest + 1 - weights.length) }, (_, i) => parsed.values.get(weights.length + i) ?? fallback)
  return serializeLayerWeights(fallback, [...weights, ...tail], emptyWhenFlat)
}
