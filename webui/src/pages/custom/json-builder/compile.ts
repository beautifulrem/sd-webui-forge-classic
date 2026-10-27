/**
 * Compiles a Layout draft into the MOSS-Image prompt (upstream
 * `sd_forge_json_builder/javascript/compile.js`, which follows OpenMOSS's
 * `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`).
 *
 * A draft is either `{ caption }` (a plain caption) or a Layout:
 * `{ width, height, bbox_coord_space: "abs", bbox_format: "xyxy", summary, items }`
 * with boxes in pixels from the top-left corner. The prompt is the JSON of
 * `{ summary, items, bbox_coord_space: "abs_centralized", bbox_format: "yxyx" }`:
 * boxes as `[y1, x1, y2, x2]` relative to the canvas centre, outward-rounded,
 * and parentheses escaped so the prompt parser does not read them as emphasis.
 * Every rule of the upstream compiler is kept, with its error messages: the
 * output is byte for byte upstream's.
 */

export class DraftError extends Error {
  override name = "DraftError"
}

export function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new DraftError(message)
}

type Json = Record<string, unknown>

export const isObject = (value: unknown): value is Json => value !== null && typeof value === "object" && !Array.isArray(value)

const has = (value: object, key: string) => Object.prototype.hasOwnProperty.call(value, key)

function fields(value: unknown, allowed: ReadonlySet<string>, required: ReadonlySet<string>, location: string): asserts value is Json {
  check(isObject(value), `${location} must be an object`)
  const keys = Object.keys(value)
  const unknown = keys.filter((key) => !allowed.has(key))
  const missing = [...required].filter((key) => !has(value, key))
  check(unknown.length === 0, `${location}: unknown fields ${JSON.stringify(unknown.sort())}`)
  check(missing.length === 0, `${location}: missing fields ${JSON.stringify(missing.sort())}`)
}

function text(value: unknown, location: string): asserts value is string {
  check(typeof value === "string" && value.trim().length > 0, `${location} must be nonempty text`)
}

function strings(value: unknown, location: string): void {
  check(Array.isArray(value), `${location} must be an array`)
  value.forEach((item, index) => text(item, `${location}[${index}]`))
}

/** Parentheses escaped for the prompt parser (`(` → `\(`, any backslashes before it folded into one). */
export function escapeParens(prompt: string): string {
  return prompt.replace(/\\*\(/g, "\\(").replace(/\\*\)/g, "\\)")
}

/** Python's `round(value, 9)` as the reference compiler applies it before flooring / ceiling. */
function pyRound9(value: number): number {
  const rounded = Number(value.toFixed(9))
  return Object.is(rounded, -0) ? 0 : rounded
}

function centeredBox(box: unknown, width: number, height: number, location: string): number[] {
  check(Array.isArray(box) && box.length === 4, `${location} must be [x1,y1,x2,y2]`)
  check(
    box.every((v) => typeof v === "number" && Number.isFinite(v)),
    `${location} must contain four finite numbers`,
  )
  const [x1, y1, x2, y2] = box as [number, number, number, number]
  check(x1 < x2 && y1 < y2, `${location} must have positive width and height`)
  check(0 <= x1 && x1 < x2 && x2 <= width && 0 <= y1 && y1 < y2 && y2 <= height, `${location} exceeds the ${width}x${height} canvas`)
  const halfW = Math.floor(width / 2)
  const halfH = Math.floor(height / 2)
  const result = [Math.floor(pyRound9(y1)) - halfH, Math.floor(pyRound9(x1)) - halfW, Math.ceil(pyRound9(y2)) - halfH, Math.ceil(pyRound9(x2)) - halfW] as [number, number, number, number]
  check(result[0] < result[2] && result[1] < result[3], `${location} is too small at export precision`)
  return result
}

/**
 * `JSON.parse` that refuses duplicate keys (the reference compiler's
 * `object_pairs_hook`), NaN / Infinity and numbers that overflow.
 */
export function parseUniqueJson(source: string): unknown {
  let pos = 0

  const skipWhitespace = () => {
    while (pos < source.length && /\s/.test(source[pos] ?? "")) pos++
  }

  const parseString = (): string => {
    const start = pos++
    while (pos < source.length) {
      if (source[pos] === "\\") pos += 2
      else if (source[pos++] === '"') return JSON.parse(source.slice(start, pos)) as string
    }
    throw new DraftError("Invalid JSON")
  }

  const parseValue = (): void => {
    skipWhitespace()
    if (source[pos] === "{") {
      pos++
      skipWhitespace()
      const keys = new Set<string>()
      if (source[pos] === "}") {
        pos++
        return
      }
      while (pos < source.length) {
        skipWhitespace()
        if (source[pos] !== '"') throw new DraftError("Invalid JSON")
        const key = parseString()
        check(!keys.has(key), `Duplicate JSON key: ${key}`)
        keys.add(key)
        skipWhitespace()
        check(source[pos++] === ":", "Invalid JSON")
        parseValue()
        skipWhitespace()
        if (source[pos] === "}") {
          pos++
          return
        }
        check(source[pos++] === ",", "Invalid JSON")
      }
      throw new DraftError("Invalid JSON")
    }
    if (source[pos] === "[") {
      pos++
      skipWhitespace()
      if (source[pos] === "]") {
        pos++
        return
      }
      while (pos < source.length) {
        parseValue()
        skipWhitespace()
        if (source[pos] === "]") {
          pos++
          return
        }
        check(source[pos++] === ",", "Invalid JSON")
      }
      throw new DraftError("Invalid JSON")
    }
    if (source[pos] === '"') {
      parseString()
      return
    }
    const match = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(source.slice(pos))
    check(match !== null, "Invalid JSON")
    const token = match[0]
    // Upstream's port also refuses true / false / null here (`Number("true")` is NaN); the
    // reference compiler parses them and the field checks refuse them: no valid draft has one.
    const literal = token === "true" || token === "false" || token === "null"
    check(literal || Number.isFinite(Number(token)), `Invalid JSON number: ${token}`)
    pos += token.length
  }

  parseValue()
  skipWhitespace()
  check(pos === source.length, "Invalid JSON")
  return JSON.parse(source)
}

const ROOT_FIELDS: ReadonlySet<string> = new Set(["width", "height", "bbox_coord_space", "bbox_format", "summary", "items"])
const SUMMARY_FIELDS: ReadonlySet<string> = new Set(["caption", "meta", "canvas", "palette"])
const ITEM_FIELDS: ReadonlySet<string> = new Set(["label", "caption", "bbox", "text", "tags", "palette"])
const ITEM_REQUIRED: ReadonlySet<string> = new Set(["label", "caption", "bbox"])

export const LABELS = ["object", "character", "text"] as const
export const CANVAS_TYPES = ["single", "multi_panel", "collage"] as const

/** The prompt for `input`: a draft object or its JSON text. Throws a {@link DraftError} naming the first problem. */
export function compilePrompt(input: unknown): string {
  const draft = typeof input === "string" ? parseUniqueJson(input) : input
  check(isObject(draft), "Draft must be a JSON object")

  if (has(draft, "caption")) {
    fields(draft, new Set(["caption"]), new Set(["caption"]), "Caption draft")
    text(draft.caption, "caption")
    return escapeParens(draft.caption)
  }

  fields(draft, ROOT_FIELDS, ROOT_FIELDS, "Layout draft")
  const { width, height } = draft
  for (const [name, value] of [
    ["width", width],
    ["height", height],
  ] as const) {
    check(Number.isInteger(value) && (value as number) > 0 && (value as number) % 16 === 0, `${name} must be a positive integer multiple of 16`)
  }
  const w = width as number
  const h = height as number
  check(draft.bbox_coord_space === "abs", "Draft bbox_coord_space must be abs")
  check(draft.bbox_format === "xyxy", "Draft bbox_format must be xyxy")

  const summary = draft.summary
  fields(summary, SUMMARY_FIELDS, new Set(["caption"]), "summary")
  text(summary.caption, "summary.caption")
  if ("canvas" in summary) {
    check((CANVAS_TYPES as readonly unknown[]).includes(summary.canvas), "summary.canvas must be single, multi_panel or collage")
  }
  for (const key of ["meta", "palette"]) {
    if (key in summary) strings(summary[key], `summary.${key}`)
  }

  const items = draft.items
  check(Array.isArray(items) && items.length > 0, "items must be a nonempty flat array")
  const compiledItems = items.map((item: unknown, index) => {
    const location = `items[${index}]`
    fields(item, ITEM_FIELDS, ITEM_REQUIRED, location)
    check((LABELS as readonly unknown[]).includes(item.label), `${location}.label must be object, character or text`)
    text(item.caption, `${location}.caption`)
    if (item.label === "text") check("text" in item, `${location}: text items require literal text`)
    if ("text" in item) text(item.text, `${location}.text`)
    for (const key of ["tags", "palette"]) {
      if (key in item) strings(item[key], `${location}.${key}`)
    }
    const compiled: Json = {}
    for (const [key, value] of Object.entries(item)) {
      compiled[key] = key === "bbox" ? centeredBox(value, w, h, `${location}.bbox`) : value
    }
    return compiled
  })

  const result = { summary, items: compiledItems, bbox_coord_space: "abs_centralized", bbox_format: "yxyx" }
  return escapeParens(JSON.stringify(result))
}
