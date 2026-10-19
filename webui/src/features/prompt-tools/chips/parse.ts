/**
 * A prompt as a list of chips (the tag view of the prompt tools) and back.
 *
 * The prompt is split at top-level commas and line breaks; `BREAK` / `AND`,
 * `/* … *\/` blocks and `#` / `//` line comments are chips of their own
 * (as Forge strips them: a line comment starts at any `#` or `//`).
 * Everything between two chips (commas, spaces, line breaks) is kept as a
 * separator string, so `serialize(parse(text)) === text` for every prompt,
 * and an edit only rewrites the chips it touches.
 *
 * A chip's weight is its effective attention: `(x)` 1.1, `((x))` 1.21,
 * `[x]` 1/1.1, `(x:1.3)` 1.3; a `<lora:name:w>`'s first number. A disabled
 * chip is a prompt comment (`/*tag, *\/`, stripped before generation when
 * `enable_prompt_comments` is on); it takes the separator after it along (or
 * the one before, at the end) so the remaining prompt has no stray comma.
 */

export type ChipKind = "tag" | "network" | "keyword" | "wildcard" | "comment"

export type Chip = {
  /** The chip's source text, exactly. */
  raw: string
  kind: ChipKind
  /** The text without weight syntax (a network's name; a disabled chip's content). */
  text: string
  /** Effective weight (1 without emphasis). */
  weight: number
  disabled: boolean
  /** Networks: `lora`, `lyco`, `hypernet`, …; also inside a comment. */
  network?: string
}

export type PromptDoc = {
  lead: string
  chips: Chip[]
  /** `seps[i]` is between `chips[i]` and `chips[i + 1]`. */
  seps: string[]
  tail: string
}

const OPEN = "([{<"
const CLOSE = ")]}>"
const ATTENTION_STEP = 1.1
const WEIGHT_TAIL = /^([\s\S]*?):\s*(-?(?:\d+\.?\d*|\.\d+))\s*$/
const NETWORK = /^<([A-Za-z_][\w-]*):([^<>]*)>$/
const NUMBER = /^\s*-?(?:\d+\.?\d*|\.\d+)\s*$/
const KEYWORD = /^(?:BREAK|AND)$/
const WILDCARD = /^__[^\s,]+__$/

type Unit = { from: number; to: number; special?: "comment" | "keyword" }

const isSpace = (c: string | undefined) => c !== undefined && /\s/.test(c)
const isBoundary = (c: string | undefined) => c === undefined || isSpace(c) || c === ","
/** A keyword ends at `j`: a boundary, or a comment (stripped before the prompt is read). */
const endsWord = (text: string, j: number) => isBoundary(text[j]) || text[j] === "#" || text.startsWith("//", j) || text.startsWith("/*", j)

/** Splits into top-level units (chips' ranges). */
function units(text: string): Unit[] {
  const found: Unit[] = []
  let depth = 0
  let start = -1
  let lastContent = -1
  const close = () => {
    if (start >= 0) found.push({ from: start, to: lastContent + 1 })
    start = -1
    lastContent = -1
  }
  let i = 0
  while (i < text.length) {
    const c = text[i]!
    if (depth === 0 && text.startsWith("/*", i)) {
      close()
      const end = text.indexOf("*/", i + 2)
      const to = end < 0 ? text.length : end + 2
      found.push({ from: i, to, special: "comment" })
      i = to
      continue
    }
    // Forge strips every `#` / `//` to the end of the line (escaped, in a
    // URL or inside brackets too): what follows on the line is not prompt
    if (c === "#" || text.startsWith("//", i)) {
      close()
      let to = text.indexOf("\n", i)
      if (to < 0) to = text.length
      while (to > i && isSpace(text[to - 1])) to--
      found.push({ from: i, to, special: "comment" })
      depth = 0
      i = to
      continue
    }
    if (depth === 0 && start < 0 && (text.startsWith("BREAK", i) || text.startsWith("AND", i))) {
      const word = text.startsWith("BREAK", i) ? "BREAK" : "AND"
      // right after a comment: the comment is stripped, so this starts the text
      const previous = found.at(-1)
      const afterComment = previous?.special === "comment" && previous.to === i
      if ((afterComment || isBoundary(text[i - 1])) && endsWord(text, i + word.length)) {
        found.push({ from: i, to: i + word.length, special: "keyword" })
        i += word.length
        continue
      }
    }
    if (depth === 0 && start >= 0 && isSpace(text[i - 1]) && (text.startsWith("BREAK", i) || text.startsWith("AND", i))) {
      const word = text.startsWith("BREAK", i) ? "BREAK" : "AND"
      if (endsWord(text, i + word.length)) {
        close()
        found.push({ from: i, to: i + word.length, special: "keyword" })
        i += word.length
        continue
      }
    }
    // an escape, except before a comment (`\#` is a `\` and a comment to Forge)
    if (c === "\\" && i + 1 < text.length && text[i + 1] !== "#" && !text.startsWith("//", i + 1)) {
      if (start < 0) start = i
      lastContent = i + 1
      i += 2
      continue
    }
    if (depth === 0 && (c === "," || c === "\n")) {
      close()
      i++
      continue
    }
    if (OPEN.includes(c)) depth++
    else if (CLOSE.includes(c) && depth > 0) depth--
    if (!isSpace(c)) {
      if (start < 0) start = i
      lastContent = i
    }
    i++
  }
  close()
  return found
}

/** Index of the bracket closing the one at `open` (escapes skipped), or -1. */
function matching(text: string, open: number): number {
  let depth = 0
  for (let i = open; i < text.length; i++) {
    const c = text[i]!
    if (c === "\\") {
      i++
      continue
    }
    if (OPEN.includes(c)) depth++
    else if (CLOSE.includes(c)) {
      depth--
      if (depth === 0) return i
    }
  }
  return -1
}

/** Whether `inner` has a top-level `:` or `|` (a schedule / alternation, not a tag). */
function hasTopLevel(inner: string, chars: string): boolean {
  let depth = 0
  for (let i = 0; i < inner.length; i++) {
    const c = inner[i]!
    if (c === "\\") {
      i++
      continue
    }
    if (OPEN.includes(c)) depth++
    else if (CLOSE.includes(c)) depth--
    else if (depth === 0 && chars.includes(c)) return true
  }
  return false
}

const round = (value: number) => Math.round(value * 10_000) / 10_000

/** The text and weight of a (non-comment) chip's source. */
function weighted(raw: string): Pick<Chip, "kind" | "text" | "weight" | "network"> {
  if (KEYWORD.test(raw)) return { kind: "keyword", text: raw, weight: 1 }
  const network = NETWORK.exec(raw)
  if (network) {
    const [name = "", first] = network[2]!.split(":")
    const weight = first !== undefined && NUMBER.test(first) ? Number(first) : 1
    return { kind: "network", text: name, weight, network: network[1]!.toLowerCase() }
  }
  if (WILDCARD.test(raw)) return { kind: "wildcard", text: raw, weight: 1 }
  let text = raw
  let weight = 1
  for (;;) {
    if (text.startsWith("(") && matching(text, 0) === text.length - 1) {
      const inner = text.slice(1, -1)
      const explicit = WEIGHT_TAIL.exec(inner)
      if (explicit && !hasTopLevel(explicit[1]!, ":")) {
        weight *= Number(explicit[2])
        text = explicit[1]!.trim()
      } else {
        weight *= ATTENTION_STEP
        text = inner.trim()
      }
      continue
    }
    if (text.startsWith("[") && matching(text, 0) === text.length - 1 && !hasTopLevel(text.slice(1, -1), ":|")) {
      weight /= ATTENTION_STEP
      text = text.slice(1, -1).trim()
      continue
    }
    break
  }
  return { kind: "tag", text, weight: round(weight) }
}

/** The separator characters a disabled chip took along (`/*, x*\/`, `/*x, *\/`). */
function commentParts(raw: string): { lead: string; body: string; trail: string } {
  const inner = raw.startsWith("/*") ? raw.slice(2, raw.endsWith("*/") && raw.length >= 4 ? -2 : undefined) : raw
  const lead = /^[\s,]*/.exec(inner)![0]
  const rest = inner.slice(lead.length)
  const trail = /[\s,]*$/.exec(rest)![0]
  return { lead, body: rest.slice(0, rest.length - trail.length), trail }
}

function chipOf(raw: string, special?: Unit["special"]): Chip {
  if (special === "keyword") return { raw, kind: "keyword", text: raw, weight: 1, disabled: false }
  if (special === "comment") {
    if (raw.startsWith("/*")) {
      const { body } = commentParts(raw)
      // a single chip commented out (a disabled chip), else a commented region
      const inner = units(body)
      if (inner.length === 1 && !inner[0]!.special && inner[0]!.from === 0 && inner[0]!.to === body.length) {
        return { ...weighted(body), raw, disabled: true }
      }
      return { raw, kind: "comment", text: body, weight: 1, disabled: true }
    }
    return { raw, kind: "comment", text: raw.replace(/^(#|\/\/)\s*/, ""), weight: 1, disabled: true }
  }
  return { ...weighted(raw), raw, disabled: false }
}

export function parsePrompt(text: string): PromptDoc {
  const found = units(text)
  if (found.length === 0) return { lead: text, chips: [], seps: [], tail: "" }
  const chips = found.map((unit) => chipOf(text.slice(unit.from, unit.to), unit.special))
  const seps = found.slice(1).map((unit, index) => text.slice(found[index]!.to, unit.from))
  return { lead: text.slice(0, found[0]!.from), chips, seps, tail: text.slice(found.at(-1)!.to) }
}

export function serializePrompt(doc: PromptDoc): string {
  let out = doc.lead
  doc.chips.forEach((chip, index) => {
    out += chip.raw
    if (index < doc.seps.length) out += doc.seps[index]
  })
  return out + doc.tail
}

// region Edits (each returns the new prompt text)

export const DEFAULT_SEPARATOR = ", "

function formatNumber(value: number): string {
  return String(Math.round(value * 100) / 100)
}

/** A chip's source with a new weight and/or text (enabled form). */
export function formatChip(chip: Pick<Chip, "kind" | "text" | "weight" | "network" | "raw" | "disabled">, changes: { text?: string; weight?: number }): string {
  const text = changes.text ?? chip.text
  const weight = changes.weight ?? chip.weight
  if (chip.kind === "network") {
    const source = chip.disabled ? commentParts(chip.raw).body : chip.raw
    const match = NETWORK.exec(source)
    const args = match ? match[2]!.split(":").slice(1) : []
    if (args.length > 0 && NUMBER.test(args[0]!)) args[0] = formatNumber(weight)
    else args.unshift(formatNumber(weight))
    return `<${chip.network ?? "lora"}:${text}:${args.join(":")}>`
  }
  if (chip.kind !== "tag") return text
  const rounded = Math.round(weight * 100) / 100
  return rounded === 1 ? text : `(${text}:${formatNumber(rounded)})`
}

/** A disabled chip's new source keeps it commented out (with the separators it took along). */
function keepDisabled(chip: Chip, raw: string): string {
  if (!chip.disabled || chip.kind === "comment") return raw
  const { lead, trail } = commentParts(chip.raw)
  return `/*${lead}${raw}${trail}*/`
}

function withChip(doc: PromptDoc, index: number, raw: string): string {
  const chips = doc.chips.slice()
  const chip = chips[index]
  if (!chip) return serializePrompt(doc)
  chips[index] = { ...chip, raw: keepDisabled(chip, raw) }
  return serializePrompt({ ...doc, chips })
}

export function setChipWeight(doc: PromptDoc, index: number, weight: number): string {
  const chip = doc.chips[index]
  if (!chip || (chip.kind !== "tag" && chip.kind !== "network")) return serializePrompt(doc)
  return withChip(doc, index, formatChip(chip, { weight }))
}

export function setChipText(doc: PromptDoc, index: number, text: string): string {
  const chip = doc.chips[index]
  if (!chip) return serializePrompt(doc)
  const trimmed = text.trim()
  if (!trimmed) return removeChip(doc, index)
  if (chip.kind === "tag" || chip.kind === "network") return withChip(doc, index, formatChip(chip, { text: trimmed }))
  return withChip(doc, index, trimmed)
}

/**
 * Replaces the text of several chips at once (indices of `doc`, so one edit
 * never shifts another, even when a new text holds commas and later parses
 * as more chips). Weights and the disabled state are kept.
 */
export function setChipTexts(doc: PromptDoc, edits: readonly { index: number; text: string }[]): string {
  const chips = doc.chips.slice()
  for (const { index, text } of edits) {
    const chip = doc.chips[index]
    const trimmed = text.trim()
    if (!chip || !trimmed) continue
    const raw = chip.kind === "tag" || chip.kind === "network" ? formatChip(chip, { text: trimmed }) : trimmed
    chips[index] = { ...chip, raw: keepDisabled(chip, raw) }
  }
  return serializePrompt({ ...doc, chips })
}

/** Whether a chip can be disabled / enabled (a comment cannot hold a `*\/`: they do not nest). */
export function canToggle(chip: Chip): boolean {
  return chip.kind !== "keyword" && (chip.disabled || !chip.raw.includes("*/"))
}

/** Disables (comments out) or enables a chip. */
export function toggleChip(doc: PromptDoc, index: number): string {
  const chip = doc.chips[index]
  if (!chip || !canToggle(chip)) return serializePrompt(doc)
  const chips = doc.chips.slice()
  const seps = doc.seps.slice()
  let { lead, tail } = doc
  const last = index === chips.length - 1
  if (chip.disabled) {
    const parts = commentParts(chip.raw)
    chips[index] = { ...chip, raw: parts.body }
    if (index > 0) seps[index - 1] = seps[index - 1] + parts.lead
    else lead += parts.lead
    if (!last) seps[index] = parts.trail + seps[index]
    else tail = parts.trail + tail
  } else if (!last) {
    chips[index] = { ...chip, raw: `/*${chip.raw}${seps[index]}*/` }
    seps[index] = ""
  } else if (index > 0) {
    // the separator before goes in, but not past a line break (a line
    // comment before it would swallow the opening `/*`)
    const before = seps[index - 1]!
    const keep = before.lastIndexOf("\n") + 1
    chips[index] = { ...chip, raw: `/*${before.slice(keep)}${chip.raw}*/` }
    seps[index - 1] = before.slice(0, keep)
  } else {
    chips[index] = { ...chip, raw: `/*${chip.raw}*/` }
  }
  return serializePrompt({ lead, chips, seps, tail })
}

/** Removes a chip and one separator next to it. */
export function removeChip(doc: PromptDoc, index: number): string {
  if (!doc.chips[index]) return serializePrompt(doc)
  const chips = doc.chips.filter((_, i) => i !== index)
  const seps = doc.seps.slice()
  if (index < seps.length) seps.splice(index, 1)
  else if (index > 0) seps.splice(index - 1, 1)
  if (!chips.length) return ""
  return serializePrompt({ ...doc, chips, seps })
}

/** Moves a chip to `to` (an index in the list without it); separators stay in place. */
export function moveChip(doc: PromptDoc, from: number, to: number): string {
  if (!doc.chips[from] || from === to) return serializePrompt(doc)
  const chips = doc.chips.slice()
  const [chip] = chips.splice(from, 1)
  chips.splice(Math.max(0, Math.min(to, chips.length)), 0, chip!)
  return serializePrompt({ ...doc, chips })
}

/** Appends text (one or more tags) as new chips. */
export function appendChips(doc: PromptDoc, text: string, separator = DEFAULT_SEPARATOR): string {
  const added = text.trim()
  if (!added) return serializePrompt(doc)
  const current = serializePrompt(doc)
  if (!doc.chips.length) return `${current.trimEnd()}${current.trim() ? separator : ""}${added}`
  const head = serializePrompt({ ...doc, tail: "" })
  // a trailing comma already separates
  const tailSeparator = /,\s*$/.test(doc.tail) ? doc.tail.replace(/\s*$/, " ") : separator
  return `${head}${tailSeparator}${added}`
}

/** Whether a tag (by text, ignoring case, spaces vs underscores and escapes) is in the prompt and enabled. */
export function tagKey(text: string): string {
  return text.replace(/\\(.)/g, "$1").trim().toLowerCase().replace(/[\s_]+/g, "_")
}

export function findTag(doc: PromptDoc, tag: string): number {
  const key = tagKey(tag)
  return doc.chips.findIndex((chip) => chip.kind === "tag" && !chip.disabled && tagKey(chip.text) === key)
}

/**
 * The prompt with tidy separators (Prompt All-in-One's prompt formatting):
 * one ", " between the tags of a line, line breaks kept (at most one empty
 * line, a comma before a break where there was one), repeated commas merged,
 * no leading or trailing commas and spaces. The chips themselves (weights,
 * networks, comments, `BREAK`) are not touched.
 */
export function tidyPrompt(text: string): string {
  const doc = parsePrompt(text)
  if (doc.chips.length === 0) return /^[\s,]*$/.test(text) ? "" : text
  const seps = doc.seps.map((sep) => {
    const breaks = sep.split("\n").length - 1
    const comma = sep.includes(",")
    if (breaks > 0) return `${comma ? "," : ""}${"\n".repeat(Math.min(breaks, 2))}`
    return comma ? DEFAULT_SEPARATOR : " "
  })
  return serializePrompt({ lead: "", chips: doc.chips, seps, tail: "" })
}

/** Adds the tag at the end, or removes it when it is there (quick tag groups). */
export function toggleTag(text: string, tag: string, separator = DEFAULT_SEPARATOR): string {
  const doc = parsePrompt(text)
  const index = findTag(doc, tag)
  return index >= 0 ? removeChip(doc, index) : appendChips(doc, tag, separator)
}

// endregion

/** Display form of a chip's text: escapes removed. */
export function displayText(text: string): string {
  return text.replace(/\\(.)/g, "$1")
}
