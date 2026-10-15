/**
 * Ctrl/Cmd+↑/↓ attention editing, ported from javascript/edit-attention.js as a
 * pure function: (text, selection) → (text, selection).
 *
 * With an empty selection it selects the enclosing `<…>`, `(…)` or `[…]`
 * block, else the word under the cursor; wraps plain text as `(text:1.0)`,
 * converts legacy `((x))` / `[x]` emphasis to an explicit weight, steps the
 * weight, and unwraps `(x:1.0)` when the weight returns to 1.
 */

export type AttentionOptions = {
  /** Step for `(text:w)`. */
  precisionAttention: number
  /** Step for `<lora:name:w>` and other extra networks. */
  precisionExtra: number
  /** Characters that end a word. */
  delimiters: string
  /** Whitespace delimiters by name: "Tab", "Carriage Return", "Line Feed". */
  whitespaceDelimiters: readonly string[]
}

export const DEFAULT_ATTENTION_OPTIONS: AttentionOptions = {
  precisionAttention: 0.1,
  precisionExtra: 0.05,
  delimiters: ".,\\/!?%^*;:{}=`~() ",
  whitespaceDelimiters: ["Tab", "Carriage Return", "Line Feed"],
}

const WHITESPACE: Record<string, string> = { Tab: "\t", "Carriage Return": "\r", "Line Feed": "\n" }

export type Edit = { text: string; from: number; to: number }

export function editAttention(
  input: string,
  from: number,
  to: number,
  direction: 1 | -1,
  options: AttentionOptions = DEFAULT_ATTENTION_OPTIONS,
): Edit | null {
  let text = input
  let start = Math.min(from, to)
  let end = Math.max(from, to)

  const selectBlock = (open: string, close: string): boolean => {
    if (start !== end) return false
    const before = text.slice(0, start)
    const openAt = before.lastIndexOf(open)
    if (openAt === -1) return false
    const closeBefore = before.lastIndexOf(close)
    if (closeBefore !== -1 && closeBefore > openAt) return false
    const after = text.slice(start)
    const closeAt = after.indexOf(close)
    if (closeAt === -1) return false
    const openAfter = after.indexOf(open)
    if (openAfter !== -1 && openAfter < closeAt) return false
    const content = text.slice(openAt + 1, start + closeAt)
    start = openAt + 1
    end = /.*:-?[\d.]+/s.test(content) ? start + content.lastIndexOf(":") : start + content.length
    return true
  }

  const selectWord = () => {
    if (start !== end) return
    let delimiters = options.delimiters
    for (const name of options.whitespaceDelimiters) delimiters += WHITESPACE[name] ?? ""
    while (start > 0 && !delimiters.includes(text.charAt(start - 1))) start--
    while (end < text.length && !delimiters.includes(text.charAt(end))) end++
    while (text[start] === " " && start < end) start++
    while (text[end - 1] === " " && end > start) end--
  }

  if (!selectBlock("<", ">") && !selectBlock("(", ")") && !selectBlock("[", "]")) selectWord()

  let close = ")"
  let delta = options.precisionAttention
  const opening = start > 0 ? text[start - 1] : ""
  const closing = text[end]

  if (opening === "<") {
    close = ">"
    delta = options.precisionExtra
  } else if ((opening === "(" && closing === ")") || (opening === "[" && closing === "]")) {
    // Legacy emphasis: count nested brackets and convert to an explicit weight.
    let depth = 0
    while (text[start - depth - 1] === opening && text[end + depth] === closing) depth++
    let weight = opening === "[" ? (1 / 1.1) ** depth : 1.1 ** depth
    weight = Number((Math.round(weight / options.precisionAttention) * options.precisionAttention).toPrecision(12))
    text = `${text.slice(0, start - depth)}(${text.slice(start, end)}:${weight})${text.slice(end + depth)}`
    start -= depth - 1
    end -= depth - 1
  } else if (opening !== "(") {
    while (end > start && text[end - 1] === " ") end--
    if (start === end) return null
    text = `${text.slice(0, start)}(${text.slice(start, end)}:1.0)${text.slice(end)}`
    start++
    end++
  }

  if (text[end] !== ":") return null
  const weightLength = text.slice(end + 1).indexOf(close) + 1
  const parsed = Number.parseFloat(text.slice(end + 1, end + weightLength))
  if (Number.isNaN(parsed)) return null

  const next = Number.parseFloat((parsed + direction * delta).toPrecision(12))
  const formatted = Number.isInteger(next) ? `${next}.0` : String(next)

  if (close === ")" && next === 1) {
    // Back to neutral: drop the parentheses and the weight.
    const closeAt = text.slice(end).indexOf(")")
    text = text.slice(0, start - 1) + text.slice(start, end) + text.slice(end + closeAt + 1)
    start--
    end--
  } else {
    text = text.slice(0, end + 1) + formatted + text.slice(end + weightLength)
  }
  return { text, from: start, to: end }
}
