/**
 * What is being typed at the cursor: a booru tag, an extra network after
 * `<`, a wildcard after `__`, or a wildcard's value after `__name__`. Pure;
 * positions are document offsets.
 */
import type { CompletionMode } from "../api"

export type CompletionTarget = {
  mode: CompletionMode
  /** What the server searches for (prefix syntax removed). */
  query: string
  /** The range a completion replaces. */
  from: number
  to: number
  /** mode `wildcard-value`: the wildcard's name. */
  wildcard?: string
  /** mode `tag`: the typed `@` artist prefix (Anima), inside the replaced range but not searched. */
  prefix?: "@"
}

/** Anima's `@artist` prefix (`@_@` is a tag, not a prefix). */
const ARTIST_PREFIX = /^@(?=[\p{L}\p{N}])/u

/** How far back the scan looks (prompts are short; a tag is shorter still). */
const LOOKBACK = 400

/** Characters that end a tag (an escaped one, `\(`, does not). */
const DELIMITERS = new Set([",", "\n", "(", ")", "[", "]", "{", "}", "|", "<", ">", ":"])
const CLOSERS = /^[)\]}>|:]/
const NUMBER = /^-?(?:\d+\.?\d*|\.\d+)$/
const KEYWORDS = new Set(["BREAK", "AND"])

const NETWORK_PREFIXES: [RegExp, CompletionMode][] = [
  [/^(?:lora|lyco|l):/i, "lora"],
  [/^(?:e|embedding|ti):/i, "embedding"],
  [/^(?:c|chant):/i, "chant"],
]

function isEscaped(text: string, index: number): boolean {
  let slashes = 0
  for (let i = index - 1; i >= 0 && text[i] === "\\"; i--) slashes++
  return slashes % 2 === 1
}

/**
 * Inside a `/* … *\/` block or after `#` / `//` on its line: what Forge's
 * prompt comments strip (`modules/processing_scripts/comments.py`: every
 * `#` and `//`, escaped or in a URL too).
 */
function inComment(before: string): boolean {
  const open = before.lastIndexOf("/*")
  if (open >= 0 && before.indexOf("*/", open) < 0) return true
  const line = before.slice(before.lastIndexOf("\n") + 1).replace(/\/\*[\s\S]*?\*\//g, "")
  return /\/\/|#/.test(line)
}

/**
 * The completion target at `cursor` in `doc`, or null (nothing to complete:
 * a weight, a keyword, a comment, a closed network tag, …).
 */
export function completionTarget(doc: string, cursor: number, options: { wildcardWrap?: string } = {}): CompletionTarget | null {
  const start = Math.max(0, cursor - LOOKBACK)
  const before = doc.slice(start, cursor)
  const wrap = options.wildcardWrap ?? "__"
  if (inComment(before)) return null

  // `__name__word`: a value of that wildcard
  const escapedWrap = wrap.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const value = new RegExp(`(?:^|[\\s,(\\[{|])(${escapedWrap}([^\\s,_][^\\s,]*?)${escapedWrap})([^\\s,]*)$`).exec(before)
  if (value) {
    const from = start + before.length - value[1]!.length - value[3]!.length
    return { mode: "wildcard-value", query: value[3]!, from, to: cursor, wildcard: value[2]! }
  }
  // `__name`: a wildcard file
  const file = new RegExp(`(?:^|[\\s,(\\[{|])${escapedWrap}([^\\s,]*)$`).exec(before)
  if (file && !file[1]!.includes(wrap)) {
    return { mode: "wildcard", query: file[1]!, from: cursor - file[1]!.length - wrap.length, to: cursor }
  }

  // `<lora:name`, `<e:name`, `<c:name`, `<name`
  const network = /<([^<>,\s]*)$/.exec(before)
  if (network) {
    const raw = network[1]!
    const from = cursor - raw.length - 1
    for (const [prefix, mode] of NETWORK_PREFIXES) {
      const match = prefix.exec(raw)
      if (match) {
        const query = raw.slice(match[0].length)
        // a second colon: the weight is being typed
        return query.includes(":") ? null : { mode, query, from, to: cursor }
      }
    }
    if (raw.includes(":")) return null
    // still typing a prefix ("<lo"): everything
    const typingPrefix = raw.length > 0 && ["lora", "lyco", "chant", "embedding"].some((word) => word.startsWith(raw.toLowerCase()) && raw.length < word.length)
    return { mode: "network", query: typingPrefix ? "" : raw, from, to: cursor }
  }

  // a booru tag: back to the last unescaped delimiter
  let i = before.length
  while (i > 0) {
    const char = before[i - 1]!
    if (DELIMITERS.has(char) && !isEscaped(before, i - 1)) break
    i--
  }
  // a closed `/* … */` before the tag on its line is not part of it
  const closed = before.lastIndexOf("*/")
  if (closed >= i) i = closed + 2
  let word = before.slice(i)
  const leading = word.length - word.trimStart().length
  word = word.trimStart()
  if (!word || NUMBER.test(word.trim()) || KEYWORDS.has(word.trim())) return null
  // "BREAK long ha": the keyword is not part of the tag
  const keyword = /^(?:BREAK|AND)\s+/.exec(word)
  const skip = keyword ? keyword[0].length : 0
  word = word.slice(skip)
  if (!word.trim()) return null
  const from = start + i + leading + skip
  if (ARTIST_PREFIX.test(word)) return { mode: "tag", query: word.slice(1), from, to: cursor, prefix: "@" }
  return { mode: "tag", query: word, from, to: cursor }
}

/**
 * The end of the token the cursor is in (a completion replaces the rest of
 * the word after the cursor too): up to the next delimiter, trailing spaces
 * excluded. Network targets also swallow their closing `>`.
 */
export function tokenEnd(doc: string, target: CompletionTarget): number {
  let end = target.to
  if (target.mode === "tag" || target.mode === "wildcard-value") {
    while (end < doc.length && !(DELIMITERS.has(doc[end]!) && !isEscaped(doc, end))) end++
    while (end > target.to && /\s/.test(doc[end - 1]!)) end--
    return end
  }
  if (target.mode === "wildcard") {
    while (end < doc.length && /[^\s,]/.test(doc[end]!)) end++
    return end
  }
  while (end < doc.length && /[^<>,\s]/.test(doc[end]!)) end++
  if (doc[end] === ">") end++
  return end
}

/** The text right after `to`: what decides the separator after an inserted tag. */
export function followingText(doc: string, to: number): string {
  return doc.slice(to, to + 8)
}

export { CLOSERS }
