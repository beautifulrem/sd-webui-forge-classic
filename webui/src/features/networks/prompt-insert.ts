/**
 * Adding a network card's text to a prompt, or taking it out again (a second
 * click toggles, like the old extra-networks cards).
 *
 * - `<type:name:args…>` tags (`<lora:style:0.8>`, `<hypernet:x:1>`) match by
 *   type and name whatever their weight, so a tag whose weight the user edited
 *   is still removed. Text the card added right after the tag (a LoRA's
 *   activation text) goes with it.
 * - Other text (an embedding name, negative text) matches as a whole word.
 * - The separator (`extra_networks_add_text_separator`) joins an added text to
 *   what is there, and is removed with it.
 */

const TAG = /<([^:<>]+):([^:<>]+)(?::[^<>]*)?>/g
const LEADING_TAG = /^<([^:<>]+):([^:<>]+)(?::[^<>]*)?>/

type Tag = { type: string; name: string; rest: string }

/** The network tag a card's text starts with, and the text after it. */
export function parseTag(text: string): Tag | null {
  const match = LEADING_TAG.exec(text.trim())
  if (!match) return null
  return { type: match[1]!.toLowerCase(), name: match[2]!, rest: text.trim().slice(match[0].length) }
}

const isWordChar = (char: string | undefined) => char !== undefined && /[\p{L}\p{N}_]/u.test(char)

/** Every `[start, end)` of `needle` in `text` that stands as a whole word at its word-character ends. */
function wholeMatches(text: string, needle: string): [number, number][] {
  const out: [number, number][] = []
  if (!needle) return out
  let from = 0
  for (;;) {
    const at = text.indexOf(needle, from)
    if (at === -1) return out
    const end = at + needle.length
    const startOk = !isWordChar(needle[0]) || !isWordChar(text[at - 1])
    const endOk = !isWordChar(needle.at(-1)) || !isWordChar(text[end])
    if (startOk && endOk) out.push([at, end])
    from = at + Math.max(1, needle.length)
  }
}

/** Ranges of the network tags of `type:name` in `text`. */
function tagMatches(text: string, tag: Tag): [number, number][] {
  const out: [number, number][] = []
  for (const match of text.matchAll(TAG)) {
    if (match[1]!.toLowerCase() === tag.type && match[2] === tag.name) out.push([match.index, match.index + match[0].length])
  }
  return out
}

/** `text` with `[start, end)` cut out, plus the separator before it (or, at the start, after it). */
function cut(text: string, start: number, end: number, separator: string): string {
  let before = text.slice(0, start)
  let after = text.slice(end)
  if (separator && before.endsWith(separator)) before = before.slice(0, -separator.length)
  else if (separator.trim() && before.trimEnd().endsWith(separator.trim())) before = before.trimEnd().slice(0, -separator.trim().length)
  else if (!before.trim()) {
    if (separator && after.startsWith(separator)) after = after.slice(separator.length)
    else if (separator.trim() && after.trimStart().startsWith(separator.trim())) after = after.trimStart().slice(separator.trim().length)
    after = after.replace(/^[ \t]+/, "")
  }
  return before + after
}

/** Removes `text` (see the module comment); `null` when it is not in `prompt`. */
export function removeFromPrompt(prompt: string, text: string, separator: string): string | null {
  const wanted = text.trim()
  if (!wanted) return null
  const tag = parseTag(wanted)
  let ranges: [number, number][]
  if (tag) {
    ranges = tagMatches(prompt, tag).map(([start, end]) => {
      const rest = tag.rest.trim()
      if (!rest) return [start, end]
      // the activation text the card wrote after the tag
      const tail = prompt.slice(end)
      const gap = tail.length - tail.trimStart().length
      return tail.trimStart().startsWith(rest) ? [start, end + gap + rest.length] : [start, end]
    })
  } else {
    ranges = wholeMatches(prompt, wanted)
  }
  if (ranges.length === 0) return null
  let out = prompt
  for (const [start, end] of ranges.toReversed()) out = cut(out, start, end, separator)
  return out
}

/** `prompt` with `text` appended after the separator (no separator into an empty prompt or after one). */
export function appendToPrompt(prompt: string, text: string, separator: string): string {
  const wanted = text.trim()
  if (!prompt.trim()) return wanted
  if (separator && prompt.endsWith(separator)) return prompt + wanted
  const bare = separator.trim()
  if (bare && prompt.trimEnd().endsWith(bare)) return `${prompt.trimEnd()}${separator.slice(separator.indexOf(bare) + bare.length) || " "}${wanted}`
  if (!bare && /\s$/.test(prompt)) return prompt + wanted
  return prompt + separator + wanted
}

/** Adds `text`, or removes it when it is there. */
export function togglePromptText(prompt: string, text: string, separator: string): { value: string; added: boolean } {
  const removed = removeFromPrompt(prompt, text, separator)
  if (removed !== null) return { value: removed, added: false }
  return { value: appendToPrompt(prompt, text, separator), added: true }
}

/** Whether the card's text is in the prompt (its tag, or the whole text). */
export function isInPrompt(prompt: string, text: string | null | undefined): boolean {
  const wanted = text?.trim()
  if (!wanted || !prompt) return false
  const tag = parseTag(wanted)
  return tag ? tagMatches(prompt, tag).length > 0 : wholeMatches(prompt, wanted).length > 0
}
